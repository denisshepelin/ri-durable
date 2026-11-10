//! Drives one streaming model request: sends it, parses the event stream, and yields [`AssistantEvent`]s.

use std::collections::{BTreeMap, VecDeque};
use std::pin::Pin;

use bytes::Bytes;
use futures_util::{Stream, StreamExt, stream};
use serde_json::Value;

use crate::sse::{SseEvent, SseParser};
use crate::types::{AssistantEvent, AssistantMessage, Content, Context, JsonObject, StopReason, ThinkingLevel, now_ms};
use crate::{Auth, Error, Model, api};

#[derive(Debug, Clone, Default)]
pub struct StreamOptions {
    /// Requested thinking level; clamped to what the model supports. `None` is `Off`.
    pub thinking: Option<ThinkingLevel>,
    pub max_tokens: Option<u64>,
    pub temperature: Option<f64>,
    /// Stable per-conversation id, used by providers for prompt caching and session affinity.
    pub session_id: Option<String>,
    /// Extra request headers. They override model and auth headers.
    pub headers: BTreeMap<String, String>,
}

impl StreamOptions {
    pub(crate) fn thinking_level(&self) -> ThinkingLevel {
        self.thinking.unwrap_or(ThinkingLevel::Off)
    }
}

/// A fully built provider request. Building is pure, so requests can be inspected and tested without a network.
#[derive(Debug, Clone, PartialEq)]
pub struct HttpRequest {
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Value,
}

impl HttpRequest {
    pub(crate) fn new(url: String, body: Value) -> Self {
        HttpRequest {
            url,
            headers: Vec::new(),
            body,
        }
    }

    /// Sets a header, replacing an earlier one of the same name (case-insensitive).
    pub(crate) fn header(mut self, name: &str, value: impl Into<String>) -> Self {
        self.headers
            .retain(|(existing, _)| !existing.eq_ignore_ascii_case(name));
        self.headers.push((name.to_owned(), value.into()));
        self
    }

    pub(crate) fn headers<'a>(self, headers: impl IntoIterator<Item = (&'a String, &'a String)>) -> Self {
        headers
            .into_iter()
            .fold(self, |request, (name, value)| request.header(name, value.clone()))
    }
}

/// Why a stream ended without the provider's own terminal event.
#[derive(Debug, Clone)]
pub struct Failure {
    pub message: String,
    pub status: Option<u16>,
}

/// Turns one protocol's server-sent events into neutral events and the final message.
pub trait Decoder: Send {
    fn push(&mut self, event: &SseEvent, out: &mut Vec<AssistantEvent>);
    /// True once the provider's terminal event arrived; the body is not read further.
    fn is_done(&self) -> bool;
    fn finish(self: Box<Self>, failure: Option<Failure>) -> AssistantMessage;
}

/// The message under construction, shared by the decoders.
pub(crate) struct Draft {
    pub message: AssistantMessage,
    /// Raw argument JSON of the tool calls still streaming, by content index.
    arguments: BTreeMap<usize, String>,
}

impl Draft {
    pub fn new(model: &Model) -> Self {
        Draft {
            message: AssistantMessage {
                content: Vec::new(),
                api: model.api.clone(),
                provider: model.provider.clone(),
                model: model.id.clone(),
                usage: Default::default(),
                stop_reason: StopReason::Stop,
                response_id: None,
                error_message: None,
                error_status: None,
                timestamp: now_ms(),
            },
            arguments: BTreeMap::new(),
        }
    }

    pub fn push(&mut self, block: Content) -> usize {
        self.message.content.push(block);
        self.message.content.len() - 1
    }

    pub fn text_delta(&mut self, index: usize, delta: &str, out: &mut Vec<AssistantEvent>) {
        if delta.is_empty() {
            return;
        }
        match self.message.content.get_mut(index) {
            Some(Content::Text { text, .. }) => {
                text.push_str(delta);
                out.push(AssistantEvent::TextDelta {
                    index,
                    delta: delta.to_owned(),
                });
            }
            Some(Content::Thinking { thinking, .. }) => {
                thinking.push_str(delta);
                out.push(AssistantEvent::ThinkingDelta {
                    index,
                    delta: delta.to_owned(),
                });
            }
            _ => {}
        }
    }

    pub fn tool_call_start(&mut self, id: String, name: String, out: &mut Vec<AssistantEvent>) -> usize {
        let index = self.push(Content::ToolCall {
            id: id.clone(),
            name: name.clone(),
            arguments: JsonObject::new(),
        });
        self.arguments.insert(index, String::new());
        out.push(AssistantEvent::ToolCallStart { index, id, name });
        index
    }

    pub fn tool_call_delta(&mut self, index: usize, delta: &str, out: &mut Vec<AssistantEvent>) {
        if let Some(raw) = self.arguments.get_mut(&index).filter(|_| !delta.is_empty()) {
            raw.push_str(delta);
            out.push(AssistantEvent::ToolCallDelta {
                index,
                delta: delta.to_owned(),
            });
        }
    }

    /// Closes a streaming tool call. `final_arguments` replaces the accumulated text when the provider repeats it.
    pub fn tool_call_end(&mut self, index: usize, final_arguments: Option<&str>, out: &mut Vec<AssistantEvent>) {
        let Some(raw) = self.arguments.remove(&index) else {
            return;
        };
        let parsed = parse_arguments(final_arguments.unwrap_or(&raw));
        if let Some(Content::ToolCall { arguments, .. }) = self.message.content.get_mut(index) {
            *arguments = parsed;
        }
        out.push(AssistantEvent::ToolCallEnd { index });
    }

    pub fn open_tool_calls(&self) -> Vec<usize> {
        self.arguments.keys().copied().collect()
    }

    pub fn has_tool_calls(&self) -> bool {
        self.message
            .content
            .iter()
            .any(|block| matches!(block, Content::ToolCall { .. }))
    }

    /// Closes what is still open and applies a failure, if any.
    pub fn finish(mut self, failure: Option<Failure>) -> AssistantMessage {
        let mut ignored = Vec::new();
        for index in self.open_tool_calls() {
            self.tool_call_end(index, None, &mut ignored);
        }
        if let Some(failure) = failure {
            self.message.stop_reason = StopReason::Error;
            self.message.error_message = Some(failure.message);
            self.message.error_status = failure.status;
        } else if self.message.stop_reason == StopReason::Stop && self.has_tool_calls() {
            self.message.stop_reason = StopReason::ToolUse;
        }
        self.message
    }
}

fn parse_arguments(raw: &str) -> JsonObject {
    match serde_json::from_str::<Value>(raw) {
        Ok(Value::Object(arguments)) => arguments,
        _ => JsonObject::new(),
    }
}

type Body = Pin<Box<dyn Stream<Item = reqwest::Result<Bytes>> + Send>>;

enum Phase {
    Send(Box<Result<HttpRequest, Error>>),
    Body(Body, SseParser),
    End,
}

struct Run {
    client: reqwest::Client,
    phase: Phase,
    decoder: Option<Box<dyn Decoder>>,
    queue: VecDeque<AssistantEvent>,
}

impl Run {
    fn end(&mut self, failure: Option<Failure>) {
        if let Some(decoder) = self.decoder.take() {
            self.queue.push_back(AssistantEvent::Done(decoder.finish(failure)));
        }
        self.phase = Phase::End;
    }

    async fn step(&mut self) {
        match std::mem::replace(&mut self.phase, Phase::End) {
            Phase::Send(request) => match send(&self.client, *request).await {
                Ok(body) => self.phase = Phase::Body(body, SseParser::default()),
                Err(failure) => self.end(Some(failure)),
            },
            Phase::Body(mut body, mut parser) => match body.next().await {
                Some(Ok(chunk)) => {
                    let mut out = Vec::new();
                    if let Some(decoder) = self.decoder.as_mut() {
                        for event in parser.push(&chunk) {
                            decoder.push(&event, &mut out);
                            if decoder.is_done() {
                                break;
                            }
                        }
                    }
                    self.queue.extend(out);
                    if self.decoder.as_ref().is_none_or(|decoder| decoder.is_done()) {
                        self.end(None);
                    } else {
                        self.phase = Phase::Body(body, parser);
                    }
                }
                Some(Err(error)) => self.end(Some(Failure {
                    message: error.to_string(),
                    status: None,
                })),
                None => self.end(None),
            },
            Phase::End => {}
        }
    }
}

async fn send(client: &reqwest::Client, request: Result<HttpRequest, Error>) -> Result<Body, Failure> {
    let request = request.map_err(|error| Failure {
        message: error.to_string(),
        status: None,
    })?;
    let response = request
        .headers
        .iter()
        .fold(client.post(&request.url), |builder, (name, value)| {
            builder.header(name, value)
        })
        .json(&request.body)
        .send()
        .await
        .map_err(|error| Failure {
            message: error.to_string(),
            status: None,
        })?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(Failure {
            message: format!("{}: {}", status.as_u16(), error_text(&body)),
            status: Some(status.as_u16()),
        });
    }
    Ok(Box::pin(response.bytes_stream()))
}

/// The provider's error message from a JSON error body, or the body itself.
pub(crate) fn error_text(body: &str) -> String {
    let parsed = serde_json::from_str::<Value>(body).ok();
    let message = parsed.as_ref().and_then(|value| {
        value
            .pointer("/error/message")
            .or_else(|| value.pointer("/detail"))
            .or_else(|| value.pointer("/message"))
            .or_else(|| value.pointer("/error"))
            .and_then(Value::as_str)
    });
    message.unwrap_or(body.trim()).chars().take(2000).collect()
}

/// Streams one model response. The stream always ends with [`AssistantEvent::Done`]; request and provider errors
/// are reported there (`stop_reason: Error`). Dropping the stream cancels the request.
pub fn stream(
    client: &reqwest::Client,
    model: &Model,
    context: &Context,
    options: &StreamOptions,
    auth: &Auth,
) -> impl Stream<Item = AssistantEvent> + Send + 'static + use<> {
    let run = Run {
        client: client.clone(),
        phase: Phase::Send(Box::new(api::build_request(model, context, options, auth))),
        decoder: Some(api::decoder(model)),
        queue: VecDeque::new(),
    };
    stream::unfold(run, |mut run| async move {
        loop {
            if let Some(event) = run.queue.pop_front() {
                return Some((event, run));
            }
            if matches!(run.phase, Phase::End) {
                return None;
            }
            run.step().await;
        }
    })
}

/// Runs a request to its end and returns the final message.
pub async fn complete(
    client: &reqwest::Client,
    model: &Model,
    context: &Context,
    options: &StreamOptions,
    auth: &Auth,
) -> AssistantMessage {
    let events = stream(client, model, context, options, auth);
    futures_util::pin_mut!(events);
    let mut last = None;
    while let Some(event) = events.next().await {
        if let AssistantEvent::Done(message) = event {
            last = Some(message);
        }
    }
    last.unwrap_or_else(|| {
        Draft::new(model).finish(Some(Failure {
            message: "stream ended without a result".into(),
            status: None,
        }))
    })
}
