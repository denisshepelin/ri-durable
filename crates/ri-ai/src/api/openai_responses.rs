//! OpenAI Responses API (`POST {base}/responses`) and the ChatGPT Codex backend that speaks the same protocol
//! with a ChatGPT login instead of an API key.

use std::collections::BTreeMap;

use serde_json::{Value, json};

use super::{data_url, endpoint, sanitize_id, tool_result_parts};
use crate::sse::SseEvent;
use crate::stream::{Decoder, Draft, Failure, HttpRequest, StreamOptions};
use crate::types::{AssistantEvent, AssistantMessage, Content, Context, Message, StopReason, ThinkingLevel, Usage};
use crate::{Auth, Error, Model};

const DEFAULT_CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api";
const MAX_ID: usize = 64;

pub(crate) fn build_request(model: &Model, context: &Context, options: &StreamOptions, auth: &Auth) -> HttpRequest {
    let role = if model.reasoning && model.compat_bool("supportsDeveloperRole") != Some(false) {
        "developer"
    } else {
        "system"
    };
    let system = context
        .system_prompt
        .as_deref()
        .filter(|prompt| !prompt.is_empty())
        .map(|prompt| json!({ "role": role, "content": prompt }));
    let mut body = json!({
        "model": model.id,
        "input": system.into_iter().chain(input(model, context)).collect::<Vec<_>>(),
        "stream": true,
        "store": false,
        "include": ["reasoning.encrypted_content"],
    });
    let fields = body.as_object_mut().expect("body is an object");
    if let Some(max_tokens) = options.max_tokens {
        fields.insert("max_output_tokens".into(), json!(max_tokens));
    }
    fields.extend(shared_fields(model, context, options));

    HttpRequest::new(endpoint(&model.base_url, "responses"), body)
        .header("authorization", format!("Bearer {}", auth.api_key))
        .header("accept", "text/event-stream")
}

pub(crate) fn build_codex_request(
    model: &Model,
    context: &Context,
    options: &StreamOptions,
    auth: &Auth,
) -> Result<HttpRequest, Error> {
    let account_id = auth
        .account_id
        .as_deref()
        .ok_or_else(|| Error::InvalidRequest("The Codex backend needs a ChatGPT account id".into()))?;
    let instructions = context.system_prompt.as_deref().filter(|prompt| !prompt.is_empty());
    let mut body = json!({
        "model": model.id,
        "store": false,
        "stream": true,
        "instructions": instructions.unwrap_or("You are a helpful assistant."),
        "input": input(model, context),
        "include": ["reasoning.encrypted_content"],
        "tool_choice": "auto",
        "parallel_tool_calls": true,
    });
    body.as_object_mut()
        .expect("body is an object")
        .extend(shared_fields(model, context, options));

    let request = HttpRequest::new(codex_url(&model.base_url), body)
        .header("authorization", format!("Bearer {}", auth.api_key))
        .header("chatgpt-account-id", account_id)
        // The backend expects an originator on every request.
        .header("originator", "ri")
        .header("user-agent", concat!("ri-ai/", env!("CARGO_PKG_VERSION")))
        .header("openai-beta", "responses=experimental")
        .header("accept", "text/event-stream");
    Ok(match &options.session_id {
        Some(session_id) => request
            .header("session-id", session_id.clone())
            .header("x-client-request-id", session_id.clone()),
        None => request,
    })
}

fn codex_url(base_url: &str) -> String {
    let base = if base_url.trim().is_empty() {
        DEFAULT_CODEX_BASE_URL
    } else {
        base_url
    };
    let base = base.trim_end_matches('/');
    if base.ends_with("/codex/responses") {
        base.to_owned()
    } else if base.ends_with("/codex") {
        format!("{base}/responses")
    } else {
        format!("{base}/codex/responses")
    }
}

/// Body fields both backends take in the same form.
fn shared_fields(model: &Model, context: &Context, options: &StreamOptions) -> Vec<(String, Value)> {
    let tools = (!context.tools.is_empty()).then(|| {
        let tools: Vec<Value> = context
            .tools
            .iter()
            .map(|tool| {
                json!({ "type": "function", "name": tool.name, "description": tool.description,
                    "parameters": tool.parameters, "strict": null })
            })
            .collect();
        ("tools".to_owned(), Value::Array(tools))
    });
    let level = model.clamp_thinking_level(options.thinking_level());
    let reasoning = model.effort(level).filter(|_| model.reasoning).map(|effort| {
        let reasoning = if level == ThinkingLevel::Off {
            json!({ "effort": effort })
        } else {
            json!({ "effort": effort, "summary": "auto" })
        };
        ("reasoning".to_owned(), reasoning)
    });
    let cache_key = options
        .session_id
        .as_ref()
        .map(|session_id| ("prompt_cache_key".to_owned(), json!(session_id)));
    let temperature = options
        .temperature
        .map(|temperature| ("temperature".to_owned(), json!(temperature)));
    tools
        .into_iter()
        .chain(reasoning)
        .chain(cache_key)
        .chain(temperature)
        .collect()
}

fn input(model: &Model, context: &Context) -> Vec<Value> {
    context
        .messages
        .iter()
        .enumerate()
        .flat_map(|(position, message)| input_items(model, position, message))
        .collect()
}

fn input_items(model: &Model, position: usize, message: &Message) -> Vec<Value> {
    match message {
        Message::User { content } => {
            let parts: Vec<Value> = content
                .iter()
                .filter_map(|block| match block {
                    Content::Text { text, .. } => Some(json!({ "type": "input_text", "text": text })),
                    Content::Image { data, mime_type } if model.accepts_images() => {
                        Some(json!({ "type": "input_image", "detail": "auto", "image_url": data_url(mime_type, data) }))
                    }
                    _ => None,
                })
                .collect();
            if parts.is_empty() {
                Vec::new()
            } else {
                vec![json!({ "role": "user", "content": parts })]
            }
        }
        Message::Assistant(assistant) => assistant_items(model, position, assistant),
        Message::ToolResult {
            tool_call_id, content, ..
        } => {
            let parts = tool_result_parts(model, content);
            let output = if parts.images.is_empty() {
                json!(parts.text)
            } else {
                let text = json!({ "type": "input_text", "text": parts.text });
                let images = parts
                    .images
                    .iter()
                    .map(|(mime_type, data)| json!({ "type": "input_image", "detail": "auto", "image_url": data_url(mime_type, data) }));
                Value::Array(std::iter::once(text).chain(images).collect())
            };
            vec![json!({ "type": "function_call_output", "call_id": call_id(tool_call_id), "output": output })]
        }
    }
}

fn assistant_items(model: &Model, position: usize, assistant: &AssistantMessage) -> Vec<Value> {
    if matches!(assistant.stop_reason, StopReason::Error | StopReason::Aborted) {
        return Vec::new();
    }
    let same_backend = assistant.provider == model.provider && assistant.api == model.api;
    let same_model = same_backend && assistant.model == model.id;
    let mut texts = 0;
    assistant
        .content
        .iter()
        .filter_map(|block| match block {
            // A reasoning item is only valid on the backend that produced it.
            Content::Thinking {
                thinking_signature: Some(item),
                ..
            } if same_backend => serde_json::from_str::<Value>(item).ok().filter(Value::is_object),
            Content::Text { text, text_signature } => {
                let signature = text_signature
                    .as_deref()
                    .filter(|_| same_backend)
                    .and_then(parse_text_signature);
                let fallback = if texts == 0 {
                    format!("msg_ri_{position}")
                } else {
                    format!("msg_ri_{position}_{texts}")
                };
                texts += 1;
                let (id, phase) = signature.unwrap_or((fallback, None));
                let mut item = json!({
                    "type": "message", "role": "assistant", "status": "completed", "id": sanitize_id(&id, MAX_ID),
                    "content": [{ "type": "output_text", "text": text, "annotations": [] }],
                });
                if let Some(phase) = phase {
                    item["phase"] = json!(phase);
                }
                Some(item)
            }
            Content::ToolCall { id, name, arguments } => {
                let mut item = json!({
                    "type": "function_call", "call_id": call_id(id), "name": name,
                    "arguments": Value::Object(arguments.clone()).to_string(),
                });
                // Item ids pair calls with reasoning items; they are only kept for the model that issued them.
                let item_id = id
                    .split_once('|')
                    .map(|(_, item_id)| item_id)
                    .filter(|item_id| same_model && item_id.starts_with("fc_"));
                if let Some(item_id) = item_id {
                    item["id"] = json!(sanitize_id(item_id, MAX_ID));
                }
                Some(item)
            }
            _ => None,
        })
        .collect()
}

/// Tool call ids are stored as `call_id|item_id`.
fn call_id(id: &str) -> String {
    sanitize_id(id.split('|').next().unwrap_or(id), MAX_ID)
}

fn text_signature(id: &str, phase: Option<&str>) -> String {
    match phase {
        Some(phase) => json!({ "v": 1, "id": id, "phase": phase }).to_string(),
        None => json!({ "v": 1, "id": id }).to_string(),
    }
}

fn parse_text_signature(signature: &str) -> Option<(String, Option<String>)> {
    let parsed = serde_json::from_str::<Value>(signature).ok()?;
    let id = parsed["id"].as_str()?.to_owned();
    Some((id, parsed["phase"].as_str().map(str::to_owned)))
}

pub(crate) struct ResponsesDecoder {
    draft: Draft,
    model: Model,
    /// Content index of each output item, by the provider's `output_index`.
    items: BTreeMap<u64, usize>,
    /// Output items whose reasoning summary already has a part, so the next part starts a new paragraph.
    summarized: Vec<u64>,
    done: bool,
    failure: Option<Failure>,
}

impl ResponsesDecoder {
    pub fn new(model: &Model) -> Self {
        ResponsesDecoder {
            draft: Draft::new(model),
            model: model.clone(),
            items: BTreeMap::new(),
            summarized: Vec::new(),
            done: false,
            failure: None,
        }
    }

    fn index(&self, event: &Value) -> Option<usize> {
        event["output_index"]
            .as_u64()
            .and_then(|output_index| self.items.get(&output_index).copied())
    }

    fn item_added(&mut self, output_index: u64, item: &Value, out: &mut Vec<AssistantEvent>) {
        let index = match item["type"].as_str() {
            Some("reasoning") => self.draft.push(Content::Thinking {
                thinking: String::new(),
                thinking_signature: None,
            }),
            Some("message") => self.draft.push(Content::text("")),
            Some("function_call") => {
                let id = format!(
                    "{}|{}",
                    item["call_id"].as_str().unwrap_or_default(),
                    item["id"].as_str().unwrap_or_default()
                );
                let name = item["name"].as_str().unwrap_or_default().to_owned();
                let index = self.draft.tool_call_start(id, name, out);
                self.draft
                    .tool_call_delta(index, item["arguments"].as_str().unwrap_or_default(), out);
                index
            }
            _ => return,
        };
        self.items.insert(output_index, index);
    }

    fn item_done(&mut self, index: usize, item: &Value, out: &mut Vec<AssistantEvent>) {
        match item["type"].as_str() {
            Some("reasoning") => {
                let summary = item["summary"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|part| part["text"].as_str())
                    .collect::<Vec<_>>()
                    .join("\n\n");
                if let Some(Content::Thinking {
                    thinking,
                    thinking_signature,
                }) = self.draft.message.content.get_mut(index)
                {
                    if !summary.is_empty() {
                        *thinking = summary;
                    }
                    *thinking_signature = Some(item.to_string());
                }
            }
            Some("message") => {
                let full = item["content"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|part| part["text"].as_str().or_else(|| part["refusal"].as_str()))
                    .collect::<String>();
                if let Some(Content::Text {
                    text,
                    text_signature: signature,
                }) = self.draft.message.content.get_mut(index)
                {
                    if !full.is_empty() {
                        *text = full;
                    }
                    *signature = item["id"].as_str().map(|id| text_signature(id, item["phase"].as_str()));
                }
            }
            Some("function_call") => self.draft.tool_call_end(index, item["arguments"].as_str(), out),
            _ => {}
        }
    }

    fn completed(&mut self, response: &Value) {
        if let Some(id) = response["id"].as_str() {
            self.draft.message.response_id = Some(id.to_owned());
        }
        if response["usage"].is_object() {
            self.draft.message.usage = self.model.price(usage(&response["usage"]));
        }
        if response["status"] == "incomplete" {
            self.draft.message.stop_reason = StopReason::Length;
        }
        self.done = true;
    }

    fn fail(&mut self, message: String) {
        self.failure = Some(Failure { message, status: None });
        self.done = true;
    }
}

impl Decoder for ResponsesDecoder {
    fn push(&mut self, event: &SseEvent, out: &mut Vec<AssistantEvent>) {
        let Ok(data) = serde_json::from_str::<Value>(&event.data) else {
            return;
        };
        let Some(kind) = data["type"].as_str().or(event.event.as_deref()) else {
            return;
        };
        match kind {
            "response.output_item.added" => {
                if let Some(output_index) = data["output_index"].as_u64() {
                    self.item_added(output_index, &data["item"], out);
                }
            }
            "response.reasoning_summary_part.added" => {
                if let (Some(index), Some(output_index)) = (self.index(&data), data["output_index"].as_u64()) {
                    if self.summarized.contains(&output_index) {
                        self.draft.text_delta(index, "\n\n", out);
                    } else {
                        self.summarized.push(output_index);
                    }
                }
            }
            "response.reasoning_summary_text.delta" | "response.output_text.delta" | "response.refusal.delta" => {
                if let Some(index) = self.index(&data) {
                    self.draft
                        .text_delta(index, data["delta"].as_str().unwrap_or_default(), out);
                }
            }
            "response.function_call_arguments.delta" => {
                if let Some(index) = self.index(&data) {
                    self.draft
                        .tool_call_delta(index, data["delta"].as_str().unwrap_or_default(), out);
                }
            }
            "response.output_item.done" => {
                if let Some(index) = self.index(&data) {
                    self.item_done(index, &data["item"], out);
                }
            }
            "response.completed" | "response.done" | "response.incomplete" => self.completed(&data["response"]),
            "response.failed" => {
                let error = &data["response"]["error"];
                let message = error["message"]
                    .as_str()
                    .or_else(|| error["code"].as_str())
                    .unwrap_or("The response failed");
                self.fail(message.to_owned());
            }
            "error" => {
                let error = if data["error"].is_object() {
                    &data["error"]
                } else {
                    &data
                };
                let message = error["message"]
                    .as_str()
                    .or_else(|| error["code"].as_str())
                    .unwrap_or("The provider reported an error");
                self.fail(message.to_owned());
            }
            _ => {}
        }
    }

    fn is_done(&self) -> bool {
        self.done
    }

    fn finish(self: Box<Self>, failure: Option<Failure>) -> AssistantMessage {
        let incomplete = (!self.done).then(|| Failure {
            message: "The response stream ended before the model finished".into(),
            status: None,
        });
        self.draft.finish(failure.or(self.failure).or(incomplete))
    }
}

fn usage(raw: &Value) -> Usage {
    let count = |value: &Value| value.as_u64().unwrap_or(0);
    let cache_read = count(&raw["input_tokens_details"]["cached_tokens"]);
    let input = count(&raw["input_tokens"]).saturating_sub(cache_read);
    let output = count(&raw["output_tokens"]);
    Usage {
        input,
        output,
        cache_read,
        cache_write: 0,
        reasoning: count(&raw["output_tokens_details"]["reasoning_tokens"]),
        total_tokens: input + output + cache_read,
        cost: Default::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Tool;

    fn codex() -> Model {
        serde_json::from_value(json!({
            "id": "gpt-5.6-luna", "api": "openai-codex-responses", "provider": "openai-codex",
            "baseUrl": "https://chatgpt.com/backend-api", "reasoning": true, "input": ["text", "image"],
            "thinkingLevelMap": { "xhigh": "xhigh", "max": "max", "minimal": "low" },
            "cost": { "input": 1.0, "output": 10.0, "cacheRead": 0.1, "cacheWrite": 0.0 }
        }))
        .unwrap()
    }

    fn auth() -> Auth {
        Auth {
            api_key: "token".into(),
            account_id: Some("acct".into()),
            headers: BTreeMap::new(),
        }
    }

    fn decode(model: &Model, events: &[Value]) -> (Vec<AssistantEvent>, AssistantMessage) {
        let mut decoder = Box::new(ResponsesDecoder::new(model));
        let mut out = Vec::new();
        for event in events {
            decoder.push(
                &SseEvent {
                    event: None,
                    data: event.to_string(),
                },
                &mut out,
            );
        }
        (out, decoder.finish(None))
    }

    fn tool_round() -> Vec<Value> {
        vec![
            json!({ "type": "response.output_item.added", "output_index": 0, "item": { "type": "reasoning", "id": "rs_1" } }),
            json!({ "type": "response.reasoning_summary_part.added", "output_index": 0 }),
            json!({ "type": "response.reasoning_summary_text.delta", "output_index": 0, "delta": "Plan" }),
            json!({ "type": "response.output_item.done", "output_index": 0,
                "item": { "type": "reasoning", "id": "rs_1", "encrypted_content": "enc", "summary": [{ "type": "summary_text", "text": "Plan" }] } }),
            json!({ "type": "response.output_item.added", "output_index": 1,
                "item": { "type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "read", "arguments": "" } }),
            json!({ "type": "response.function_call_arguments.delta", "output_index": 1, "delta": "{\"path\":" }),
            json!({ "type": "response.function_call_arguments.delta", "output_index": 1, "delta": "\"a\"}" }),
            json!({ "type": "response.output_item.done", "output_index": 1,
                "item": { "type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "read", "arguments": "{\"path\":\"a\"}" } }),
            json!({ "type": "response.completed", "response": { "id": "resp_1", "status": "completed",
                "usage": { "input_tokens": 100, "output_tokens": 10, "input_tokens_details": { "cached_tokens": 30 } } } }),
        ]
    }

    #[test]
    fn decodes_reasoning_and_a_tool_call() {
        let (events, message) = decode(&codex(), &tool_round());
        assert_eq!(message.stop_reason, StopReason::ToolUse);
        assert_eq!(message.response_id.as_deref(), Some("resp_1"));
        assert_eq!(
            (message.usage.input, message.usage.cache_read, message.usage.output),
            (70, 30, 10)
        );
        assert!((message.usage.cost.total - (70.0 + 3.0 + 100.0) / 1e6).abs() < 1e-12);
        assert_eq!(message.tool_calls().collect::<Vec<_>>()[0].0, "call_1|fc_1");
        assert!(events.contains(&AssistantEvent::ThinkingDelta {
            index: 0,
            delta: "Plan".into()
        }));
        assert!(events.contains(&AssistantEvent::ToolCallEnd { index: 1 }));
    }

    #[test]
    fn replays_a_tool_round_to_the_model_that_made_it() {
        let model = codex();
        let (_, assistant) = decode(&model, &tool_round());
        let context = Context {
            system_prompt: Some("Be brief.".into()),
            messages: vec![
                Message::user("go"),
                Message::Assistant(assistant),
                Message::ToolResult {
                    tool_call_id: "call_1|fc_1".into(),
                    tool_name: "read".into(),
                    content: vec![Content::text("data")],
                    is_error: false,
                },
            ],
            tools: vec![Tool {
                name: "read".into(),
                description: "Read".into(),
                parameters: json!({ "type": "object" }),
            }],
        };
        let options = StreamOptions {
            thinking: Some(ThinkingLevel::Minimal),
            session_id: Some("s1".into()),
            ..Default::default()
        };
        let request = build_codex_request(&model, &context, &options, &auth()).unwrap();
        assert_eq!(request.url, "https://chatgpt.com/backend-api/codex/responses");
        assert_eq!(request.body["instructions"], "Be brief.");
        assert_eq!(request.body["reasoning"], json!({ "effort": "low", "summary": "auto" }));
        assert_eq!(request.body["prompt_cache_key"], "s1");
        let input = request.body["input"].as_array().unwrap();
        assert_eq!(input[1]["encrypted_content"], "enc");
        assert_eq!(
            input[2],
            json!({ "type": "function_call", "call_id": "call_1", "name": "read", "arguments": "{\"path\":\"a\"}", "id": "fc_1" })
        );
        assert_eq!(
            input[3],
            json!({ "type": "function_call_output", "call_id": "call_1", "output": "data" })
        );
        assert!(request.headers.contains(&("chatgpt-account-id".into(), "acct".into())));
    }

    #[test]
    fn drops_reasoning_and_item_ids_from_another_backend() {
        let (_, mut assistant) = decode(&codex(), &tool_round());
        assistant.provider = "elsewhere".into();
        let context = Context {
            system_prompt: None,
            messages: vec![Message::Assistant(assistant)],
            tools: vec![],
        };
        let request = build_codex_request(&codex(), &context, &StreamOptions::default(), &auth()).unwrap();
        let input = request.body["input"].as_array().unwrap();
        assert_eq!(input.len(), 1);
        assert!(input[0].get("id").is_none());
    }

    #[test]
    fn reports_a_failed_response() {
        let (_, message) = decode(
            &codex(),
            &[json!({ "type": "response.failed", "response": { "error": { "message": "overloaded" } } })],
        );
        assert_eq!(message.stop_reason, StopReason::Error);
        assert_eq!(message.error_message.as_deref(), Some("overloaded"));
    }
}
