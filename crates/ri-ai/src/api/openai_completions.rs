//! OpenAI Chat Completions (`POST {base}/chat/completions`) and the many servers that imitate it.

use std::collections::BTreeMap;

use serde_json::{Value, json};

use super::{data_url, endpoint, sanitize_id, tool_result_parts};
use crate::sse::SseEvent;
use crate::stream::{Decoder, Draft, Failure, HttpRequest, StreamOptions};
use crate::types::{
    AssistantEvent, AssistantMessage, Content, Context, Message, StopReason, ThinkingLevel, Usage, joined_text,
};
use crate::{Auth, Model};

const REASONING_FIELDS: [&str; 3] = ["reasoning_content", "reasoning", "reasoning_text"];
const MAX_TOOL_CALL_ID: usize = 40;

/// Provider quirks: the model's `compat` entry where set, otherwise detected from the provider and URL.
struct Compat<'a> {
    developer_role: bool,
    store: bool,
    usage_in_streaming: bool,
    reasoning_effort: bool,
    max_tokens_field: &'a str,
    reasoning_content_required: bool,
    thinking_format: Option<&'a str>,
}

fn compat(model: &Model) -> Compat<'_> {
    let openai = model.base_url.contains("api.openai.com");
    let deepseek = model.provider == "deepseek" || model.base_url.contains("deepseek.com");
    Compat {
        developer_role: model.compat_bool("supportsDeveloperRole").unwrap_or(openai),
        store: model.compat_bool("supportsStore").unwrap_or(openai),
        usage_in_streaming: model.compat_bool("supportsUsageInStreaming").unwrap_or(true),
        reasoning_effort: model.compat_bool("supportsReasoningEffort").unwrap_or(true),
        max_tokens_field: model.compat_str("maxTokensField").unwrap_or(if openai {
            "max_completion_tokens"
        } else {
            "max_tokens"
        }),
        reasoning_content_required: model
            .compat_bool("requiresReasoningContentOnAssistantMessages")
            .unwrap_or(deepseek),
        thinking_format: model.compat_str("thinkingFormat").or(deepseek.then_some("deepseek")),
    }
}

pub(crate) fn build_request(model: &Model, context: &Context, options: &StreamOptions, auth: &Auth) -> HttpRequest {
    let compat = compat(model);
    let mut body = json!({
        "model": model.id,
        "messages": messages(model, context, &compat),
        "stream": true,
    });
    let fields = body.as_object_mut().expect("body is an object");
    if compat.usage_in_streaming {
        fields.insert("stream_options".into(), json!({ "include_usage": true }));
    }
    if compat.store {
        fields.insert("store".into(), json!(false));
    }
    if let Some(max_tokens) = options.max_tokens {
        fields.insert(compat.max_tokens_field.into(), json!(max_tokens));
    }
    if let Some(temperature) = options.temperature {
        fields.insert("temperature".into(), json!(temperature));
    }
    if !context.tools.is_empty() {
        let tools: Vec<Value> = context
            .tools
            .iter()
            .map(|tool| {
                json!({ "type": "function", "function": {
                    "name": tool.name, "description": tool.description, "parameters": tool.parameters,
                } })
            })
            .collect();
        fields.insert("tools".into(), Value::Array(tools));
    }
    fields.extend(thinking_fields(model, options.thinking_level(), &compat));

    HttpRequest::new(endpoint(&model.base_url, "chat/completions"), body)
        .header("authorization", format!("Bearer {}", auth.api_key))
        .header("accept", "text/event-stream")
}

fn thinking_fields(model: &Model, level: ThinkingLevel, compat: &Compat<'_>) -> Vec<(String, Value)> {
    if !model.reasoning {
        return Vec::new();
    }
    let level = model.clamp_thinking_level(level);
    let thinking = level != ThinkingLevel::Off;
    let effort = model.effort(level).filter(|_| compat.reasoning_effort);
    match compat.thinking_format {
        Some("deepseek") => {
            let switch = if thinking { "enabled" } else { "disabled" };
            let toggle = (thinking || model.supports_thinking_level(ThinkingLevel::Off))
                .then(|| ("thinking".to_owned(), json!({ "type": switch })));
            let effort = effort
                .filter(|_| thinking)
                .map(|effort| ("reasoning_effort".to_owned(), json!(effort)));
            toggle.into_iter().chain(effort).collect()
        }
        Some("openrouter") => effort
            .map(|effort| ("reasoning".to_owned(), json!({ "effort": effort })))
            .into_iter()
            .collect(),
        _ => effort
            .map(|effort| ("reasoning_effort".to_owned(), json!(effort)))
            .into_iter()
            .collect(),
    }
}

fn messages(model: &Model, context: &Context, compat: &Compat<'_>) -> Vec<Value> {
    let system = context
        .system_prompt
        .as_deref()
        .filter(|prompt| !prompt.is_empty())
        .map(|prompt| {
            let role = if model.reasoning && compat.developer_role {
                "developer"
            } else {
                "system"
            };
            json!({ "role": role, "content": prompt })
        });
    system
        .into_iter()
        .chain(
            context
                .messages
                .iter()
                .flat_map(|message| wire_message(model, message, compat)),
        )
        .collect()
}

fn wire_message(model: &Model, message: &Message, compat: &Compat<'_>) -> Vec<Value> {
    match message {
        Message::User { content } => user_message(model, content).into_iter().collect(),
        Message::Assistant(assistant) => assistant_message(assistant, compat).into_iter().collect(),
        Message::ToolResult {
            tool_call_id, content, ..
        } => {
            let parts = tool_result_parts(model, content);
            let result =
                json!({ "role": "tool", "tool_call_id": wire_tool_call_id(tool_call_id), "content": parts.text });
            let images: Vec<Value> = parts
                .images
                .iter()
                .map(|(mime_type, data)| json!({ "type": "image_url", "image_url": { "url": data_url(mime_type, data) } }))
                .collect();
            let attached = (!images.is_empty()).then(|| json!({ "role": "user", "content": images }));
            std::iter::once(result).chain(attached).collect()
        }
    }
}

fn user_message(model: &Model, content: &[Content]) -> Option<Value> {
    let parts: Vec<Value> = content
        .iter()
        .filter_map(|block| match block {
            Content::Text { text, .. } => Some(json!({ "type": "text", "text": text })),
            Content::Image { data, mime_type } if model.accepts_images() => {
                Some(json!({ "type": "image_url", "image_url": { "url": data_url(mime_type, data) } }))
            }
            _ => None,
        })
        .collect();
    match parts.as_slice() {
        [] => None,
        [only] if only["type"] == "text" => Some(json!({ "role": "user", "content": only["text"] })),
        _ => Some(json!({ "role": "user", "content": parts })),
    }
}

fn assistant_message(assistant: &AssistantMessage, compat: &Compat<'_>) -> Option<Value> {
    if matches!(assistant.stop_reason, StopReason::Error | StopReason::Aborted) {
        return None;
    }
    let text = joined_text(&assistant.content, "");
    let tool_calls: Vec<Value> = assistant
        .tool_calls()
        .map(|(id, name, arguments)| {
            json!({ "id": wire_tool_call_id(id), "type": "function", "function": {
                "name": name, "arguments": Value::Object(arguments.clone()).to_string(),
            } })
        })
        .collect();
    if text.is_empty() && tool_calls.is_empty() {
        return None;
    }
    let mut message =
        json!({ "role": "assistant", "content": if text.is_empty() { Value::Null } else { json!(text) } });
    let fields = message.as_object_mut().expect("message is an object");
    if !tool_calls.is_empty() {
        fields.insert("tool_calls".into(), Value::Array(tool_calls));
    }
    let reasoning = assistant.content.iter().find_map(|block| match block {
        Content::Thinking {
            thinking,
            thinking_signature: Some(field),
        } if REASONING_FIELDS.contains(&field.as_str()) => Some((field.clone(), thinking.clone())),
        _ => None,
    });
    match reasoning {
        Some((field, thinking)) => {
            fields.insert(field, json!(thinking));
        }
        None if compat.reasoning_content_required => {
            fields.insert("reasoning_content".into(), json!(""));
        }
        None => {}
    }
    Some(message)
}

/// Tool call ids from the Responses API are `call_id|item_id`; other servers accept only the first part.
fn wire_tool_call_id(id: &str) -> String {
    sanitize_id(id.split('|').next().unwrap_or(id), MAX_TOOL_CALL_ID)
}

pub(crate) struct CompletionsDecoder {
    draft: Draft,
    model: Model,
    text: Option<usize>,
    thinking: Option<usize>,
    /// Content index of each streaming tool call, by the provider's `index`.
    tool_calls: BTreeMap<u64, usize>,
    finished: bool,
    done: bool,
    failure: Option<Failure>,
}

impl CompletionsDecoder {
    pub fn new(model: &Model) -> Self {
        CompletionsDecoder {
            draft: Draft::new(model),
            model: model.clone(),
            text: None,
            thinking: None,
            tool_calls: BTreeMap::new(),
            finished: false,
            done: false,
            failure: None,
        }
    }

    fn delta(&mut self, delta: &Value, out: &mut Vec<AssistantEvent>) {
        if let Some(text) = delta["content"].as_str().filter(|text| !text.is_empty()) {
            let index = match self.text {
                Some(index) => index,
                None => *self.text.insert(self.draft.push(Content::text(""))),
            };
            self.draft.text_delta(index, text, out);
        }
        let reasoning = REASONING_FIELDS.iter().find_map(|field| {
            delta[*field]
                .as_str()
                .filter(|text| !text.is_empty())
                .map(|text| (*field, text))
        });
        if let Some((field, text)) = reasoning {
            let index = match self.thinking {
                Some(index) => index,
                None => *self.thinking.insert(self.draft.push(Content::Thinking {
                    thinking: String::new(),
                    thinking_signature: Some(field.to_owned()),
                })),
            };
            self.draft.text_delta(index, text, out);
        }
        for call in delta["tool_calls"].as_array().into_iter().flatten() {
            let wire_index = call["index"].as_u64().unwrap_or(self.tool_calls.len() as u64);
            let index = match self.tool_calls.get(&wire_index) {
                Some(index) => *index,
                None => {
                    let id = call["id"].as_str().unwrap_or_default().to_owned();
                    let name = call["function"]["name"].as_str().unwrap_or_default().to_owned();
                    let index = self.draft.tool_call_start(id, name, out);
                    self.tool_calls.insert(wire_index, index);
                    index
                }
            };
            if let Some(arguments) = call["function"]["arguments"].as_str() {
                self.draft.tool_call_delta(index, arguments, out);
            }
        }
    }
}

impl Decoder for CompletionsDecoder {
    fn push(&mut self, event: &SseEvent, out: &mut Vec<AssistantEvent>) {
        if event.data.trim() == "[DONE]" {
            self.done = true;
            return;
        }
        let Ok(chunk) = serde_json::from_str::<Value>(&event.data) else {
            return;
        };
        if let Some(error) = chunk.get("error").filter(|error| !error.is_null()) {
            let message = error["message"]
                .as_str()
                .map_or_else(|| error.to_string(), str::to_owned);
            self.failure = Some(Failure { message, status: None });
            self.done = true;
            return;
        }
        if let Some(id) = chunk["id"]
            .as_str()
            .filter(|_| self.draft.message.response_id.is_none())
        {
            self.draft.message.response_id = Some(id.to_owned());
        }
        if chunk["usage"].is_object() {
            self.draft.message.usage = self.model.price(usage(&chunk["usage"]));
        }
        let choice = &chunk["choices"][0];
        if choice["delta"].is_object() {
            self.delta(&choice["delta"], out);
        }
        if let Some(reason) = choice["finish_reason"].as_str() {
            self.finished = true;
            match reason {
                "length" => self.draft.message.stop_reason = StopReason::Length,
                "tool_calls" | "function_call" => self.draft.message.stop_reason = StopReason::ToolUse,
                "content_filter" => {
                    self.failure = Some(Failure {
                        message: "The provider's content filter stopped the response".into(),
                        status: None,
                    });
                }
                _ => {}
            }
            for index in self.draft.open_tool_calls() {
                self.draft.tool_call_end(index, None, out);
            }
        }
    }

    fn is_done(&self) -> bool {
        self.done
    }

    fn finish(self: Box<Self>, failure: Option<Failure>) -> AssistantMessage {
        let incomplete = (!self.finished).then(|| Failure {
            message: "The response stream ended before the model finished".into(),
            status: None,
        });
        self.draft.finish(failure.or(self.failure).or(incomplete))
    }
}

/// Providers disagree on where cache hits are reported: OpenAI and OpenRouter use
/// `prompt_tokens_details.cached_tokens`, DeepSeek `prompt_cache_hit_tokens`, Kimi a top-level `cached_tokens`.
fn usage(raw: &Value) -> Usage {
    let count = |value: &Value| value.as_u64().unwrap_or(0);
    let prompt = count(&raw["prompt_tokens"]);
    let cache_read = [
        &raw["prompt_tokens_details"]["cached_tokens"],
        &raw["prompt_cache_hit_tokens"],
        &raw["cached_tokens"],
    ]
    .into_iter()
    .find_map(Value::as_u64)
    .unwrap_or(0);
    let cache_write = count(&raw["prompt_tokens_details"]["cache_write_tokens"]);
    let input = prompt.saturating_sub(cache_read + cache_write);
    let output = count(&raw["completion_tokens"]);
    Usage {
        input,
        output,
        cache_read,
        cache_write,
        reasoning: count(&raw["completion_tokens_details"]["reasoning_tokens"]),
        total_tokens: input + output + cache_read + cache_write,
        cost: Default::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Tool;

    fn deepseek() -> Model {
        serde_json::from_value(json!({
            "id": "deepseek-flash", "api": "openai-completions", "provider": "deepseek",
            "baseUrl": "https://api.deepseek.com", "reasoning": true, "input": ["text"],
            "thinkingLevelMap": { "minimal": null, "low": "low", "medium": null, "high": "high", "max": "max" },
            "cost": { "input": 1.0, "output": 2.0, "cacheRead": 0.1, "cacheWrite": 0.0 },
            "compat": { "supportsStore": false, "supportsDeveloperRole": false, "maxTokensField": "max_tokens",
                "requiresReasoningContentOnAssistantMessages": true, "thinkingFormat": "deepseek" }
        }))
        .unwrap()
    }

    fn decode(model: &Model, lines: &[Value]) -> (Vec<AssistantEvent>, AssistantMessage) {
        let mut decoder = Box::new(CompletionsDecoder::new(model));
        let mut events = Vec::new();
        for line in lines {
            decoder.push(
                &SseEvent {
                    event: None,
                    data: line.to_string(),
                },
                &mut events,
            );
        }
        (events, decoder.finish(None))
    }

    #[test]
    fn builds_a_deepseek_request() {
        let model = deepseek();
        let context = Context {
            system_prompt: Some("Be brief.".into()),
            messages: vec![Message::user("hi")],
            tools: vec![Tool {
                name: "read".into(),
                description: "Read a file".into(),
                parameters: json!({ "type": "object" }),
            }],
        };
        let options = StreamOptions {
            thinking: Some(ThinkingLevel::Medium),
            max_tokens: Some(100),
            ..Default::default()
        };
        let request = build_request(&model, &context, &options, &Auth::api_key("k"));
        assert_eq!(request.url, "https://api.deepseek.com/chat/completions");
        assert_eq!(
            request.body["messages"],
            json!([{ "role": "system", "content": "Be brief." }, { "role": "user", "content": "hi" }])
        );
        assert_eq!(request.body["thinking"], json!({ "type": "enabled" }));
        assert_eq!(request.body["reasoning_effort"], "high");
        assert_eq!(request.body["max_tokens"], 100);
        assert!(request.body.get("store").is_none());
        assert_eq!(request.body["tools"][0]["function"]["name"], "read");
    }

    #[test]
    fn replays_tool_calls_with_reasoning_content() {
        let model = deepseek();
        let (_, assistant) = decode(
            &model,
            &[
                json!({ "id": "r1", "choices": [{ "delta": { "reasoning_content": "hm" } }] }),
                json!({ "choices": [{ "delta": { "tool_calls": [{ "index": 0, "id": "call_1", "function": { "name": "read", "arguments": "{\"pa" } }] } }] }),
                json!({ "choices": [{ "delta": { "tool_calls": [{ "index": 0, "function": { "arguments": "th\":\"a\"}" } }] } }] }),
                json!({ "choices": [{ "delta": {}, "finish_reason": "tool_calls" }] }),
                json!({ "choices": [], "usage": { "prompt_tokens": 100, "completion_tokens": 20, "prompt_cache_hit_tokens": 40 } }),
            ],
        );
        assert_eq!(assistant.stop_reason, StopReason::ToolUse);
        assert_eq!(assistant.usage.input, 60);
        assert_eq!(assistant.usage.cache_read, 40);
        let calls: Vec<_> = assistant.tool_calls().collect();
        assert_eq!(calls[0].2["path"], "a");

        let context = Context {
            system_prompt: None,
            messages: vec![
                Message::user("go"),
                Message::Assistant(assistant),
                Message::ToolResult {
                    tool_call_id: "call_1".into(),
                    tool_name: "read".into(),
                    content: vec![Content::text("data")],
                    is_error: false,
                },
            ],
            tools: vec![],
        };
        let request = build_request(&model, &context, &StreamOptions::default(), &Auth::api_key("k"));
        let messages = request.body["messages"].as_array().unwrap();
        assert_eq!(messages[1]["reasoning_content"], "hm");
        assert_eq!(messages[1]["content"], Value::Null);
        assert_eq!(
            messages[1]["tool_calls"][0]["function"]["arguments"],
            "{\"path\":\"a\"}"
        );
        assert_eq!(
            messages[2],
            json!({ "role": "tool", "tool_call_id": "call_1", "content": "data" })
        );
        assert_eq!(request.body["thinking"], json!({ "type": "disabled" }));
    }

    #[test]
    fn a_stream_cut_before_its_finish_reason_is_an_error() {
        let (events, assistant) = decode(
            &deepseek(),
            &[json!({ "choices": [{ "delta": { "content": "par" } }] })],
        );
        assert_eq!(
            events,
            vec![AssistantEvent::TextDelta {
                index: 0,
                delta: "par".into()
            }]
        );
        assert_eq!(assistant.stop_reason, StopReason::Error);
        assert_eq!(assistant.text(), "par");
    }
}
