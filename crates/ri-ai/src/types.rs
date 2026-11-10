//! Provider-neutral conversation types. JSON field names follow pi-ai so stored transcripts stay readable by both.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub type JsonObject = Map<String, Value>;

/// One block of message content. Which variants are valid depends on the message role:
/// users send `Text` and `Image`, assistants produce `Text`, `Thinking` and `ToolCall`,
/// tool results carry `Text` and `Image`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum Content {
    #[serde(rename_all = "camelCase")]
    Text {
        text: String,
        /// Opaque provider data needed to send this block back unchanged.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        text_signature: Option<String>,
    },
    #[serde(rename_all = "camelCase")]
    Thinking {
        thinking: String,
        /// Opaque provider data (for example an encrypted reasoning item) needed to send this block back.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        thinking_signature: Option<String>,
    },
    #[serde(rename_all = "camelCase")]
    Image {
        /// Base64 without a data URL prefix.
        data: String,
        mime_type: String,
    },
    #[serde(rename_all = "camelCase")]
    ToolCall {
        id: String,
        name: String,
        arguments: JsonObject,
    },
}

impl Content {
    pub fn text(text: impl Into<String>) -> Self {
        Content::Text {
            text: text.into(),
            text_signature: None,
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            Content::Text { text, .. } => Some(text),
            _ => None,
        }
    }
}

/// Joins the text blocks of `content` with `separator`, ignoring every other block.
pub fn joined_text(content: &[Content], separator: &str) -> String {
    content
        .iter()
        .filter_map(Content::as_text)
        .collect::<Vec<_>>()
        .join(separator)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "camelCase")]
pub enum Message {
    User {
        content: Vec<Content>,
    },
    Assistant(AssistantMessage),
    #[serde(rename_all = "camelCase")]
    ToolResult {
        tool_call_id: String,
        tool_name: String,
        content: Vec<Content>,
        #[serde(default)]
        is_error: bool,
    },
}

impl Message {
    pub fn user(text: impl Into<String>) -> Self {
        Message::User {
            content: vec![Content::text(text)],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantMessage {
    pub content: Vec<Content>,
    pub api: String,
    pub provider: String,
    pub model: String,
    #[serde(default)]
    pub usage: Usage,
    pub stop_reason: StopReason,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// HTTP status of a failed request, when the failure was an HTTP error. Callers use it to decide on retries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_status: Option<u16>,
    /// Unix time in milliseconds.
    pub timestamp: u64,
}

impl AssistantMessage {
    pub fn text(&self) -> String {
        joined_text(&self.content, "")
    }

    pub fn tool_calls(&self) -> impl Iterator<Item = (&str, &str, &JsonObject)> {
        self.content.iter().filter_map(|block| match block {
            Content::ToolCall { id, name, arguments } => Some((id.as_str(), name.as_str(), arguments)),
            _ => None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StopReason {
    Stop,
    Length,
    ToolUse,
    Error,
    Aborted,
}

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    /// Uncached input tokens.
    pub input: u64,
    /// Output tokens, reasoning included.
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    #[serde(default)]
    pub reasoning: u64,
    pub total_tokens: u64,
    #[serde(default)]
    pub cost: Cost,
}

/// US dollars.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Cost {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
    pub total: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    Off,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl ThinkingLevel {
    pub const ALL: [ThinkingLevel; 7] = [
        ThinkingLevel::Off,
        ThinkingLevel::Minimal,
        ThinkingLevel::Low,
        ThinkingLevel::Medium,
        ThinkingLevel::High,
        ThinkingLevel::Xhigh,
        ThinkingLevel::Max,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            ThinkingLevel::Off => "off",
            ThinkingLevel::Minimal => "minimal",
            ThinkingLevel::Low => "low",
            ThinkingLevel::Medium => "medium",
            ThinkingLevel::High => "high",
            ThinkingLevel::Xhigh => "xhigh",
            ThinkingLevel::Max => "max",
        }
    }
}

impl std::str::FromStr for ThinkingLevel {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        ThinkingLevel::ALL
            .into_iter()
            .find(|level| level.as_str() == value)
            .ok_or_else(|| format!("unknown thinking level: {value}"))
    }
}

/// A tool offered to the model. `parameters` is a JSON Schema object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// Everything one model request is built from.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Context {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
}

/// Incremental output of one model response. `index` is the position of the block in the final message's content.
/// The last event of every stream is `Done`, also for failed requests (`stop_reason` is then `Error`).
#[derive(Debug, Clone, PartialEq)]
pub enum AssistantEvent {
    TextDelta {
        index: usize,
        delta: String,
    },
    ThinkingDelta {
        index: usize,
        delta: String,
    },
    ToolCallStart {
        index: usize,
        id: String,
        name: String,
    },
    /// A fragment of the call's arguments as raw JSON text.
    ToolCallDelta {
        index: usize,
        delta: String,
    },
    ToolCallEnd {
        index: usize,
    },
    Done(AssistantMessage),
}

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as u64)
        .unwrap_or(0)
}
