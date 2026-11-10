//! A scripted model for tests and examples: it answers requests from a list instead of calling a provider.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use futures_util::{StreamExt, stream};
use ri_ai::{
    AssistantEvent, AssistantMessage, Content, Context, EventStream, JsonObject, Model, Models, StopReason,
    StreamOptions, now_ms,
};
use serde_json::Value;

/// One scripted response.
#[derive(Debug, Clone)]
pub enum Reply {
    Text(String),
    /// Tool calls as `(name, arguments)`.
    Calls(Vec<(String, Value)>),
    /// A failed request with an optional HTTP status.
    Error(Option<u16>, String),
    /// A request that never finishes, as if the provider hung.
    Hang,
}

impl Reply {
    pub fn text(text: &str) -> Reply {
        Reply::Text(text.to_owned())
    }

    pub fn call(name: &str, arguments: Value) -> Reply {
        Reply::Calls(vec![(name.to_owned(), arguments)])
    }
}

#[derive(Clone, Default)]
pub struct ScriptedModels {
    replies: Arc<Mutex<VecDeque<Reply>>>,
    requests: Arc<Mutex<Vec<Context>>>,
    context_window: Option<u64>,
}

impl ScriptedModels {
    pub fn new(replies: impl IntoIterator<Item = Reply>) -> Self {
        let models = ScriptedModels::default();
        models.push(replies);
        models
    }

    /// Gives the scripted model a context window of this many tokens.
    pub fn context_window(mut self, tokens: u64) -> Self {
        self.context_window = Some(tokens);
        self
    }

    /// Adds replies after the ones still waiting.
    pub fn push(&self, replies: impl IntoIterator<Item = Reply>) {
        self.replies.lock().expect("replies lock").extend(replies);
    }

    /// Every request received so far.
    pub fn requests(&self) -> Vec<Context> {
        self.requests.lock().expect("requests lock").clone()
    }
}

fn scripted_model() -> Model {
    serde_json::from_value(serde_json::json!({ "id": "scripted", "api": "scripted", "provider": "test" }))
        .expect("model")
}

impl Models for ScriptedModels {
    fn model(&self, reference: Option<&str>) -> Result<Model, String> {
        match reference {
            Some("missing/model") => Err("unknown model `missing/model`".to_owned()),
            _ => {
                let model = scripted_model();
                Ok(Model {
                    context_window: self.context_window.unwrap_or(model.context_window),
                    ..model
                })
            }
        }
    }

    fn stream(&self, model: &Model, context: Context, _options: StreamOptions) -> EventStream {
        let request = self.requests.lock().expect("requests lock").len();
        self.requests.lock().expect("requests lock").push(context);
        let reply = self.replies.lock().expect("replies lock").pop_front();
        let message = |content: Vec<Content>, stop_reason| AssistantMessage {
            content,
            api: model.api.clone(),
            provider: model.provider.clone(),
            model: model.id.clone(),
            usage: ri_ai::Usage {
                input: 10,
                output: 5,
                total_tokens: 15,
                ..Default::default()
            },
            stop_reason,
            response_id: None,
            error_message: None,
            error_status: None,
            timestamp: now_ms(),
        };
        match reply {
            Some(Reply::Text(text)) => {
                let delta = AssistantEvent::TextDelta {
                    index: 0,
                    delta: text.clone(),
                };
                let done = AssistantEvent::Done(message(vec![Content::text(text)], StopReason::Stop));
                stream::iter([delta, done]).boxed()
            }
            Some(Reply::Calls(calls)) => {
                let content = calls
                    .into_iter()
                    .enumerate()
                    .map(|(position, (name, arguments))| Content::ToolCall {
                        id: format!("call_{request}_{position}"),
                        name,
                        arguments: arguments.as_object().cloned().unwrap_or_else(JsonObject::new),
                    })
                    .collect();
                stream::iter([AssistantEvent::Done(message(content, StopReason::ToolUse))]).boxed()
            }
            Some(Reply::Error(status, text)) => {
                stream::iter([AssistantEvent::Done(AssistantMessage::failed(model, text, status))]).boxed()
            }
            Some(Reply::Hang) => stream::pending().boxed(),
            None => {
                let failed = AssistantMessage::failed(model, "the script has no reply left", Some(400));
                stream::iter([AssistantEvent::Done(failed)]).boxed()
            }
        }
    }
}
