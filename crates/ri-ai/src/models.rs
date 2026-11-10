//! The seam between an agent loop and whatever supplies models and credentials.

use std::pin::Pin;

use futures_util::Stream;

use crate::stream::{Draft, Failure};
use crate::{AssistantEvent, AssistantMessage, Context, Model, StreamOptions};

pub type EventStream = Pin<Box<dyn Stream<Item = AssistantEvent> + Send>>;

/// A source of models that can answer requests. Implementations own credential lookup and transport, so callers
/// only name a model.
pub trait Models: Send + Sync + 'static {
    /// Describes the model named by `reference` (`provider/id`), or the default model for `None`.
    fn model(&self, reference: Option<&str>) -> Result<Model, String>;

    /// Streams one response. The stream ends with [`AssistantEvent::Done`], also on failure.
    fn stream(&self, model: &Model, context: Context, options: StreamOptions) -> EventStream;
}

impl AssistantMessage {
    /// A message that records a failed request to `model`.
    pub fn failed(model: &Model, message: impl Into<String>, status: Option<u16>) -> AssistantMessage {
        Draft::new(model).finish(Some(Failure {
            message: message.into(),
            status,
        }))
    }
}
