//! Wire protocols, selected by [`Model::api`].

pub mod openai_completions;
pub mod openai_responses;

use crate::sse::SseEvent;
use crate::stream::{Decoder, Draft, Failure, HttpRequest, StreamOptions};
use crate::types::{AssistantEvent, AssistantMessage, Content, Context, joined_text};
use crate::{Auth, Error, Model};

pub const OPENAI_COMPLETIONS: &str = "openai-completions";
pub const OPENAI_RESPONSES: &str = "openai-responses";
pub const OPENAI_CODEX_RESPONSES: &str = "openai-codex-responses";

/// The chat APIs [`crate::stream`] can speak.
pub const CHAT_APIS: [&str; 3] = [OPENAI_COMPLETIONS, OPENAI_RESPONSES, OPENAI_CODEX_RESPONSES];

/// Builds the provider request for one model response without sending it.
pub fn build_request(
    model: &Model,
    context: &Context,
    options: &StreamOptions,
    auth: &Auth,
) -> Result<HttpRequest, Error> {
    let request = match model.api.as_str() {
        OPENAI_COMPLETIONS => openai_completions::build_request(model, context, options, auth),
        OPENAI_RESPONSES => openai_responses::build_request(model, context, options, auth),
        OPENAI_CODEX_RESPONSES => openai_responses::build_codex_request(model, context, options, auth)?,
        _ => {
            return Err(Error::UnsupportedApi {
                api: model.api.clone(),
                model: model.id.clone(),
            });
        }
    };
    Ok(request
        .headers(&model.headers)
        .headers(&auth.headers)
        .headers(&options.headers))
}

pub(crate) fn decoder(model: &Model) -> Box<dyn Decoder> {
    match model.api.as_str() {
        OPENAI_COMPLETIONS => Box::new(openai_completions::CompletionsDecoder::new(model)),
        OPENAI_RESPONSES | OPENAI_CODEX_RESPONSES => Box::new(openai_responses::ResponsesDecoder::new(model)),
        _ => Box::new(Unsupported(Draft::new(model))),
    }
}

struct Unsupported(Draft);

impl Decoder for Unsupported {
    fn push(&mut self, _event: &SseEvent, _out: &mut Vec<AssistantEvent>) {}

    fn is_done(&self) -> bool {
        true
    }

    fn finish(self: Box<Self>, failure: Option<Failure>) -> AssistantMessage {
        self.0.finish(failure)
    }
}

pub(crate) fn endpoint(base_url: &str, path: &str) -> String {
    format!("{}/{}", base_url.trim_end_matches('/'), path)
}

pub(crate) fn data_url(mime_type: &str, data: &str) -> String {
    format!("data:{mime_type};base64,{data}")
}

/// Keeps `[A-Za-z0-9_-]`, replaces the rest with `_`, and cuts to `max` characters.
pub(crate) fn sanitize_id(id: &str, max: usize) -> String {
    id.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' || character == '-' {
                character
            } else {
                '_'
            }
        })
        .take(max)
        .collect()
}

pub(crate) struct ToolResultParts<'a> {
    pub text: String,
    pub images: Vec<(&'a str, &'a str)>,
}

/// Splits a tool result into the text the model reads and its images as `(mime_type, data)`.
pub(crate) fn tool_result_parts<'a>(model: &Model, content: &'a [Content]) -> ToolResultParts<'a> {
    let text = joined_text(content, "\n");
    let images: Vec<_> = content
        .iter()
        .filter_map(|block| match block {
            Content::Image { data, mime_type } => Some((mime_type.as_str(), data.as_str())),
            _ => None,
        })
        .collect();
    let has_images = !images.is_empty();
    let images = if model.accepts_images() { images } else { Vec::new() };
    let text = match (text.trim().is_empty(), has_images) {
        (false, _) => text,
        (true, true) => "(see attached image)".to_owned(),
        (true, false) => "(no tool output)".to_owned(),
    };
    ToolResultParts { text, images }
}
