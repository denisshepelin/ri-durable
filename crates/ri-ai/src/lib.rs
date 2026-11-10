//! Provider-neutral LLM access: conversation types, model descriptions, and the wire protocols that stream
//! a response from a provider. Credentials are passed in as [`Auth`]; this crate reads no files or environment.

pub mod api;
pub mod classify;
pub mod model;
pub mod sse;
pub mod stream;
pub mod types;

use std::collections::BTreeMap;

pub use model::{CostTier, Model, ModelCost, ModelKind};
pub use stream::{HttpRequest, StreamOptions, stream};
pub use types::*;

/// A resolved credential for one provider.
#[derive(Clone, Default)]
pub struct Auth {
    pub api_key: String,
    /// ChatGPT account id, required by the Codex backend.
    pub account_id: Option<String>,
    /// Extra request headers. They override the model's headers.
    pub headers: BTreeMap<String, String>,
}

impl Auth {
    pub fn api_key(api_key: impl Into<String>) -> Self {
        Auth {
            api_key: api_key.into(),
            ..Auth::default()
        }
    }
}

impl std::fmt::Debug for Auth {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Auth")
            .field("api_key", &"<redacted>")
            .field("account_id", &self.account_id)
            .field("headers", &self.headers.keys().collect::<Vec<_>>())
            .finish()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("unsupported API `{api}` for model {model}")]
    UnsupportedApi { api: String, model: String },
    #[error("{0}")]
    InvalidRequest(String),
    #[error("{label} returned {status}: {body}")]
    Http { label: String, status: u16, body: String },
    #[error("{0}")]
    Transport(#[from] reqwest::Error),
    #[error("{0}")]
    InvalidResponse(String),
}
