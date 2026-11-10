//! Reuses what the pi coding agent keeps in its agent directory (`~/.pi/agent`): provider credentials, the model
//! catalog, and default model settings. An app built on this never signs in itself; `pi` stays the place to log in.

pub mod auth;
pub mod catalog;
pub mod config_value;
pub mod lock;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use futures_util::{StreamExt, stream};
use ri_ai::{
    AssistantEvent, AssistantMessage, Auth, Context, EventStream, Model, Models, StreamOptions, ThinkingLevel,
};
use serde_json::Value;

pub use auth::{AuthStore, Credential, OAuthCredential};
pub use catalog::Catalog;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("cannot read {path}: {source}")]
    Io { path: PathBuf, source: std::io::Error },
    #[error("cannot parse {path}: {source}")]
    Parse { path: PathBuf, source: serde_json::Error },
    #[error("no credential for provider `{0}`; sign in with pi or set its API key variable")]
    NoCredential(String),
    #[error("the API key of provider `{0}` references a variable that is not set, or its command failed")]
    UnresolvedKey(String),
    #[error("the login for provider `{0}` has expired and ri cannot refresh it; run pi once to renew it")]
    RefreshUnsupported(String),
    #[error("refreshing the login for provider `{provider}` failed: {detail}")]
    RefreshFailed { provider: String, detail: String },
    #[error("unknown model `{0}`")]
    UnknownModel(String),
    #[error("pi has no default model configured")]
    NoDefaultModel,
}

/// pi's agent directory: `$PI_CODING_AGENT_DIR`, otherwise `~/.pi/agent`.
pub fn agent_dir() -> PathBuf {
    let home = || PathBuf::from(std::env::var_os("HOME").unwrap_or_default());
    match std::env::var("PI_CODING_AGENT_DIR").ok().filter(|dir| !dir.is_empty()) {
        Some(dir) if dir == "~" => home(),
        Some(dir) => dir
            .strip_prefix("~/")
            .map_or_else(|| PathBuf::from(&dir), |rest| home().join(rest)),
        None => home().join(".pi").join("agent"),
    }
}

/// Reads a JSON file that may contain comments. A missing file is `None`.
pub(crate) fn read_json(path: &Path) -> Result<Option<Value>, Error> {
    match std::fs::read_to_string(path) {
        Ok(text) => serde_json::from_str(&strip_comments(text.trim_start_matches('\u{feff}')))
            .map(Some)
            .map_err(|source| Error::Parse {
                path: path.to_owned(),
                source,
            }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(Error::Io {
            path: path.to_owned(),
            source,
        }),
    }
}

/// Blanks out `//` and `/* */` comments outside of strings.
fn strip_comments(text: &str) -> String {
    #[derive(Clone, Copy, PartialEq)]
    enum State {
        Code,
        String,
        Escape,
        Line,
        Block,
    }
    let mut state = State::Code;
    let mut characters = text.chars().peekable();
    let mut stripped = String::with_capacity(text.len());
    while let Some(character) = characters.next() {
        let next = characters.peek().copied();
        state = match (state, character, next) {
            (State::Code, '"', _) => State::String,
            (State::Code, '/', Some('/')) => State::Line,
            (State::Code, '/', Some('*')) => State::Block,
            (State::String, '\\', _) => State::Escape,
            (State::String, '"', _) => State::Code,
            (State::Escape, _, _) => State::String,
            (State::Line, '\n', _) => State::Code,
            (State::Block, '*', Some('/')) => {
                characters.next();
                stripped.push(' ');
                state = State::Code;
                continue;
            }
            (state, _, _) => state,
        };
        let comment = matches!(state, State::Line | State::Block);
        stripped.push(if comment && character != '\n' { ' ' } else { character });
    }
    stripped
}

/// The provider's conventional API key variable.
fn api_key_variable(provider: &str) -> String {
    match provider {
        "google" => "GEMINI_API_KEY".to_owned(),
        "opencode" | "opencode-go" => "OPENCODE_API_KEY".to_owned(),
        "azure" => "AZURE_OPENAI_API_KEY".to_owned(),
        other => format!("{}_API_KEY", other.to_uppercase().replace('-', "_")),
    }
}

/// pi's configuration, loaded once. Credentials are read again at each use, so a login made in pi meanwhile is seen.
#[derive(Debug, Clone)]
pub struct Pi {
    catalog: Catalog,
    settings: Value,
    auth: AuthStore,
    client: reqwest::Client,
}

impl Pi {
    pub fn load() -> Result<Pi, Error> {
        Pi::load_from(&agent_dir())
    }

    pub fn load_from(agent_dir: &Path) -> Result<Pi, Error> {
        Ok(Pi {
            catalog: Catalog::load(agent_dir)?,
            settings: read_json(&agent_dir.join("settings.json"))?.unwrap_or(Value::Null),
            auth: AuthStore::new(agent_dir.join("auth.json")),
            client: reqwest::Client::new(),
        })
    }

    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    pub fn auth_store(&self) -> &AuthStore {
        &self.auth
    }

    /// A shared HTTP client for model requests.
    pub fn client(&self) -> &reqwest::Client {
        &self.client
    }

    /// Finds a model by `provider/id`, or by a bare id when it is unambiguous.
    pub fn model(&self, reference: &str) -> Result<&Model, Error> {
        self.catalog
            .find(reference)
            .ok_or_else(|| Error::UnknownModel(reference.to_owned()))
    }

    /// The model pi starts with: `defaultProvider` and `defaultModel` of its settings.
    pub fn default_model(&self) -> Result<&Model, Error> {
        let provider = self.settings["defaultProvider"].as_str().ok_or(Error::NoDefaultModel)?;
        let id = self.settings["defaultModel"].as_str().ok_or(Error::NoDefaultModel)?;
        self.catalog
            .get(provider, id)
            .ok_or_else(|| Error::UnknownModel(format!("{provider}/{id}")))
    }

    pub fn default_thinking_level(&self) -> ThinkingLevel {
        self.settings["defaultThinkingLevel"]
            .as_str()
            .and_then(|level| level.parse().ok())
            .unwrap_or(ThinkingLevel::Medium)
    }

    /// Whether a credential for `provider` can be found, without refreshing or validating it.
    pub fn has_credential(&self, provider: &str) -> bool {
        let stored = self.auth.credential(provider).ok().flatten().is_some();
        let configured = self
            .catalog
            .provider(provider)
            .is_some_and(|config| config.api_key.is_some());
        stored || configured || std::env::var(api_key_variable(provider)).is_ok_and(|key| !key.is_empty())
    }

    /// Resolves the credential for a model's provider, in pi's order: the stored login or key, then the key in
    /// `models.json`, then the provider's environment variable. A stored credential is never bypassed: when it
    /// cannot be used, that is the error. Expiring OAuth tokens are refreshed and written back.
    pub async fn auth(&self, model: &Model) -> Result<Auth, Error> {
        let provider = model.provider.as_str();
        let unresolved = || Error::UnresolvedKey(provider.to_owned());
        match self.auth.credential(provider)? {
            Some(Credential::ApiKey { key: Some(key), env }) => {
                return config_value::resolve(&key, &env)
                    .map(Auth::api_key)
                    .ok_or_else(unresolved);
            }
            Some(Credential::OAuth(_)) => {
                let tokens = self.auth.fresh_oauth(provider, &self.client).await?;
                let account_id = tokens.extra.get("accountId").and_then(Value::as_str).map(str::to_owned);
                let account_id = account_id.or_else(|| auth::codex_account_id(&tokens.access));
                return Ok(Auth {
                    api_key: tokens.access,
                    account_id,
                    headers: BTreeMap::new(),
                });
            }
            _ => {}
        }
        if let Some(key) = self
            .catalog
            .provider(provider)
            .and_then(|config| config.api_key.as_deref())
        {
            return config_value::resolve(key, &BTreeMap::new())
                .map(Auth::api_key)
                .ok_or_else(unresolved);
        }
        std::env::var(api_key_variable(provider))
            .ok()
            .filter(|key| !key.is_empty())
            .map(Auth::api_key)
            .ok_or_else(|| Error::NoCredential(provider.to_owned()))
    }
}

impl Models for Pi {
    fn model(&self, reference: Option<&str>) -> Result<Model, String> {
        let model = match reference {
            Some(reference) => Pi::model(self, reference),
            None => self.default_model(),
        };
        model.cloned().map_err(|error| error.to_string())
    }

    fn stream(&self, model: &Model, context: Context, options: StreamOptions) -> EventStream {
        let (pi, model) = (self.clone(), model.clone());
        let started = async move {
            match pi.auth(&model).await {
                Ok(auth) => ri_ai::stream(&pi.client, &model, &context, &options, &auth).boxed(),
                Err(error) => {
                    let failed = AssistantMessage::failed(&model, error.to_string(), None);
                    stream::iter([AssistantEvent::Done(failed)]).boxed()
                }
            }
        };
        stream::once(started).flatten().boxed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn strips_comments_but_not_string_contents() {
        let text = "{ // note\n \"url\": \"http://x/*y*/\", /* gone */ \"a\": 1 }";
        let parsed: Value = serde_json::from_str(&strip_comments(text)).unwrap();
        assert_eq!(parsed, json!({ "url": "http://x/*y*/", "a": 1 }));
    }

    #[tokio::test]
    async fn resolves_stored_keys_and_reports_missing_credentials() {
        let directory = tempfile::tempdir().unwrap();
        let write = |name: &str, value: Value| std::fs::write(directory.path().join(name), value.to_string()).unwrap();
        write(
            "auth.json",
            json!({ "deepseek": { "type": "api_key", "key": "${RI_TEST_KEY}", "env": { "RI_TEST_KEY": "sk-stored" } } }),
        );
        write(
            "settings.json",
            json!({ "defaultProvider": "deepseek", "defaultModel": "flash", "defaultThinkingLevel": "high" }),
        );
        write(
            "models-store.json",
            json!({
                "deepseek": { "models": [{ "id": "flash", "api": "openai-completions" }] },
                "nokey": { "models": [{ "id": "m", "api": "openai-completions" }] },
            }),
        );
        let pi = Pi::load_from(directory.path()).unwrap();

        let model = pi.default_model().unwrap();
        assert_eq!(pi.default_thinking_level(), ThinkingLevel::High);
        assert_eq!(pi.auth(model).await.unwrap().api_key, "sk-stored");
        assert!(pi.has_credential("deepseek"));
        assert!(matches!(
            pi.auth(pi.model("nokey/m").unwrap()).await,
            Err(Error::NoCredential(_))
        ));
    }
}
