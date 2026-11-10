//! pi's credential file, `auth.json`: one credential per provider id, either an API key or OAuth tokens.
//!
//! Refreshing OAuth tokens rewrites the file. Refresh tokens are single-use, so a refresh runs under the same
//! cross-process lock pi takes and re-checks the stored credential inside it: whichever process gets there first
//! refreshes, the others reuse its result.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::Error;
use crate::lock::FileLock;

/// Tokens this close to expiry are refreshed before use.
const MINIMUM_VALIDITY_MS: f64 = 5.0 * 60.0 * 1000.0;
const REFRESH_TIMEOUT: Duration = Duration::from_secs(15);

const CODEX_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const CODEX_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const CODEX_JWT_CLAIM: &str = "https://api.openai.com/auth";

#[derive(Clone, PartialEq, Deserialize)]
#[serde(tag = "type")]
pub enum Credential {
    #[serde(rename = "api_key")]
    ApiKey {
        /// A literal key, or pi's `$VAR` / `!command` syntax.
        key: Option<String>,
        /// Variables consulted before the process environment when `key` references them.
        #[serde(default)]
        env: BTreeMap<String, String>,
    },
    #[serde(rename = "oauth")]
    OAuth(OAuthCredential),
}

#[derive(Clone, PartialEq, Deserialize)]
pub struct OAuthCredential {
    pub access: String,
    pub refresh: String,
    /// Unix time in milliseconds.
    pub expires: f64,
    /// Provider-specific fields, for example `accountId` for Codex.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl OAuthCredential {
    pub fn needs_refresh(&self, now_ms: f64) -> bool {
        now_ms + MINIMUM_VALIDITY_MS >= self.expires
    }

    fn to_json(&self) -> Value {
        let mut stored =
            json!({ "type": "oauth", "access": self.access, "refresh": self.refresh, "expires": self.expires as u64 });
        stored
            .as_object_mut()
            .expect("credential is an object")
            .extend(self.extra.clone());
        stored
    }
}

impl std::fmt::Debug for Credential {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Credential::ApiKey { .. } => "Credential::ApiKey(<redacted>)",
            Credential::OAuth(_) => "Credential::OAuth(<redacted>)",
        })
    }
}

#[derive(Debug, Clone)]
pub struct AuthStore {
    path: PathBuf,
}

impl AuthStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        AuthStore { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn read(&self) -> Result<Map<String, Value>, Error> {
        match fs::read_to_string(&self.path) {
            Ok(text) => serde_json::from_str(text.trim_start_matches('\u{feff}')).map_err(|source| Error::Parse {
                path: self.path.clone(),
                source,
            }),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Map::new()),
            Err(source) => Err(Error::Io {
                path: self.path.clone(),
                source,
            }),
        }
    }

    /// The stored credential of `provider`, without refreshing it.
    pub fn credential(&self, provider: &str) -> Result<Option<Credential>, Error> {
        Ok(self
            .read()?
            .get(provider)
            .and_then(|stored| serde_json::from_value(stored.clone()).ok()))
    }

    /// Provider ids that have a stored credential.
    pub fn providers(&self) -> Result<Vec<String>, Error> {
        Ok(self.read()?.keys().cloned().collect())
    }

    /// Returns OAuth tokens valid for at least five more minutes, refreshing and storing them when needed.
    pub async fn fresh_oauth(&self, provider: &str, client: &reqwest::Client) -> Result<OAuthCredential, Error> {
        self.fresh_oauth_with(
            provider,
            |current| async move { refresh(provider, &current, client).await },
        )
        .await
    }

    pub(crate) async fn fresh_oauth_with<F, Fut>(&self, provider: &str, refresh: F) -> Result<OAuthCredential, Error>
    where
        F: FnOnce(OAuthCredential) -> Fut,
        Fut: Future<Output = Result<OAuthCredential, Error>>,
    {
        let stored = |store: &Self| match store.credential(provider)? {
            Some(Credential::OAuth(credential)) => Ok(credential),
            _ => Err(Error::NoCredential(provider.to_owned())),
        };
        let current = stored(self)?;
        if !current.needs_refresh(now_ms()) {
            return Ok(current);
        }
        let _lock = FileLock::acquire(&self.path).await.map_err(|source| Error::Io {
            path: self.path.clone(),
            source,
        })?;
        // Another process may have refreshed while this one waited for the lock.
        let current = stored(self)?;
        if !current.needs_refresh(now_ms()) {
            return Ok(current);
        }
        let refreshed = refresh(current).await?;
        let mut all = self.read()?;
        all.insert(provider.to_owned(), refreshed.to_json());
        self.write(&all)?;
        Ok(refreshed)
    }

    /// Replaces the file atomically, so a concurrent reader never sees half of it.
    fn write(&self, all: &Map<String, Value>) -> Result<(), Error> {
        let io = |source| Error::Io {
            path: self.path.clone(),
            source,
        };
        let target = fs::canonicalize(&self.path).map_err(io)?;
        let temporary = target.with_extension(format!("json.{}.tmp", std::process::id()));
        let text = serde_json::to_string_pretty(all).expect("JSON values serialize");
        let written = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&temporary)
            .and_then(|mut file| file.write_all(text.as_bytes()).and_then(|()| file.sync_all()))
            .and_then(|()| fs::rename(&temporary, &target));
        written.map_err(|source| {
            let _ = fs::remove_file(&temporary);
            io(source)
        })
    }
}

async fn refresh(
    provider: &str,
    current: &OAuthCredential,
    client: &reqwest::Client,
) -> Result<OAuthCredential, Error> {
    match provider {
        "openai-codex" => refresh_codex(current, client).await,
        _ => Err(Error::RefreshUnsupported(provider.to_owned())),
    }
}

async fn refresh_codex(current: &OAuthCredential, client: &reqwest::Client) -> Result<OAuthCredential, Error> {
    let failed = |detail: String| Error::RefreshFailed {
        provider: "openai-codex".to_owned(),
        detail,
    };
    let form = [
        ("grant_type", "refresh_token"),
        ("refresh_token", current.refresh.as_str()),
        ("client_id", CODEX_CLIENT_ID),
    ];
    let response = client
        .post(CODEX_TOKEN_URL)
        .timeout(REFRESH_TIMEOUT)
        .form(&form)
        .send()
        .await
        .map_err(|error| failed(error.to_string()))?;
    let status = response.status();
    let body = response.text().await.map_err(|error| failed(error.to_string()))?;
    if !status.is_success() {
        return Err(failed(format!(
            "{status}: {}",
            body.chars().take(500).collect::<String>()
        )));
    }
    let token: Value = serde_json::from_str(&body).map_err(|error| failed(error.to_string()))?;
    let field = |name: &str| {
        token[name]
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| failed(format!("the token response has no {name}")))
    };
    let access = field("access_token")?;
    let expires_in = token["expires_in"]
        .as_f64()
        .ok_or_else(|| failed("the token response has no expires_in".into()))?;
    let account_id =
        codex_account_id(&access).ok_or_else(|| failed("the access token names no ChatGPT account".into()))?;
    let mut extra = current.extra.clone();
    extra.insert("accountId".to_owned(), json!(account_id));
    Ok(OAuthCredential {
        access,
        refresh: field("refresh_token")?,
        expires: now_ms() + expires_in * 1000.0,
        extra,
    })
}

/// The ChatGPT account id carried in a Codex access token.
pub fn codex_account_id(access_token: &str) -> Option<String> {
    let payload = access_token.split('.').nth(1)?;
    let claims: Value = serde_json::from_slice(&URL_SAFE_NO_PAD.decode(payload.trim_end_matches('=')).ok()?).ok()?;
    claims[CODEX_JWT_CLAIM]["chatgpt_account_id"]
        .as_str()
        .filter(|id| !id.is_empty())
        .map(str::to_owned)
}

fn now_ms() -> f64 {
    ri_ai::now_ms() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(contents: Value) -> (tempfile::TempDir, AuthStore) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("auth.json");
        fs::write(&path, contents.to_string()).unwrap();
        (directory, AuthStore::new(path))
    }

    fn oauth(access: &str, expires: f64) -> Value {
        json!({ "type": "oauth", "access": access, "refresh": "r1", "expires": expires, "accountId": "acct" })
    }

    #[tokio::test]
    async fn valid_tokens_are_returned_without_refreshing() {
        let (_directory, store) = store(json!({ "openai-codex": oauth("a1", now_ms() + 3_600_000.0) }));
        let credential = store
            .fresh_oauth_with("openai-codex", |_| async { panic!("must not refresh") })
            .await
            .unwrap();
        assert_eq!(credential.access, "a1");
    }

    #[tokio::test]
    async fn expiring_tokens_are_refreshed_and_stored_next_to_the_other_credentials() {
        let (_directory, store) = store(json!({
            "deepseek": { "type": "api_key", "key": "sk-1", "future": true },
            "openai-codex": oauth("a1", now_ms() + 60_000.0),
        }));
        let refreshed = store
            .fresh_oauth_with("openai-codex", |current| async move {
                assert_eq!(current.refresh, "r1");
                Ok(OAuthCredential {
                    access: "a2".into(),
                    refresh: "r2".into(),
                    expires: now_ms() + 3_600_000.0,
                    extra: current.extra,
                })
            })
            .await
            .unwrap();
        assert_eq!(refreshed.access, "a2");

        let file: Value = serde_json::from_str(&fs::read_to_string(store.path()).unwrap()).unwrap();
        assert_eq!(file["openai-codex"]["refresh"], "r2");
        assert_eq!(file["openai-codex"]["accountId"], "acct");
        assert_eq!(
            file["deepseek"],
            json!({ "type": "api_key", "key": "sk-1", "future": true })
        );
        assert_eq!(
            file.as_object().unwrap().keys().collect::<Vec<_>>(),
            ["deepseek", "openai-codex"]
        );
        let mode = std::os::unix::fs::PermissionsExt::mode(&fs::metadata(store.path()).unwrap().permissions());
        assert_eq!(mode & 0o777, 0o600);
        assert!(!store.path().with_extension("json.lock").exists());
    }

    #[tokio::test]
    async fn a_failed_refresh_keeps_the_stored_tokens() {
        let (_directory, store) = store(json!({ "openai-codex": oauth("a1", 0.0) }));
        let before = fs::read_to_string(store.path()).unwrap();
        let result = store
            .fresh_oauth_with("openai-codex", |_| async { Err(Error::RefreshUnsupported("x".into())) })
            .await;
        assert!(result.is_err());
        assert_eq!(fs::read_to_string(store.path()).unwrap(), before);
    }

    #[test]
    fn reads_the_account_id_from_a_codex_token() {
        let claims = URL_SAFE_NO_PAD.encode(json!({ CODEX_JWT_CLAIM: { "chatgpt_account_id": "acct-1" } }).to_string());
        assert_eq!(codex_account_id(&format!("h.{claims}.s")).as_deref(), Some("acct-1"));
        assert_eq!(codex_account_id("not a token"), None);
    }
}
