//! The models pi knows: its cached provider catalogs (`models-store.json`) with the user's own providers and
//! overrides (`models.json`) applied on top.

use std::collections::BTreeMap;
use std::path::Path;

use ri_ai::Model;
use serde_json::{Map, Value, json};

use crate::{Error, read_json};

/// What `models.json` says about one provider, apart from its models.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProviderConfig {
    /// A literal key, or pi's `$VAR` / `!command` syntax.
    pub api_key: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct Catalog {
    models: Vec<Model>,
    providers: BTreeMap<String, ProviderConfig>,
}

impl Catalog {
    /// Reads both files from pi's agent directory. A missing file contributes nothing.
    pub fn load(agent_dir: &Path) -> Result<Catalog, Error> {
        let store = read_json(&agent_dir.join("models-store.json"))?.unwrap_or_else(|| json!({}));
        let custom = read_json(&agent_dir.join("models.json"))?.unwrap_or_else(|| json!({}));
        Ok(Catalog::from_json(&store, &custom))
    }

    pub fn from_json(store: &Value, custom: &Value) -> Catalog {
        let stored = store.as_object().into_iter().flatten().flat_map(|(provider, entry)| {
            let defaults = json!({ "provider": provider });
            entry["models"]
                .as_array()
                .into_iter()
                .flatten()
                .map(move |model| merged(&defaults, model))
        });
        let no_providers = Map::new();
        let custom_providers = custom["providers"].as_object().unwrap_or(&no_providers);
        let models = custom_providers
            .iter()
            .fold(stored.collect::<Vec<Value>>(), |models, (provider, config)| {
                apply_provider(models, provider, config)
            });
        Catalog {
            // Entries this crate cannot describe, such as ones without an API, are left out.
            models: models
                .into_iter()
                .filter_map(|model| serde_json::from_value(model).ok())
                .collect(),
            providers: custom_providers
                .iter()
                .map(|(provider, config)| {
                    (
                        provider.clone(),
                        ProviderConfig {
                            api_key: config["apiKey"].as_str().map(str::to_owned),
                        },
                    )
                })
                .collect(),
        }
    }

    pub fn models(&self) -> &[Model] {
        &self.models
    }

    pub fn get(&self, provider: &str, id: &str) -> Option<&Model> {
        self.models
            .iter()
            .find(|model| model.provider == provider && model.id == id)
    }

    /// Finds a model by `provider/id`, or by a bare id when exactly one provider has it.
    pub fn find(&self, reference: &str) -> Option<&Model> {
        let qualified = reference
            .split_once('/')
            .and_then(|(provider, id)| self.get(provider, id));
        qualified.or_else(|| {
            let mut matches = self.models.iter().filter(|model| model.id == reference);
            matches.next().filter(|_| matches.next().is_none())
        })
    }

    pub fn provider(&self, provider: &str) -> Option<&ProviderConfig> {
        self.providers.get(provider)
    }
}

/// `overlay`'s fields over `base`'s; nested objects are merged, everything else is replaced.
fn merged(base: &Value, overlay: &Value) -> Value {
    match (base, overlay) {
        (Value::Object(base), Value::Object(overlay)) => {
            let mut result = base.clone();
            for (key, value) in overlay {
                let combined = result
                    .get(key)
                    .map_or_else(|| value.clone(), |existing| merged(existing, value));
                result.insert(key.clone(), combined);
            }
            Value::Object(result)
        }
        _ => overlay.clone(),
    }
}

fn is_model(model: &Value, provider: &str, id: &str) -> bool {
    model["provider"] == provider && model["id"] == id
}

/// Applies one `models.json` provider: its connection settings reach the provider's known models, its own models
/// are added or replace same-id ones, and `modelOverrides` are applied last.
fn apply_provider(models: Vec<Value>, provider: &str, config: &Value) -> Vec<Value> {
    let connection: Map<String, Value> = ["baseUrl", "headers", "compat"]
        .into_iter()
        .filter_map(|key| config.get(key).map(|value| (key.to_owned(), value.clone())))
        .collect();
    let connection = Value::Object(connection);
    let own_defaults = merged(&json!({ "provider": provider, "api": config["api"] }), &connection);
    let own: Vec<Value> = config["models"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|model| {
            let model = merged(&own_defaults, model);
            if model["name"].is_null() {
                merged(&model, &json!({ "name": model["id"] }))
            } else {
                model
            }
        })
        .collect();
    let replaced = |model: &Value| {
        own.iter()
            .any(|custom| model["provider"] == provider && model["id"] == custom["id"])
    };
    let overrides = config["modelOverrides"].as_object();
    models
        .into_iter()
        .filter(|model| !replaced(model))
        .map(|model| {
            if model["provider"] == provider {
                merged(&model, &connection)
            } else {
                model
            }
        })
        .chain(own.iter().cloned())
        .map(|model| {
            let patch = overrides.and_then(|overrides| overrides.iter().find(|(id, _)| is_model(&model, provider, id)));
            patch.map_or(model.clone(), |(_, patch)| merged(&model, patch))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> Value {
        json!({
            "deepseek": { "models": [
                { "id": "deepseek-flash", "api": "openai-completions", "provider": "deepseek", "baseUrl": "https://api.deepseek.com", "contextWindow": 1000 },
                { "id": "broken" },
            ] },
            "openai": { "models": [{ "id": "deepseek-flash", "api": "openai-responses", "baseUrl": "https://api.openai.com/v1" }] },
        })
    }

    #[test]
    fn reads_stored_models_and_skips_unusable_entries() {
        let catalog = Catalog::from_json(&store(), &json!({}));
        assert_eq!(catalog.models().len(), 2);
        assert_eq!(catalog.get("openai", "deepseek-flash").unwrap().provider, "openai");
        assert!(
            catalog.find("deepseek-flash").is_none(),
            "a bare id two providers share is ambiguous"
        );
        assert_eq!(catalog.find("deepseek/deepseek-flash").unwrap().context_window, 1000);
    }

    #[test]
    fn applies_custom_providers_and_overrides() {
        let custom = json!({ "providers": {
            "deepseek": { "baseUrl": "http://proxy", "modelOverrides": { "deepseek-flash": { "contextWindow": 5, "compat": { "supportsStore": true } } } },
            "local": { "baseUrl": "http://localhost:8080/v1", "api": "openai-completions", "apiKey": "$LOCAL_KEY",
                "models": [{ "id": "qwen", "reasoning": true }] },
        } });
        let catalog = Catalog::from_json(&store(), &custom);
        let deepseek = catalog.get("deepseek", "deepseek-flash").unwrap();
        assert_eq!(
            (deepseek.base_url.as_str(), deepseek.context_window),
            ("http://proxy", 5)
        );
        assert_eq!(deepseek.compat_bool("supportsStore"), Some(true));
        let local = catalog.find("qwen").unwrap();
        assert_eq!(
            (local.provider.as_str(), local.api.as_str(), local.base_url.as_str()),
            ("local", "openai-completions", "http://localhost:8080/v1")
        );
        assert_eq!(local.name, "qwen");
        assert_eq!(
            catalog.provider("local").unwrap().api_key.as_deref(),
            Some("$LOCAL_KEY")
        );
    }
}
