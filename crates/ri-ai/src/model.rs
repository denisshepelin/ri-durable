//! Model descriptions in the JSON shape pi stores in `models-store.json` and `models.json`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::types::{Cost, JsonObject, ThinkingLevel, Usage};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelKind {
    #[default]
    Chat,
    Image,
    Classifier,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    pub id: String,
    #[serde(default)]
    pub name: String,
    /// Wire protocol, for example `openai-completions` or `openai-codex-responses`.
    pub api: String,
    #[serde(default)]
    pub provider: String,
    #[serde(default)]
    pub base_url: String,
    #[serde(rename = "type", default)]
    pub kind: ModelKind,
    #[serde(default)]
    pub reasoning: bool,
    #[serde(default)]
    pub input: Vec<String>,
    #[serde(default)]
    pub cost: ModelCost,
    #[serde(default = "default_context_window")]
    pub context_window: u64,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u64,
    /// Thinking level to the provider's effort name. A `None` value marks a level the model does not support;
    /// a missing level is sent under its own name.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub thinking_level_map: BTreeMap<String, Option<String>>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
    /// Provider quirks, read with [`Model::compat_bool`] and [`Model::compat_str`].
    #[serde(default, skip_serializing_if = "JsonObject::is_empty")]
    pub compat: JsonObject,
}

fn default_context_window() -> u64 {
    128_000
}

fn default_max_tokens() -> u64 {
    16_384
}

/// US dollars per million tokens.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCost {
    #[serde(default)]
    pub input: f64,
    #[serde(default)]
    pub output: f64,
    #[serde(default)]
    pub cache_read: f64,
    #[serde(default)]
    pub cache_write: f64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tiers: Vec<CostTier>,
}

/// Rates that replace the base rates once a request's input exceeds `input_tokens_above`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CostTier {
    pub input_tokens_above: u64,
    #[serde(default)]
    pub input: f64,
    #[serde(default)]
    pub output: f64,
    #[serde(default)]
    pub cache_read: f64,
    #[serde(default)]
    pub cache_write: f64,
}

impl Model {
    pub fn accepts_images(&self) -> bool {
        self.input.iter().any(|kind| kind == "image")
    }

    pub fn compat_bool(&self, key: &str) -> Option<bool> {
        self.compat.get(key).and_then(Value::as_bool)
    }

    pub fn compat_str(&self, key: &str) -> Option<&str> {
        self.compat.get(key).and_then(Value::as_str)
    }

    pub fn supports_thinking_level(&self, level: ThinkingLevel) -> bool {
        match level {
            ThinkingLevel::Off => !matches!(self.thinking_level_map.get("off"), Some(None)),
            _ => self.reasoning && !matches!(self.thinking_level_map.get(level.as_str()), Some(None)),
        }
    }

    /// The supported level closest to `level`, preferring the next higher one.
    pub fn clamp_thinking_level(&self, level: ThinkingLevel) -> ThinkingLevel {
        if !self.reasoning {
            return ThinkingLevel::Off;
        }
        let supported = |candidate: &ThinkingLevel| self.supports_thinking_level(*candidate);
        ThinkingLevel::ALL
            .into_iter()
            .filter(|candidate| *candidate >= level)
            .find(supported)
            .or_else(|| {
                ThinkingLevel::ALL
                    .into_iter()
                    .rev()
                    .filter(|candidate| *candidate < level)
                    .find(supported)
            })
            .unwrap_or(ThinkingLevel::Off)
    }

    /// The provider's effort name for `level` after clamping, or `None` when no effort should be sent.
    pub fn effort(&self, level: ThinkingLevel) -> Option<String> {
        let level = self.clamp_thinking_level(level);
        match self.thinking_level_map.get(level.as_str()) {
            Some(mapped) => mapped.clone(),
            None if level == ThinkingLevel::Off => Some("none".to_owned()),
            None => Some(level.as_str().to_owned()),
        }
    }

    /// Fills `usage.cost` from this model's rates, using the highest tier the request's input exceeds.
    pub fn price(&self, usage: Usage) -> Usage {
        let input_tokens = usage.input + usage.cache_read + usage.cache_write;
        let base = (
            self.cost.input,
            self.cost.output,
            self.cost.cache_read,
            self.cost.cache_write,
        );
        let (input, output, cache_read, cache_write) = self
            .cost
            .tiers
            .iter()
            .filter(|tier| input_tokens > tier.input_tokens_above)
            .max_by_key(|tier| tier.input_tokens_above)
            .map_or(base, |tier| {
                (tier.input, tier.output, tier.cache_read, tier.cache_write)
            });
        let per_token = |rate: f64, tokens: u64| rate / 1_000_000.0 * tokens as f64;
        let cost = Cost {
            input: per_token(input, usage.input),
            output: per_token(output, usage.output),
            cache_read: per_token(cache_read, usage.cache_read),
            cache_write: per_token(cache_write, usage.cache_write),
            total: 0.0,
        };
        Usage {
            cost: Cost {
                total: cost.input + cost.output + cost.cache_read + cost.cache_write,
                ..cost
            },
            ..usage
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deepseek() -> Model {
        serde_json::from_value(serde_json::json!({
            "id": "deepseek-flash",
            "api": "openai-completions",
            "reasoning": true,
            "thinkingLevelMap": { "minimal": null, "low": "low", "medium": null, "high": "high", "max": "max" },
            "cost": { "input": 1.0, "output": 2.0, "cacheRead": 0.5, "cacheWrite": 0.0,
                "tiers": [{ "inputTokensAbove": 100, "input": 10.0, "output": 20.0, "cacheRead": 5.0, "cacheWrite": 0.0 }] }
        }))
        .unwrap()
    }

    #[test]
    fn clamps_unsupported_levels_upwards_then_downwards() {
        let model = deepseek();
        assert_eq!(model.clamp_thinking_level(ThinkingLevel::Medium), ThinkingLevel::High);
        assert_eq!(model.clamp_thinking_level(ThinkingLevel::Minimal), ThinkingLevel::Low);
        assert_eq!(model.effort(ThinkingLevel::Medium).as_deref(), Some("high"));
        assert_eq!(model.effort(ThinkingLevel::Off).as_deref(), Some("none"));
    }

    #[test]
    fn prices_with_the_matching_tier() {
        let model = deepseek();
        let small = model.price(Usage {
            input: 50,
            output: 10,
            ..Usage::default()
        });
        assert!((small.cost.total - (50.0 + 20.0) / 1e6).abs() < 1e-12);
        let large = model.price(Usage {
            input: 80,
            cache_read: 40,
            output: 10,
            ..Usage::default()
        });
        assert!((large.cost.total - (800.0 + 200.0 + 200.0) / 1e6).abs() < 1e-12);
    }
}
