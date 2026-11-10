//! Classifier models: they read structured JSON state, and optionally images, and answer typed questions with
//! probabilities instead of text. Two wire protocols are spoken: TypeSafe's System One (`typesafe-system-one`,
//! also served by OpenRouter and llama.cpp) and OpenAI Decisions (`openai-decisions`).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::api::{data_url, endpoint};
use crate::stream::error_text;
use crate::types::{JsonObject, Usage};
use crate::{Auth, Error, Model, ModelKind};

pub const TYPESAFE_SYSTEM_ONE: &str = "typesafe-system-one";
pub const OPENAI_DECISIONS: &str = "openai-decisions";

/// The classifier APIs [`classify`] can speak.
pub const CLASSIFIER_APIS: [&str; 2] = [TYPESAFE_SYSTEM_ONE, OPENAI_DECISIONS];

const MAX_DECISIONS_IMAGES: usize = 128;
const MAX_DATA_URL_IMAGES: usize = 4;

#[derive(Debug, Clone, PartialEq)]
pub enum Question {
    /// Pick one of `choices`, given as `(value, what it means)`.
    Choice {
        instructions: String,
        choices: Vec<(String, String)>,
    },
    /// Rate on a scale whose levels are described from lowest to highest.
    Score { instructions: String, levels: Vec<String> },
    /// Yes or no, with what each answer means.
    Bool {
        instructions: String,
        when_true: String,
        when_false: String,
    },
}

impl Question {
    pub fn choice<V: Into<String>, D: Into<String>>(
        instructions: impl Into<String>,
        choices: impl IntoIterator<Item = (V, D)>,
    ) -> Self {
        Question::Choice {
            instructions: instructions.into(),
            choices: choices
                .into_iter()
                .map(|(value, description)| (value.into(), description.into()))
                .collect(),
        }
    }

    pub fn score<L: Into<String>>(instructions: impl Into<String>, levels: impl IntoIterator<Item = L>) -> Self {
        Question::Score {
            instructions: instructions.into(),
            levels: levels.into_iter().map(Into::into).collect(),
        }
    }

    pub fn bool(instructions: impl Into<String>, when_true: impl Into<String>, when_false: impl Into<String>) -> Self {
        Question::Bool {
            instructions: instructions.into(),
            when_true: when_true.into(),
            when_false: when_false.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ClassifierContext {
    pub state: JsonObject,
    /// Images judged together with `state`, as `(mime_type, base64)`. Only models with image input accept them.
    pub images: Vec<(String, String)>,
    pub questions: BTreeMap<String, Question>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Choice {
        choice: String,
        probabilities: BTreeMap<String, f64>,
        confidence: f64,
    },
    Score {
        score: f64,
        confidence: f64,
    },
    /// Probability that the answer is yes.
    Bool {
        probability: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Classification {
    /// One answer per question, under the question's key.
    pub answers: BTreeMap<String, Answer>,
    /// Present when the service reports token counts.
    pub usage: Option<Usage>,
}

/// Answers `context.questions` with a classifier model.
pub async fn classify(
    client: &reqwest::Client,
    model: &Model,
    context: &ClassifierContext,
    auth: &Auth,
) -> Result<Classification, Error> {
    let protocol = Protocol::of(model)?;
    if !context.images.is_empty() && !model.accepts_images() {
        return Err(Error::InvalidRequest(format!(
            "{} does not accept images for {}",
            protocol.label(),
            model.id
        )));
    }
    let (url, extra_headers, body) = match protocol {
        Protocol::SystemOne => system_one_request(model, context)?,
        Protocol::Decisions => (
            endpoint(&model.base_url, "decisions"),
            Vec::new(),
            decisions_request(model, context)?,
        ),
    };
    let headers = model
        .headers
        .iter()
        .chain(&auth.headers)
        .map(|(name, value)| (name.as_str(), value.as_str()));
    let response = extra_headers
        .into_iter()
        .chain(headers)
        .fold(client.post(url).bearer_auth(&auth.api_key), |builder, (name, value)| {
            builder.header(name, value)
        })
        .json(&body)
        .send()
        .await?;
    let status = response.status();
    if !status.is_success() {
        return Err(Error::Http {
            label: protocol.label().to_owned(),
            status: status.as_u16(),
            body: error_text(&response.text().await.unwrap_or_default()),
        });
    }
    let body: Value = response.json().await?;
    let answers = match protocol {
        Protocol::SystemOne => system_one_answers(&body["answers"], context),
        Protocol::Decisions => decisions_answers(&body["answers"], context),
    }
    .map_err(|problem| Error::InvalidResponse(format!("{} {problem}", protocol.label())))?;
    Ok(Classification {
        answers,
        usage: usage(&body["usage"], model),
    })
}

#[derive(Clone, Copy)]
enum Protocol {
    SystemOne,
    Decisions,
}

impl Protocol {
    fn of(model: &Model) -> Result<Self, Error> {
        match (model.kind, model.api.as_str()) {
            (ModelKind::Classifier, TYPESAFE_SYSTEM_ONE) => Ok(Protocol::SystemOne),
            (ModelKind::Classifier, OPENAI_DECISIONS) => Ok(Protocol::Decisions),
            _ => Err(Error::UnsupportedApi {
                api: model.api.clone(),
                model: model.id.clone(),
            }),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Protocol::SystemOne => "System One API",
            Protocol::Decisions => "OpenAI Decisions",
        }
    }
}

type Headers = Vec<(&'static str, &'static str)>;

fn system_one_request(model: &Model, context: &ClassifierContext) -> Result<(String, Headers, Value), Error> {
    let questions: Map<String, Value> = context
        .questions
        .iter()
        .map(|(id, question)| {
            let wire = match question {
                Question::Choice { instructions, choices } => {
                    let criteria: Map<String, Value> = choices.iter().map(|(value, meaning)| (value.clone(), json!(meaning))).collect();
                    json!({ "type": "choice", "instructions": instructions, "criteria": criteria })
                }
                Question::Score { instructions, levels } => json!({ "type": "score", "instructions": instructions, "criteria": levels }),
                // System One calls a yes/no question "noul".
                Question::Bool { instructions, when_true, when_false } => {
                    json!({ "type": "noul", "instructions": instructions, "criteria": { "true": when_true, "false": when_false } })
                }
            };
            (id.clone(), wire)
        })
        .collect();
    let state = Value::Object(context.state.clone());
    let mut body = json!({ "model": model.id, "state": state, "questions": questions });
    let mut headers = Headers::new();
    if !context.images.is_empty() {
        let openrouter = model.provider == "openrouter" || model.base_url.contains("openrouter.ai");
        let form = model
            .compat_str("imageInput")
            .unwrap_or(if openrouter { "state-image-urls" } else { "assets" });
        match form {
            "images" => {
                if context.images.len() > MAX_DATA_URL_IMAGES {
                    return Err(Error::InvalidRequest(format!(
                        "{} accepts at most {MAX_DATA_URL_IMAGES} images",
                        model.id
                    )));
                }
                body["images"] = context
                    .images
                    .iter()
                    .map(|(mime_type, data)| json!(data_url(mime_type, data)))
                    .collect();
            }
            "state-image-urls" => {
                let parts = context
                    .images
                    .iter()
                    .map(|(mime_type, data)| json!({ "type": "image_url", "image_url": { "url": data_url(mime_type, data) } }));
                body["state"] = parts.chain([state]).collect();
            }
            _ => {
                let assets = context
                    .images
                    .iter()
                    .enumerate()
                    .map(|(position, (mime_type, data))| {
                        let format = match mime_type.as_str() {
                            "image/png" => "png",
                            "image/jpeg" => "jpeg",
                            other => {
                                return Err(Error::InvalidRequest(format!(
                                    "{} accepts PNG and JPEG images, got {other}",
                                    model.id
                                )));
                            }
                        };
                        Ok((
                            format!("<image-{}>", position + 1),
                            json!({ "type": "image", "format": format, "data_base64": data }),
                        ))
                    })
                    .collect::<Result<Map<String, Value>, Error>>()?;
                // Every asset must be referenced from the state.
                body["state"] = assets.keys().map(|name| json!(name)).chain([state]).collect();
                body["assets"] = Value::Object(assets);
            }
        }
        if model.compat_bool("sendImageOptInHeader").unwrap_or(!openrouter) {
            headers.push(("TypeSafe-Danger-Zone-Images", "true"));
        }
    }
    Ok((endpoint(&model.base_url, "systemone"), headers, body))
}

fn decisions_request(model: &Model, context: &ClassifierContext) -> Result<Value, Error> {
    if context.images.len() > MAX_DECISIONS_IMAGES {
        return Err(Error::InvalidRequest(format!(
            "OpenAI Decisions accepts at most {MAX_DECISIONS_IMAGES} images"
        )));
    }
    let questions: Vec<Value> = context
        .questions
        .iter()
        .map(|(name, question)| match question {
            Question::Choice { instructions, choices } => {
                let choices: Vec<Value> = choices
                    .iter()
                    .map(|(value, description)| {
                        if description.is_empty() {
                            json!({ "value": value })
                        } else {
                            json!({ "value": value, "description": description })
                        }
                    })
                    .collect();
                json!({ "type": "choice", "name": name, "instructions": instructions, "choices": choices })
            }
            Question::Score { instructions, levels } => {
                let levels: Vec<Value> = levels.iter().map(|label| json!({ "label": label })).collect();
                json!({ "type": "score", "name": name, "instructions": instructions, "levels": levels })
            }
            // Predicates have no criteria field, so the meanings go into the instructions.
            Question::Bool {
                instructions,
                when_true,
                when_false,
            } => {
                let meanings: Vec<String> = [("True", when_true), ("False", when_false)]
                    .into_iter()
                    .filter(|(_, meaning)| !meaning.is_empty())
                    .map(|(label, meaning)| format!("{label} means: {meaning}"))
                    .collect();
                let instructions = if meanings.is_empty() {
                    instructions.clone()
                } else {
                    format!("{instructions}\n\n{}", meanings.join("\n"))
                };
                json!({ "type": "predicate", "name": name, "instructions": instructions })
            }
        })
        .collect();
    let state = Value::Object(context.state.clone()).to_string();
    let input = if context.images.is_empty() {
        json!(state)
    } else {
        let images = context
            .images
            .iter()
            .map(|(mime_type, data)| json!({ "type": "input_image", "image_url": data_url(mime_type, data) }));
        let content: Vec<Value> = std::iter::once(json!({ "type": "input_text", "text": state }))
            .chain(images)
            .collect();
        json!([{ "role": "user", "content": content }])
    };
    Ok(json!({ "model": model.id, "input": input, "questions": questions }))
}

fn number(value: &Value, what: &str) -> Result<f64, String> {
    value
        .as_f64()
        .filter(|number| number.is_finite())
        .ok_or_else(|| format!("returned an invalid {what}"))
}

fn system_one_answers(answers: &Value, context: &ClassifierContext) -> Result<BTreeMap<String, Answer>, String> {
    context
        .questions
        .iter()
        .map(|(id, question)| {
            let answer = answers
                .get(id)
                .filter(|answer| answer.is_object())
                .ok_or_else(|| format!("did not return an answer for {id}"))?;
            let kind = answer["type"].as_str();
            let parsed = match question {
                Question::Choice { .. } if kind == Some("choice") => {
                    let probabilities = answer["probabilities"]
                        .as_object()
                        .ok_or_else(|| format!("returned invalid probabilities for {id}"))?
                        .iter()
                        .map(|(value, probability)| {
                            Ok((
                                value.clone(),
                                number(probability, &format!("probability for {id}.{value}"))?,
                            ))
                        })
                        .collect::<Result<_, String>>()?;
                    choice_answer(id, answer, probabilities)?
                }
                Question::Score { .. } if kind == Some("score") => score_answer(id, answer)?,
                Question::Bool { .. } if kind == Some("noul") => Answer::Bool {
                    probability: number(&answer["noul"], &format!("probability for {id}"))?,
                },
                _ => return Err(format!("returned the wrong kind of answer for {id}")),
            };
            Ok((id.clone(), parsed))
        })
        .collect()
}

fn decisions_answers(answers: &Value, context: &ClassifierContext) -> Result<BTreeMap<String, Answer>, String> {
    let answers = answers.as_array().ok_or("returned an unexpected response")?;
    context
        .questions
        .iter()
        .map(|(id, question)| {
            let answer = answers
                .iter()
                .find(|answer| answer["name"] == id.as_str())
                .ok_or_else(|| format!("did not return an answer for {id}"))?;
            let kind = answer["type"].as_str();
            let parsed = match question {
                _ if kind == Some("refusal") => return Err(format!("refused to answer {id}")),
                Question::Choice { .. } if kind == Some("choice") => {
                    let probabilities = answer["probabilities"]
                        .as_array()
                        .ok_or_else(|| format!("returned invalid probabilities for {id}"))?
                        .iter()
                        .map(|entry| {
                            let value = entry["value"]
                                .as_str()
                                .ok_or_else(|| format!("returned invalid probabilities for {id}"))?;
                            Ok((
                                value.to_owned(),
                                number(&entry["probability"], &format!("probability for {id}.{value}"))?,
                            ))
                        })
                        .collect::<Result<_, String>>()?;
                    choice_answer(id, answer, probabilities)?
                }
                Question::Score { .. } if kind == Some("score") => score_answer(id, answer)?,
                Question::Bool { .. } if kind == Some("predicate") => Answer::Bool {
                    probability: number(&answer["probability"], &format!("probability for {id}"))?,
                },
                _ => return Err(format!("returned the wrong kind of answer for {id}")),
            };
            Ok((id.clone(), parsed))
        })
        .collect()
}

fn choice_answer(id: &str, answer: &Value, probabilities: BTreeMap<String, f64>) -> Result<Answer, String> {
    Ok(Answer::Choice {
        choice: answer["choice"]
            .as_str()
            .ok_or_else(|| format!("did not return a choice for {id}"))?
            .to_owned(),
        probabilities,
        confidence: number(&answer["confidence"], &format!("confidence for {id}"))?,
    })
}

fn score_answer(id: &str, answer: &Value) -> Result<Answer, String> {
    Ok(Answer::Score {
        score: number(&answer["score"], &format!("score for {id}"))?,
        confidence: number(&answer["confidence"], &format!("confidence for {id}"))?,
    })
}

fn usage(raw: &Value, model: &Model) -> Option<Usage> {
    if raw["input_tokens"].is_null() && raw["output_tokens"].is_null() {
        return None;
    }
    let input = raw["input_tokens"].as_u64().unwrap_or(0);
    let output = raw["output_tokens"].as_u64().unwrap_or(0);
    Some(model.price(Usage {
        input,
        output,
        total_tokens: input + output,
        ..Usage::default()
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn jev() -> Model {
        serde_json::from_value(json!({
            "type": "classifier", "id": "jev-latest", "api": "typesafe-system-one", "provider": "typesafe",
            "baseUrl": "https://api.typesafe.ai/v1/", "input": ["text", "image"],
            "compat": { "imageInput": "assets", "sendImageOptInHeader": true }
        }))
        .unwrap()
    }

    fn context() -> ClassifierContext {
        ClassifierContext {
            state: json!({ "message": "thanks, works" }).as_object().unwrap().clone(),
            images: Vec::new(),
            questions: BTreeMap::from([
                (
                    "approved".to_owned(),
                    Question::bool("Does the user approve?", "Approval", "No approval"),
                ),
                (
                    "category".to_owned(),
                    Question::choice("Classify.", [("approval", "approves"), ("correction", "corrects")]),
                ),
                ("satisfaction".to_owned(), Question::score("Score.", ["low", "high"])),
            ]),
        }
    }

    #[test]
    fn system_one_sends_bool_as_noul_and_images_as_assets() {
        let mut context = context();
        context.images.push(("image/png".into(), "AAAA".into()));
        let (url, headers, body) = system_one_request(&jev(), &context).unwrap();
        assert_eq!(url, "https://api.typesafe.ai/v1/systemone");
        assert_eq!(headers, vec![("TypeSafe-Danger-Zone-Images", "true")]);
        assert_eq!(body["questions"]["approved"]["type"], "noul");
        assert_eq!(
            body["questions"]["category"]["criteria"],
            json!({ "approval": "approves", "correction": "corrects" })
        );
        assert_eq!(body["state"], json!(["<image-1>", { "message": "thanks, works" }]));
        assert_eq!(
            body["assets"]["<image-1>"],
            json!({ "type": "image", "format": "png", "data_base64": "AAAA" })
        );
    }

    #[test]
    fn system_one_answers_map_back_to_typed_answers() {
        let answers = json!({
            "approved": { "type": "noul", "noul": 0.9 },
            "category": { "type": "choice", "choice": "approval", "probabilities": { "approval": 0.8, "correction": 0.2 }, "confidence": 0.8 },
            "satisfaction": { "type": "score", "score": 1.7, "confidence": 0.6 },
        });
        let parsed = system_one_answers(&answers, &context()).unwrap();
        assert_eq!(parsed["approved"], Answer::Bool { probability: 0.9 });
        assert_eq!(
            parsed["satisfaction"],
            Answer::Score {
                score: 1.7,
                confidence: 0.6
            }
        );
        assert!(
            system_one_answers(&json!({}), &context())
                .unwrap_err()
                .contains("approved")
        );
    }

    #[test]
    fn decisions_uses_predicates_and_named_answers() {
        let body = decisions_request(&jev(), &context()).unwrap();
        assert_eq!(body["input"], "{\"message\":\"thanks, works\"}");
        assert_eq!(body["questions"][0]["type"], "predicate");
        assert_eq!(
            body["questions"][0]["instructions"],
            "Does the user approve?\n\nTrue means: Approval\nFalse means: No approval"
        );
        let answers = json!([
            { "name": "approved", "type": "predicate", "probability": 0.4 },
            { "name": "category", "type": "choice", "choice": "correction", "probabilities": [{ "value": "correction", "probability": 1.0 }], "confidence": 1.0 },
            { "name": "satisfaction", "type": "score", "score": 0.0, "confidence": 0.5 },
        ]);
        assert_eq!(
            decisions_answers(&answers, &context()).unwrap()["approved"],
            Answer::Bool { probability: 0.4 }
        );
    }
}
