//! Asks a classifier model typed questions about a piece of state.
//!
//!     cargo run -p ri-pi --example classify -- "The change works perfectly, thanks."

use std::collections::BTreeMap;

use ri_ai::classify::{ClassifierContext, Question, classify};
use ri_pi::Pi;
use serde_json::json;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pi = Pi::load()?;
    let model = pi.model("typesafe/jev-latest")?;
    let auth = pi.auth(model).await?;
    let message = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "The change works perfectly, thanks.".to_owned());
    let context = ClassifierContext {
        state: json!({ "message": message }).as_object().cloned().unwrap_or_default(),
        images: Vec::new(),
        questions: BTreeMap::from([
            (
                "category".to_owned(),
                Question::choice(
                    "Classify the message.",
                    [
                        ("approval", "The user approves of the result"),
                        ("correction", "The user requests a correction"),
                    ],
                ),
            ),
            (
                "satisfaction".to_owned(),
                Question::score("Score user satisfaction.", ["dissatisfied", "neutral", "satisfied"]),
            ),
            (
                "approved".to_owned(),
                Question::bool("Does the user approve?", "Approval", "No approval"),
            ),
        ]),
    };
    let result = classify(pi.client(), model, &context, &auth).await?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
