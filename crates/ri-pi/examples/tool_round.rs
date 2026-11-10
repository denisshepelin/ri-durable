//! One tool round by hand: the model calls a tool, gets its result, and answers.
//!
//!     cargo run -p ri-pi --example tool_round -- [provider/model]

use ri_ai::stream::complete;
use ri_ai::{Content, Context, Message, StopReason, StreamOptions, Tool};
use ri_pi::Pi;
use serde_json::json;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pi = Pi::load()?;
    let model = match std::env::args().nth(1) {
        Some(reference) => pi.model(&reference)?.clone(),
        None => pi.default_model()?.clone(),
    };
    let auth = pi.auth(&model).await?;
    let options = StreamOptions {
        thinking: Some(pi.default_thinking_level()),
        session_id: Some("ri-example-tool-round".to_owned()),
        ..StreamOptions::default()
    };
    let mut context = Context {
        system_prompt: Some("Use the tools you are given.".to_owned()),
        messages: vec![Message::user(
            "What is the weather in Paris and in Rome? Answer in one sentence.",
        )],
        tools: vec![Tool {
            name: "weather".to_owned(),
            description: "Current weather in a city".to_owned(),
            parameters: json!({ "type": "object", "properties": { "city": { "type": "string" } }, "required": ["city"] }),
        }],
    };

    loop {
        let answer = complete(pi.client(), &model, &context, &options, &auth).await;
        if answer.stop_reason == StopReason::Error {
            return Err(answer.error_message.unwrap_or_default().into());
        }
        let results: Vec<Message> = answer
            .tool_calls()
            .map(|(id, name, arguments)| {
                println!("-> {name}({})", json!(arguments));
                let city = arguments["city"].as_str().unwrap_or_default();
                let weather = if city == "Paris" {
                    "rain, 9 degrees"
                } else {
                    "sun, 18 degrees"
                };
                Message::ToolResult {
                    tool_call_id: id.to_owned(),
                    tool_name: name.to_owned(),
                    content: vec![Content::text(weather)],
                    is_error: false,
                }
            })
            .collect();
        let text = answer.text();
        context.messages.push(Message::Assistant(answer));
        if results.is_empty() {
            println!("{text}");
            return Ok(());
        }
        context.messages.extend(results);
    }
}
