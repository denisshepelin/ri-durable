//! Asks one question with pi's stored credentials and streams the answer.
//!
//!     cargo run -p ri-pi --example ask -- "What is the capital of France?"
//!     cargo run -p ri-pi --example ask -- --model deepseek/deepseek-flash --thinking high "Why is the sky blue?"
//!     cargo run -p ri-pi --example ask -- --models

use std::io::Write;

use futures_util::StreamExt;
use ri_ai::{AssistantEvent, Context, Message, ModelKind, StopReason, StreamOptions, ThinkingLevel};
use ri_pi::Pi;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pi = Pi::load()?;
    let mut arguments = std::env::args().skip(1);
    let mut model = None;
    let mut thinking: Option<ThinkingLevel> = None;
    let mut prompt = Vec::new();
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--models" => {
                for model in pi
                    .catalog()
                    .models()
                    .iter()
                    .filter(|model| pi.has_credential(&model.provider))
                {
                    let kind = if model.kind == ModelKind::Chat {
                        ""
                    } else {
                        " (classifier or image)"
                    };
                    println!("{}/{}  [{}]{kind}", model.provider, model.id, model.api);
                }
                return Ok(());
            }
            "--model" => model = arguments.next(),
            "--thinking" => thinking = Some(arguments.next().ok_or("--thinking needs a level")?.parse()?),
            _ => prompt.push(argument),
        }
    }
    let model = match &model {
        Some(reference) => pi.model(reference)?,
        None => pi.default_model()?,
    };
    let auth = pi.auth(model).await?;
    let context = Context {
        system_prompt: Some("You are a concise assistant.".to_owned()),
        messages: vec![Message::user(prompt.join(" "))],
        tools: Vec::new(),
    };
    let options = StreamOptions {
        thinking: Some(thinking.unwrap_or_else(|| pi.default_thinking_level())),
        ..StreamOptions::default()
    };

    eprintln!("[{}/{}]", model.provider, model.id);
    let events = ri_ai::stream(pi.client(), model, &context, &options, &auth);
    futures_util::pin_mut!(events);
    let mut stdout = std::io::stdout();
    while let Some(event) = events.next().await {
        match event {
            AssistantEvent::TextDelta { delta, .. } => {
                write!(stdout, "{delta}")?;
                stdout.flush()?;
            }
            AssistantEvent::ThinkingDelta { delta, .. } => eprint!("{delta}"),
            AssistantEvent::Done(message) => {
                println!();
                if message.stop_reason == StopReason::Error {
                    return Err(message.error_message.unwrap_or_default().into());
                }
                let usage = message.usage;
                eprintln!(
                    "[{} in, {} cached, {} out, ${:.5}]",
                    usage.input, usage.cache_read, usage.output, usage.cost.total
                );
            }
            _ => {}
        }
    }
    Ok(())
}
