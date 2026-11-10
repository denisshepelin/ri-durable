//! A persistent chat with pi's default model and one tool. The conversation is stored under
//! `~/.local/share/ri-durable/example-chat`, so each run continues the last one.
//!
//!     cargo run -p ri-durable --example chat -- "What time is it?"
//!     cargo run -p ri-durable --example chat -- "What did I just ask you?"
//!     cargo run -p ri-durable --example chat -- --reset

use std::io::Write;

use ri_ai::AssistantEvent;
use ri_durable::{AgentConfig, Event, Harness, JsonlStorage, Options, ToolOutput, Update, data_dir, tool};
use ri_pi::Pi;

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct Clock {}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pi = Pi::load()?;
    let clock = tool("clock", "The current Unix time in seconds", |_: Clock, _| async {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|error| error.to_string())?;
        Ok(ToolOutput::text(now.as_secs().to_string()))
    });
    let defaults = AgentConfig {
        thinking: Some(pi.default_thinking_level()),
        instructions: Some("You are a concise assistant.".to_owned()),
        ..AgentConfig::default()
    };
    let storage = JsonlStorage::open(data_dir("example-chat"))?;
    let harness = Harness::open(storage, Options::new(pi).tool(clock).defaults(defaults))?;
    let root = harness.root()?;

    let prompt = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if prompt == "--reset" {
        root.reset_blocking(None)?;
        println!("Started a new context.");
        return Ok(());
    }

    let (_, mut updates) = root.watch()?;
    let submission = root.submit(prompt)?;
    let printer = std::thread::spawn(move || {
        let mut stdout = std::io::stdout();
        while let Some(update) = updates.next_blocking() {
            match update {
                Update::Partial {
                    event: AssistantEvent::TextDelta { delta, .. },
                    ..
                } => {
                    let _ = write!(stdout, "{delta}");
                    let _ = stdout.flush();
                }
                Update::Partial {
                    event: AssistantEvent::ToolCallStart { name, .. },
                    ..
                } => eprintln!("[calling {name}]"),
                Update::Event(Event::RunEnded { .. }) => break,
                _ => {}
            }
        }
    });
    let answered = submission.answer_blocking();
    let _ = printer.join();
    println!();
    answered?;
    for (model, usage) in harness.usage() {
        eprintln!(
            "[{model}: {} tokens, ${:.4} in this session]",
            usage.total_tokens, usage.cost.total
        );
    }
    Ok(())
}
