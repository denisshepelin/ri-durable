//! The model answers by writing one script that calls tools, instead of making a tool call per step.
//!
//!     cargo run -p ri-codemode --example codemode -- "Which of these cities is warmest right now?"

use ri_codemode::codemode_tool;
use ri_durable::{AgentConfig, EntryKind, Harness, MemoryStorage, Options, ToolOutput, tool};
use ri_pi::Pi;

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct NoArguments {}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct City {
    /// A city name as returned by `cities`
    city: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cities = tool(
        "cities",
        "The cities weather is known for, as a JSON array of names",
        |_: NoArguments, _| async {
            Ok(ToolOutput::json(serde_json::json!([
                "Oslo",
                "Lisbon",
                "Cairo",
                "Reykjavik",
                "Athens"
            ])))
        },
    );
    let weather = tool(
        "weather",
        "Current temperature in a city, in degrees Celsius, as a number",
        |args: City, _| async move {
            let degrees = match args.city.as_str() {
                "Oslo" => 3,
                "Lisbon" => 17,
                "Cairo" => 26,
                "Reykjavik" => -1,
                "Athens" => 19,
                other => return Err(format!("No weather for {other}")),
            };
            Ok(ToolOutput::json(serde_json::json!(degrees)))
        },
    );
    let tools = vec![cities, weather];
    let specs: Vec<_> = tools.iter().map(|tool| tool.spec().clone()).collect();
    let pi = Pi::load()?;
    let defaults = AgentConfig {
        thinking: Some(pi.default_thinking_level()),
        tools: Some(vec![ri_codemode::TOOL_NAME.to_owned()]),
        ..AgentConfig::default()
    };
    let options = tools
        .into_iter()
        .fold(Options::new(pi).defaults(defaults), Options::tool);
    let harness = Harness::open(
        MemoryStorage::new(),
        options.tool(codemode_tool(&specs, Default::default())),
    )?;
    let root = harness.root()?;

    let prompt = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let answer = root.submit(prompt)?.answer_blocking()?;
    for entry in root.snapshot()?.entries {
        match entry.kind {
            EntryKind::Assistant { message } => {
                for (_, name, arguments) in message.tool_calls() {
                    eprintln!("--- {name} ---\n{}", arguments["code"].as_str().unwrap_or_default());
                }
            }
            EntryKind::ToolResult { content, .. } => eprintln!("--- output ---\n{}", ri_ai::joined_text(&content, "")),
            _ => {}
        }
    }
    println!("{}", answer.text());
    Ok(())
}
