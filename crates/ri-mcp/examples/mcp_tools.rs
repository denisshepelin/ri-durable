//! Lists the tools of the MCP servers in pi's `mcp.json`, or asks the default model a question with them.
//!
//!     cargo run -p ri-mcp --example mcp_tools -- --server browse
//!     cargo run -p ri-mcp --example mcp_tools -- --server browse "Which tools do you have? Name three."

use ri_durable::{Harness, MemoryStorage, Options};
use ri_mcp::{ClientOptions, Mcp};
use ri_pi::Pi;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1).peekable();
    let mut only = Vec::new();
    while arguments.peek().is_some_and(|argument| argument == "--server") {
        arguments.next();
        only.extend(arguments.next());
    }
    let prompt = arguments.collect::<Vec<_>>().join(" ");

    let servers = ri_mcp::config::load(&ri_pi::agent_dir(), None)?;
    let servers = servers
        .into_iter()
        .filter(|server| only.is_empty() || only.contains(&server.name))
        .collect();
    let mcp = Mcp::connect(
        servers,
        ClientOptions {
            cwd: std::env::current_dir().ok(),
        },
    );
    for (server, problem) in mcp.failures() {
        eprintln!("{server}: {problem}");
    }
    if prompt.is_empty() {
        for tool in mcp.tools() {
            let description = tool
                .spec()
                .description
                .lines()
                .next()
                .unwrap_or_default()
                .chars()
                .take(90)
                .collect::<String>();
            println!("{}  {description}", tool.spec().name);
        }
        return Ok(());
    }

    let options = mcp.tools().into_iter().fold(Options::new(Pi::load()?), Options::tool);
    let harness = Harness::open(MemoryStorage::new(), options)?;
    let answer = harness.root()?.submit(prompt)?.answer_blocking()?;
    println!("{}", answer.text());
    Ok(())
}
