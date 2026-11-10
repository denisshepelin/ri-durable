//! MCP servers as tools for a [`ri_durable::Harness`], configured the way pi configures them.
//!
//! ```no_run
//! let servers = ri_mcp::config::load(&ri_pi::agent_dir(), None)?;
//! let mcp = ri_mcp::Mcp::connect(servers, Default::default());
//! for (server, problem) in mcp.failures() {
//!     eprintln!("{server}: {problem}");
//! }
//! let tools = mcp.tools(); // pass each to Options::tool(); keep `mcp` alive while they are used
//! # Ok::<(), ri_mcp::Error>(())
//! ```

pub mod client;
pub mod config;

use std::sync::Arc;
use std::time::Duration;

use futures_util::future::join_all;
use ri_ai::{Content, JsonObject};
use ri_durable::tool::BoxFuture;
use ri_durable::{Replay, Tool, ToolContext, ToolOutput, ToolSpec};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::runtime::Runtime;

pub use client::{Client, ClientOptions, ToolInfo};
pub use config::{ServerConfig, Transport};

/// Provider tool names are limited to 64 characters.
const MAX_TOOL_NAME: usize = 64;
/// Text results longer than this are cut in the middle.
const MAX_TEXT_BYTES: usize = 20 * 1024;
/// How long connecting and listing tools may take per server.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Config(String),
    #[error("MCP server `{0}`: {1}")]
    Connect(String, String),
    #[error("MCP server `{0}`: {1}")]
    Request(String, String),
}

/// `mcp__<server>__<tool>` with everything but `[A-Za-z0-9_]` replaced by `_`, as pi and Codex name MCP tools.
/// Names that are too long or `taken` get a hash suffix.
pub fn tool_name(server: &str, tool: &str, taken: impl Fn(&str) -> bool) -> String {
    let name: String = format!("mcp__{server}__{tool}")
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect();
    if name.len() <= MAX_TOOL_NAME && !taken(&name) {
        return name;
    }
    let digest = Sha256::digest(format!("{server}\0{tool}").as_bytes());
    let hash: String = digest.iter().take(4).map(|byte| format!("{byte:02x}")).collect();
    format!("{}_{hash}", &name[..name.len().min(MAX_TOOL_NAME - hash.len() - 1)])
}

/// Connected MCP servers. Dropping it stops the server processes, after which its tools fail.
pub struct Mcp {
    runtime: Option<Runtime>,
    tools: Vec<Arc<dyn Tool>>,
    failures: Vec<(String, String)>,
}

impl Mcp {
    /// Connects to every enabled server at once and lists its tools. Servers that fail are reported in
    /// [`Mcp::failures`]; the others still work. Blocks until all have answered or timed out.
    pub fn connect(servers: Vec<ServerConfig>, options: ClientOptions) -> Mcp {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("ri-mcp")
            .enable_all()
            .build();
        let runtime = match runtime {
            Ok(runtime) => runtime,
            Err(error) => {
                return Mcp {
                    runtime: None,
                    tools: Vec::new(),
                    failures: vec![("runtime".to_owned(), error.to_string())],
                };
            }
        };
        let connecting = servers.into_iter().filter(|server| server.enabled).map(|server| {
            let options = options.clone();
            async move {
                let listed = tokio::time::timeout(STARTUP_TIMEOUT, async {
                    let client = Client::connect(&server, &options).await?;
                    let tools = client.list_tools().await?;
                    Ok::<_, Error>((Arc::new(client), tools))
                })
                .await
                .unwrap_or_else(|_| Err(Error::Connect(server.name.clone(), "did not answer in time".to_owned())));
                (server.name, listed)
            }
        });
        let connected = runtime.block_on(join_all(connecting));
        let (tools, failures) = connected.into_iter().fold(
            (Vec::<Arc<dyn Tool>>::new(), Vec::new()),
            |(mut tools, mut failures), (server, listed)| {
                match listed {
                    Ok((client, listed)) => {
                        for info in listed {
                            let name = tool_name(&server, &info.name, |name| {
                                tools.iter().any(|tool| tool.spec().name == name)
                            });
                            tools.push(Arc::new(McpTool::new(name, client.clone(), info)));
                        }
                    }
                    Err(error) => failures.push((server, error.to_string())),
                }
                (tools, failures)
            },
        );
        Mcp {
            runtime: Some(runtime),
            tools,
            failures,
        }
    }

    /// Every connected server's tools, named `mcp__<server>__<tool>`.
    pub fn tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.clone()
    }

    /// Servers that could not be used, with the reason.
    pub fn failures(&self) -> &[(String, String)] {
        &self.failures
    }
}

impl Drop for Mcp {
    fn drop(&mut self) {
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_background();
        }
    }
}

struct McpTool {
    spec: ToolSpec,
    client: Arc<Client>,
    remote_name: String,
}

impl McpTool {
    fn new(name: String, client: Arc<Client>, info: ToolInfo) -> Self {
        let parameters = if info.input_schema.is_object() {
            info.input_schema
        } else {
            json!({ "type": "object", "properties": {} })
        };
        let spec = ToolSpec {
            name,
            description: info.description,
            parameters,
            replay: Replay::Never,
        };
        McpTool {
            spec,
            client,
            remote_name: info.name,
        }
    }
}

impl Tool for McpTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn call<'a>(&'a self, arguments: JsonObject, context: ToolContext) -> BoxFuture<'a, Result<ToolOutput, String>> {
        Box::pin(async move {
            let result = tokio::select! {
                _ = context.cancel.cancelled() => return Err("The call was aborted.".to_owned()),
                result = self.client.call_tool(&self.remote_name, Value::Object(arguments)) => result,
            };
            result
                .map(|result| tool_output(&result))
                .map_err(|error| error.to_string())
        })
    }
}

/// Keeps the start and end of a long text and says how much was left out.
fn limit_text(text: &str) -> String {
    if text.len() <= MAX_TEXT_BYTES {
        return text.to_owned();
    }
    let half = MAX_TEXT_BYTES / 2;
    let head_end = (0..=half)
        .rev()
        .find(|index| text.is_char_boundary(*index))
        .unwrap_or(0);
    let tail_start = (text.len() - half..text.len())
        .find(|index| text.is_char_boundary(*index))
        .unwrap_or(text.len());
    format!(
        "{}\n\n[... {} bytes left out ...]\n\n{}",
        &text[..head_end],
        tail_start - head_end,
        &text[tail_start..]
    )
}

/// Maps a server's `CallToolResult` to what the model reads.
pub fn tool_output(result: &Value) -> ToolOutput {
    let blocks = result["content"].as_array().map(Vec::as_slice).unwrap_or_default();
    let content: Vec<Content> = blocks
        .iter()
        .filter_map(|block| {
            let text = |text: String| Some(Content::text(limit_text(&text)));
            let image = |data: &Value, mime_type: &Value| {
                Some(Content::Image {
                    data: data.as_str()?.to_owned(),
                    mime_type: mime_type.as_str()?.to_owned(),
                })
            };
            match block["type"].as_str()? {
                "text" => text(block["text"].as_str()?.to_owned()),
                "image" => image(&block["data"], &block["mimeType"]),
                "audio" => text("[audio omitted]".to_owned()),
                "resource_link" => text(format!(
                    "{}: {}",
                    block["name"].as_str().unwrap_or("resource"),
                    block["uri"].as_str().unwrap_or_default()
                )),
                "resource" => {
                    let resource = &block["resource"];
                    let is_image = resource["mimeType"]
                        .as_str()
                        .is_some_and(|kind| kind.starts_with("image/"));
                    match (resource["text"].as_str(), is_image) {
                        (Some(contents), _) => text(contents.to_owned()),
                        (None, true) => image(&resource["blob"], &resource["mimeType"]),
                        (None, false) => text(format!(
                            "[binary resource {} omitted]",
                            resource["uri"].as_str().unwrap_or_default()
                        )),
                    }
                }
                _ => None,
            }
        })
        .collect();
    let content = match (content.is_empty(), result.get("structuredContent")) {
        (true, Some(structured)) if !structured.is_null() => vec![Content::text(limit_text(&structured.to_string()))],
        _ => content,
    };
    let structured = result
        .get("structuredContent")
        .filter(|structured| !structured.is_null())
        .cloned();
    ToolOutput {
        content,
        is_error: result["isError"].as_bool().unwrap_or(false),
        details: None,
        structured,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    #[test]
    fn names_tools_like_pi() {
        assert_eq!(
            tool_name("my-server", "search.docs", |_| false),
            "mcp__my_server__search_docs"
        );
        let long = tool_name("server", &"t".repeat(80), |_| false);
        assert_eq!(long.len(), MAX_TOOL_NAME);
        assert_ne!(tool_name("s", "a-b", |name| name == "mcp__s__a_b"), "mcp__s__a_b");
    }

    #[test]
    fn maps_results_to_model_content() {
        let result = json!({ "content": [
            { "type": "text", "text": "hello" },
            { "type": "image", "data": "AAAA", "mimeType": "image/png" },
            { "type": "resource_link", "name": "spec", "uri": "file:///spec.md" },
            { "type": "resource", "resource": { "uri": "file:///a.txt", "text": "contents" } },
        ], "isError": true });
        let output = tool_output(&result);
        assert!(output.is_error);
        assert_eq!(output.content.len(), 4);
        assert_eq!(output.content[2], Content::text("spec: file:///spec.md"));

        let structured = tool_output(&json!({ "content": [], "structuredContent": { "n": 1 } }));
        assert_eq!(structured.content, vec![Content::text("{\"n\":1}")]);
        let long = tool_output(&json!({ "content": [{ "type": "text", "text": "é".repeat(30_000) }] }));
        assert!(long.content[0].as_text().unwrap().contains("bytes left out"));
    }

    /// A server over an in-memory pipe that pings the client once, lists one tool in two pages and echoes calls.
    async fn serve(stream: tokio::io::DuplexStream) {
        let (reader, mut writer) = tokio::io::split(stream);
        let mut lines = BufReader::new(reader).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let message: Value = serde_json::from_str(&line).unwrap();
            let Some(id) = message.get("id").cloned() else { continue };
            if message.get("method").is_none() {
                continue;
            }
            let params = &message["params"];
            let result = match message["method"].as_str().unwrap() {
                "initialize" => {
                    json!({ "protocolVersion": "2025-06-18", "capabilities": { "tools": {} }, "serverInfo": { "name": "fake", "version": "1" } })
                }
                "tools/list" if params["cursor"].is_null() => {
                    let ping = json!({ "jsonrpc": "2.0", "id": "server-1", "method": "ping" });
                    writer.write_all(format!("{ping}\n").as_bytes()).await.unwrap();
                    json!({ "tools": [{ "name": "echo", "description": "Echo", "inputSchema": { "type": "object" } }], "nextCursor": "2" })
                }
                "tools/list" => json!({ "tools": [{ "name": "slow", "inputSchema": { "type": "object" } }] }),
                "tools/call" if params["name"] == "slow" => continue,
                "tools/call" => {
                    json!({ "content": [{ "type": "text", "text": format!("echo {}", params["arguments"]["text"]) }] })
                }
                _ => {
                    let error = json!({ "jsonrpc": "2.0", "id": id, "error": { "code": -32601, "message": "nope" } });
                    writer.write_all(format!("{error}\n").as_bytes()).await.unwrap();
                    continue;
                }
            };
            let response = json!({ "jsonrpc": "2.0", "id": id, "result": result });
            writer.write_all(format!("{response}\n").as_bytes()).await.unwrap();
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn talks_to_a_server_over_a_stream() {
        let (ours, theirs) = tokio::io::duplex(64 * 1024);
        tokio::spawn(serve(theirs));
        let (reader, writer) = tokio::io::split(ours);
        let client = Client::over_stream("fake", reader, writer, Duration::from_millis(200))
            .await
            .unwrap();

        let tools = client.list_tools().await.unwrap();
        assert_eq!(
            tools.iter().map(|tool| tool.name.as_str()).collect::<Vec<_>>(),
            ["echo", "slow"]
        );

        let result = client.call_tool("echo", json!({ "text": "hi" })).await.unwrap();
        assert_eq!(tool_output(&result).content, vec![Content::text("echo \"hi\"")]);
        assert!(
            client
                .request("unknown/method", json!({}))
                .await
                .unwrap_err()
                .to_string()
                .contains("nope")
        );
        assert!(
            client
                .call_tool("slow", json!({}))
                .await
                .unwrap_err()
                .to_string()
                .contains("timed out")
        );
    }
}
