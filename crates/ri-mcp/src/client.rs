//! A minimal MCP client: initialize, list tools, call tools. It speaks JSON-RPC over a byte stream (a local
//! process's standard input and output) or over Streamable HTTP.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::StreamExt;
use ri_ai::sse::SseParser;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::oneshot;

use crate::Error;
use crate::config::{ServerConfig, Transport};

const PROTOCOL_VERSION: &str = "2025-06-18";

type Pending = Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value, String>>>>>;
type Writer = Arc<tokio::sync::Mutex<Box<dyn AsyncWrite + Send + Unpin>>>;

/// A tool as its server describes it.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolInfo {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

enum Wire {
    Stream {
        writer: Writer,
        pending: Pending,
        _child: Option<Child>,
    },
    Http {
        client: reqwest::Client,
        url: String,
        headers: BTreeMap<String, String>,
        session: Mutex<Option<String>>,
    },
}

pub struct Client {
    name: String,
    timeout: Duration,
    wire: Wire,
    next_id: AtomicU64,
}

/// Where the client runs, told to servers that ask for roots.
#[derive(Debug, Clone, Default)]
pub struct ClientOptions {
    pub cwd: Option<PathBuf>,
}

fn rpc_error(error: &Value) -> String {
    error["message"]
        .as_str()
        .map_or_else(|| error.to_string(), str::to_owned)
}

impl Client {
    /// Starts or reaches the server and performs the MCP handshake. Must run inside a Tokio runtime.
    pub async fn connect(config: &ServerConfig, options: &ClientOptions) -> Result<Client, Error> {
        let resolve = |values: &BTreeMap<String, String>| {
            values
                .iter()
                .map(|(key, value)| {
                    let resolved = ri_pi::config_value::resolve(value, &BTreeMap::new());
                    resolved.map(|resolved| (key.clone(), resolved)).ok_or_else(|| {
                        Error::Config(format!(
                            "MCP server `{}`: `{key}` references a variable that is not set",
                            config.name
                        ))
                    })
                })
                .collect::<Result<BTreeMap<String, String>, Error>>()
        };
        let wire = match &config.transport {
            Transport::Stdio {
                command,
                args,
                env,
                cwd,
            } => {
                let mut process = Command::new(command);
                process
                    .args(args)
                    .envs(resolve(env)?)
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::null());
                if let Some(cwd) = cwd.as_ref().or(options.cwd.as_ref()) {
                    process.current_dir(cwd);
                }
                let mut child = process.kill_on_drop(true).spawn().map_err(|error| {
                    Error::Connect(config.name.clone(), format!("cannot start `{command}`: {error}"))
                })?;
                let (stdin, stdout) = (
                    child.stdin.take().expect("piped stdin"),
                    child.stdout.take().expect("piped stdout"),
                );
                stream_wire(stdout, stdin, options.clone(), Some(child))
            }
            Transport::Http { url, headers } => Wire::Http {
                client: reqwest::Client::new(),
                url: url.clone(),
                headers: resolve(headers)?,
                session: Mutex::new(None),
            },
        };
        let client = Client {
            name: config.name.clone(),
            timeout: config.timeout,
            wire,
            next_id: AtomicU64::new(1),
        };
        client.initialize().await?;
        Ok(client)
    }

    /// A client over an already open byte stream. Must run inside a Tokio runtime.
    pub async fn over_stream(
        name: &str,
        reader: impl AsyncRead + Send + Unpin + 'static,
        writer: impl AsyncWrite + Send + Unpin + 'static,
        timeout: Duration,
    ) -> Result<Client, Error> {
        let wire = stream_wire(reader, writer, ClientOptions::default(), None);
        let client = Client {
            name: name.to_owned(),
            timeout,
            wire,
            next_id: AtomicU64::new(1),
        };
        client.initialize().await?;
        Ok(client)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    async fn initialize(&self) -> Result<(), Error> {
        let params = json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": { "roots": {} },
            "clientInfo": { "name": "ri-mcp", "version": env!("CARGO_PKG_VERSION") },
        });
        self.request("initialize", params).await?;
        self.notify("notifications/initialized", json!({})).await
    }

    pub async fn list_tools(&self) -> Result<Vec<ToolInfo>, Error> {
        let mut tools = Vec::new();
        let mut cursor: Option<String> = None;
        loop {
            let params = cursor
                .as_ref()
                .map_or_else(|| json!({}), |cursor| json!({ "cursor": cursor }));
            let page = self.request("tools/list", params).await?;
            tools.extend(page["tools"].as_array().into_iter().flatten().filter_map(|tool| {
                Some(ToolInfo {
                    name: tool["name"].as_str()?.to_owned(),
                    description: tool["description"].as_str().unwrap_or_default().to_owned(),
                    input_schema: tool["inputSchema"].clone(),
                })
            }));
            cursor = page["nextCursor"].as_str().map(str::to_owned);
            if cursor.is_none() {
                return Ok(tools);
            }
        }
    }

    /// Calls a tool and returns the server's `CallToolResult`.
    pub async fn call_tool(&self, tool: &str, arguments: Value) -> Result<Value, Error> {
        self.request("tools/call", json!({ "name": tool, "arguments": arguments }))
            .await
    }

    async fn notify(&self, method: &str, params: Value) -> Result<(), Error> {
        let message = json!({ "jsonrpc": "2.0", "method": method, "params": params });
        match &self.wire {
            Wire::Stream { writer, .. } => write_line(writer, &message)
                .await
                .map_err(|error| self.failed(error.to_string())),
            Wire::Http { .. } => self.post(&message, None).await.map(|_| ()),
        }
    }

    fn failed(&self, detail: String) -> Error {
        Error::Request(self.name.clone(), detail)
    }

    pub async fn request(&self, method: &str, params: Value) -> Result<Value, Error> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let message = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        let answered = async {
            match &self.wire {
                Wire::Stream { writer, pending, .. } => {
                    let (sender, receiver) = oneshot::channel();
                    pending.lock().expect("pending lock").insert(id, sender);
                    write_line(writer, &message)
                        .await
                        .map_err(|error| self.failed(error.to_string()))?;
                    receiver
                        .await
                        .map_err(|_| self.failed("the server closed the connection".to_owned()))?
                        .map_err(|error| self.failed(error))
                }
                Wire::Http { .. } => self.post(&message, Some(id)).await,
            }
        };
        match tokio::time::timeout(self.timeout, answered).await {
            Ok(result) => result,
            Err(_) => {
                if let Wire::Stream { pending, .. } = &self.wire {
                    pending.lock().expect("pending lock").remove(&id);
                }
                let _ = self
                    .notify(
                        "notifications/cancelled",
                        json!({ "requestId": id, "reason": "timeout" }),
                    )
                    .await;
                Err(self.failed(format!("`{method}` timed out after {} s", self.timeout.as_secs())))
            }
        }
    }

    /// Posts one message. With `id`, returns that request's result from the JSON or event-stream response.
    async fn post(&self, message: &Value, id: Option<u64>) -> Result<Value, Error> {
        let Wire::Http {
            client,
            url,
            headers,
            session,
        } = &self.wire
        else {
            unreachable!("post is only used over HTTP")
        };
        let session_id = session.lock().expect("session lock").clone();
        let request = headers
            .iter()
            .fold(client.post(url), |request, (name, value)| request.header(name, value))
            .header("accept", "application/json, text/event-stream")
            .header("mcp-protocol-version", PROTOCOL_VERSION);
        let request = session_id.iter().fold(request, |request, session_id| {
            request.header("mcp-session-id", session_id)
        });
        let response = request
            .json(message)
            .send()
            .await
            .map_err(|error| self.failed(error.to_string()))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            let hint = if status.as_u16() == 401 {
                " (sign in with `pi mcp login` or configure a header)"
            } else {
                ""
            };
            return Err(self.failed(format!(
                "{status}{hint}: {}",
                body.chars().take(300).collect::<String>()
            )));
        }
        if let Some(assigned) = response
            .headers()
            .get("mcp-session-id")
            .and_then(|value| value.to_str().ok())
        {
            *session.lock().expect("session lock") = Some(assigned.to_owned());
        }
        let Some(id) = id else { return Ok(Value::Null) };
        let answer = |message: &Value| {
            (message["id"].as_u64() == Some(id)).then(|| match message.get("error") {
                Some(error) => Err(self.failed(rpc_error(error))),
                None => Ok(message["result"].clone()),
            })
        };
        let is_stream = response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .is_some_and(|kind| kind.contains("text/event-stream"));
        if !is_stream {
            let body: Value = response.json().await.map_err(|error| self.failed(error.to_string()))?;
            return answer(&body)
                .unwrap_or_else(|| Err(self.failed("the response does not answer the request".to_owned())));
        }
        let mut parser = SseParser::default();
        let mut body = response.bytes_stream();
        while let Some(chunk) = body.next().await {
            let chunk = chunk.map_err(|error| self.failed(error.to_string()))?;
            let found = parser
                .push(&chunk)
                .iter()
                .filter_map(|event| serde_json::from_str::<Value>(&event.data).ok())
                .find_map(|message| answer(&message));
            if let Some(result) = found {
                return result;
            }
        }
        Err(self.failed("the event stream ended without an answer".to_owned()))
    }
}

async fn write_line(writer: &Writer, message: &Value) -> std::io::Result<()> {
    let mut line = message.to_string().into_bytes();
    line.push(b'\n');
    let mut writer = writer.lock().await;
    writer.write_all(&line).await?;
    writer.flush().await
}

fn stream_wire(
    reader: impl AsyncRead + Send + Unpin + 'static,
    writer: impl AsyncWrite + Send + Unpin + 'static,
    options: ClientOptions,
    child: Option<Child>,
) -> Wire {
    let writer: Writer = Arc::new(tokio::sync::Mutex::new(Box::new(writer)));
    let pending: Pending = Arc::new(Mutex::new(HashMap::new()));
    tokio::spawn(read_messages(reader, writer.clone(), pending.clone(), options));
    Wire::Stream {
        writer,
        pending,
        _child: child,
    }
}

/// Routes the server's messages: responses to their waiting requests, the few requests a server may make to their
/// answers. Notifications are ignored.
async fn read_messages(reader: impl AsyncRead + Unpin, writer: Writer, pending: Pending, options: ClientOptions) {
    let mut lines = BufReader::new(reader).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        let Ok(message) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        match (message.get("id"), message["method"].as_str()) {
            (Some(id), Some(method)) => {
                let reply = match method {
                    "ping" => json!({ "jsonrpc": "2.0", "id": id, "result": {} }),
                    "roots/list" => {
                        let roots: Vec<Value> = options
                            .cwd
                            .iter()
                            .map(|cwd| json!({ "uri": format!("file://{}", cwd.display()) }))
                            .collect();
                        json!({ "jsonrpc": "2.0", "id": id, "result": { "roots": roots } })
                    }
                    _ => {
                        json!({ "jsonrpc": "2.0", "id": id, "error": { "code": -32601, "message": "Method not found" } })
                    }
                };
                if write_line(&writer, &reply).await.is_err() {
                    break;
                }
            }
            (Some(id), None) => {
                let waiting = id
                    .as_u64()
                    .and_then(|id| pending.lock().expect("pending lock").remove(&id));
                if let Some(sender) = waiting {
                    let result = match message.get("error") {
                        Some(error) => Err(rpc_error(error)),
                        None => Ok(message["result"].clone()),
                    };
                    let _ = sender.send(result);
                }
            }
            _ => {}
        }
    }
    // Dropping the senders fails every request still waiting.
    pending.lock().expect("pending lock").clear();
}
