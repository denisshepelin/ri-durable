//! Code mode: instead of emitting one tool call per step, the model writes JavaScript that calls the tools as
//! `await tools.<name>(args)`, and only the script's output comes back to it.
//!
//! Scripts run in a QuickJS sandbox ([`sandbox`]) with no access to the host other than the tools. A script
//! survives a restart: the sandbox is deterministic and every result handed to the script is journaled first, so
//! after a crash the script runs again, gets the journaled results in the same order without the tools running
//! again, and continues from where it was.
//!
//! ```no_run
//! # use ri_durable::{Options, Tool};
//! # fn example(options: Options, tools: Vec<std::sync::Arc<dyn Tool>>) -> Options {
//! let specs: Vec<_> = tools.iter().map(|tool| tool.spec().clone()).collect();
//! let options = tools.into_iter().fold(options, Options::tool);
//! options.tool(ri_codemode::codemode_tool(&specs, Default::default()))
//! # }
//! ```
//!
//! To have the model work only through scripts, offer it just this tool with
//! `AgentConfig { tools: Some(vec!["codemode".into()]), .. }`; scripts can still call every registered tool.

pub mod declare;
pub mod sandbox;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;
use futures_util::stream::FuturesUnordered;
use ri_ai::{Content, JsonObject};
use ri_durable::tool::BoxFuture;
use ri_durable::{Replay, Tool, ToolContext, ToolOutput, ToolSpec};
use serde_json::{Value, json};

use sandbox::{CatalogEntry, Determinism, HostCall, Limits};

pub const TOOL_NAME: &str = "codemode";

#[derive(Debug, Clone, PartialEq)]
pub struct CodemodeOptions {
    pub limits: Limits,
    /// Bytes of tool declarations shown in the tool's description. Tools that do not fit are found by scripts with
    /// `searchTools()` and `describeTool()`.
    pub declaration_budget: usize,
    /// Bytes of script output the model gets; longer output is cut in the middle.
    pub max_output_bytes: usize,
}

impl Default for CodemodeOptions {
    fn default() -> Self {
        CodemodeOptions {
            limits: Limits::default(),
            declaration_budget: 12_000,
            max_output_bytes: 40_000,
        }
    }
}

/// The `codemode` tool over the tools described by `specs`. Scripts call them through the harness, so the tools
/// must be registered with it too.
pub fn codemode_tool(specs: &[ToolSpec], options: CodemodeOptions) -> Arc<dyn Tool> {
    let callable: Vec<&ToolSpec> = specs.iter().filter(|spec| spec.name != TOOL_NAME).collect();
    let owned: Vec<ToolSpec> = callable.iter().map(|spec| (*spec).clone()).collect();
    let (declarations, hidden) = declare::declarations(&owned, options.declaration_budget);
    let more = if hidden.is_empty() {
        String::new()
    } else {
        format!(
            "\n\n{} more tools are not listed: {}. Find them with `searchTools(query)`, which returns `{{ name, description }}[]`, and read a tool's signature with `describeTool(name)`.",
            hidden.len(),
            hidden.join(", ")
        )
    };
    let description = format!(
        "Run JavaScript that calls tools as functions. Prefer one script over many separate tool calls: loop, branch, \
         run calls in parallel with Promise.all, and return only what matters.\n\n\
         `code` is the body of an async function in plain JavaScript (no TypeScript, no imports, no timers, no network). \
         Call tools with `await tools.<name>(args)`; a failed call throws. A tool's structured result comes back as that value; \
         otherwise its text comes back as a string, parsed when it is a JSON object or array. `print(...)` adds to the output, and the value the script returns is appended to it.\n\n\
         ```ts\n{declarations}\n```{more}"
    );
    let spec = ToolSpec {
        name: TOOL_NAME.to_owned(),
        description,
        parameters: json!({
            "type": "object",
            "required": ["code"],
            "properties": {
                "code": { "type": "string", "description": "The body of an async JavaScript function" },
                "timeout_ms": { "type": "integer", "description": "Wall time the script may take, at most the configured limit" },
            },
        }),
        // A rerun replays the journal, so finished tool calls do not run twice.
        replay: Replay::Safe,
    };
    let catalog = owned
        .iter()
        .map(|spec| CatalogEntry {
            name: spec.name.clone(),
            description: spec.description.lines().next().unwrap_or_default().to_owned(),
            declaration: declare::declaration(spec),
        })
        .collect();
    let replay = owned.iter().map(|spec| (spec.name.clone(), spec.replay)).collect();
    Arc::new(Codemode {
        spec,
        options,
        catalog,
        replay,
    })
}

struct Codemode {
    spec: ToolSpec,
    options: CodemodeOptions,
    catalog: Vec<CatalogEntry>,
    replay: BTreeMap<String, Replay>,
}

/// A stable fingerprint of text, to notice that a rerun is not the run it claims to continue.
fn fingerprint(text: &str) -> String {
    let hash = text.bytes().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    });
    format!("{hash:016x}")
}

/// What a script gets for a tool's result: its structured value when it has one, a lone text as a string, a lone
/// image as an object, otherwise the content blocks. A failed call rejects with its text.
fn script_value(output: &ToolOutput) -> Result<Value, String> {
    if let Some(structured) = &output.structured {
        return Ok(structured.clone());
    }
    if output.is_error {
        return Err(ri_ai::joined_text(&output.content, "\n"));
    }
    let block = |content: &Content| match content {
        Content::Text { text, .. } => Some(json!({ "type": "text", "text": text })),
        Content::Image { data, mime_type } => Some(json!({ "type": "image", "data": data, "mimeType": mime_type })),
        _ => None,
    };
    Ok(match output.content.as_slice() {
        [] => json!(""),
        // Tools without a structured value often print JSON; scripts get that as data too.
        [Content::Text { text, .. }] => match serde_json::from_str::<Value>(text) {
            Ok(parsed @ (Value::Object(_) | Value::Array(_))) => parsed,
            _ => json!(text),
        },
        [image @ Content::Image { .. }] => block(image).unwrap_or_default(),
        blocks => blocks.iter().filter_map(block).collect(),
    })
}

fn stored_result(result: &Result<Value, String>) -> Value {
    match result {
        Ok(value) => json!({ "ok": true, "value": value }),
        Err(message) => json!({ "ok": false, "error": message }),
    }
}

fn loaded_result(stored: &Value) -> Result<Value, String> {
    if stored["ok"] == true {
        Ok(stored["value"].clone())
    } else {
        Err(stored["error"].as_str().unwrap_or_default().to_owned())
    }
}

/// Keeps the start and end of long output and says how much was left out.
fn limit_output(text: &str, limit: usize) -> String {
    if text.len() <= limit {
        return text.to_owned();
    }
    let half = limit / 2;
    let head = (0..=half)
        .rev()
        .find(|index| text.is_char_boundary(*index))
        .unwrap_or(0);
    let tail = (text.len() - half..text.len())
        .find(|index| text.is_char_boundary(*index))
        .unwrap_or(text.len());
    format!(
        "{}\n\n[... {} bytes of output left out ...]\n\n{}",
        &text[..head],
        tail - head,
        &text[tail..]
    )
}

impl Codemode {
    async fn run(&self, arguments: JsonObject, context: ToolContext) -> Result<ToolOutput, String> {
        let code = arguments
            .get("code")
            .and_then(Value::as_str)
            .ok_or("`code` must be a string")?
            .to_owned();
        let requested = arguments
            .get("timeout_ms")
            .and_then(Value::as_u64)
            .map(Duration::from_millis);
        let timeout = requested.map_or(self.options.limits.timeout, |requested| {
            requested.min(self.options.limits.timeout)
        });
        let stored = |error: ri_durable::Error| format!("Could not store the script's progress: {error}");

        // The first run records what makes a rerun repeat it; a rerun must be of the same script.
        let run = match context.load("run") {
            Some(run) if run["code"] == fingerprint(&code) => run,
            Some(_) => return Err("The script changed since this call started, so it cannot be continued.".to_owned()),
            None => {
                let run =
                    json!({ "code": fingerprint(&code), "seed": ri_ai::now_ms() as u32, "clock": ri_ai::now_ms() });
                context.store("run", Some(run.clone())).map_err(stored)?;
                run
            }
        };
        let determinism = Determinism {
            seed: run["seed"].as_u64().unwrap_or(0) as u32,
            clock: run["clock"].as_u64().unwrap_or(0),
        };
        // The results the script already got before a restart, in the order it got them.
        let journal: Vec<Value> = (0..)
            .map_while(|position| context.load(&format!("delivered/{position}")))
            .collect();
        let replayed: BTreeMap<u64, &Value> = journal
            .iter()
            .filter_map(|entry| Some((entry["id"].as_u64()?, entry)))
            .collect();

        let limits = Limits {
            timeout,
            ..self.options.limits
        };
        let mut script = sandbox::start(code, determinism, self.catalog.clone(), limits);
        for entry in &journal {
            script.deliver(
                entry["id"].as_u64().unwrap_or(0) as u32,
                loaded_result(&entry["result"]),
            );
        }
        let mut delivered = journal.len();
        let mut running = FuturesUnordered::new();
        let mut made: Vec<String> = Vec::new();
        let finished = loop {
            tokio::select! {
                _ = context.cancel.cancelled() => return Err("The script was aborted.".to_owned()),
                finished = &mut script.finished => break finished.map_err(|_| "The script stopped unexpectedly.".to_owned())?,
                Some(call) = script.calls.recv() => {
                    let HostCall { id, name, arguments } = call;
                    let signature = json!({ "name": name, "arguments": arguments });
                    made.push(name.clone());
                    let started_key = format!("started/{id}");
                    let started = context.load(&started_key);
                    if started.as_ref().is_some_and(|started| *started != signature) {
                        return Err("The script took a different path than before the restart, so it cannot be continued.".to_owned());
                    }
                    if replayed.contains_key(&u64::from(id)) {
                        continue;
                    }
                    // Started before a restart without a result: the tool's own replay policy decides.
                    let interrupted = started.is_some() && self.replay.get(&name) != Some(&Replay::Safe);
                    if started.is_none() {
                        context.store(&started_key, Some(signature)).map_err(stored)?;
                    }
                    let context = context.clone();
                    running.push(async move {
                        let result = if name == TOOL_NAME {
                            Err("Scripts cannot start scripts.".to_owned())
                        } else if interrupted {
                            Err("The call was interrupted by a restart before it finished (interrupted). Its effects are unknown.".to_owned())
                        } else {
                            script_value(&context.call_tool(&name, arguments.as_object().cloned().unwrap_or_default()).await)
                        };
                        (id, result)
                    });
                }
                Some((id, result)) = running.next() => {
                    // Journaled before the script sees it, so a rerun hands over exactly the same results.
                    let entry = json!({ "id": id, "result": stored_result(&result) });
                    context.store(&format!("delivered/{delivered}"), Some(entry)).map_err(stored)?;
                    delivered += 1;
                    script.deliver(id, result);
                }
            }
        };

        let output = limit_output(&finished.output, self.options.max_output_bytes);
        Ok(match finished.error {
            None if output.is_empty() => ToolOutput::text("(the script finished without output)"),
            None => ToolOutput::text(output),
            Some(error) => {
                let calls = if made.is_empty() {
                    String::new()
                } else {
                    format!("\n\nTool calls made before the failure: {}", made.join(", "))
                };
                let printed = if output.is_empty() {
                    String::new()
                } else {
                    format!("\n\nOutput before the failure:\n{output}")
                };
                ToolOutput::error(format!("The script failed: {error}{calls}{printed}"))
            }
        })
    }
}

impl Tool for Codemode {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn call<'a>(&'a self, arguments: JsonObject, context: ToolContext) -> BoxFuture<'a, Result<ToolOutput, String>> {
        Box::pin(self.run(arguments, context))
    }
}
