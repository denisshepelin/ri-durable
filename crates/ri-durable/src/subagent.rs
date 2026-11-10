//! A tool that hands a self-contained task to a child conversation and returns its answer.

use std::sync::Arc;

use ri_ai::JsonObject;
use serde_json::{Value, json};

use crate::event::AgentConfig;
use crate::harness::Input;
use crate::tool::{BoxFuture, Replay, Tool, ToolContext, ToolOutput, ToolSpec};

#[derive(Debug, Clone, PartialEq)]
pub struct SubagentOptions {
    /// The tool's name.
    pub name: String,
    pub description: String,
    /// The model children run on; `None` uses the calling conversation's.
    pub model: Option<String>,
    /// The children's system prompt; `None` uses the calling conversation's.
    pub instructions: Option<String>,
}

impl Default for SubagentOptions {
    fn default() -> Self {
        SubagentOptions {
            name: "subagent".to_owned(),
            description: "Delegate a self-contained task to a subagent and get its answer back. The subagent has the \
                same tools but none of this conversation, so the task must say everything it needs to know."
                .to_owned(),
            model: None,
            instructions: None,
        }
    }
}

/// A tool whose every call runs a child conversation, owned by that call: aborting the call aborts the child, and
/// a call that reruns after a crash finds its child and waits for the same answer.
pub fn subagent_tool(options: SubagentOptions) -> Arc<dyn Tool> {
    let spec = ToolSpec {
        name: options.name.clone(),
        description: options.description.clone(),
        parameters: json!({
            "type": "object",
            "required": ["task"],
            "properties": { "task": { "type": "string", "description": "What the subagent should do, with all the context it needs" } },
        }),
        replay: Replay::Safe,
    };
    Arc::new(Subagent { spec, options })
}

struct Subagent {
    spec: ToolSpec,
    options: SubagentOptions,
}

impl Subagent {
    async fn run(&self, arguments: JsonObject, context: ToolContext) -> Result<ToolOutput, String> {
        let task = arguments
            .get("task")
            .and_then(Value::as_str)
            .ok_or("`task` must be a string")?;
        let failed = |error: crate::Error| error.to_string();
        let existing = context
            .load("child")
            .and_then(|child| child.as_str().and_then(|id| context.conversation(id)));
        let child = match existing {
            Some(child) => child,
            None => {
                let parent = context.agent().map_err(failed)?;
                // Children get no subagents of their own.
                let offered = parent
                    .tools
                    .clone()
                    .unwrap_or_else(|| context.tool_specs().into_iter().map(|spec| spec.name).collect());
                let agent = AgentConfig {
                    model: self.options.model.clone().or(parent.model),
                    instructions: self.options.instructions.clone().or(parent.instructions),
                    tools: Some(offered.into_iter().filter(|name| *name != self.options.name).collect()),
                    ..parent
                };
                let child = context.create_conversation(agent).map_err(failed)?;
                context.store("child", Some(json!(child.id()))).map_err(failed)?;
                child
            }
        };
        // The request id makes a rerun wait for the answer already under way instead of asking again.
        let submission = child
            .submit(Input::text(task).request_id(format!("subagent:{}", context.call_id)))
            .map_err(failed)?;
        let answer = tokio::select! {
            _ = context.cancel.cancelled() => {
                let _ = child.abort().await;
                return Err("The subagent was aborted.".to_owned());
            }
            answer = submission.answer() => answer,
        };
        let details = json!({ "conversation": child.id() });
        match answer {
            Ok(message) => Ok(ToolOutput::text(message.text()).with_details(details)),
            Err(error) => Ok(ToolOutput::error(format!("The subagent did not finish: {error}")).with_details(details)),
        }
    }
}

impl Tool for Subagent {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn call<'a>(&'a self, arguments: JsonObject, context: ToolContext) -> BoxFuture<'a, Result<ToolOutput, String>> {
        Box::pin(self.run(arguments, context))
    }
}
