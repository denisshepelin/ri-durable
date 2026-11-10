//! Tools the model can call, and hooks that observe or adjust the harness's work.

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use ri_ai::{Content, Context, JsonObject};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::Error;
use crate::event::{AgentConfig, ConversationId, Event};
use crate::harness::{Conversation, Shared, Update};

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// What happens to a call that a crash cut off.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Replay {
    /// The model gets an `interrupted` error result. The default, for tools with effects.
    #[default]
    Never,
    /// The call runs again from the start.
    Safe,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// JSON Schema of the arguments.
    pub parameters: Value,
    pub replay: Replay,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ToolOutput {
    /// What the model reads.
    pub content: Vec<Content>,
    pub is_error: bool,
    /// Data for user interfaces; the model never sees it.
    pub details: Option<Value>,
    /// The result as data, for programs that call the tool with [`ToolContext::call_tool`]. Not stored.
    pub structured: Option<Value>,
}

impl ToolOutput {
    pub fn text(text: impl Into<String>) -> Self {
        ToolOutput {
            content: vec![Content::text(text)],
            ..ToolOutput::default()
        }
    }

    /// A result that is data: the model reads it as JSON text, and programs calling the tool get the value.
    pub fn json(value: Value) -> Self {
        ToolOutput {
            structured: Some(value.clone()),
            ..ToolOutput::text(value.to_string())
        }
    }

    pub fn error(text: impl Into<String>) -> Self {
        ToolOutput {
            is_error: true,
            ..ToolOutput::text(text)
        }
    }

    pub fn with_details(self, details: Value) -> Self {
        ToolOutput {
            details: Some(details),
            ..self
        }
    }
}

impl From<String> for ToolOutput {
    fn from(text: String) -> Self {
        ToolOutput::text(text)
    }
}

impl From<&str> for ToolOutput {
    fn from(text: &str) -> Self {
        ToolOutput::text(text)
    }
}

/// What a running call knows about itself, and what it may ask of the harness.
#[derive(Clone)]
pub struct ToolContext {
    pub conversation: ConversationId,
    pub call_id: String,
    pub cwd: Option<PathBuf>,
    /// Cancelled when the call is aborted. Long-running tools should stop when it fires.
    pub cancel: CancellationToken,
    /// How often a crash cut this call off before this run of it. Above zero only for replay-safe tools.
    pub restarts: u32,
    pub(crate) shared: Arc<Shared>,
}

impl ToolContext {
    /// Shows running output to watchers. It is not stored and not part of the result.
    pub fn output(&self, text: &str) {
        let update = Update::ToolOutput {
            conversation: self.conversation.clone(),
            call: self.call_id.clone(),
            text: text.to_owned(),
        };
        self.shared.notify(&self.conversation, update);
    }

    /// Calls another registered tool, as a script or a bridge does. The call runs the tool hooks and returns its
    /// result to the caller instead of the transcript.
    pub async fn call_tool(&self, name: &str, arguments: JsonObject) -> ToolOutput {
        crate::run::nested_call(self, name, arguments).await
    }

    /// Every registered tool, also those not offered to the model.
    pub fn tool_specs(&self) -> Vec<ToolSpec> {
        self.shared.tools.values().map(|tool| tool.spec().clone()).collect()
    }

    /// What the calling conversation runs with, harness defaults applied.
    pub fn agent(&self) -> Result<AgentConfig, Error> {
        self.shared
            .read(&self.conversation, |state| state.agent.or(&self.shared.defaults))
    }

    /// Creates a conversation whose parent is the calling one, for work this call delegates.
    pub fn create_conversation(&self, agent: AgentConfig) -> Result<Conversation, Error> {
        let id = uuid::Uuid::now_v7().to_string();
        let created = Event::ConversationCreated {
            conversation: id.clone(),
            parent: Some(self.conversation.clone()),
            agent,
        };
        self.shared.commit(|_| Ok((vec![created], ())))?;
        Ok(Conversation::new(self.shared.clone(), id))
    }

    pub fn conversation(&self, id: &str) -> Option<Conversation> {
        let known = self.shared.read(id, |_| ()).is_ok();
        known.then(|| Conversation::new(self.shared.clone(), id.to_owned()))
    }

    fn stored_key(&self, key: &str) -> String {
        format!("call/{}/{key}", self.call_id)
    }

    /// Durably keeps a value for this call, so a replay-safe tool finds it again when it reruns after a crash.
    /// `None` removes it. Everything a call stored is dropped when its result is stored.
    pub fn store(&self, key: &str, value: Option<Value>) -> Result<(), Error> {
        let event = Event::Value {
            conversation: self.conversation.clone(),
            key: self.stored_key(key),
            value,
        };
        self.shared.commit(|_| Ok((vec![event], ())))
    }

    pub fn load(&self, key: &str) -> Option<Value> {
        let key = self.stored_key(key);
        self.shared
            .read(&self.conversation, |state| state.values.get(&key).cloned())
            .ok()
            .flatten()
    }
}

pub trait Tool: Send + Sync + 'static {
    fn spec(&self) -> &ToolSpec;

    /// Runs the call. An `Err` becomes an error result the model sees.
    fn call<'a>(&'a self, arguments: JsonObject, context: ToolContext) -> BoxFuture<'a, Result<ToolOutput, String>>;
}

struct FnTool<F> {
    spec: ToolSpec,
    run: F,
}

impl<F, Fut> Tool for FnTool<F>
where
    F: Fn(JsonObject, ToolContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<ToolOutput, String>> + Send + 'static,
{
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn call<'a>(&'a self, arguments: JsonObject, context: ToolContext) -> BoxFuture<'a, Result<ToolOutput, String>> {
        Box::pin((self.run)(arguments, context))
    }
}

/// A tool whose arguments are a Rust type: the schema is derived from `A` and calls are deserialized into it.
///
/// ```
/// # use ri_durable::{tool, ToolOutput};
/// #[derive(serde::Deserialize, schemars::JsonSchema)]
/// struct Add { a: i64, b: i64 }
///
/// let add = tool("add", "Add two numbers", |args: Add, _context| async move {
///     Ok(ToolOutput::text((args.a + args.b).to_string()))
/// });
/// assert_eq!(add.spec().parameters["required"], serde_json::json!(["a", "b"]));
/// ```
pub fn tool<A, F, Fut>(name: &str, description: &str, run: F) -> Arc<dyn Tool>
where
    A: DeserializeOwned + JsonSchema + Send + 'static,
    F: Fn(A, ToolContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<ToolOutput, String>> + Send + 'static,
{
    let run = Arc::new(run);
    raw_tool(name, description, schema_of::<A>(), move |arguments, context| {
        let run = run.clone();
        async move {
            let arguments = serde_json::from_value::<A>(Value::Object(arguments))
                .map_err(|error| format!("Invalid arguments: {error}"))?;
            run(arguments, context).await
        }
    })
}

/// A tool with a hand-written JSON Schema that receives its arguments as JSON.
pub fn raw_tool<F, Fut>(name: &str, description: &str, parameters: Value, run: F) -> Arc<dyn Tool>
where
    F: Fn(JsonObject, ToolContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<ToolOutput, String>> + Send + 'static,
{
    let spec = ToolSpec {
        name: name.to_owned(),
        description: description.to_owned(),
        parameters,
        replay: Replay::Never,
    };
    Arc::new(FnTool { spec, run })
}

/// Marks a tool as safe to run again after a crash interrupted it.
pub fn replay_safe(tool: Arc<dyn Tool>) -> Arc<dyn Tool> {
    struct Safe(Arc<dyn Tool>, ToolSpec);
    impl Tool for Safe {
        fn spec(&self) -> &ToolSpec {
            &self.1
        }
        fn call<'a>(
            &'a self,
            arguments: JsonObject,
            context: ToolContext,
        ) -> BoxFuture<'a, Result<ToolOutput, String>> {
            self.0.call(arguments, context)
        }
    }
    let spec = ToolSpec {
        replay: Replay::Safe,
        ..tool.spec().clone()
    };
    Arc::new(Safe(tool, spec))
}

/// The JSON Schema of `A`, without the metadata providers reject.
pub fn schema_of<A: JsonSchema>() -> Value {
    let mut schema = serde_json::to_value(schemars::schema_for!(A)).unwrap_or_default();
    if let Some(fields) = schema.as_object_mut() {
        fields.remove("$schema");
        fields.remove("title");
        fields.entry("type").or_insert_with(|| Value::from("object"));
        fields
            .entry("properties")
            .or_insert_with(|| Value::Object(Default::default()));
    }
    schema
}

/// A tool call as hooks see it.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub conversation: ConversationId,
    pub id: String,
    pub name: String,
    pub arguments: JsonObject,
}

pub enum BeforeTool {
    Continue,
    /// Run the call with these arguments instead.
    Rewrite(JsonObject),
    /// Do not run the call; the model gets this text as an error result.
    Block(String),
}

/// Observes or adjusts the harness's work. Every method has a default that changes nothing.
pub trait Hook: Send + Sync + 'static {
    /// Runs before a model request is sent and may change what is sent. The transcript is not changed.
    fn before_request(&self, _conversation: &ConversationId, _context: &mut Context) {}

    fn before_tool(&self, _call: &ToolCall) -> BeforeTool {
        BeforeTool::Continue
    }

    /// May replace a tool's result before it is stored.
    fn after_tool(&self, _call: &ToolCall, output: ToolOutput) -> ToolOutput {
        output
    }
}
