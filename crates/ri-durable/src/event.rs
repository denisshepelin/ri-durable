//! What is stored: every change to a session is an [`Event`], and a session's state is the fold of its events.

use std::path::PathBuf;

use ri_ai::{AssistantMessage, Content, JsonObject, ThinkingLevel};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub type ConversationId = String;
pub type SubmissionId = String;
/// Position of an entry in its conversation's transcript.
pub type EntryId = u64;

/// What a conversation runs with. Unset fields follow the harness defaults.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct AgentConfig {
    /// `provider/id`.
    pub model: Option<String>,
    pub thinking: Option<ThinkingLevel>,
    /// The system prompt.
    pub instructions: Option<String>,
    /// Names of the tools offered to the model; `None` offers every registered tool.
    pub tools: Option<Vec<String>>,
    /// Tool rounds one run may take.
    pub max_turns: Option<u32>,
    pub cwd: Option<PathBuf>,
}

impl AgentConfig {
    /// This config with unset fields taken from `defaults`.
    pub fn or(&self, defaults: &AgentConfig) -> AgentConfig {
        AgentConfig {
            model: self.model.clone().or_else(|| defaults.model.clone()),
            thinking: self.thinking.or(defaults.thinking),
            instructions: self.instructions.clone().or_else(|| defaults.instructions.clone()),
            tools: self.tools.clone().or_else(|| defaults.tools.clone()),
            max_turns: self.max_turns.or(defaults.max_turns),
            cwd: self.cwd.clone().or_else(|| defaults.cwd.clone()),
        }
    }
}

/// One immutable transcript record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Entry {
    pub id: EntryId,
    /// Unix time in milliseconds.
    pub timestamp: u64,
    #[serde(flatten)]
    pub kind: EntryKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum EntryKind {
    #[serde(rename_all = "camelCase")]
    User {
        content: Vec<Content>,
        /// The submission this input came from.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        submission: Option<SubmissionId>,
    },
    Assistant {
        message: AssistantMessage,
    },
    #[serde(rename_all = "camelCase")]
    ToolResult {
        tool_call_id: String,
        tool_name: String,
        content: Vec<Content>,
        #[serde(default)]
        is_error: bool,
        /// Data for user interfaces; the model never sees it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        details: Option<Value>,
    },
    /// Starts a new context: the model no longer sees earlier entries. `note` opens the new context.
    Reset {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        note: Option<String>,
    },
    /// Replaces the entries before `first_kept` with `summary` in what the model sees.
    #[serde(rename_all = "camelCase")]
    Compaction {
        summary: String,
        first_kept: EntryId,
    },
    /// An application record the model does not see.
    Custom {
        name: String,
        data: Value,
    },
}

/// How an input behaves when it arrives while a run is working.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum WhenBusy {
    /// Wait for the run to answer, then start the next run.
    #[default]
    FollowUp,
    /// Join the running work after its current tool round.
    Steer,
}

/// An input waiting in a conversation's inbox.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Queued {
    pub id: SubmissionId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    pub content: Vec<Content>,
    #[serde(default)]
    pub when_busy: WhenBusy,
}

/// The work answering one or more inputs, checkpointed at every step so a restarted process continues it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Run {
    /// The submissions this run answers.
    pub inputs: Vec<SubmissionId>,
    /// Tool rounds finished so far.
    pub turns: u32,
    pub phase: Phase,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "camelCase")]
pub enum Phase {
    /// A model request is due or in flight. `attempt` counts the failed attempts before it.
    Request { attempt: u32 },
    /// The model's tool calls are running.
    Tools { calls: Vec<Call> },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Call {
    pub id: String,
    pub name: String,
    pub arguments: JsonObject,
    pub status: CallStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "camelCase")]
pub enum CallStatus {
    /// Not started. `restarts` counts how often a crash cut the call off.
    Pending { restarts: u32 },
    /// Started: its intent was stored before the tool ran.
    Running { restarts: u32 },
    /// Its result is in the transcript.
    Done,
}

/// Why a run ended without an answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Reason {
    ModelError,
    NoModel,
    Aborted,
    Reset,
    /// The process stopped during a model request and no retry was left.
    Interrupted,
    TurnLimit,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunFailure {
    pub reason: Reason,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// How a submission ended.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "camelCase")]
pub enum Outcome {
    /// Answered by the assistant entry `entry`.
    Done {
        entry: EntryId,
    },
    Unanswered(RunFailure),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum Event {
    ConversationCreated {
        conversation: ConversationId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent: Option<ConversationId>,
        agent: AgentConfig,
    },
    AgentChanged {
        conversation: ConversationId,
        agent: AgentConfig,
    },
    /// An input was admitted to the inbox.
    Queued {
        conversation: ConversationId,
        submission: Queued,
    },
    /// A transcript entry was appended. A user entry takes its submission out of the inbox.
    Entry { conversation: ConversationId, entry: Entry },
    /// A run started or reached its next checkpoint.
    Run { conversation: ConversationId, run: Run },
    Call {
        conversation: ConversationId,
        call: String,
        status: CallStatus,
    },
    /// The run is over. `failure` stays visible until the next run starts.
    RunEnded {
        conversation: ConversationId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        failure: Option<RunFailure>,
    },
    /// A submission ended. One still in the inbox leaves it.
    Settled {
        conversation: ConversationId,
        submission: SubmissionId,
        outcome: Outcome,
    },
    /// An application value stored with the conversation; `None` removes it.
    Value {
        conversation: ConversationId,
        key: String,
        value: Option<Value>,
    },
    /// A process opened the session for writing.
    Opened,
    /// The harness closed in an orderly way: nothing before this was interrupted by a crash.
    Closed,
}
