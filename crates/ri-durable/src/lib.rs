//! A durable agent harness.
//!
//! Conversations, model turns and tool calls are committed to storage before anything is shown. When the process
//! dies mid-turn, reopening the storage continues the work where it stopped: a model request it died during counts
//! as a failed attempt and is retried, and a tool call it died during reruns only when the tool is declared safe
//! to replay.
//!
//! The harness runs its work on threads of its own, so it needs no async runtime from the host. Calls that wait
//! come in an `async` form and a `_blocking` form for plain threads.
//!
//! ```no_run
//! use ri_durable::{Harness, JsonlStorage, Options, data_dir};
//! # fn models() -> ri_durable::testing::ScriptedModels { unimplemented!() }
//!
//! let storage = JsonlStorage::open(data_dir("my-app").join("main"))?;
//! let harness = Harness::open(storage, Options::new(models()))?;
//! let answer = harness.root()?.submit("What is the capital of France?")?.answer_blocking()?;
//! println!("{}", answer.text());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod event;
mod harness;
mod run;
pub mod state;
pub mod storage;
pub mod testing;
pub mod tool;

use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::thread::{self, Thread};

pub use event::{
    AgentConfig, Call, CallStatus, ConversationId, Entry, EntryId, EntryKind, Event, Outcome, Phase, Queued, Reason,
    Run, RunFailure, SubmissionId, WhenBusy,
};
pub use harness::{Conversation, Harness, Input, Options, Settings, Submission, Update, Updates};
pub use state::{ConversationState, State};
#[cfg(feature = "sqlite")]
pub use storage::SqliteStorage;
pub use storage::{JsonlStorage, MemoryStorage, Storage, data_dir};
pub use tool::{
    BeforeTool, Hook, Replay, Tool, ToolCall, ToolContext, ToolOutput, ToolSpec, raw_tool, replay_safe, schema_of, tool,
};

#[derive(Debug, Clone, thiserror::Error)]
pub enum Error {
    #[error("the harness is closed")]
    Closed,
    /// Storage failed. The session is consistent on disk; reopen it to continue.
    #[error("the harness failed: {0}")]
    Failed(String),
    #[error("unknown conversation")]
    UnknownConversation,
    #[error("the input was not answered: {:?}{}", .0.reason, .0.detail.as_ref().map(|detail| format!(" ({detail})")).unwrap_or_default())]
    Unanswered(RunFailure),
}

struct ThreadWaker(Thread);

impl Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

/// Runs a future to completion on the calling thread. Must not be called from inside an async runtime.
pub fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    let waker = Waker::from(Arc::new(ThreadWaker(thread::current())));
    let mut context = Context::from_waker(&waker);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(output) => return output,
            Poll::Pending => thread::park(),
        }
    }
}
