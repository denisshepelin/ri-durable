//! The harness: one open session and the machinery that runs agents on it.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use ri_ai::{AssistantEvent, AssistantMessage, Content, Models, Usage, now_ms};
use serde_json::Value;
use tokio::runtime::{Handle, Runtime};
use tokio::sync::{mpsc, watch};
use tokio_util::sync::CancellationToken;

use crate::compaction::{Compacted, Compaction};
use crate::event::{
    AgentConfig, CallStatus, ConversationId, Entry, EntryId, EntryKind, Event, Outcome, Phase, Queued, Reason, Run,
    RunFailure, SubmissionId, WhenBusy,
};
use crate::state::{ConversationState, State};
use crate::storage::Storage;
use crate::tool::{Hook, Replay, Tool};
use crate::{Error, block_on, run};

/// Run policy shared by every conversation.
#[derive(Debug, Clone)]
pub struct Settings {
    /// Failed model requests retried before the run fails.
    pub max_retries: u32,
    /// Wait before the first retry; it doubles with each further one.
    pub retry_delay: Duration,
    /// How often a crash may cut a replay-safe tool call off before it ends with an error result.
    pub restart_limit: u32,
    /// Continue unfinished work as soon as the session opens.
    pub resume: bool,
    pub compaction: Compaction,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            max_retries: 3,
            retry_delay: Duration::from_secs(1),
            restart_limit: 5,
            resume: true,
            compaction: Compaction::default(),
        }
    }
}

pub struct Options {
    pub models: Arc<dyn Models>,
    pub tools: Vec<Arc<dyn Tool>>,
    pub hooks: Vec<Arc<dyn Hook>>,
    /// What conversations run with where their own configuration is unset.
    pub defaults: AgentConfig,
    pub settings: Settings,
}

impl Options {
    pub fn new(models: impl Models) -> Self {
        Options {
            models: Arc::new(models),
            tools: Vec::new(),
            hooks: Vec::new(),
            defaults: AgentConfig::default(),
            settings: Settings::default(),
        }
    }

    pub fn tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    pub fn hook(mut self, hook: impl Hook) -> Self {
        self.hooks.push(Arc::new(hook));
        self
    }

    pub fn defaults(mut self, defaults: AgentConfig) -> Self {
        self.defaults = defaults;
        self
    }

    pub fn settings(mut self, settings: Settings) -> Self {
        self.settings = settings;
        self
    }
}

/// Something a watcher is told. Events are sent after they are stored; the other updates are transient progress
/// that is never stored.
#[derive(Debug, Clone, PartialEq)]
pub enum Update {
    Event(Event),
    /// A fragment of the model response being streamed.
    Partial {
        conversation: ConversationId,
        event: AssistantEvent,
    },
    /// Running output of a tool call.
    ToolOutput {
        conversation: ConversationId,
        call: String,
        text: String,
    },
}

pub struct Updates {
    receiver: mpsc::UnboundedReceiver<Update>,
}

impl Updates {
    /// The next update, or `None` once the harness has closed.
    pub async fn next(&mut self) -> Option<Update> {
        self.receiver.recv().await
    }

    pub fn next_blocking(&mut self) -> Option<Update> {
        block_on(self.next())
    }

    /// An update that is already waiting, without blocking.
    pub fn try_next(&mut self) -> Option<Update> {
        self.receiver.try_recv().ok()
    }
}

struct Watcher {
    conversation: Option<ConversationId>,
    sender: mpsc::UnboundedSender<Update>,
}

pub(crate) struct Inner {
    pub state: State,
    storage: Box<dyn Storage>,
    watchers: Vec<Watcher>,
    /// Conversations a task is driving, with the token that aborts it.
    active: HashMap<ConversationId, CancellationToken>,
    /// Set once storage failed; every later call fails with it.
    failed: Option<String>,
    closed: bool,
}

pub(crate) struct Shared {
    inner: Mutex<Inner>,
    /// Bumped after every commit, so waiters re-check the state.
    version: watch::Sender<u64>,
    pub models: Arc<dyn Models>,
    pub tools: BTreeMap<String, Arc<dyn Tool>>,
    pub hooks: Vec<Arc<dyn Hook>>,
    pub defaults: AgentConfig,
    pub settings: Settings,
    runtime: Handle,
}

fn event_conversation(event: &Event) -> Option<&ConversationId> {
    match event {
        Event::ConversationCreated { conversation, .. }
        | Event::AgentChanged { conversation, .. }
        | Event::Queued { conversation, .. }
        | Event::Entry { conversation, .. }
        | Event::Run { conversation, .. }
        | Event::Call { conversation, .. }
        | Event::RunEnded { conversation, .. }
        | Event::Settled { conversation, .. }
        | Event::Value { conversation, .. } => Some(conversation),
        Event::Opened | Event::Closed => None,
    }
}

impl Inner {
    fn check(&self) -> Result<(), Error> {
        match &self.failed {
            Some(cause) => Err(Error::Failed(cause.clone())),
            None if self.closed => Err(Error::Closed),
            None => Ok(()),
        }
    }

    fn send(&mut self, conversation: Option<&ConversationId>, update: &Update) {
        self.watchers.retain(|watcher| {
            let wanted = watcher.conversation.is_none()
                || conversation.is_none()
                || watcher.conversation.as_ref() == conversation;
            !wanted || watcher.sender.send(update.clone()).is_ok()
        });
    }

    /// Stores `events` as one commit, then applies them to memory and tells watchers. Nothing is shown before it
    /// is stored; a storage error fails the harness.
    fn store(&mut self, events: Vec<Event>) -> Result<(), Error> {
        if events.is_empty() {
            return Ok(());
        }
        if let Err(error) = self.storage.append(&events) {
            let cause = error.to_string();
            self.failed = Some(cause.clone());
            return Err(Error::Failed(cause));
        }
        for event in events {
            self.state.apply(&event);
            let conversation = event_conversation(&event).cloned();
            self.send(conversation.as_ref(), &Update::Event(event));
        }
        Ok(())
    }
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Builds events from the current state and stores them as one commit.
    pub fn commit<R>(&self, build: impl FnOnce(&State) -> Result<(Vec<Event>, R), Error>) -> Result<R, Error> {
        let mut inner = self.lock();
        inner.check()?;
        let (events, result) = build(&inner.state)?;
        let stored = inner.store(events);
        drop(inner);
        self.version.send_modify(|version| *version += 1);
        stored.map(|()| result)
    }

    /// Like [`Shared::commit`], for changes to one conversation.
    pub fn commit_in<R>(
        &self,
        conversation: &str,
        build: impl FnOnce(&ConversationState) -> Result<(Vec<Event>, R), Error>,
    ) -> Result<R, Error> {
        self.commit(|state| {
            build(
                state
                    .conversations
                    .get(conversation)
                    .ok_or(Error::UnknownConversation)?,
            )
        })
    }

    pub fn read<R>(&self, conversation: &str, read: impl FnOnce(&ConversationState) -> R) -> Result<R, Error> {
        let inner = self.lock();
        inner
            .state
            .conversations
            .get(conversation)
            .map(read)
            .ok_or(Error::UnknownConversation)
    }

    pub fn notify(&self, conversation: &ConversationId, update: Update) {
        self.lock().send(Some(conversation), &update);
    }

    /// Starts a task for the conversation when it has work and none is driving it.
    pub fn start(self: &Arc<Self>, conversation: &str) {
        let mut inner = self.lock();
        let busy = inner
            .state
            .conversations
            .get(conversation)
            .is_some_and(ConversationState::is_busy);
        if !busy || inner.closed || inner.failed.is_some() || inner.active.contains_key(conversation) {
            return;
        }
        let cancel = CancellationToken::new();
        inner.active.insert(conversation.to_owned(), cancel.clone());
        drop(inner);
        self.runtime
            .spawn(run::drive(self.clone(), conversation.to_owned(), cancel));
    }

    /// Called by a task that found nothing left to do. Returns false when work arrived meanwhile.
    pub fn retire(&self, conversation: &str) -> bool {
        let mut inner = self.lock();
        let busy = inner
            .state
            .conversations
            .get(conversation)
            .is_some_and(ConversationState::is_busy);
        if busy && !inner.closed && inner.failed.is_none() {
            return false;
        }
        inner.active.remove(conversation);
        drop(inner);
        self.version.send_modify(|version| *version += 1);
        true
    }

    /// Stops the conversation's work: a task driving it is told to stop, work nothing drives is ended here. The
    /// conversations it started for delegated work are stopped with it.
    pub fn abort(&self, conversation: &str) -> Result<(), Error> {
        let (driven, children) = {
            let inner = self.lock();
            inner.check()?;
            let children: Vec<ConversationId> = inner
                .state
                .conversations
                .values()
                .filter(|child| child.parent.as_deref() == Some(conversation))
                .map(|child| child.id.clone())
                .collect();
            (
                inner.active.get(conversation).map(|cancel| cancel.cancel()).is_some(),
                children,
            )
        };
        if !driven {
            self.commit_in(conversation, |state| {
                let failure = RunFailure {
                    reason: Reason::Aborted,
                    detail: None,
                };
                Ok((end_events(state, failure, "The tool call was aborted."), ()))
            })?;
        }
        children.iter().try_for_each(|child| self.abort(child))
    }

    /// Takes the conversation away from its task after an abort, so new work gets a fresh one.
    pub fn release(&self, conversation: &str) {
        self.lock().active.remove(conversation);
        self.version.send_modify(|version| *version += 1);
    }

    async fn wait_for<R>(&self, mut check: impl FnMut(&Inner) -> Option<R>) -> Result<R, Error> {
        let mut changes = self.version.subscribe();
        loop {
            {
                let inner = self.lock();
                if let Some(result) = check(&inner) {
                    return Ok(result);
                }
                inner.check()?;
            }
            changes.changed().await.map_err(|_| Error::Closed)?;
        }
    }
}

/// The events that end a conversation's current work without an answer: unfinished tool calls get an error result
/// so the transcript stays valid, and the run's inputs and everything queued settle unanswered.
pub(crate) fn end_events(state: &ConversationState, failure: RunFailure, call_error: &str) -> Vec<Event> {
    let conversation = &state.id;
    let results = state
        .pending_calls()
        .into_iter()
        .enumerate()
        .flat_map(|(offset, call)| {
            let entry = Entry {
                id: state.next_entry_id() + offset as EntryId,
                timestamp: now_ms(),
                kind: EntryKind::ToolResult {
                    tool_call_id: call.id.clone(),
                    tool_name: call.name,
                    content: vec![Content::text(call_error)],
                    is_error: true,
                    details: None,
                },
            };
            [
                Event::Entry {
                    conversation: conversation.clone(),
                    entry,
                },
                Event::Call {
                    conversation: conversation.clone(),
                    call: call.id,
                    status: CallStatus::Done,
                },
            ]
        });
    let inputs = state.run.iter().flat_map(|run| run.inputs.iter().cloned());
    let settled = inputs
        .chain(state.inbox.iter().map(|queued| queued.id.clone()))
        .map(|submission| Event::Settled {
            conversation: conversation.clone(),
            submission,
            outcome: Outcome::Unanswered(failure.clone()),
        });
    let ended = state.run.is_some().then(|| Event::RunEnded {
        conversation: conversation.clone(),
        failure: Some(failure.clone()),
    });
    results.chain(settled).chain(ended).collect()
}

/// What reopening a session changes: work a crash cut off is counted and either retried or ended.
fn recovery_events(state: &State, tools: &BTreeMap<String, Arc<dyn Tool>>, settings: &Settings) -> Vec<Event> {
    let crashed = !state.closed_cleanly;
    state
        .conversations
        .values()
        .flat_map(|conversation| {
            let id = &conversation.id;
            match &conversation.run {
                // A request the process died during is a failed attempt, so one that keeps killing it ends.
                Some(run @ Run { phase: Phase::Request { attempt }, .. }) if crashed => {
                    if *attempt >= settings.max_retries {
                        let failure = RunFailure { reason: Reason::Interrupted, detail: None };
                        end_events(conversation, failure, "interrupted")
                    } else {
                        let run = Run { phase: Phase::Request { attempt: attempt + 1 }, ..run.clone() };
                        vec![Event::Run { conversation: id.clone(), run }]
                    }
                }
                Some(Run { phase: Phase::Tools { calls }, .. }) => {
                    let mut next_entry = conversation.next_entry_id();
                    calls
                        .iter()
                        .filter_map(|call| match call.status {
                            CallStatus::Running { restarts } => Some((call, restarts)),
                            _ => None,
                        })
                        .flat_map(|(call, restarts)| {
                            let restarts = restarts + u32::from(crashed);
                            let safe = tools.get(&call.name).is_some_and(|tool| tool.spec().replay == Replay::Safe);
                            if safe && restarts < settings.restart_limit {
                                let status = CallStatus::Pending { restarts };
                                return vec![Event::Call { conversation: id.clone(), call: call.id.clone(), status }];
                            }
                            let text = if safe {
                                "The tool call was interrupted too many times and was not run again (restart_limit)."
                            } else {
                                "The tool call was interrupted before it finished (interrupted). Its effects are unknown."
                            };
                            let entry = Entry {
                                id: next_entry,
                                timestamp: now_ms(),
                                kind: EntryKind::ToolResult {
                                    tool_call_id: call.id.clone(),
                                    tool_name: call.name.clone(),
                                    content: vec![Content::text(text)],
                                    is_error: true,
                                    details: None,
                                },
                            };
                            next_entry += 1;
                            vec![
                                Event::Entry { conversation: id.clone(), entry },
                                Event::Call { conversation: id.clone(), call: call.id.clone(), status: CallStatus::Done },
                            ]
                        })
                        .collect()
                }
                _ => Vec::new(),
            }
        })
        .collect()
}

/// One open session. Dropping it closes it.
pub struct Harness {
    shared: Arc<Shared>,
    runtime: Option<Runtime>,
}

impl Harness {
    /// Opens a session: reads its history, settles what a crash interrupted, and continues unfinished work.
    pub fn open(storage: impl Storage, options: Options) -> Result<Harness, Error> {
        let mut storage: Box<dyn Storage> = Box::new(storage);
        let events = storage.load().map_err(|error| Error::Failed(error.to_string()))?;
        let state = State::from_events(&events);
        let tools: BTreeMap<String, Arc<dyn Tool>> = options
            .tools
            .into_iter()
            .map(|tool| (tool.spec().name.clone(), tool))
            .collect();
        let mut recovery = if events.is_empty() {
            Vec::new()
        } else {
            recovery_events(&state, &tools, &options.settings)
        };
        recovery.push(Event::Opened);
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("ri-durable")
            .enable_all()
            .build()
            .map_err(|error| Error::Failed(error.to_string()))?;
        let shared = Arc::new(Shared {
            inner: Mutex::new(Inner {
                state,
                storage,
                watchers: Vec::new(),
                active: HashMap::new(),
                failed: None,
                closed: false,
            }),
            version: watch::channel(0).0,
            models: options.models,
            tools,
            hooks: options.hooks,
            defaults: options.defaults,
            settings: options.settings,
            runtime: runtime.handle().clone(),
        });
        shared.commit(|_| Ok((recovery, ())))?;
        let harness = Harness {
            shared,
            runtime: Some(runtime),
        };
        if harness.shared.settings.resume {
            harness.resume();
        }
        Ok(harness)
    }

    /// Continues every conversation that has unfinished work. Submitting to a conversation resumes it too.
    pub fn resume(&self) {
        let conversations: Vec<ConversationId> = self.shared.lock().state.conversations.keys().cloned().collect();
        for conversation in conversations {
            self.shared.start(&conversation);
        }
    }

    /// The root conversation, created on first use.
    pub fn root(&self) -> Result<Conversation, Error> {
        let existing = self.shared.lock().state.root.clone();
        match existing {
            Some(id) => Ok(self.handle(id)),
            None => self.create_conversation(AgentConfig::default()),
        }
    }

    pub fn create_conversation(&self, agent: AgentConfig) -> Result<Conversation, Error> {
        let id = uuid::Uuid::now_v7().to_string();
        let created = Event::ConversationCreated {
            conversation: id.clone(),
            parent: None,
            agent,
        };
        self.shared.commit(|_| Ok((vec![created], ())))?;
        Ok(self.handle(id))
    }

    pub fn conversation(&self, id: &str) -> Option<Conversation> {
        self.shared
            .lock()
            .state
            .conversations
            .contains_key(id)
            .then(|| self.handle(id.to_owned()))
    }

    pub fn conversations(&self) -> Vec<Conversation> {
        let ids: Vec<ConversationId> = self.shared.lock().state.conversations.keys().cloned().collect();
        ids.into_iter().map(|id| self.handle(id)).collect()
    }

    fn handle(&self, id: ConversationId) -> Conversation {
        Conversation {
            shared: self.shared.clone(),
            id,
        }
    }

    /// Every update of the session, for all conversations.
    pub fn watch(&self) -> Updates {
        let (sender, receiver) = mpsc::unbounded_channel();
        self.shared.lock().watchers.push(Watcher {
            conversation: None,
            sender,
        });
        Updates { receiver }
    }

    /// Token and cost totals of the whole session, by `provider/model`.
    pub fn usage(&self) -> BTreeMap<String, Usage> {
        let inner = self.shared.lock();
        inner
            .state
            .conversations
            .values()
            .flat_map(ConversationState::usage)
            .fold(BTreeMap::new(), |mut totals, (model, usage)| {
                let total: &mut Usage = totals.entry(model).or_default();
                *total = crate::state::add_usage(*total, usage);
                totals
            })
    }

    /// Closes the session in an orderly way. Work in flight stops and continues when the session is opened again;
    /// a model request stopped this way does not count as a failed attempt.
    pub fn close(mut self) {
        self.shutdown(true);
    }

    /// Stops without the orderly-close record, as a crash would. For tests of recovery.
    #[doc(hidden)]
    pub fn abandon(mut self) {
        self.shutdown(false);
    }

    fn shutdown(&mut self, orderly: bool) {
        let Some(runtime) = self.runtime.take() else { return };
        {
            let mut inner = self.shared.lock();
            if orderly && !inner.closed && inner.failed.is_none() {
                let _ = inner.store(vec![Event::Closed]);
            }
            inner.closed = true;
            inner.watchers.clear();
            // Dropping the storage releases its writer lock even while stopped tasks still hold the session.
            inner.storage = Box::new(crate::storage::MemoryStorage::new());
        }
        self.shared.version.send_modify(|version| *version += 1);
        runtime.shutdown_background();
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        self.shutdown(true);
    }
}

/// An input handed to a conversation.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Input {
    pub content: Vec<Content>,
    /// Makes the submission idempotent: submitting the same id again returns the first submission.
    pub request_id: Option<String>,
    pub when_busy: WhenBusy,
}

impl Input {
    pub fn text(text: impl Into<String>) -> Self {
        Input {
            content: vec![Content::text(text)],
            ..Input::default()
        }
    }

    pub fn request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = Some(request_id.into());
        self
    }

    /// Joins the running work after its current tool round instead of waiting for the answer.
    pub fn steer(mut self) -> Self {
        self.when_busy = WhenBusy::Steer;
        self
    }
}

impl From<&str> for Input {
    fn from(text: &str) -> Self {
        Input::text(text)
    }
}

impl From<String> for Input {
    fn from(text: String) -> Self {
        Input::text(text)
    }
}

/// A handle to one conversation. It holds no state; compare handles by [`Conversation::id`].
#[derive(Clone)]
pub struct Conversation {
    shared: Arc<Shared>,
    id: ConversationId,
}

impl Conversation {
    pub(crate) fn new(shared: Arc<Shared>, id: ConversationId) -> Self {
        Conversation { shared, id }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    /// Durably admits an input and returns its submission. A run answers it; while one is already working, the
    /// input waits in the inbox.
    pub fn submit(&self, input: impl Into<Input>) -> Result<Submission, Error> {
        let input: Input = input.into();
        let (conversation, id) = self.shared.commit(|state| {
            let known = input
                .request_id
                .as_deref()
                .and_then(|request_id| state.submission_for_request(request_id));
            if let Some(existing) = known {
                return Ok((Vec::new(), existing.clone()));
            }
            if !state.conversations.contains_key(&self.id) {
                return Err(Error::UnknownConversation);
            }
            let id = uuid::Uuid::now_v7().to_string();
            let submission = Queued {
                id: id.clone(),
                request_id: input.request_id.clone(),
                content: input.content.clone(),
                when_busy: input.when_busy,
            };
            Ok((
                vec![Event::Queued {
                    conversation: self.id.clone(),
                    submission,
                }],
                (self.id.clone(), id),
            ))
        })?;
        self.shared.start(&conversation);
        Ok(Submission {
            shared: self.shared.clone(),
            conversation,
            id,
        })
    }

    /// Reacquires a submission by id, for example to wait for it after a restart.
    pub fn submission(&self, id: &str) -> Submission {
        Submission {
            shared: self.shared.clone(),
            conversation: self.id.clone(),
            id: id.to_owned(),
        }
    }

    /// The conversation's committed state right now.
    pub fn snapshot(&self) -> Result<ConversationState, Error> {
        self.shared.read(&self.id, Clone::clone)
    }

    /// The current state together with every later update, with nothing missed in between.
    pub fn watch(&self) -> Result<(ConversationState, Updates), Error> {
        let mut inner = self.shared.lock();
        let state = inner
            .state
            .conversations
            .get(&self.id)
            .cloned()
            .ok_or(Error::UnknownConversation)?;
        let (sender, receiver) = mpsc::unbounded_channel();
        inner.watchers.push(Watcher {
            conversation: Some(self.id.clone()),
            sender,
        });
        Ok((state, Updates { receiver }))
    }

    pub fn entry(&self, id: EntryId) -> Result<Option<Entry>, Error> {
        self.shared
            .read(&self.id, |state| state.entries.get(id as usize).cloned())
    }

    /// Changes what the conversation runs with. The change applies from the next model request.
    pub fn configure(&self, change: impl FnOnce(&mut AgentConfig)) -> Result<(), Error> {
        self.shared.commit_in(&self.id, |state| {
            let mut agent = state.agent.clone();
            change(&mut agent);
            Ok((
                vec![Event::AgentChanged {
                    conversation: self.id.clone(),
                    agent,
                }],
                (),
            ))
        })
    }

    /// Appends an application record to the transcript. The model does not see it.
    pub fn append(&self, name: &str, data: Value) -> Result<EntryId, Error> {
        self.shared.commit_in(&self.id, |state| {
            let id = state.next_entry_id();
            let kind = EntryKind::Custom {
                name: name.to_owned(),
                data,
            };
            let entry = Entry {
                id,
                timestamp: now_ms(),
                kind,
            };
            Ok((
                vec![Event::Entry {
                    conversation: self.id.clone(),
                    entry,
                }],
                id,
            ))
        })
    }

    /// Stores an application value with the conversation; `None` removes it.
    pub fn set_value(&self, key: &str, value: Option<Value>) -> Result<(), Error> {
        let event = Event::Value {
            conversation: self.id.clone(),
            key: key.to_owned(),
            value,
        };
        self.shared.commit_in(&self.id, |_| Ok((vec![event], ())))
    }

    pub fn value(&self, key: &str) -> Result<Option<Value>, Error> {
        self.shared.read(&self.id, |state| state.values.get(key).cloned())
    }

    /// Stops the conversation: queued inputs are withdrawn, running tool calls are cancelled, and the call
    /// returns once the conversation is idle.
    pub async fn abort(&self) -> Result<(), Error> {
        self.shared.abort(&self.id)?;
        self.wait_idle().await
    }

    pub fn abort_blocking(&self) -> Result<(), Error> {
        block_on(self.abort())
    }

    /// Starts a new context: the model no longer sees older entries, which stay stored. `note` opens the new
    /// context, for example with a handoff summary. Work in progress is aborted first.
    pub async fn reset(&self, note: Option<String>) -> Result<(), Error> {
        self.abort().await?;
        self.shared.commit_in(&self.id, |state| {
            let entry = Entry {
                id: state.next_entry_id(),
                timestamp: now_ms(),
                kind: EntryKind::Reset { note },
            };
            Ok((
                vec![Event::Entry {
                    conversation: self.id.clone(),
                    entry,
                }],
                (),
            ))
        })
    }

    pub fn reset_blocking(&self, note: Option<String>) -> Result<(), Error> {
        block_on(self.reset(note))
    }

    /// Summarizes the older part of the context now, keeping the recent part verbatim. `instructions` tell the
    /// summary what to pay attention to. Returns false when there was nothing older to summarize.
    pub async fn compact(&self, instructions: Option<&str>) -> Result<bool, Error> {
        let shared = &self.shared;
        let agent = shared.read(&self.id, |state| state.agent.or(&shared.defaults))?;
        let model = shared.models.model(agent.model.as_deref()).map_err(Error::Compaction)?;
        let keep = shared.settings.compaction.keep_recent_tokens;
        let cancel = CancellationToken::new();
        // The summary request needs the harness's runtime, whatever thread asks for it.
        let (shared, id, instructions) = (shared.clone(), self.id.clone(), instructions.map(str::to_owned));
        let task = self.shared.runtime.spawn(async move {
            crate::compaction::compact(&shared, &id, &model, instructions.as_deref(), keep, &cancel).await
        });
        match task.await.map_err(|error| Error::Compaction(error.to_string()))?? {
            Compacted::Done => Ok(true),
            Compacted::Nothing => Ok(false),
            Compacted::Failed(detail) => Err(Error::Compaction(detail)),
        }
    }

    pub fn compact_blocking(&self, instructions: Option<&str>) -> Result<bool, Error> {
        block_on(self.compact(instructions))
    }

    /// Resolves once no run is working and nothing is queued.
    pub async fn wait_idle(&self) -> Result<(), Error> {
        let id = self.id.clone();
        self.shared
            .wait_for(move |inner| {
                let busy = inner
                    .state
                    .conversations
                    .get(&id)
                    .is_some_and(ConversationState::is_busy);
                (!busy && !inner.active.contains_key(&id)).then_some(())
            })
            .await
    }

    pub fn wait_idle_blocking(&self) -> Result<(), Error> {
        block_on(self.wait_idle())
    }
}

/// An input that was admitted and can be waited for.
#[derive(Clone)]
pub struct Submission {
    shared: Arc<Shared>,
    conversation: ConversationId,
    id: SubmissionId,
}

impl Submission {
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn conversation(&self) -> &str {
        &self.conversation
    }

    /// Resolves once the input is answered or has failed. Dropping the wait does not stop the work.
    pub async fn wait(&self) -> Result<Outcome, Error> {
        let (conversation, id) = (self.conversation.clone(), self.id.clone());
        self.shared
            .wait_for(move |inner| inner.state.conversations.get(&conversation)?.settled.get(&id).cloned())
            .await
    }

    pub fn wait_blocking(&self) -> Result<Outcome, Error> {
        block_on(self.wait())
    }

    /// Waits for the input's answer. An input that ended without one is [`Error::Unanswered`].
    pub async fn answer(&self) -> Result<AssistantMessage, Error> {
        match self.wait().await? {
            Outcome::Done { entry } => {
                let found = self
                    .shared
                    .read(&self.conversation, |state| state.entries.get(entry as usize).cloned())?;
                match found.map(|entry| entry.kind) {
                    Some(EntryKind::Assistant { message }) => Ok(message),
                    _ => Err(Error::Failed("the answer entry is missing".to_owned())),
                }
            }
            Outcome::Unanswered(failure) => Err(Error::Unanswered(failure)),
        }
    }

    pub fn answer_blocking(&self) -> Result<AssistantMessage, Error> {
        block_on(self.answer())
    }
}
