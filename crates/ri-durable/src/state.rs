//! A session's state in memory: the fold of its events.

use std::collections::BTreeMap;

use ri_ai::{Content, Message, Usage};
use serde_json::Value;

use crate::event::{
    AgentConfig, Call, CallStatus, ConversationId, Entry, EntryId, EntryKind, Event, Outcome, Phase, Queued, Run,
    RunFailure, SubmissionId,
};

#[derive(Debug, Clone, Default)]
pub struct State {
    pub conversations: BTreeMap<ConversationId, ConversationState>,
    /// The first conversation created.
    pub root: Option<ConversationId>,
    /// Whether the last stored event was an orderly close: the process that wrote the session last did not crash.
    pub closed_cleanly: bool,
    request_ids: BTreeMap<String, (ConversationId, SubmissionId)>,
}

#[derive(Debug, Clone, Default)]
pub struct ConversationState {
    pub id: ConversationId,
    pub parent: Option<ConversationId>,
    pub agent: AgentConfig,
    pub entries: Vec<Entry>,
    pub inbox: Vec<Queued>,
    pub run: Option<Run>,
    /// Why the last run ended without an answer, until the next run starts.
    pub failure: Option<RunFailure>,
    /// Outcomes of settled submissions.
    pub settled: BTreeMap<SubmissionId, Outcome>,
    pub values: BTreeMap<String, Value>,
}

impl State {
    pub fn from_events<'a>(events: impl IntoIterator<Item = &'a Event>) -> State {
        events.into_iter().fold(State::default(), |mut state, event| {
            state.apply(event);
            state
        })
    }

    pub fn apply(&mut self, event: &Event) {
        self.closed_cleanly = matches!(event, Event::Closed);
        match event {
            Event::ConversationCreated {
                conversation,
                parent,
                agent,
            } => {
                self.root.get_or_insert_with(|| conversation.clone());
                self.conversations.insert(
                    conversation.clone(),
                    ConversationState {
                        id: conversation.clone(),
                        parent: parent.clone(),
                        agent: agent.clone(),
                        ..ConversationState::default()
                    },
                );
            }
            Event::Queued {
                conversation,
                submission,
            } => {
                if let Some(request_id) = &submission.request_id {
                    self.request_ids
                        .insert(request_id.clone(), (conversation.clone(), submission.id.clone()));
                }
                self.update(conversation, |state| state.inbox.push(submission.clone()));
            }
            Event::AgentChanged { conversation, agent } => {
                self.update(conversation, |state| state.agent = agent.clone())
            }
            Event::Entry { conversation, entry } => self.update(conversation, |state| {
                if let EntryKind::User {
                    submission: Some(submission),
                    ..
                } = &entry.kind
                {
                    state.inbox.retain(|queued| queued.id != *submission);
                }
                state.entries.push(entry.clone());
            }),
            Event::Run { conversation, run } => self.update(conversation, |state| {
                state.run = Some(run.clone());
                state.failure = None;
            }),
            Event::Call {
                conversation,
                call,
                status,
            } => self.update(conversation, |state| {
                if *status == CallStatus::Done {
                    // What a call stored for its own reruns goes with the commit that stores its result.
                    let (own, nested) = (format!("call/{call}/"), format!("call/{call}>"));
                    state
                        .values
                        .retain(|key, _| !key.starts_with(&own) && !key.starts_with(&nested));
                }
                let calls = match &mut state.run {
                    Some(Run {
                        phase: Phase::Tools { calls },
                        ..
                    }) => calls,
                    _ => return,
                };
                if let Some(found) = calls.iter_mut().find(|candidate| candidate.id == *call) {
                    found.status = *status;
                }
            }),
            Event::RunEnded { conversation, failure } => self.update(conversation, |state| {
                state.run = None;
                state.failure = failure.clone();
            }),
            Event::Settled {
                conversation,
                submission,
                outcome,
            } => self.update(conversation, |state| {
                state.inbox.retain(|queued| queued.id != *submission);
                state.settled.insert(submission.clone(), outcome.clone());
            }),
            Event::Value {
                conversation,
                key,
                value,
            } => self.update(conversation, |state| match value {
                Some(value) => {
                    state.values.insert(key.clone(), value.clone());
                }
                None => {
                    state.values.remove(key);
                }
            }),
            Event::Opened | Event::Closed => {}
        }
    }

    fn update(&mut self, conversation: &str, change: impl FnOnce(&mut ConversationState)) {
        if let Some(state) = self.conversations.get_mut(conversation) {
            change(state);
        }
    }

    /// The submission an earlier input with this `request_id` created.
    pub fn submission_for_request(&self, request_id: &str) -> Option<&(ConversationId, SubmissionId)> {
        self.request_ids.get(request_id)
    }
}

impl ConversationState {
    pub fn next_entry_id(&self) -> EntryId {
        self.entries.len() as EntryId
    }

    /// Busy while a run is working or inputs wait to start one.
    pub fn is_busy(&self) -> bool {
        self.run.is_some() || !self.inbox.is_empty()
    }

    pub fn pending_calls(&self) -> Vec<Call> {
        match &self.run {
            Some(Run {
                phase: Phase::Tools { calls },
                ..
            }) => calls
                .iter()
                .filter(|call| call.status != CallStatus::Done)
                .cloned()
                .collect(),
            _ => Vec::new(),
        }
    }

    /// The entries the model sees: those after the newest reset, with compacted ones replaced by their summary.
    pub fn context_entries(&self) -> (Vec<Message>, &[Entry]) {
        let reset = self
            .entries
            .iter()
            .rposition(|entry| matches!(entry.kind, EntryKind::Reset { .. }));
        let active = &self.entries[reset.unwrap_or(0)..];
        let compaction = active.iter().rev().find_map(|entry| match &entry.kind {
            EntryKind::Compaction { summary, first_kept } => Some((summary.as_str(), *first_kept)),
            _ => None,
        });
        match compaction {
            Some((summary, first_kept)) => {
                let kept = active
                    .iter()
                    .position(|entry| entry.id >= first_kept)
                    .unwrap_or(active.len());
                (vec![Message::user(summary)], &active[kept..])
            }
            None => (Vec::new(), active),
        }
    }

    /// The messages of the next model request.
    pub fn context(&self) -> Vec<Message> {
        let (opening, entries) = self.context_entries();
        opening.into_iter().chain(entries.iter().filter_map(message)).collect()
    }

    /// Token and cost totals of the model responses in this conversation, by `provider/model`.
    pub fn usage(&self) -> BTreeMap<String, Usage> {
        self.entries.iter().fold(BTreeMap::new(), |mut totals, entry| {
            if let EntryKind::Assistant { message } = &entry.kind {
                let total: &mut Usage = totals
                    .entry(format!("{}/{}", message.provider, message.model))
                    .or_default();
                *total = add_usage(*total, message.usage);
            }
            totals
        })
    }
}

fn message(entry: &Entry) -> Option<Message> {
    match &entry.kind {
        EntryKind::User { content, .. } => Some(Message::User {
            content: content.clone(),
        }),
        EntryKind::Assistant { message } => Some(Message::Assistant(message.clone())),
        EntryKind::ToolResult {
            tool_call_id,
            tool_name,
            content,
            is_error,
            ..
        } => Some(Message::ToolResult {
            tool_call_id: tool_call_id.clone(),
            tool_name: tool_name.clone(),
            content: content.clone(),
            is_error: *is_error,
        }),
        EntryKind::Reset { note } => note.as_ref().map(|note| Message::User {
            content: vec![Content::text(note)],
        }),
        EntryKind::Compaction { .. } | EntryKind::Custom { .. } => None,
    }
}

pub fn add_usage(left: Usage, right: Usage) -> Usage {
    Usage {
        input: left.input + right.input,
        output: left.output + right.output,
        cache_read: left.cache_read + right.cache_read,
        cache_write: left.cache_write + right.cache_write,
        reasoning: left.reasoning + right.reasoning,
        total_tokens: left.total_tokens + right.total_tokens,
        cost: ri_ai::Cost {
            input: left.cost.input + right.cost.input,
            output: left.cost.output + right.cost.output,
            cache_read: left.cost.cache_read + right.cost.cache_read,
            cache_write: left.cost.cache_write + right.cost.cache_write,
            total: left.cost.total + right.cost.total,
        },
    }
}
