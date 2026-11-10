//! Compaction shrinks what the model sees: older entries are summarized by the model and replaced, in its context,
//! by that summary. The entries themselves stay stored.

use futures_util::StreamExt;
use ri_ai::{AssistantEvent, Content, Context, Message, Model, StopReason, StreamOptions, now_ms};
use tokio_util::sync::CancellationToken;

use crate::Error;
use crate::event::{ConversationId, Entry, EntryId, EntryKind, Event};
use crate::harness::Shared;
use crate::state::ConversationState;

const SUMMARY_PROMPT: &str = "You summarize a conversation between a user and an AI assistant so the assistant can \
continue it without the original messages. Write a dense summary that keeps: what the user wants and any \
constraints or preferences they stated, decisions made and why, facts and results learned from tools, names, \
paths, identifiers and numbers that later work depends on, what was done, and what is still open. Leave out \
pleasantries and anything that no longer matters. Write only the summary.";
const SUMMARY_LEAD: &str = "The earlier part of this conversation was summarized to save space. Summary:";
/// Tool results are cut to this many characters in the text given to the summarizer.
const MAX_RESULT_CHARS: usize = 2000;
const IMAGE_TOKENS: u64 = 1200;

/// When and how much to compact.
#[derive(Debug, Clone, PartialEq)]
pub struct Compaction {
    /// Compact automatically before a request that would not fit. Manual compaction always works.
    pub enabled: bool,
    /// A request is preceded by a compaction once the context exceeds the model's window minus this.
    pub reserve_tokens: u64,
    /// Roughly how much recent context stays verbatim.
    pub keep_recent_tokens: u64,
}

impl Default for Compaction {
    fn default() -> Self {
        Compaction {
            enabled: true,
            reserve_tokens: 16_384,
            keep_recent_tokens: 20_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Compacted {
    Done,
    /// The context is a single turn or empty, so there is nothing older to summarize.
    Nothing,
    Failed(String),
}

fn content_tokens(content: &[Content]) -> u64 {
    content
        .iter()
        .map(|block| match block {
            Content::Text { text, .. } => text.len() as u64 / 4,
            Content::Thinking { thinking, .. } => thinking.len() as u64 / 4,
            Content::Image { .. } => IMAGE_TOKENS,
            Content::ToolCall { name, arguments, .. } => {
                (name.len() + serde_json::Value::Object(arguments.clone()).to_string().len()) as u64 / 4
            }
        })
        .sum()
}

/// A rough token count of an entry: four characters per token.
pub(crate) fn estimate_tokens(entry: &Entry) -> u64 {
    match &entry.kind {
        EntryKind::User { content, .. } | EntryKind::ToolResult { content, .. } => content_tokens(content) + 4,
        EntryKind::Assistant { message } => content_tokens(&message.content) + 4,
        EntryKind::Reset { note } => note.as_ref().map_or(0, |note| note.len() as u64 / 4),
        EntryKind::Compaction { .. } | EntryKind::Custom { .. } => 0,
    }
}

/// How many tokens the next request's context holds: what the provider counted for the newest response, plus an
/// estimate of everything after it.
pub(crate) fn context_tokens(state: &ConversationState) -> u64 {
    let (opening, entries) = state.context_entries();
    let opening: u64 = opening
        .iter()
        .map(|message| match message {
            Message::User { content } => content_tokens(content),
            _ => 0,
        })
        .sum();
    let counted = entries
        .iter()
        .rposition(|entry| matches!(&entry.kind, EntryKind::Assistant { message } if message.usage.total_tokens > 0));
    match counted {
        Some(position) => {
            let EntryKind::Assistant { message } = &entries[position].kind else {
                unreachable!("position is an assistant entry")
            };
            let usage = message.usage;
            let known = usage.input + usage.cache_read + usage.cache_write + usage.output;
            known + entries[position + 1..].iter().map(estimate_tokens).sum::<u64>()
        }
        None => opening + entries.iter().map(estimate_tokens).sum::<u64>(),
    }
}

struct Plan {
    first_kept: EntryId,
    /// The text to summarize: an earlier summary, if any, and the entries that leave the context.
    transcript: String,
}

/// Picks the cut: the earliest user entry from which the rest fits `keep_recent_tokens`, so no tool call is
/// separated from its result. When even the newest turn is larger, the cut is at that turn.
fn plan(state: &ConversationState, keep_recent_tokens: u64) -> Option<Plan> {
    let (opening, entries) = state.context_entries();
    let turn_starts = entries
        .iter()
        .enumerate()
        .skip(1)
        .filter(|(_, entry)| matches!(entry.kind, EntryKind::User { .. }));
    let turn_starts: Vec<usize> = turn_starts.map(|(position, _)| position).collect();
    let tail_tokens = |position: usize| entries[position..].iter().map(estimate_tokens).sum::<u64>();
    let cut = turn_starts
        .iter()
        .copied()
        .find(|position| tail_tokens(*position) <= keep_recent_tokens)
        .or(turn_starts.last().copied())?;
    let earlier = opening.iter().filter_map(|message| match message {
        Message::User { content } => Some(format!(
            "Summary of what came before:\n{}",
            ri_ai::joined_text(content, "\n")
        )),
        _ => None,
    });
    let lines = earlier
        .chain(entries[..cut].iter().filter_map(transcript_line))
        .collect::<Vec<_>>()
        .join("\n\n");
    (!lines.trim().is_empty()).then(|| Plan {
        first_kept: entries[cut].id,
        transcript: lines,
    })
}

fn transcript_line(entry: &Entry) -> Option<String> {
    let text = |content: &[Content]| {
        let images = content
            .iter()
            .filter(|block| matches!(block, Content::Image { .. }))
            .count();
        let note = if images == 0 {
            String::new()
        } else {
            format!(" [{images} image(s)]")
        };
        format!("{}{note}", ri_ai::joined_text(content, "\n"))
    };
    match &entry.kind {
        EntryKind::User { content, .. } => Some(format!("User: {}", text(content))),
        EntryKind::Assistant { message } => {
            let calls = message.tool_calls().map(|(_, name, arguments)| {
                format!("[called {name} with {}]", serde_json::Value::Object(arguments.clone()))
            });
            let parts: Vec<String> = std::iter::once(message.text())
                .filter(|said| !said.is_empty())
                .chain(calls)
                .collect();
            (!parts.is_empty()).then(|| format!("Assistant: {}", parts.join("\n")))
        }
        EntryKind::ToolResult {
            tool_name,
            content,
            is_error,
            ..
        } => {
            let result: String = text(content).chars().take(MAX_RESULT_CHARS).collect();
            Some(format!(
                "Tool {tool_name} {}: {result}",
                if *is_error { "failed" } else { "returned" }
            ))
        }
        EntryKind::Reset { note } => note.as_ref().map(|note| format!("User: {note}")),
        EntryKind::Compaction { .. } | EntryKind::Custom { .. } => None,
    }
}

/// Summarizes the older part of the conversation's context with `model` and stores the summary.
pub(crate) async fn compact(
    shared: &Shared,
    conversation: &ConversationId,
    model: &Model,
    instructions: Option<&str>,
    keep_recent_tokens: u64,
    cancel: &CancellationToken,
) -> Result<Compacted, Error> {
    let Some(plan) = shared.read(conversation, |state| plan(state, keep_recent_tokens))? else {
        return Ok(Compacted::Nothing);
    };
    let focus = instructions
        .map(|instructions| format!("\n\nPay particular attention to: {instructions}"))
        .unwrap_or_default();
    let request = format!(
        "Summarize this conversation.{focus}\n\n<conversation>\n{}\n</conversation>",
        plan.transcript
    );
    let context = Context {
        system_prompt: Some(SUMMARY_PROMPT.to_owned()),
        messages: vec![Message::user(request)],
        tools: Vec::new(),
    };
    let options = StreamOptions {
        session_id: Some(format!("{conversation}-compaction")),
        ..StreamOptions::default()
    };
    let mut events = shared.models.stream(model, context, options);
    let message = loop {
        let event = tokio::select! {
            _ = cancel.cancelled() => return Ok(Compacted::Failed("aborted".to_owned())),
            event = events.next() => event,
        };
        match event {
            Some(AssistantEvent::Done(message)) => break message,
            Some(_) => {}
            None => return Ok(Compacted::Failed("the model stream ended without a result".to_owned())),
        }
    };
    let summary = message.text();
    if matches!(message.stop_reason, StopReason::Error | StopReason::Aborted) || summary.trim().is_empty() {
        return Ok(Compacted::Failed(
            message
                .error_message
                .unwrap_or_else(|| "the model returned no summary".to_owned()),
        ));
    }
    shared.commit_in(conversation, |state| {
        let kind = EntryKind::Compaction {
            summary: format!("{SUMMARY_LEAD}\n\n{}", summary.trim()),
            first_kept: plan.first_kept,
        };
        let entry = Entry {
            id: state.next_entry_id(),
            timestamp: now_ms(),
            kind,
        };
        Ok((
            vec![Event::Entry {
                conversation: conversation.clone(),
                entry,
            }],
            Compacted::Done,
        ))
    })
}

/// Whether a provider rejected a request because its context is too long.
pub(crate) fn is_context_overflow(error: &str) -> bool {
    let error = error.to_lowercase();
    [
        "context length",
        "context window",
        "maximum context",
        "context_length_exceeded",
        "prompt is too long",
        "input is too long",
        "too many tokens",
        "exceeds the maximum",
    ]
    .iter()
    .any(|needle| error.contains(needle))
}
