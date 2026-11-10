//! The task that drives one conversation: it places queued inputs, asks the model, runs tool calls, and stores a
//! checkpoint at every step, so the work can continue from the last one after a restart.

use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::time::Duration;

use futures_util::future::join_all;
use futures_util::{FutureExt, StreamExt};
use ri_ai::{
    AssistantEvent, AssistantMessage, Content, Context, StopReason, StreamOptions, Tool as ToolDeclaration, now_ms,
};
use tokio_util::sync::CancellationToken;

use crate::Error;
use crate::compaction::{self, Compacted};
use crate::event::{
    Call, CallStatus, ConversationId, Entry, EntryId, EntryKind, Event, Outcome, Phase, Reason, Run, RunFailure,
    WhenBusy,
};
use crate::harness::{Shared, Update, end_events};
use crate::state::ConversationState;
use crate::tool::{BeforeTool, ToolCall, ToolContext, ToolOutput};
use ri_ai::JsonObject;

const MAX_RETRY_DELAY: Duration = Duration::from_secs(60);

enum Step {
    Request { attempt: u32 },
    Tools { calls: Vec<Call> },
    Idle,
}

pub(crate) async fn drive(shared: Arc<Shared>, conversation: ConversationId, cancel: CancellationToken) {
    loop {
        let step = match next_step(&shared, &conversation) {
            Ok(Step::Idle) if shared.retire(&conversation) => return,
            Ok(Step::Idle) => continue,
            Ok(step) => step,
            Err(_) => break,
        };
        let done = match step {
            Step::Request { attempt } => request(&shared, &conversation, attempt, &cancel).await,
            Step::Tools { calls } => tools(&shared, &conversation, calls, &cancel).await,
            Step::Idle => Ok(()),
        };
        if done.is_err() {
            break;
        }
        if cancel.is_cancelled() {
            let aborted = shared.commit_in(&conversation, |state| {
                let failure = RunFailure {
                    reason: Reason::Aborted,
                    detail: None,
                };
                Ok((end_events(state, failure, "The tool call was aborted."), ()))
            });
            shared.release(&conversation);
            if aborted.is_ok() {
                // Inputs submitted after the abort start fresh work.
                shared.start(&conversation);
            }
            return;
        }
    }
    shared.release(&conversation);
}

fn user_entry(id: EntryId, queued: &crate::event::Queued) -> Entry {
    let kind = EntryKind::User {
        content: queued.content.clone(),
        submission: Some(queued.id.clone()),
    };
    Entry {
        id,
        timestamp: now_ms(),
        kind,
    }
}

/// Reads where the conversation stands. With no run and inputs waiting, it places them and starts a run.
fn next_step(shared: &Shared, conversation: &str) -> Result<Step, Error> {
    shared.commit_in(conversation, |state| match &state.run {
        Some(Run {
            phase: Phase::Request { attempt },
            ..
        }) => Ok((Vec::new(), Step::Request { attempt: *attempt })),
        Some(Run {
            phase: Phase::Tools { .. },
            ..
        }) => Ok((
            Vec::new(),
            Step::Tools {
                calls: state.pending_calls(),
            },
        )),
        None if state.inbox.is_empty() => Ok((Vec::new(), Step::Idle)),
        None => {
            let placed = state.inbox.iter().enumerate().map(|(offset, queued)| Event::Entry {
                conversation: state.id.clone(),
                entry: user_entry(state.next_entry_id() + offset as EntryId, queued),
            });
            let run = Run {
                inputs: state.inbox.iter().map(|queued| queued.id.clone()).collect(),
                turns: 0,
                phase: Phase::Request { attempt: 0 },
            };
            let started = Event::Run {
                conversation: state.id.clone(),
                run,
            };
            Ok((placed.chain([started]).collect(), Step::Request { attempt: 0 }))
        }
    })
}

fn fail(shared: &Shared, conversation: &str, reason: Reason, detail: Option<String>) -> Result<(), Error> {
    shared.commit_in(conversation, |state| {
        let failure = RunFailure { reason, detail };
        Ok((
            end_events(state, failure, "The run ended before this tool call finished."),
            (),
        ))
    })
}

fn is_retryable(message: &AssistantMessage) -> bool {
    message
        .error_status
        .is_none_or(|status| status == 408 || status == 429 || status >= 500)
}

async fn request(
    shared: &Arc<Shared>,
    conversation: &ConversationId,
    attempt: u32,
    cancel: &CancellationToken,
) -> Result<(), Error> {
    let agent = shared.read(conversation, |state| state.agent.or(&shared.defaults))?;
    let model = match shared.models.model(agent.model.as_deref()) {
        Ok(model) => model,
        Err(detail) => return fail(shared, conversation, Reason::NoModel, Some(detail)),
    };
    let policy = &shared.settings.compaction;
    let used = shared.read(conversation, compaction::context_tokens)?;
    if policy.enabled && used + policy.reserve_tokens > model.context_window {
        let keep = policy.keep_recent_tokens.min(model.context_window / 2);
        if let Compacted::Failed(detail) = compaction::compact(shared, conversation, &model, None, keep, cancel).await?
        {
            if cancel.is_cancelled() {
                return Ok(());
            }
            return fail(
                shared,
                conversation,
                Reason::ModelError,
                Some(format!("Compaction failed: {detail}")),
            );
        }
    }
    let messages = shared.read(conversation, ConversationState::context)?;
    let offered = |name: &String| agent.tools.as_ref().is_none_or(|enabled| enabled.contains(name));
    let tools = shared
        .tools
        .values()
        .map(|tool| tool.spec())
        .filter(|spec| offered(&spec.name))
        .map(|spec| ToolDeclaration {
            name: spec.name.clone(),
            description: spec.description.clone(),
            parameters: spec.parameters.clone(),
        })
        .collect();
    let mut context = Context {
        system_prompt: agent.instructions.clone(),
        messages,
        tools,
    };
    for hook in &shared.hooks {
        hook.before_request(conversation, &mut context);
    }
    let options = StreamOptions {
        thinking: agent.thinking,
        session_id: Some(conversation.clone()),
        ..StreamOptions::default()
    };

    let mut events = shared.models.stream(&model, context, options);
    let message = loop {
        let event = tokio::select! {
            _ = cancel.cancelled() => return Ok(()),
            event = events.next() => event,
        };
        match event {
            Some(AssistantEvent::Done(message)) => break message,
            Some(event) => shared.notify(
                conversation,
                Update::Partial {
                    conversation: conversation.clone(),
                    event,
                },
            ),
            None => break AssistantMessage::failed(&model, "The model stream ended without a result", None),
        }
    };
    drop(events);

    if matches!(message.stop_reason, StopReason::Error | StopReason::Aborted) {
        let detail = message.error_message.clone();
        if detail.as_deref().is_some_and(compaction::is_context_overflow) {
            // The provider says the context is too long: shrink it and send the request again.
            let keep = policy.keep_recent_tokens.min(model.context_window / 4);
            return match compaction::compact(shared, conversation, &model, None, keep, cancel).await? {
                Compacted::Done => Ok(()),
                _ => fail(shared, conversation, Reason::ModelError, detail),
            };
        }
        if !is_retryable(&message) || attempt >= shared.settings.max_retries {
            return fail(shared, conversation, Reason::ModelError, detail);
        }
        shared.commit_in(conversation, |state| {
            let run = state.run.clone().map(|run| Run {
                phase: Phase::Request { attempt: attempt + 1 },
                ..run
            });
            Ok((
                run.map(|run| Event::Run {
                    conversation: conversation.clone(),
                    run,
                })
                .into_iter()
                .collect(),
                (),
            ))
        })?;
        let delay = shared
            .settings
            .retry_delay
            .saturating_mul(2u32.saturating_pow(attempt))
            .min(MAX_RETRY_DELAY);
        tokio::select! {
            _ = cancel.cancelled() => {}
            _ = tokio::time::sleep(delay) => {}
        }
        return Ok(());
    }
    shared.commit_in(conversation, |state| Ok((answer_events(state, message), ())))
}

/// The events that store a model response: with tool calls the run moves on to them, without it is answered.
fn answer_events(state: &ConversationState, message: AssistantMessage) -> Vec<Event> {
    let conversation = &state.id;
    let Some(run) = &state.run else { return Vec::new() };
    let entry_id = state.next_entry_id();
    let calls: Vec<Call> = message
        .tool_calls()
        .map(|(id, name, arguments)| Call {
            id: id.to_owned(),
            name: name.to_owned(),
            arguments: arguments.clone(),
            status: CallStatus::Pending { restarts: 0 },
        })
        .collect();
    let cut_off = message.stop_reason == StopReason::Length;
    let entry = Entry {
        id: entry_id,
        timestamp: now_ms(),
        kind: EntryKind::Assistant { message },
    };
    let stored = Event::Entry {
        conversation: conversation.clone(),
        entry,
    };
    if calls.is_empty() {
        let settled = run.inputs.iter().map(|submission| Event::Settled {
            conversation: conversation.clone(),
            submission: submission.clone(),
            outcome: Outcome::Done { entry: entry_id },
        });
        let ended = Event::RunEnded {
            conversation: conversation.clone(),
            failure: None,
        };
        return std::iter::once(stored).chain(settled).chain([ended]).collect();
    }
    if cut_off {
        // The response hit its output limit, so the calls' arguments may be incomplete: they are not run.
        let results = calls.iter().enumerate().map(|(offset, call)| Event::Entry {
            conversation: conversation.clone(),
            entry: Entry {
                id: entry_id + 1 + offset as EntryId,
                timestamp: now_ms(),
                kind: EntryKind::ToolResult {
                    tool_call_id: call.id.clone(),
                    tool_name: call.name.clone(),
                    content: vec![Content::text(
                        "The response was cut off at its output limit before this call was complete. It was not run.",
                    )],
                    is_error: true,
                    details: None,
                },
            },
        });
        let run = Run {
            turns: run.turns + 1,
            phase: Phase::Request { attempt: 0 },
            ..run.clone()
        };
        return std::iter::once(stored)
            .chain(results)
            .chain([Event::Run {
                conversation: conversation.clone(),
                run,
            }])
            .collect();
    }
    let run = Run {
        phase: Phase::Tools { calls },
        ..run.clone()
    };
    vec![
        stored,
        Event::Run {
            conversation: conversation.clone(),
            run,
        },
    ]
}

async fn tools(
    shared: &Arc<Shared>,
    conversation: &ConversationId,
    calls: Vec<Call>,
    cancel: &CancellationToken,
) -> Result<(), Error> {
    let cwd = shared.read(conversation, |state| state.agent.or(&shared.defaults).cwd)?;
    let results = join_all(
        calls
            .into_iter()
            .map(|call| run_call(shared, conversation, call, cwd.clone(), cancel)),
    )
    .await;
    results.into_iter().collect::<Result<Vec<()>, Error>>()?;
    if cancel.is_cancelled() {
        return Ok(());
    }
    shared.commit_in(conversation, |state| {
        let Some(run) = &state.run else {
            return Ok((Vec::new(), ()));
        };
        let turns = run.turns + 1;
        let limit = state.agent.or(&shared.defaults).max_turns;
        if limit.is_some_and(|limit| turns >= limit) {
            let failure = RunFailure {
                reason: Reason::TurnLimit,
                detail: None,
            };
            return Ok((end_events(state, failure, "The run reached its turn limit."), ()));
        }
        // Steers join the run here, after the tool round.
        let steers: Vec<_> = state
            .inbox
            .iter()
            .filter(|queued| queued.when_busy == WhenBusy::Steer)
            .collect();
        let placed = steers.iter().enumerate().map(|(offset, queued)| Event::Entry {
            conversation: conversation.clone(),
            entry: user_entry(state.next_entry_id() + offset as EntryId, queued),
        });
        let inputs = run
            .inputs
            .iter()
            .cloned()
            .chain(steers.iter().map(|queued| queued.id.clone()))
            .collect();
        let run = Run {
            inputs,
            turns,
            phase: Phase::Request { attempt: 0 },
        };
        Ok((
            placed
                .chain([Event::Run {
                    conversation: conversation.clone(),
                    run,
                }])
                .collect(),
            (),
        ))
    })
}

async fn run_call(
    shared: &Arc<Shared>,
    conversation: &ConversationId,
    call: Call,
    cwd: Option<std::path::PathBuf>,
    cancel: &CancellationToken,
) -> Result<(), Error> {
    let restarts = match call.status {
        CallStatus::Pending { restarts } | CallStatus::Running { restarts } => restarts,
        CallStatus::Done => return Ok(()),
    };
    let seen = ToolCall {
        conversation: conversation.clone(),
        id: call.id.clone(),
        name: call.name.clone(),
        arguments: call.arguments.clone(),
    };
    let decision = before_tool(shared, &seen);
    // The intent is stored before the tool runs, so a crash mid-call is known to have happened.
    let running = Event::Call {
        conversation: conversation.clone(),
        call: call.id.clone(),
        status: CallStatus::Running { restarts },
    };
    shared.commit(|_| Ok((vec![running], ())))?;

    let output = match decision {
        Err(reason) => ToolOutput::error(reason),
        Ok(arguments) => {
            let context = ToolContext {
                conversation: conversation.clone(),
                call_id: call.id.clone(),
                cwd,
                cancel: cancel.child_token(),
                restarts,
                shared: shared.clone(),
            };
            tokio::select! {
                // The abort that follows writes this call's result.
                _ = cancel.cancelled() => return Ok(()),
                output = invoke(shared, &call.name, arguments, context) => output,
            }
        }
    };
    let output = shared
        .hooks
        .iter()
        .fold(output, |output, hook| hook.after_tool(&seen, output));
    shared.commit_in(conversation, |state| {
        let kind = EntryKind::ToolResult {
            tool_call_id: call.id.clone(),
            tool_name: call.name.clone(),
            content: if output.content.is_empty() {
                vec![Content::text("")]
            } else {
                output.content
            },
            is_error: output.is_error,
            details: output.details,
        };
        let entry = Entry {
            id: state.next_entry_id(),
            timestamp: now_ms(),
            kind,
        };
        let events = vec![
            Event::Entry {
                conversation: conversation.clone(),
                entry,
            },
            Event::Call {
                conversation: conversation.clone(),
                call: call.id.clone(),
                status: CallStatus::Done,
            },
        ];
        Ok((events, ()))
    })
}

/// Asks the hooks about a call. `Ok` carries the arguments to run it with, `Err` the reason it is blocked.
fn before_tool(shared: &Shared, call: &ToolCall) -> Result<JsonObject, String> {
    shared.hooks.iter().try_fold(call.arguments.clone(), |arguments, hook| {
        match hook.before_tool(&ToolCall {
            arguments: arguments.clone(),
            ..call.clone()
        }) {
            BeforeTool::Continue => Ok(arguments),
            BeforeTool::Rewrite(rewritten) => Ok(rewritten),
            BeforeTool::Block(reason) => Err(reason),
        }
    })
}

/// Runs a registered tool. Failures of every kind become an error result.
async fn invoke(shared: &Shared, name: &str, arguments: JsonObject, context: ToolContext) -> ToolOutput {
    let Some(tool) = shared.tools.get(name) else {
        return ToolOutput::error(format!("Unknown tool: {name}"));
    };
    match AssertUnwindSafe(tool.call(arguments, context)).catch_unwind().await {
        Ok(Ok(output)) => output,
        Ok(Err(message)) => ToolOutput::error(message),
        Err(_) => ToolOutput::error("The tool crashed while running this call."),
    }
}

/// A call one tool makes to another: hooks run as for the model's calls, the result goes to the caller.
pub(crate) async fn nested_call(parent: &ToolContext, name: &str, arguments: JsonObject) -> ToolOutput {
    let shared = &parent.shared;
    let seen = ToolCall {
        conversation: parent.conversation.clone(),
        id: format!("{}>{name}", parent.call_id),
        name: name.to_owned(),
        arguments,
    };
    let output = match before_tool(shared, &seen) {
        Err(reason) => ToolOutput::error(reason),
        Ok(arguments) => {
            let context = ToolContext {
                call_id: seen.id.clone(),
                cancel: parent.cancel.child_token(),
                restarts: 0,
                ..parent.clone()
            };
            tokio::select! {
                _ = parent.cancel.cancelled() => ToolOutput::error("The call was aborted."),
                output = invoke(shared, name, arguments, context) => output,
            }
        }
    };
    shared
        .hooks
        .iter()
        .fold(output, |output, hook| hook.after_tool(&seen, output))
}
