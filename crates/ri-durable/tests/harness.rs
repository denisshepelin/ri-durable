use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use ri_ai::{AssistantEvent, Content, Message};
use ri_durable::testing::{Reply, ScriptedModels};
use ri_durable::{
    AgentConfig, BeforeTool, Conversation, EntryKind, Error, Event, Harness, Hook, Input, JsonlStorage, MemoryStorage,
    Options, Outcome, Reason, Settings, Tool, ToolCall, ToolOutput, Update, replay_safe, tool,
};
use serde_json::json;

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct Echo {
    text: String,
}

fn echo() -> Arc<dyn Tool> {
    tool("echo", "Repeat the text", |args: Echo, _| async move {
        Ok(ToolOutput::text(format!("echo: {}", args.text)))
    })
}

/// A tool that tells the test it started and then never finishes, as a call a crash cuts off would.
fn stuck(started: mpsc::Sender<()>, finish_after: Arc<AtomicU32>) -> Arc<dyn Tool> {
    let runs = Arc::new(AtomicU32::new(0));
    tool("stuck", "Hangs", move |_: Echo, _| {
        let (started, runs, finish_after) = (started.clone(), runs.clone(), finish_after.clone());
        async move {
            let run = runs.fetch_add(1, Ordering::SeqCst) + 1;
            let _ = started.send(());
            if run <= finish_after.load(Ordering::SeqCst) {
                std::future::pending::<()>().await;
            }
            Ok(ToolOutput::text("finished"))
        }
    })
}

fn fast() -> Settings {
    Settings {
        retry_delay: Duration::from_millis(1),
        ..Settings::default()
    }
}

fn open(models: &ScriptedModels, tools: Vec<Arc<dyn Tool>>) -> Harness {
    let options = tools
        .into_iter()
        .fold(Options::new(models.clone()).settings(fast()), Options::tool);
    Harness::open(MemoryStorage::new(), options).unwrap()
}

fn kinds(conversation: &Conversation) -> Vec<String> {
    let entries = conversation.snapshot().unwrap().entries;
    entries
        .iter()
        .map(|entry| match &entry.kind {
            EntryKind::User { content, .. } => format!("user:{}", ri_ai::joined_text(content, "")),
            EntryKind::Assistant { message } if message.tool_calls().next().is_some() => "calls".to_owned(),
            EntryKind::Assistant { message } => format!("assistant:{}", message.text()),
            EntryKind::ToolResult { content, is_error, .. } => {
                format!(
                    "{}:{}",
                    if *is_error { "error" } else { "result" },
                    ri_ai::joined_text(content, "")
                )
            }
            EntryKind::Reset { .. } => "reset".to_owned(),
            EntryKind::Compaction { .. } => "compaction".to_owned(),
            EntryKind::Custom { name, .. } => format!("custom:{name}"),
        })
        .collect()
}

fn wait_until(what: &str, check: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !check() {
        assert!(Instant::now() < deadline, "timed out waiting until {what}");
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn answers_an_input_and_records_usage() {
    let models = ScriptedModels::new([Reply::text("Paris")]);
    let harness = open(&models, vec![]);
    let root = harness.root().unwrap();

    let answer = root.submit("Capital of France?").unwrap().answer_blocking().unwrap();

    assert_eq!(answer.text(), "Paris");
    assert_eq!(kinds(&root), ["user:Capital of France?", "assistant:Paris"]);
    assert_eq!(harness.usage()["test/scripted"].total_tokens, 15);
    assert_eq!(harness.root().unwrap().id(), root.id(), "the root is created once");
}

#[test]
fn runs_tool_calls_and_feeds_their_results_back() {
    let models = ScriptedModels::new([
        Reply::Calls(vec![
            ("echo".into(), json!({ "text": "a" })),
            ("echo".into(), json!({ "text": "b" })),
        ]),
        Reply::text("done"),
    ]);
    let harness = open(&models, vec![echo()]);
    let root = harness.root().unwrap();
    root.configure(|agent| agent.instructions = Some("Be brief.".into()))
        .unwrap();

    assert_eq!(root.submit("go").unwrap().answer_blocking().unwrap().text(), "done");

    let transcript = kinds(&root);
    assert_eq!(transcript.len(), 5);
    assert!(transcript.contains(&"result:echo: a".to_owned()) && transcript.contains(&"result:echo: b".to_owned()));
    let requests = models.requests();
    assert_eq!(requests[0].system_prompt.as_deref(), Some("Be brief."));
    assert_eq!(requests[0].tools[0].name, "echo");
    assert_eq!(requests[1].messages.len(), 4);
    assert!(matches!(
        &requests[1].messages[2],
        Message::ToolResult { is_error: false, .. }
    ));
}

#[test]
fn a_bad_call_becomes_an_error_result_the_model_can_react_to() {
    let models = ScriptedModels::new([
        Reply::Calls(vec![("echo".into(), json!({ "wrong": 1 })), ("nope".into(), json!({}))]),
        Reply::text("sorry"),
    ]);
    let harness = open(&models, vec![echo()]);
    let root = harness.root().unwrap();

    root.submit("go").unwrap().answer_blocking().unwrap();

    let transcript = kinds(&root);
    assert!(
        transcript
            .iter()
            .any(|kind| kind.starts_with("error:Invalid arguments"))
    );
    assert!(transcript.contains(&"error:Unknown tool: nope".to_owned()));
}

#[test]
fn retries_transient_model_errors_and_gives_up_on_others() {
    let models = ScriptedModels::new([
        Reply::Error(Some(503), "overloaded".into()),
        Reply::Error(None, "connection reset".into()),
        Reply::text("ok"),
        Reply::Error(Some(401), "bad key".into()),
    ]);
    let harness = open(&models, vec![]);
    let root = harness.root().unwrap();

    assert_eq!(root.submit("one").unwrap().answer_blocking().unwrap().text(), "ok");
    assert_eq!(models.requests().len(), 3);

    let failed = root.submit("two").unwrap().answer_blocking().unwrap_err();
    assert!(
        matches!(&failed, Error::Unanswered(failure) if failure.reason == Reason::ModelError && failure.detail.as_deref() == Some("bad key"))
    );
    assert_eq!(models.requests().len(), 4, "a rejected request is not retried");
    assert_eq!(root.snapshot().unwrap().failure.unwrap().reason, Reason::ModelError);
    assert_eq!(kinds(&root), ["user:one", "assistant:ok", "user:two"]);
}

#[test]
fn the_same_request_id_submits_once() {
    let models = ScriptedModels::new([Reply::text("hi")]);
    let harness = open(&models, vec![]);
    let root = harness.root().unwrap();

    let first = root.submit(Input::text("Hello").request_id("greeting-1")).unwrap();
    first.answer_blocking().unwrap();
    let again = root.submit(Input::text("Hello").request_id("greeting-1")).unwrap();

    assert_eq!(again.id(), first.id());
    assert_eq!(again.answer_blocking().unwrap().text(), "hi");
    assert_eq!(models.requests().len(), 1);
}

#[test]
fn steers_join_the_run_and_follow_ups_start_the_next_one() {
    let (started, starts) = mpsc::channel();
    let gate = Arc::new(tokio::sync::Notify::new());
    let wait = {
        let gate = gate.clone();
        tool("wait", "Waits for the test", move |_: Echo, _| {
            let (started, gate) = (started.clone(), gate.clone());
            async move {
                let _ = started.send(());
                gate.notified().await;
                Ok(ToolOutput::text("waited"))
            }
        })
    };
    let models = ScriptedModels::new([
        Reply::call("wait", json!({ "text": "" })),
        Reply::text("first"),
        Reply::text("second"),
    ]);
    let harness = open(&models, vec![wait]);
    let root = harness.root().unwrap();

    let first = root.submit("start").unwrap();
    starts.recv_timeout(Duration::from_secs(5)).unwrap();
    let follow_up = root.submit("afterwards").unwrap();
    let steer = root.submit(Input::text("use pnpm").steer()).unwrap();
    assert_eq!(root.snapshot().unwrap().inbox.len(), 2);
    gate.notify_one();

    assert_eq!(steer.answer_blocking().unwrap().text(), "first");
    assert_eq!(first.answer_blocking().unwrap().text(), "first");
    assert_eq!(follow_up.answer_blocking().unwrap().text(), "second");
    assert_eq!(
        kinds(&root),
        [
            "user:start",
            "calls",
            "result:waited",
            "user:use pnpm",
            "assistant:first",
            "user:afterwards",
            "assistant:second"
        ]
    );
}

#[test]
fn abort_ends_the_run_and_leaves_the_conversation_usable() {
    let (started, starts) = mpsc::channel();
    let models = ScriptedModels::new([Reply::call("stuck", json!({ "text": "" })), Reply::text("fresh")]);
    let harness = open(&models, vec![stuck(started, Arc::new(AtomicU32::new(u32::MAX)))]);
    let root = harness.root().unwrap();

    let running = root.submit("go").unwrap();
    starts.recv_timeout(Duration::from_secs(5)).unwrap();
    let queued = root.submit("later").unwrap();
    root.abort_blocking().unwrap();

    for submission in [&running, &queued] {
        assert!(
            matches!(submission.wait_blocking().unwrap(), Outcome::Unanswered(failure) if failure.reason == Reason::Aborted)
        );
    }
    assert_eq!(kinds(&root), ["user:go", "calls", "error:The tool call was aborted."]);
    assert_eq!(root.submit("again").unwrap().answer_blocking().unwrap().text(), "fresh");
    let last_request = models.requests().pop().unwrap();
    assert_eq!(
        last_request.messages.len(),
        4,
        "the aborted call still has its result, so the context is valid"
    );
}

#[test]
fn a_turn_limit_ends_a_run_that_keeps_calling_tools() {
    let call = || Reply::call("echo", json!({ "text": "x" }));
    let models = ScriptedModels::new([call(), call(), call()]);
    let harness = open(&models, vec![echo()]);
    let root = harness
        .create_conversation(AgentConfig {
            max_turns: Some(2),
            ..AgentConfig::default()
        })
        .unwrap();

    let failed = root.submit("loop").unwrap().answer_blocking().unwrap_err();

    assert!(matches!(failed, Error::Unanswered(failure) if failure.reason == Reason::TurnLimit));
    assert_eq!(models.requests().len(), 2);
}

#[test]
fn hooks_block_calls_and_rewrite_requests() {
    struct Guard;
    impl Hook for Guard {
        fn before_request(&self, _conversation: &String, context: &mut ri_ai::Context) {
            context.system_prompt = Some("guarded".into());
        }
        fn before_tool(&self, call: &ToolCall) -> BeforeTool {
            if call.arguments["text"] == "secret" {
                BeforeTool::Block("Needs approval".into())
            } else {
                BeforeTool::Continue
            }
        }
        fn after_tool(&self, _call: &ToolCall, output: ToolOutput) -> ToolOutput {
            if output.is_error {
                output
            } else {
                ToolOutput::text("redacted")
            }
        }
    }
    let models = ScriptedModels::new([
        Reply::Calls(vec![
            ("echo".into(), json!({ "text": "secret" })),
            ("echo".into(), json!({ "text": "fine" })),
        ]),
        Reply::text("ok"),
    ]);
    let options = Options::new(models.clone()).tool(echo()).hook(Guard);
    let harness = Harness::open(MemoryStorage::new(), options).unwrap();
    let root = harness.root().unwrap();

    root.submit("go").unwrap().answer_blocking().unwrap();

    let transcript = kinds(&root);
    assert!(
        transcript.contains(&"error:Needs approval".to_owned()) && transcript.contains(&"result:redacted".to_owned())
    );
    assert_eq!(models.requests()[0].system_prompt.as_deref(), Some("guarded"));
}

#[test]
fn reset_starts_a_new_context_and_keeps_the_history() {
    let models = ScriptedModels::new([Reply::text("one"), Reply::text("two")]);
    let harness = open(&models, vec![]);
    let root = harness.root().unwrap();
    root.submit("first").unwrap().answer_blocking().unwrap();

    root.reset_blocking(Some("We were testing resets.".into())).unwrap();
    root.submit("second").unwrap().answer_blocking().unwrap();

    assert_eq!(kinds(&root).len(), 5);
    let request = models.requests().pop().unwrap();
    assert_eq!(
        request.messages,
        vec![Message::user("We were testing resets."), Message::user("second")]
    );
}

#[test]
fn watchers_see_streamed_text_and_then_the_stored_events() {
    let models = ScriptedModels::new([Reply::text("Paris")]);
    let harness = open(&models, vec![]);
    let root = harness.root().unwrap();
    let (before, mut updates) = root.watch().unwrap();
    assert!(before.entries.is_empty());

    root.submit("Capital?").unwrap().answer_blocking().unwrap();
    root.wait_idle_blocking().unwrap();

    let seen: Vec<Update> = std::iter::from_fn(|| updates.try_next()).collect();
    let partial = seen.iter().position(|update| matches!(update, Update::Partial { event: AssistantEvent::TextDelta { delta, .. }, .. } if delta == "Paris"));
    let stored = seen.iter().position(|update| matches!(update, Update::Event(Event::Entry { entry, .. }) if matches!(entry.kind, EntryKind::Assistant { .. })));
    assert!(partial.unwrap() < stored.unwrap());
    assert!(matches!(
        seen.last(),
        Some(Update::Event(Event::RunEnded { failure: None, .. }))
    ));
}

#[test]
fn application_values_and_records_are_stored_with_the_conversation() {
    let storage = MemoryStorage::new();
    let models = ScriptedModels::new([Reply::text("ok")]);
    {
        let harness = Harness::open(storage.clone(), Options::new(models.clone())).unwrap();
        let root = harness.root().unwrap();
        root.set_value("draft", Some(json!({ "text": "hello" }))).unwrap();
        root.append("app.note", json!("user opened a file")).unwrap();
        root.submit("hi").unwrap().answer_blocking().unwrap();
    }
    let harness = Harness::open(storage, Options::new(models.clone())).unwrap();
    let root = harness.root().unwrap();

    assert_eq!(root.value("draft").unwrap(), Some(json!({ "text": "hello" })));
    assert_eq!(kinds(&root), ["custom:app.note", "user:hi", "assistant:ok"]);
    assert_eq!(
        models.requests()[0].messages,
        vec![Message::user("hi")],
        "the model does not see application records"
    );
}

#[test]
fn a_tool_call_cut_off_by_a_crash_gets_an_interrupted_result() {
    let directory = tempfile::tempdir().unwrap();
    let (started, starts) = mpsc::channel();
    let models = ScriptedModels::new([Reply::call("stuck", json!({ "text": "" })), Reply::text("recovered")]);
    let options = || {
        Options::new(models.clone())
            .tool(stuck(started.clone(), Arc::new(AtomicU32::new(u32::MAX))))
            .settings(fast())
    };

    let harness = Harness::open(JsonlStorage::open(directory.path()).unwrap(), options()).unwrap();
    let submission = harness
        .root()
        .unwrap()
        .submit(Input::text("go").request_id("r1"))
        .unwrap();
    starts.recv_timeout(Duration::from_secs(5)).unwrap();
    harness.abandon();

    let harness = Harness::open(JsonlStorage::open(directory.path()).unwrap(), options()).unwrap();
    let root = harness.root().unwrap();
    let again = root.submit(Input::text("go").request_id("r1")).unwrap();

    assert_eq!(again.id(), submission.id());
    assert_eq!(again.answer_blocking().unwrap().text(), "recovered");
    let transcript = kinds(&root);
    assert_eq!(transcript.len(), 4);
    assert!(transcript[2].starts_with("error:The tool call was interrupted before it finished"));
    assert!(
        starts.try_recv().is_err(),
        "a tool that is not replay-safe never runs twice"
    );
}

#[test]
fn a_replay_safe_tool_call_runs_again_after_a_crash() {
    let directory = tempfile::tempdir().unwrap();
    let (started, starts) = mpsc::channel();
    let models = ScriptedModels::new([Reply::call("stuck", json!({ "text": "" })), Reply::text("recovered")]);
    // The first run hangs; the rerun after the crash finishes.
    let hangs = Arc::new(AtomicU32::new(1));
    let tool = replay_safe(stuck(started, hangs));
    let options = || Options::new(models.clone()).tool(tool.clone()).settings(fast());

    let harness = Harness::open(JsonlStorage::open(directory.path()).unwrap(), options()).unwrap();
    harness.root().unwrap().submit("go").unwrap();
    starts.recv_timeout(Duration::from_secs(5)).unwrap();
    harness.abandon();

    let harness = Harness::open(JsonlStorage::open(directory.path()).unwrap(), options()).unwrap();
    let root = harness.root().unwrap();
    root.wait_idle_blocking().unwrap();

    assert_eq!(
        kinds(&root),
        ["user:go", "calls", "result:finished", "assistant:recovered"]
    );
}

#[test]
fn a_model_request_that_keeps_crashing_the_process_ends_the_run() {
    let directory = tempfile::tempdir().unwrap();
    let models = ScriptedModels::new(std::iter::repeat_n(Reply::Hang, 8));
    let settings = Settings {
        max_retries: 2,
        resume: true,
        ..fast()
    };
    let options = || Options::new(models.clone()).settings(settings.clone());
    let open = || Harness::open(JsonlStorage::open(directory.path()).unwrap(), options()).unwrap();

    let harness = open();
    let submission = harness.root().unwrap().submit("go").unwrap();
    let submission_id = submission.id().to_owned();
    wait_until("the first request is sent", || models.requests().len() == 1);

    // An orderly close stops the request without counting it as a failed attempt.
    harness.close();
    let harness = open();
    wait_until("the request is sent again", || models.requests().len() == 2);
    harness.abandon();

    let harness = open();
    wait_until("the retry after the first crash", || models.requests().len() == 3);
    harness.abandon();
    let harness = open();
    wait_until("the retry after the second crash", || models.requests().len() == 4);
    harness.abandon();

    let harness = open();
    let root = harness.root().unwrap();
    let outcome = root.submission(&submission_id).wait_blocking().unwrap();
    assert!(matches!(outcome, Outcome::Unanswered(failure) if failure.reason == Reason::Interrupted));
    assert_eq!(models.requests().len(), 4, "no retry was left");
    assert_eq!(kinds(&root), ["user:go"]);
}

#[test]
fn a_missing_model_fails_the_run_without_a_request() {
    let models = ScriptedModels::new([Reply::text("unused")]);
    let harness = open(&models, vec![]);
    let conversation = harness
        .create_conversation(AgentConfig {
            model: Some("missing/model".into()),
            ..AgentConfig::default()
        })
        .unwrap();

    let failed = conversation.submit("hi").unwrap().answer_blocking().unwrap_err();

    assert!(matches!(failed, Error::Unanswered(failure) if failure.reason == Reason::NoModel));
    assert!(models.requests().is_empty());
}

#[test]
fn images_reach_the_model_as_content() {
    let models = ScriptedModels::new([Reply::text("a cat")]);
    let harness = open(&models, vec![]);
    let input = Input {
        content: vec![
            Content::text("What is this?"),
            Content::Image {
                data: "AAAA".into(),
                mime_type: "image/png".into(),
            },
        ],
        ..Input::default()
    };

    harness
        .root()
        .unwrap()
        .submit(input.clone())
        .unwrap()
        .answer_blocking()
        .unwrap();

    assert_eq!(
        models.requests()[0].messages,
        vec![Message::User { content: input.content }]
    );
}

#[test]
fn manual_compaction_replaces_older_turns_with_a_summary() {
    let models = ScriptedModels::new([
        Reply::text("one"),
        Reply::text("two"),
        Reply::text("SUMMARY"),
        Reply::text("three"),
    ]);
    let settings = Settings {
        compaction: ri_durable::Compaction {
            keep_recent_tokens: 0,
            ..Default::default()
        },
        ..fast()
    };
    let harness = Harness::open(MemoryStorage::new(), Options::new(models.clone()).settings(settings)).unwrap();
    let root = harness.root().unwrap();
    root.submit("first").unwrap().answer_blocking().unwrap();
    assert!(
        !root.compact_blocking(None).unwrap(),
        "a single turn has nothing older to summarize"
    );
    root.submit("second").unwrap().answer_blocking().unwrap();

    assert!(root.compact_blocking(Some("the numbers")).unwrap());
    root.submit("third").unwrap().answer_blocking().unwrap();

    let requests = models.requests();
    let summary_request = ri_ai::joined_text(
        match &requests[2].messages[0] {
            Message::User { content } => content,
            _ => panic!(),
        },
        "",
    );
    assert!(summary_request.contains("User: first") && summary_request.contains("Assistant: one"));
    assert!(
        summary_request.contains("the numbers") && !summary_request.contains("User: second"),
        "the newest turn stays verbatim"
    );
    let texts: Vec<String> = requests[3]
        .messages
        .iter()
        .map(|message| match message {
            Message::User { content } => ri_ai::joined_text(content, ""),
            Message::Assistant(assistant) => assistant.text(),
            _ => String::new(),
        })
        .collect();
    assert!(texts[0].ends_with("SUMMARY"));
    assert_eq!(&texts[1..], ["second", "two", "third"]);
    assert_eq!(kinds(&root).len(), 7, "every entry is still stored");
}

#[test]
fn a_context_that_no_longer_fits_is_compacted_before_the_request() {
    let models =
        ScriptedModels::new([Reply::text("one"), Reply::text("SUMMARY"), Reply::text("two")]).context_window(30);
    let settings = Settings {
        compaction: ri_durable::Compaction {
            reserve_tokens: 0,
            keep_recent_tokens: 1000,
            ..Default::default()
        },
        ..fast()
    };
    let harness = Harness::open(MemoryStorage::new(), Options::new(models.clone()).settings(settings)).unwrap();
    let root = harness.root().unwrap();
    root.submit("first").unwrap().answer_blocking().unwrap();

    root.submit("x".repeat(100)).unwrap().answer_blocking().unwrap();

    assert!(kinds(&root).contains(&"compaction".to_owned()));
    let last = models.requests().pop().unwrap();
    assert_eq!(last.messages.len(), 2, "the summary and the new input");
}

#[test]
fn a_context_overflow_error_compacts_and_sends_the_request_again() {
    let overflow = Reply::Error(Some(400), "This model's maximum context length is 8192 tokens".into());
    let models = ScriptedModels::new([Reply::text("one"), overflow, Reply::text("SUMMARY"), Reply::text("two")]);
    let harness = open(&models, vec![]);
    let root = harness.root().unwrap();
    root.submit("first").unwrap().answer_blocking().unwrap();

    assert_eq!(root.submit("second").unwrap().answer_blocking().unwrap().text(), "two");

    assert_eq!(
        kinds(&root),
        [
            "user:first",
            "assistant:one",
            "user:second",
            "compaction",
            "assistant:two"
        ]
    );
}

#[test]
fn a_subagent_answers_in_a_child_conversation_of_its_own() {
    let models = ScriptedModels::new([
        Reply::call("subagent", json!({ "task": "Count the files" })),
        Reply::call("echo", json!({ "text": "from the child" })),
        Reply::text("42 files"),
        Reply::text("There are 42 files."),
    ]);
    let subagent = ri_durable::subagent_tool(ri_durable::SubagentOptions {
        instructions: Some("You are a helper.".into()),
        ..Default::default()
    });
    let harness = open(&models, vec![echo(), subagent]);
    let root = harness.root().unwrap();

    assert_eq!(
        root.submit("How many files?")
            .unwrap()
            .answer_blocking()
            .unwrap()
            .text(),
        "There are 42 files."
    );

    assert_eq!(
        kinds(&root),
        [
            "user:How many files?",
            "calls",
            "result:42 files",
            "assistant:There are 42 files."
        ]
    );
    let child = harness
        .conversations()
        .into_iter()
        .find(|conversation| conversation.id() != root.id())
        .unwrap();
    let state = child.snapshot().unwrap();
    assert_eq!(state.parent.as_deref(), Some(root.id()));
    assert_eq!(
        state.agent.tools,
        Some(vec!["echo".to_owned()]),
        "children get no subagents of their own"
    );
    assert_eq!(
        kinds(&child),
        [
            "user:Count the files",
            "calls",
            "result:echo: from the child",
            "assistant:42 files"
        ]
    );
    let requests = models.requests();
    assert_eq!(requests[1].system_prompt.as_deref(), Some("You are a helper."));
    assert_eq!(
        requests[1].messages,
        vec![Message::user("Count the files")],
        "the child sees only its task"
    );
}

#[test]
fn aborting_the_parent_aborts_its_subagent() {
    let models = ScriptedModels::new([Reply::call("subagent", json!({ "task": "wait" })), Reply::Hang]);
    let harness = open(&models, vec![ri_durable::subagent_tool(Default::default())]);
    let root = harness.root().unwrap();
    root.submit("go").unwrap();
    wait_until("the child asks the model", || models.requests().len() == 2);

    root.abort_blocking().unwrap();

    let child = harness
        .conversations()
        .into_iter()
        .find(|conversation| conversation.id() != root.id())
        .unwrap();
    wait_until("the child is idle", || !child.snapshot().unwrap().is_busy());
    assert_eq!(child.snapshot().unwrap().failure.unwrap().reason, Reason::Aborted);
}
