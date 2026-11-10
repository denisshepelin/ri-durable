use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use ri_codemode::{CodemodeOptions, codemode_tool};
use ri_durable::testing::{Reply, ScriptedModels};
use ri_durable::{
    AgentConfig, EntryKind, Harness, JsonlStorage, MemoryStorage, Options, Settings, Tool, ToolOutput, replay_safe,
    tool,
};
use serde_json::json;

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct Text {
    text: String,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct Nothing {}

fn upper() -> Arc<dyn Tool> {
    tool("upper", "Uppercase the text", |args: Text, _| async move {
        Ok(ToolOutput::text(args.text.to_uppercase()))
    })
}

fn with_codemode(models: &ScriptedModels, tools: Vec<Arc<dyn Tool>>) -> Options {
    let specs: Vec<_> = tools.iter().map(|tool| tool.spec().clone()).collect();
    let settings = Settings {
        retry_delay: Duration::from_millis(1),
        ..Settings::default()
    };
    let defaults = AgentConfig {
        tools: Some(vec!["codemode".into()]),
        ..AgentConfig::default()
    };
    tools
        .into_iter()
        .fold(
            Options::new(models.clone()).settings(settings).defaults(defaults),
            Options::tool,
        )
        .tool(codemode_tool(&specs, CodemodeOptions::default()))
}

fn results(harness: &Harness) -> Vec<(String, bool)> {
    let entries = harness.root().unwrap().snapshot().unwrap().entries;
    entries
        .iter()
        .filter_map(|entry| match &entry.kind {
            EntryKind::ToolResult { content, is_error, .. } => Some((ri_ai::joined_text(content, ""), *is_error)),
            _ => None,
        })
        .collect()
}

#[test]
fn a_script_calls_tools_and_only_its_output_reaches_the_model() {
    let code = "const words = await Promise.all(['a', 'b'].map((text) => tools.upper({ text })));\nprint(words.join('-'));\nreturn { count: words.length };";
    let models = ScriptedModels::new([Reply::call("codemode", json!({ "code": code })), Reply::text("done")]);
    let harness = Harness::open(MemoryStorage::new(), with_codemode(&models, vec![upper()])).unwrap();

    harness.root().unwrap().submit("go").unwrap().answer_blocking().unwrap();

    assert_eq!(results(&harness), [("A-B\n{\n  \"count\": 2\n}".to_owned(), false)]);
    let offered = &models.requests()[0].tools;
    assert_eq!(offered.len(), 1, "the model is offered only codemode");
    assert!(
        offered[0]
            .description
            .contains("upper(args: {\n    text: string;\n  }): Promise<any>;"),
        "{}",
        offered[0].description
    );
    assert!(
        harness.root().unwrap().snapshot().unwrap().values.is_empty(),
        "the journal is gone once the result is stored"
    );
}

#[test]
fn scripts_get_structured_results_and_parsed_json_text_as_data() {
    let data = tool("data", "Returns data", |_: Nothing, _| async {
        Ok(ToolOutput::json(json!({ "n": 2 })))
    });
    let printed = tool("printed", "Prints JSON", |_: Nothing, _| async {
        Ok(ToolOutput::text("[1, 2, 3]"))
    });
    let code = "const [a, b, c] = [await tools.data({}), await tools.printed({}), await tools.upper({ text: '7' })];\nreturn [a.n + 1, b.length, typeof c];";
    let models = ScriptedModels::new([Reply::call("codemode", json!({ "code": code })), Reply::text("done")]);
    let harness = Harness::open(
        MemoryStorage::new(),
        with_codemode(&models, vec![upper(), data, printed]),
    )
    .unwrap();

    harness.root().unwrap().submit("go").unwrap().answer_blocking().unwrap();

    let value: serde_json::Value = serde_json::from_str(&results(&harness)[0].0).unwrap();
    assert_eq!(value, json!([3, 3, "string"]));
}

#[test]
fn a_failing_script_reports_its_error_calls_and_output() {
    let code = "print('step 1');\nawait tools.upper({ text: 'x' });\nawait tools.missing({});";
    let models = ScriptedModels::new([Reply::call("codemode", json!({ "code": code })), Reply::text("done")]);
    let harness = Harness::open(MemoryStorage::new(), with_codemode(&models, vec![upper()])).unwrap();

    harness.root().unwrap().submit("go").unwrap().answer_blocking().unwrap();

    let (text, is_error) = results(&harness).remove(0);
    assert!(is_error);
    assert!(
        text.starts_with("The script failed: Error: Unknown tool: missing"),
        "{text}"
    );
    assert!(text.contains("Tool calls made before the failure: upper, missing") && text.contains("step 1"));
}

#[test]
fn a_script_continues_after_a_crash_without_repeating_finished_calls() {
    let directory = tempfile::tempdir().unwrap();
    let counted = Arc::new(AtomicU32::new(0));
    let count = {
        let counted = counted.clone();
        tool("count", "Counts its calls", move |_: Nothing, _| {
            let counted = counted.clone();
            async move {
                Ok(ToolOutput::text(
                    (counted.fetch_add(1, Ordering::SeqCst) + 1).to_string(),
                ))
            }
        })
    };
    // Hangs the first time, as a call a crash cuts off; finishes when it runs again.
    let (started, starts) = mpsc::channel();
    let attempts = Arc::new(AtomicU32::new(0));
    let slow = replay_safe(tool("slow", "Slow", move |_: Nothing, _| {
        let (started, attempts) = (started.clone(), attempts.clone());
        async move {
            let _ = started.send(());
            if attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                std::future::pending::<()>().await;
            }
            Ok(ToolOutput::text("slow done"))
        }
    }));
    let code = "const first = await tools.count({});\nconst random = Math.random();\nconst waited = await tools.slow({});\nconst second = await tools.count({});\nreturn [first, waited, second, random === Math.random()];";
    let models = ScriptedModels::new([Reply::call("codemode", json!({ "code": code })), Reply::text("done")]);
    let open = || {
        Harness::open(
            JsonlStorage::open(directory.path()).unwrap(),
            with_codemode(&models, vec![count.clone(), slow.clone()]),
        )
        .unwrap()
    };

    let harness = open();
    harness.root().unwrap().submit("go").unwrap();
    starts.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(counted.load(Ordering::SeqCst), 1);
    harness.abandon();

    let harness = open();
    harness.root().unwrap().wait_idle_blocking().unwrap();

    let (text, is_error) = results(&harness).remove(0);
    assert!(!is_error, "{text}");
    let value: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(value, json!(["1", "slow done", "2", false]));
    assert_eq!(
        counted.load(Ordering::SeqCst),
        2,
        "the call that finished before the crash was not made again"
    );
}
