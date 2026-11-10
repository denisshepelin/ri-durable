//! Runs one script in a fresh QuickJS VM on a thread of its own. The script reaches the host only through tool
//! calls, which the host answers one at a time.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc as blocking;
use std::time::{Duration, Instant};

use rquickjs::promise::PromiseState;
use rquickjs::{CatchResultExt, Context, Function, Promise, Runtime};
use serde_json::{Value, json};
use tokio::sync::{mpsc, oneshot};

const PRELUDE: &str = include_str!("prelude.js");

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Limits {
    /// Wall time one run of the script may take, waits for tools included.
    pub timeout: Duration,
    pub memory_bytes: usize,
    /// Printed output beyond this many bytes is dropped.
    pub max_output_bytes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            timeout: Duration::from_secs(300),
            memory_bytes: 256 * 1024 * 1024,
            max_output_bytes: 1024 * 1024,
        }
    }
}

/// What makes two runs of one script take the same path.
#[derive(Debug, Clone, PartialEq)]
pub struct Determinism {
    pub seed: u32,
    /// The virtual clock's start, as Unix milliseconds. It advances one millisecond per reading.
    pub clock: u64,
}

/// A tool as scripts can look it up with `searchTools()` and `describeTool()`.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct CatalogEntry {
    pub name: String,
    pub description: String,
    pub declaration: String,
}

/// A tool call the script made. Ids count up from zero in the order the script makes its calls.
#[derive(Debug, Clone, PartialEq)]
pub struct HostCall {
    pub id: u32,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Finished {
    /// What the script printed, followed by its return value.
    pub output: String,
    /// Why the script failed, if it did.
    pub error: Option<String>,
}

/// A running script. Dropping it stops the script.
pub struct Script {
    /// The script's tool calls, as it makes them.
    pub calls: mpsc::UnboundedReceiver<HostCall>,
    /// Resolves when the script ends.
    pub finished: oneshot::Receiver<Finished>,
    results: blocking::Sender<(u32, Result<Value, String>)>,
    stop: Arc<AtomicBool>,
}

impl Script {
    /// Hands the script a call's result. Results are applied in the order they are delivered, each once the
    /// script has nothing else to do.
    pub fn deliver(&self, id: u32, result: Result<Value, String>) {
        let _ = self.results.send((id, result));
    }
}

impl Drop for Script {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
    }
}

/// Starts `code`, the body of an async function, on a new thread.
pub fn start(code: String, determinism: Determinism, catalog: Vec<CatalogEntry>, limits: Limits) -> Script {
    let (call_sender, calls) = mpsc::unbounded_channel();
    let (results, result_receiver) = blocking::channel();
    let (finish, finished) = oneshot::channel();
    let stop = Arc::new(AtomicBool::new(false));
    let stopped = stop.clone();
    let spawned = std::thread::Builder::new()
        .name("ri-codemode".to_owned())
        .spawn(move || {
            let output = Rc::new(RefCell::new(String::new()));
            let error = execute(
                &code,
                &determinism,
                &catalog,
                limits,
                &call_sender,
                &result_receiver,
                &stopped,
                &output,
            )
            .err();
            let output = output.borrow().clone();
            let _ = finish.send(Finished { output, error });
        });
    if spawned.is_err() {
        stop.store(true, Ordering::SeqCst);
    }
    Script {
        calls,
        finished,
        results,
        stop,
    }
}

#[allow(clippy::too_many_arguments)]
fn execute(
    code: &str,
    determinism: &Determinism,
    catalog: &[CatalogEntry],
    limits: Limits,
    calls: &mpsc::UnboundedSender<HostCall>,
    results: &blocking::Receiver<(u32, Result<Value, String>)>,
    stop: &Arc<AtomicBool>,
    output: &Rc<RefCell<String>>,
) -> Result<(), String> {
    let deadline = Instant::now() + limits.timeout;
    let timed_out = || format!("The script ran longer than its limit of {} s", limits.timeout.as_secs());
    let runtime = Runtime::new().map_err(|error| error.to_string())?;
    runtime.set_memory_limit(limits.memory_bytes);
    runtime.set_max_stack_size(512 * 1024);
    let interrupt = stop.clone();
    runtime.set_interrupt_handler(Some(Box::new(move || {
        interrupt.load(Ordering::SeqCst) || Instant::now() > deadline
    })));
    let context = Context::full(&runtime).map_err(|error| error.to_string())?;

    context.with(|ctx| -> Result<(), String> {
        let describe = |error: rquickjs::CaughtError<'_>| match error {
            rquickjs::CaughtError::Exception(exception) => {
                exception.message().unwrap_or_else(|| "The script threw an exception".to_owned())
            }
            rquickjs::CaughtError::Value(value) => format!("The script threw {value:?}"),
            rquickjs::CaughtError::Error(error) => error.to_string(),
        };
        let globals = ctx.globals();
        let issued = Rc::new(RefCell::new(Vec::<HostCall>::new()));
        let bridge = {
            let issued = issued.clone();
            Function::new(ctx.clone(), move |id: u32, name: String, arguments: String| {
                let arguments = serde_json::from_str(&arguments).unwrap_or(Value::Null);
                issued.borrow_mut().push(HostCall { id, name, arguments });
            })
        };
        let emit = {
            let (output, limit) = (output.clone(), limits.max_output_bytes);
            Function::new(ctx.clone(), move |line: String| {
                let mut output = output.borrow_mut();
                if output.len() + line.len() < limit {
                    output.push_str(&line);
                    output.push('\n');
                }
            })
        };
        let config = json!({ "seed": determinism.seed, "clock": determinism.clock, "catalog": catalog }).to_string();
        let installed = bridge
            .and_then(|bridge| globals.set("__bridge", bridge))
            .and(emit)
            .and_then(|emit| globals.set("__emit", emit))
            .and_then(|()| globals.set("__config", config));
        installed.map_err(|error| error.to_string())?;
        let settle: Function = ctx.eval(PRELUDE).catch(&ctx).map_err(describe)?;

        // The script's outcome always arrives as a resolved JSON string, so thrown values need no special path.
        let wrapped = format!(
            "(async () => {{\n{code}\n}})().then(\n  (value) => JSON.stringify({{ ok: true, value: value === undefined ? null : value }}),\n  (error) => JSON.stringify({{ ok: false, error: error instanceof Error ? `${{error.name}}: ${{error.message}}` : String(error) }}),\n)"
        );
        let promise: Promise = ctx.eval(wrapped).catch(&ctx).map_err(describe)?;
        let mut outstanding = 0usize;
        loop {
            while ctx.execute_pending_job() {}
            if stop.load(Ordering::SeqCst) {
                return Err("The script was stopped".to_owned());
            }
            if Instant::now() > deadline {
                return Err(timed_out());
            }
            if promise.state() != PromiseState::Pending {
                break;
            }
            for call in issued.borrow_mut().drain(..) {
                outstanding += 1;
                if calls.send(call).is_err() {
                    return Err("The script was stopped".to_owned());
                }
            }
            if outstanding == 0 {
                return Err("The script is waiting for something that can never happen (a promise no tool call settles)".to_owned());
            }
            // Short waits keep the stop flag and the deadline in view.
            let (id, result) = loop {
                match results.recv_timeout(Duration::from_millis(50)) {
                    Ok(result) => break result,
                    Err(blocking::RecvTimeoutError::Timeout) if stop.load(Ordering::SeqCst) => return Err("The script was stopped".to_owned()),
                    Err(blocking::RecvTimeoutError::Timeout) if Instant::now() > deadline => return Err(timed_out()),
                    Err(blocking::RecvTimeoutError::Timeout) => {}
                    Err(blocking::RecvTimeoutError::Disconnected) => return Err("The script was stopped".to_owned()),
                }
            };
            outstanding -= 1;
            let (ok, payload) = match result {
                Ok(value) => (true, value.to_string()),
                Err(message) => (false, Value::String(message).to_string()),
            };
            settle.call::<_, ()>((id, ok, payload)).catch(&ctx).map_err(describe)?;
        }
        let outcome: String = promise.result::<String>().ok_or("The script did not finish")?.catch(&ctx).map_err(describe)?;
        let outcome: Value = serde_json::from_str(&outcome).map_err(|error| error.to_string())?;
        if outcome["ok"] != true {
            return Err(outcome["error"].as_str().unwrap_or("The script failed").to_owned());
        }
        match &outcome["value"] {
            Value::Null => {}
            Value::String(text) => output.borrow_mut().push_str(text),
            value => output.borrow_mut().push_str(&serde_json::to_string_pretty(value).unwrap_or_default()),
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixed() -> Determinism {
        Determinism {
            seed: 42,
            clock: 1_700_000_000_000,
        }
    }

    /// Runs a script, answering every tool call with its name and arguments.
    fn run(code: &str, limits: Limits) -> (Finished, Vec<HostCall>) {
        let mut script = start(code.to_owned(), fixed(), Vec::new(), limits);
        let mut seen = Vec::new();
        loop {
            match script.finished.try_recv() {
                Ok(finished) => return (finished, seen),
                Err(oneshot::error::TryRecvError::Closed) => panic!("the script thread vanished"),
                Err(oneshot::error::TryRecvError::Empty) => {}
            }
            while let Ok(call) = script.calls.try_recv() {
                let result = if call.name == "fail" {
                    Err("it broke".to_owned())
                } else {
                    Ok(json!({ "tool": call.name, "args": call.arguments }))
                };
                script.deliver(call.id, result);
                seen.push(call);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn calls_tools_in_parallel_and_returns_a_value() {
        let code = "const [a, b] = await Promise.all([tools.read({ path: 'a' }), tools.read({ path: 'b' })]);
            print('got', a.args.path, b.args.path);
            class Box { #v = 1; get v() { return this.#v; } }
            return { sum: new Box().v + 1 };";
        let (finished, calls) = run(code, Limits::default());
        assert_eq!(finished.error, None);
        assert_eq!(finished.output, "got a b\n{\n  \"sum\": 2\n}");
        assert_eq!(calls.iter().map(|call| call.id).collect::<Vec<_>>(), [0, 1]);
    }

    #[test]
    fn a_failed_call_rejects_and_can_be_caught() {
        let (finished, _) = run(
            "try { await tools.fail(); } catch (error) { return error.message; }",
            Limits::default(),
        );
        assert_eq!((finished.output.as_str(), finished.error), ("it broke", None));
        let (finished, _) = run("print('before'); await tools.fail();", Limits::default());
        assert_eq!(
            (finished.output.as_str(), finished.error.as_deref()),
            ("before\n", Some("Error: it broke"))
        );
    }

    #[test]
    fn reports_syntax_errors_and_thrown_values() {
        assert!(run("const = 1;", Limits::default()).0.error.is_some());
        assert_eq!(
            run("throw new TypeError('nope')", Limits::default()).0.error.as_deref(),
            Some("TypeError: nope")
        );
        assert_eq!(
            run("throw 'plain'", Limits::default()).0.error.as_deref(),
            Some("plain")
        );
    }

    #[test]
    fn stops_endless_loops_memory_bombs_and_waits_that_cannot_end() {
        let quick = Limits {
            timeout: Duration::from_millis(100),
            ..Limits::default()
        };
        assert!(run("for (;;) {}", quick).0.error.is_some());
        let small = Limits {
            memory_bytes: 8 * 1024 * 1024,
            ..Limits::default()
        };
        assert!(
            run(
                "const all = []; for (;;) all.push('x'.repeat(1e6) + all.length);",
                small
            )
            .0
            .error
            .is_some()
        );
        assert!(
            run("await new Promise(() => {});", Limits::default())
                .0
                .error
                .unwrap()
                .contains("never happen")
        );
    }

    #[test]
    fn randomness_and_time_repeat_with_the_same_seed_and_clock() {
        let code = "return [Math.random(), Math.random(), Date.now(), new Date().getTime(), new Date(5).getTime()];";
        let first = run(code, Limits::default()).0.output;
        assert_eq!(first, run(code, Limits::default()).0.output);
        let values: Vec<f64> = serde_json::from_str(&first).unwrap();
        assert_eq!(&values[2..], [1_700_000_000_000.0, 1_700_000_000_001.0, 5.0]);
    }

    #[test]
    fn scripts_can_search_the_tool_catalog() {
        let catalog = vec![
            CatalogEntry {
                name: "read".into(),
                description: "Read a file".into(),
                declaration: "read(args: { path: string })".into(),
            },
            CatalogEntry {
                name: "bash".into(),
                description: "Run a command".into(),
                declaration: "bash(args: {})".into(),
            },
        ];
        let mut script = start(
            "return [searchTools('file'), describeTool('bash')];".into(),
            fixed(),
            catalog,
            Limits::default(),
        );
        let finished = loop {
            if let Ok(finished) = script.finished.try_recv() {
                break finished;
            }
            std::thread::sleep(Duration::from_millis(1));
        };
        let value: Value = serde_json::from_str(&finished.output).unwrap();
        assert_eq!(
            value,
            json!([[{ "name": "read", "description": "Read a file" }], "bash(args: {})"])
        );
    }
}
