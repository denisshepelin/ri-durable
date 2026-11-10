# ri-durable

> Experimental. The API changes without notice.

LLM-powered operations for Rust applications: a durable agent harness in the spirit of
[pi-durable](https://github.com/earendil-works/pi/blob/main/packages/durable/README.md), plus access to the models
and logins you already have in [pi](https://github.com/earendil-works/pi). An app built on it never signs in itself.

| Crate | What it is |
|---|---|
| `ri-ai` | Provider-neutral messages, models and usage. Streams responses over the OpenAI Completions, OpenAI Responses and ChatGPT Codex protocols, and asks classifier models (TypeSafe System One, OpenAI Decisions) typed questions. |
| `ri-pi` | Reads pi's agent directory (`~/.pi/agent`): `auth.json`, `models-store.json`, `models.json`, `settings.json`. Refreshes OAuth tokens under the lock pi uses. |
| `ri-durable` | The harness: conversations, model turns and tool calls are committed before they are shown, and continue after a crash. |

## One request

```rust
let pi = ri_pi::Pi::load()?;
let model = pi.default_model()?;
let auth = pi.auth(model).await?;
let context = ri_ai::Context { messages: vec![ri_ai::Message::user("Hello")], ..Default::default() };
let answer = ri_ai::stream::complete(pi.client(), model, &context, &Default::default(), &auth).await;
println!("{}", answer.text());
```

## A durable conversation

```rust
use ri_durable::{Harness, JsonlStorage, Options, ToolOutput, data_dir, tool};

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct Add { a: i64, b: i64 }

let add = tool("add", "Add two numbers", |args: Add, _| async move {
    Ok(ToolOutput::text((args.a + args.b).to_string()))
});
let storage = JsonlStorage::open(data_dir("my-app").join("main"))?;
let harness = Harness::open(storage, Options::new(ri_pi::Pi::load()?).tool(add))?;
let answer = harness.root()?.submit("What is 17 + 25?")?.answer_blocking()?;
println!("{}", answer.text());
```

The harness runs on threads of its own, so the host needs no async runtime: every call that waits has an `async`
form and a `_blocking` form. `conversation.watch()` gives the committed state and then every update, including
streamed text, for a UI to render.

What survives a restart:

- **Inputs.** `submit()` stores the input before returning. A `request_id` makes it idempotent: submitting the same
  id again, also after a restart, returns the first submission.
- **Model requests.** A request the process died during counts as a failed attempt and is retried, so a request
  that keeps killing the process ends the run (`Reason::Interrupted`). An orderly `close()` is not a crash.
- **Tool calls.** A call's intent is stored before the tool runs. After a crash it runs again only when the tool is
  declared `replay_safe`; otherwise the model gets an `interrupted` error result.

While a run is working, further inputs wait in the inbox: a follow-up (the default) starts the next run, a steer
(`Input::text(..).steer()`) joins the running one after its current tool round. `abort()` stops a conversation,
`reset()` starts a new context, hooks can block or rewrite tool calls and requests.

## Storage

Sessions are append-only logs of events; a session's state is the fold of its events.

| Backend | Notes |
|---|---|
| `MemoryStorage` | Nothing is kept. |
| `JsonlStorage::open(dir)` | One JSON line per commit in `dir/log.jsonl`, one writer at a time. A torn last line is dropped on open. |
| `SqliteStorage::open(file)` | Behind the `sqlite` feature. WAL mode. |

`data_dir(app)` is `$RI_DURABLE_DIR/<app>`, otherwise `$XDG_DATA_HOME/ri-durable/<app>`, otherwise
`~/.local/share/ri-durable/<app>`.

## Examples

```bash
cargo run -p ri-pi --example ask -- "What is the capital of France?"
cargo run -p ri-pi --example ask -- --models
cargo run -p ri-pi --example tool_round
cargo run -p ri-pi --example classify -- "The change works perfectly, thanks."
cargo run -p ri-durable --example chat -- "What time is it?"
```

## Not here yet

MCP servers from pi's `mcp.json`, code mode (QuickJS), compaction and subagents are planned. The Anthropic and
Google wire protocols are not implemented, and only Codex logins can be refreshed; other expired OAuth logins ask
you to run pi once.

## License

MIT
