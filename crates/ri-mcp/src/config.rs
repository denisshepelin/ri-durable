//! pi's MCP server configuration: `<agent dir>/mcp.json`, with a project's `.pi/mcp.json` on top.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::{Map, Value};

use crate::Error;

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, PartialEq)]
pub enum Transport {
    /// A local process speaking JSON-RPC over its standard input and output.
    Stdio {
        command: String,
        args: Vec<String>,
        env: BTreeMap<String, String>,
        cwd: Option<PathBuf>,
    },
    /// A Streamable HTTP endpoint.
    Http {
        url: String,
        headers: BTreeMap<String, String>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ServerConfig {
    pub name: String,
    pub transport: Transport,
    pub enabled: bool,
    /// How long one request may take.
    pub timeout: Duration,
}

/// Reads the global configuration and, when given, the project's. A project entry replaces the global one of the
/// same name; one without a command or URL only overrides `enabled`.
pub fn load(agent_dir: &Path, project_dir: Option<&Path>) -> Result<Vec<ServerConfig>, Error> {
    let global = read(&agent_dir.join("mcp.json"))?;
    let project = match project_dir {
        Some(directory) => read(&directory.join(".pi").join("mcp.json"))?,
        None => Map::new(),
    };
    let merged = project.into_iter().fold(global, |mut servers, (name, entry)| {
        let is_override = ["command", "url", "type"].iter().all(|key| entry.get(key).is_none());
        match (is_override, servers.get_mut(&name)) {
            (true, Some(Value::Object(existing))) => {
                if let Some(enabled) = entry.get("enabled") {
                    existing.insert("enabled".to_owned(), enabled.clone());
                }
            }
            (true, _) => {}
            (false, _) => {
                servers.insert(name, entry);
            }
        }
        servers
    });
    merged.iter().map(|(name, entry)| parse(name, entry)).collect()
}

fn read(path: &Path) -> Result<Map<String, Value>, Error> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Map::new()),
        Err(error) => return Err(Error::Config(format!("cannot read {}: {error}", path.display()))),
    };
    let parsed: Value = serde_json::from_str(text.trim_start_matches('\u{feff}'))
        .map_err(|error| Error::Config(format!("cannot parse {}: {error}", path.display())))?;
    Ok(parsed["mcpServers"].as_object().cloned().unwrap_or_default())
}

fn expand_home(value: &str) -> String {
    let home = || std::env::var("HOME").unwrap_or_default();
    match value.strip_prefix("~/") {
        Some(rest) => format!("{}/{rest}", home()),
        None if value == "~" => home(),
        None => value.to_owned(),
    }
}

fn strings(value: &Value) -> BTreeMap<String, String> {
    let entries = value.as_object().into_iter().flatten();
    entries
        .filter_map(|(key, value)| Some((key.clone(), value.as_str()?.to_owned())))
        .collect()
}

fn parse(name: &str, entry: &Value) -> Result<ServerConfig, Error> {
    let invalid = |problem: &str| Error::Config(format!("MCP server `{name}`: {problem}"));
    if !name
        .chars()
        .all(|character| character.is_ascii_alphanumeric() || character == '_' || character == '-')
        || name.is_empty()
    {
        return Err(invalid("names may only contain letters, digits, `_` and `-`"));
    }
    let transport = match (entry["type"].as_str(), entry["command"].as_str(), entry["url"].as_str()) {
        (Some("sse"), _, _) => {
            return Err(invalid(
                "the legacy SSE transport is not supported; use Streamable HTTP",
            ));
        }
        (None | Some("stdio"), Some(command), _) => Transport::Stdio {
            command: expand_home(command),
            args: entry["args"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(expand_home)
                .collect(),
            env: strings(&entry["env"]),
            cwd: entry["cwd"].as_str().map(|cwd| PathBuf::from(expand_home(cwd))),
        },
        (None | Some("http" | "streamable-http"), None, Some(url)) => Transport::Http {
            url: url.to_owned(),
            headers: strings(&entry["headers"]),
        },
        _ => return Err(invalid("needs either a `command` or a `url`")),
    };
    Ok(ServerConfig {
        name: name.to_owned(),
        transport,
        enabled: entry["enabled"].as_bool().unwrap_or(true),
        timeout: entry["timeout"]
            .as_f64()
            .map_or(DEFAULT_TIMEOUT, Duration::from_secs_f64),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn merges_the_project_file_over_the_global_one() {
        let directory = tempfile::tempdir().unwrap();
        let project = directory.path().join("project");
        std::fs::create_dir_all(project.join(".pi")).unwrap();
        let global = json!({ "mcpServers": {
            "browse": { "command": "~/bin/browse", "args": ["mcp"], "env": { "A": "1" } },
            "docs": { "url": "https://example.com/mcp", "headers": { "X-Key": "$KEY" }, "timeout": 5 },
            "old": { "command": "old" },
        } });
        let local = json!({ "mcpServers": { "old": { "enabled": false }, "docs": { "command": "docs-local" }, "ghost": { "enabled": true } } });
        std::fs::write(directory.path().join("mcp.json"), global.to_string()).unwrap();
        std::fs::write(project.join(".pi").join("mcp.json"), local.to_string()).unwrap();

        let servers = load(directory.path(), Some(&project)).unwrap();
        let by_name = |name: &str| servers.iter().find(|server| server.name == name).unwrap();

        assert_eq!(servers.len(), 3);
        assert!(
            matches!(&by_name("browse").transport, Transport::Stdio { command, args, .. } if command.ends_with("/bin/browse") && !command.starts_with('~') && args == &["mcp"])
        );
        assert!(matches!(&by_name("docs").transport, Transport::Stdio { command, .. } if command == "docs-local"));
        assert!(!by_name("old").enabled);
        assert_eq!(load(directory.path(), None).unwrap()[1].timeout, Duration::from_secs(5));
    }

    #[test]
    fn rejects_the_legacy_sse_transport() {
        let error = parse("x", &json!({ "type": "sse", "url": "https://example.com/sse" })).unwrap_err();
        assert!(error.to_string().contains("SSE"));
    }
}
