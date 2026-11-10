//! pi's syntax for configured secrets: a leading `!` runs the rest as a shell command and uses its output;
//! otherwise `$NAME` and `${NAME}` are replaced from the environment, `$$` is a literal `$` and `$!` a literal `!`.

use std::collections::BTreeMap;
use std::process::Command;

fn is_name_start(character: char) -> bool {
    character.is_ascii_alphabetic() || character == '_'
}

fn is_name_part(character: char) -> bool {
    character.is_ascii_alphanumeric() || character == '_'
}

fn is_name(name: &str) -> bool {
    name.starts_with(is_name_start) && name.chars().all(is_name_part)
}

/// Expands a template with `lookup`. Returns `None` when a referenced variable is missing.
pub fn expand(template: &str, lookup: impl Fn(&str) -> Option<String>) -> Option<String> {
    let mut resolved = String::new();
    let mut rest = template;
    while let Some(dollar) = rest.find('$') {
        resolved.push_str(&rest[..dollar]);
        let after = &rest[dollar + 1..];
        let braced = after
            .strip_prefix('{')
            .and_then(|inner| inner.find('}').map(|end| (&inner[..end], end + 2)));
        if let Some(escaped) = after.chars().next().filter(|next| matches!(next, '$' | '!')) {
            resolved.push(escaped);
            rest = &after[1..];
        } else if let Some((name, length)) = braced {
            if is_name(name) {
                resolved.push_str(&lookup(name)?);
            } else {
                resolved.push_str(&rest[dollar..dollar + 1 + length]);
            }
            rest = &after[length..];
        } else if after.starts_with(is_name_start) {
            let length = after.find(|character| !is_name_part(character)).unwrap_or(after.len());
            resolved.push_str(&lookup(&after[..length])?);
            rest = &after[length..];
        } else {
            resolved.push('$');
            rest = after;
        }
    }
    resolved.push_str(rest);
    Some(resolved)
}

/// Resolves a configured value. `env` is consulted before the process environment.
pub fn resolve(config: &str, env: &BTreeMap<String, String>) -> Option<String> {
    match config.strip_prefix('!') {
        Some(command) => run(command),
        None => expand(config, |name| {
            env.get(name)
                .cloned()
                .or_else(|| std::env::var(name).ok())
                .filter(|value| !value.is_empty())
        }),
    }
}

fn run(command: &str) -> Option<String> {
    let output = Command::new("sh").arg("-c").arg(command).output().ok()?;
    let value = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    (output.status.success() && !value.is_empty()).then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_references_and_escapes() {
        let lookup = |name: &str| (name == "KEY").then(|| "secret".to_owned());
        assert_eq!(expand("plain", lookup).as_deref(), Some("plain"));
        assert_eq!(expand("$KEY", lookup).as_deref(), Some("secret"));
        assert_eq!(expand("a-${KEY}-b$$c$!", lookup).as_deref(), Some("a-secret-b$c!"));
        assert_eq!(
            expand("cost: $5 ${not a name}", lookup).as_deref(),
            Some("cost: $5 ${not a name}")
        );
        assert_eq!(expand("$MISSING", lookup), None);
    }

    #[test]
    fn runs_commands() {
        assert_eq!(resolve("!printf ' abc \n'", &BTreeMap::new()).as_deref(), Some("abc"));
        assert_eq!(resolve("!exit 1", &BTreeMap::new()), None);
    }
}
