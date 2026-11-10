//! Renders tools as TypeScript declarations, the form models read function signatures in most reliably.

use ri_durable::ToolSpec;
use serde_json::Value;

const MAX_DEPTH: usize = 6;
const MAX_PROPERTY_COMMENT: usize = 100;

fn literal(value: &Value) -> String {
    value.to_string()
}

fn union(types: impl IntoIterator<Item = String>) -> String {
    let mut distinct: Vec<String> = Vec::new();
    for kind in types {
        if !distinct.contains(&kind) {
            distinct.push(kind);
        }
    }
    if distinct.is_empty() {
        "unknown".to_owned()
    } else {
        distinct.join(" | ")
    }
}

fn is_identifier(name: &str) -> bool {
    let mut characters = name.chars();
    characters
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic() || first == '_' || first == '$')
        && characters.all(|character| character.is_ascii_alphanumeric() || character == '_' || character == '$')
}

/// The TypeScript type a JSON Schema describes. What it cannot express becomes `unknown`.
pub fn schema_type(schema: &Value, depth: usize) -> String {
    if depth > MAX_DEPTH {
        return "unknown".to_owned();
    }
    if let Some(values) = schema["enum"].as_array() {
        return union(values.iter().map(literal));
    }
    if let Some(value) = schema.get("const") {
        return literal(value);
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(variants) = schema[key].as_array() {
            return union(variants.iter().map(|variant| schema_type(variant, depth + 1)));
        }
    }
    let named = |kind: &str| match kind {
        "string" => "string".to_owned(),
        "number" | "integer" => "number".to_owned(),
        "boolean" => "boolean".to_owned(),
        "null" => "null".to_owned(),
        "array" => {
            let item = schema_type(&schema["items"], depth + 1);
            if item.contains(' ') {
                format!("Array<{item}>")
            } else {
                format!("{item}[]")
            }
        }
        "object" => object_type(schema, depth),
        _ => "unknown".to_owned(),
    };
    match &schema["type"] {
        Value::String(kind) => named(kind),
        Value::Array(kinds) => union(kinds.iter().filter_map(Value::as_str).map(named)),
        _ if schema["properties"].is_object() => object_type(schema, depth),
        _ => "unknown".to_owned(),
    }
}

fn object_type(schema: &Value, depth: usize) -> String {
    let Some(properties) = schema["properties"]
        .as_object()
        .filter(|properties| !properties.is_empty())
    else {
        return match &schema["additionalProperties"] {
            Value::Object(_) => format!(
                "Record<string, {}>",
                schema_type(&schema["additionalProperties"], depth + 1)
            ),
            _ => "Record<string, unknown>".to_owned(),
        };
    };
    let required: Vec<&str> = schema["required"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    let indent = "  ".repeat(depth + 2);
    let fields: Vec<String> = properties
        .iter()
        .map(|(name, property)| {
            let comment = property["description"]
                .as_str()
                .map(|text| text.split_whitespace().collect::<Vec<_>>().join(" "))
                .filter(|text| !text.is_empty() && text.len() <= MAX_PROPERTY_COMMENT)
                .map(|text| format!("{indent}/** {} */\n", text.replace("*/", "* /")))
                .unwrap_or_default();
            let key = if is_identifier(name) {
                name.clone()
            } else {
                literal(&Value::String(name.clone()))
            };
            let optional = if required.contains(&name.as_str()) { "" } else { "?" };
            format!(
                "{comment}{indent}{key}{optional}: {};",
                schema_type(property, depth + 1)
            )
        })
        .collect();
    format!("{{\n{}\n{}}}", fields.join("\n"), "  ".repeat(depth + 1))
}

/// One tool as a method of the `tools` object.
pub fn declaration(spec: &ToolSpec) -> String {
    let description = spec.description.trim().replace("*/", "* /");
    let comment = if description.is_empty() {
        String::new()
    } else {
        let lines: Vec<String> = description
            .lines()
            .map(|line| format!("   * {}", line.trim_end()))
            .collect();
        format!("  /**\n{}\n   */\n", lines.join("\n"))
    };
    format!(
        "{comment}  {}(args: {}): Promise<any>;",
        spec.name,
        schema_type(&spec.parameters, 0)
    )
}

/// The declarations of as many tools as fit in `budget` bytes, and the names of the rest.
pub fn declarations(specs: &[ToolSpec], budget: usize) -> (String, Vec<String>) {
    let (shown, hidden, _) = specs.iter().fold(
        (Vec::new(), Vec::new(), 0usize),
        |(mut shown, mut hidden, used), spec| {
            let rendered = declaration(spec);
            if used + rendered.len() <= budget {
                let used = used + rendered.len();
                shown.push(rendered);
                (shown, hidden, used)
            } else {
                hidden.push(spec.name.clone());
                (shown, hidden, used)
            }
        },
    );
    (format!("declare const tools: {{\n{}\n}};", shown.join("\n")), hidden)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ri_durable::Replay;
    use serde_json::json;

    fn spec(name: &str, parameters: Value) -> ToolSpec {
        ToolSpec {
            name: name.into(),
            description: "Search the docs".into(),
            parameters,
            replay: Replay::Never,
        }
    }

    #[test]
    fn renders_a_schema_as_a_typescript_type() {
        let schema = json!({ "type": "object", "required": ["query"], "properties": {
            "query": { "type": "string", "description": "What to look for" },
            "limit": { "type": ["integer", "null"] },
            "mode": { "enum": ["fast", "deep"] },
            "tags": { "type": "array", "items": { "type": "string" } },
            "filter-by": { "anyOf": [{ "type": "string" }, { "type": "object", "additionalProperties": { "type": "number" } }] },
        } });
        let rendered = declaration(&spec("search", schema));
        assert!(
            rendered.contains("/** What to look for */\n    query: string;"),
            "{rendered}"
        );
        assert!(rendered.contains("limit?: number | null;"));
        assert!(rendered.contains("mode?: \"fast\" | \"deep\";"));
        assert!(rendered.contains("tags?: string[];"));
        assert!(rendered.contains("\"filter-by\"?: string | Record<string, number>;"));
        assert!(rendered.starts_with("  /**\n   * Search the docs\n   */\n  search(args: {"));
    }

    #[test]
    fn leaves_out_what_does_not_fit_the_budget() {
        let specs = [
            spec("a", json!({ "type": "object" })),
            spec("b", json!({ "type": "object" })),
        ];
        let one = declaration(&specs[0]).len();
        let (text, hidden) = declarations(&specs, one);
        assert!(text.contains("a(args: Record<string, unknown>)") && !text.contains("b(args"));
        assert_eq!(hidden, ["b"]);
    }
}
