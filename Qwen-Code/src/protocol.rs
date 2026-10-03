//! Agent communication protocol. Thinking and Coder models talk to the agent exclusively through
//! JSON objects. Free text is never interpreted as a tool call or a shell command.
//!
//! Pipeline: model text -> `<think>` split -> JSON extraction -> schema validation -> typed message.

use crate::fileedit::EditOp;
use crate::schema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Thinking,
    Coder,
}

impl Role {
    pub fn name(&self) -> &'static str {
        match self {
            Role::Thinking => "thinking",
            Role::Coder => "coder",
        }
    }

    pub fn allowed_types(&self) -> &'static [&'static str] {
        match self {
            Role::Thinking => &["plan", "instruction", "tool_call", "complete"],
            Role::Coder => &["tool_call", "edit", "report", "complete"],
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Verification {
    #[serde(default)]
    pub build: Option<bool>,
    #[serde(default)]
    pub test: Option<bool>,
    #[serde(default)]
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CompleteStatus {
    #[default]
    Done,
    NeedsInput,
    Blocked,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentMessage {
    ToolCall {
        tool: String,
        #[serde(default)]
        arguments: Value,
    },
    Edit {
        path: String,
        operations: Vec<EditOp>,
        #[serde(default)]
        expected_hash: Option<String>,
    },
    Plan {
        steps: Vec<String>,
        #[serde(default)]
        notes: Option<String>,
    },
    Instruction {
        instruction: String,
        #[serde(default)]
        files: Vec<String>,
        #[serde(default)]
        acceptance: Vec<String>,
    },
    Report {
        summary: String,
        #[serde(default)]
        files_changed: Vec<String>,
        #[serde(default)]
        status: Option<String>,
    },
    Complete {
        summary: String,
        #[serde(default)]
        verification: Verification,
        #[serde(default)]
        status: CompleteStatus,
    },
}

#[derive(Debug, Clone)]
pub struct Parsed {
    pub message: AgentMessage,
    /// The validated JSON object exactly as the model produced it.
    pub raw: Value,
    pub reasoning: String,
    /// Further JSON objects in the same response that were ignored.
    pub extra_objects: usize,
}

#[derive(Debug, Clone, thiserror::Error)]
#[error("{message}")]
pub struct ProtocolError {
    pub message: String,
}

fn perr(m: impl Into<String>) -> ProtocolError {
    ProtocolError { message: m.into() }
}

// ---------------------------------------------------------------------------------------------
// Text handling
// ---------------------------------------------------------------------------------------------

/// Splits `<think>…</think>` reasoning from the answer. Returns `(reasoning, answer, unterminated)`.
pub fn split_reasoning(text: &str) -> (String, String, bool) {
    if let Some(end) = text.rfind("</think>") {
        let before = &text[..end];
        let reasoning = before.rsplit_once("<think>").map(|(_, r)| r).unwrap_or(before);
        let answer = &text[end + "</think>".len()..];
        return (reasoning.trim().to_string(), answer.trim().to_string(), false);
    }
    if let Some(start) = text.find("<think>") {
        return (text[start + 7..].trim().to_string(), String::new(), true);
    }
    (String::new(), text.trim().to_string(), false)
}

/// Escapes raw control characters inside JSON strings (a very common model mistake).
fn escape_control_chars_in_strings(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 16);
    let mut in_str = false;
    let mut esc = false;
    for c in s.chars() {
        if in_str {
            if esc {
                esc = false;
                out.push(c);
                continue;
            }
            match c {
                '\\' => {
                    esc = true;
                    out.push(c);
                }
                '"' => {
                    in_str = false;
                    out.push(c);
                }
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                _ => out.push(c),
            }
        } else {
            if c == '"' {
                in_str = true;
            }
            out.push(c);
        }
    }
    out
}

fn strip_trailing_commas(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut in_str = false;
    let mut esc = false;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if in_str {
            out.push(c);
            if esc {
                esc = false;
            } else if c == '\\' {
                esc = true;
            } else if c == '"' {
                in_str = false;
            }
        } else if c == '"' {
            in_str = true;
            out.push(c);
        } else if c == ',' {
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            if j < chars.len() && (chars[j] == '}' || chars[j] == ']') {
                // drop the trailing comma
            } else {
                out.push(c);
            }
        } else {
            out.push(c);
        }
        i += 1;
    }
    out
}

fn try_parse_object(candidate: &str) -> Option<Value> {
    let attempts = [candidate.to_string(), escape_control_chars_in_strings(candidate), strip_trailing_commas(&escape_control_chars_in_strings(candidate))];
    for a in attempts {
        if let Ok(v @ Value::Object(_)) = serde_json::from_str::<Value>(&a) {
            return Some(v);
        }
    }
    None
}

/// Finds all top-level JSON objects in `text` (string-aware brace matching).
pub fn extract_json_objects(text: &str) -> Vec<Value> {
    let bytes: Vec<char> = text.chars().collect();
    let mut found = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != '{' {
            i += 1;
            continue;
        }
        // scan for the matching brace
        let mut depth = 0usize;
        let mut in_str = false;
        let mut esc = false;
        let mut end = None;
        for (j, &c) in bytes.iter().enumerate().skip(i) {
            if in_str {
                if esc {
                    esc = false;
                } else if c == '\\' {
                    esc = true;
                } else if c == '"' {
                    in_str = false;
                }
                continue;
            }
            match c {
                '"' => in_str = true,
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(j);
                        break;
                    }
                }
                _ => {}
            }
        }
        match end {
            Some(e) => {
                let candidate: String = bytes[i..=e].iter().collect();
                match try_parse_object(&candidate) {
                    Some(v) => {
                        found.push(v);
                        i = e + 1;
                    }
                    None => i += 1, // skip this brace and look for an inner/next object
                }
            }
            None => break,
        }
    }
    found
}

/// `true` when `text` (after the reasoning block) already contains a complete JSON object.
/// Used by runtimes to stop generation early.
pub fn has_complete_json(text: &str) -> bool {
    let (_, answer, unterminated) = split_reasoning(text);
    if unterminated {
        return false;
    }
    !extract_json_objects(&answer).is_empty()
}

// ---------------------------------------------------------------------------------------------
// Schemas
// ---------------------------------------------------------------------------------------------

pub fn edit_op_schema() -> Value {
    json!({
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "type": {"const": "create"},
                    "content": {"type": "string"},
                    "overwrite": {"type": "boolean"}
                },
                "required": ["type", "content"],
                "additionalProperties": false
            },
            {
                "type": "object",
                "properties": {
                    "type": {"const": "replace"},
                    "old": {"type": "string", "minLength": 1},
                    "new": {"type": "string"},
                    "all": {"type": "boolean"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                    "expected_old": {"type": "string"}
                },
                "required": ["type", "new"],
                "additionalProperties": false
            },
            {
                "type": "object",
                "properties": {
                    "type": {"const": "insert"},
                    "content": {"type": "string"},
                    "line": {"type": "integer", "minimum": 1},
                    "before": {"type": "string", "minLength": 1},
                    "after": {"type": "string", "minLength": 1}
                },
                "required": ["type", "content"],
                "additionalProperties": false
            },
            {
                "type": "object",
                "properties": {
                    "type": {"const": "delete"},
                    "old": {"type": "string", "minLength": 1},
                    "all": {"type": "boolean"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                    "expected_old": {"type": "string"}
                },
                "required": ["type"],
                "additionalProperties": false
            },
            {
                "type": "object",
                "properties": {
                    "type": {"const": "append"},
                    "content": {"type": "string"}
                },
                "required": ["type", "content"],
                "additionalProperties": false
            }
        ]
    })
}

/// JSON schema of one message type.
pub fn message_schema(kind: &str) -> Option<Value> {
    Some(match kind {
        "tool_call" => json!({
            "type": "object",
            "properties": {
                "type": {"const": "tool_call"},
                "tool": {"type": "string", "minLength": 1},
                "arguments": {"type": "object"}
            },
            "required": ["type", "tool"],
            "additionalProperties": false
        }),
        "edit" => json!({
            "type": "object",
            "properties": {
                "type": {"const": "edit"},
                "path": {"type": "string", "minLength": 1},
                "operations": {"type": "array", "minItems": 1, "maxItems": 50, "items": edit_op_schema()},
                "expected_hash": {"type": "string"}
            },
            "required": ["type", "path", "operations"],
            "additionalProperties": false
        }),
        "plan" => json!({
            "type": "object",
            "properties": {
                "type": {"const": "plan"},
                "steps": {"type": "array", "minItems": 1, "maxItems": 40, "items": {"type": "string", "minLength": 1}},
                "notes": {"type": "string"}
            },
            "required": ["type", "steps"],
            "additionalProperties": false
        }),
        "instruction" => json!({
            "type": "object",
            "properties": {
                "type": {"const": "instruction"},
                "instruction": {"type": "string", "minLength": 1},
                "files": {"type": "array", "items": {"type": "string"}},
                "acceptance": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["type", "instruction"],
            "additionalProperties": false
        }),
        "report" => json!({
            "type": "object",
            "properties": {
                "type": {"const": "report"},
                "summary": {"type": "string", "minLength": 1},
                "files_changed": {"type": "array", "items": {"type": "string"}},
                "status": {"type": "string"}
            },
            "required": ["type", "summary"],
            "additionalProperties": false
        }),
        "complete" => json!({
            "type": "object",
            "properties": {
                "type": {"const": "complete"},
                "summary": {"type": "string", "minLength": 1},
                "verification": {
                    "type": "object",
                    "properties": {
                        "build": {"type": ["boolean", "null"]},
                        "test": {"type": ["boolean", "null"]},
                        "notes": {"type": "string"}
                    }
                },
                "status": {"enum": ["done", "needs_input", "blocked"]}
            },
            "required": ["type", "summary"],
            "additionalProperties": false
        }),
        _ => return None,
    })
}

/// Parses and validates one model response for `role`.
pub fn parse_response(text: &str, role: Role) -> Result<Parsed, ProtocolError> {
    let (reasoning, answer, unterminated) = split_reasoning(text);
    if unterminated {
        return Err(perr("The response ended while still inside <think>. Think briefly, then output exactly one JSON object after </think>."));
    }
    if answer.trim().is_empty() {
        return Err(perr("The response was empty. Output exactly one JSON object."));
    }
    let mut objects = extract_json_objects(&answer);
    if objects.is_empty() {
        let hint = if answer.contains('{') { "The JSON is malformed (check quotes, escapes and braces)." } else { "No JSON object was found." };
        return Err(perr(format!(
            "{hint} Plain text is never executed. Reply with exactly one JSON object such as {{\"type\": \"tool_call\", \"tool\": \"read_file\", \"arguments\": {{\"path\": \"src/main.rs\"}}}}."
        )));
    }
    let first = objects.remove(0);
    let extra = objects.len();

    let kind = first.get("type").and_then(|t| t.as_str()).ok_or_else(|| perr("The JSON object has no string field \"type\"."))?.to_string();
    let Some(schema) = message_schema(&kind) else {
        return Err(perr(format!("Unknown message type \"{kind}\". Allowed types for the {} model: {}.", role.name(), role.allowed_types().join(", "))));
    };
    if !role.allowed_types().contains(&kind.as_str()) {
        return Err(perr(format!(
            "Message type \"{kind}\" is not allowed for the {} model. Allowed types: {}.",
            role.name(),
            role.allowed_types().join(", ")
        )));
    }
    schema::validate(&first, &schema).map_err(|errs| perr(format!("The \"{kind}\" message does not match its schema:\n- {}", errs.join("\n- "))))?;
    let raw = first.clone();
    let mut message: AgentMessage = serde_json::from_value(first).map_err(|e| perr(format!("Cannot decode the \"{kind}\" message: {e}")))?;

    // The Coder reports back to the Thinking model; `complete` is accepted as an alias for `report`.
    if role == Role::Coder {
        if let AgentMessage::Complete { summary, .. } = &message {
            message = AgentMessage::Report { summary: summary.clone(), files_changed: Vec::new(), status: None };
        }
    }
    Ok(Parsed { message, raw, reasoning, extra_objects: extra })
}

/// Short human-readable summary of a message (for logs and the UI).
pub fn summarize(msg: &AgentMessage) -> String {
    match msg {
        AgentMessage::ToolCall { tool, arguments } => format!("{tool} {}", compact_json(arguments, 120)),
        AgentMessage::Edit { path, operations, .. } => format!("edit {path} ({} op)", operations.len()),
        AgentMessage::Plan { steps, .. } => format!("plan ({} steps)", steps.len()),
        AgentMessage::Instruction { instruction, .. } => format!("instruction: {}", instruction.chars().take(100).collect::<String>()),
        AgentMessage::Report { summary, .. } => format!("report: {}", summary.chars().take(100).collect::<String>()),
        AgentMessage::Complete { summary, .. } => format!("complete: {}", summary.chars().take(100).collect::<String>()),
    }
}

pub fn compact_json(v: &Value, max: usize) -> String {
    let s = v.to_string();
    if s.chars().count() > max {
        format!("{}…", s.chars().take(max).collect::<String>())
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tool_call_with_noise_and_fences() {
        let text = "Sure! Here you go:\n```json\n{\"type\":\"tool_call\",\"tool\":\"read_file\",\"arguments\":{\"path\":\"src/main.rs\"}}\n```\nDone.";
        let p = parse_response(text, Role::Coder).unwrap();
        match p.message {
            AgentMessage::ToolCall { tool, arguments } => {
                assert_eq!(tool, "read_file");
                assert_eq!(arguments["path"], "src/main.rs");
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn think_block_is_split_off() {
        let text = "<think>I should look at {the file} first</think>\n{\"type\":\"plan\",\"steps\":[\"a\",\"b\"]}";
        let p = parse_response(text, Role::Thinking).unwrap();
        assert!(p.reasoning.contains("look at"));
        assert!(matches!(p.message, AgentMessage::Plan { ref steps, .. } if steps.len() == 2));
    }

    #[test]
    fn unterminated_think_is_an_error() {
        let e = parse_response("<think>still thinking {\"type\":\"plan\"", Role::Thinking).unwrap_err();
        assert!(e.message.contains("inside <think>"));
    }

    #[test]
    fn plain_text_is_never_a_tool_call() {
        let e = parse_response("run: rm -rf /", Role::Coder).unwrap_err();
        assert!(e.message.contains("No JSON object"));
    }

    #[test]
    fn schema_violations_are_reported() {
        let e = parse_response("{\"type\":\"edit\",\"path\":\"a\",\"operations\":[{\"type\":\"replace\",\"old\":\"x\"}]}", Role::Coder).unwrap_err();
        assert!(e.message.contains("schema"), "{}", e.message);
        let e = parse_response("{\"type\":\"tool_call\"}", Role::Coder).unwrap_err();
        assert!(e.message.contains("missing required property \"tool\""));
    }

    #[test]
    fn role_restrictions() {
        let e = parse_response("{\"type\":\"edit\",\"path\":\"a\",\"operations\":[{\"type\":\"append\",\"content\":\"x\"}]}", Role::Thinking).unwrap_err();
        assert!(e.message.contains("not allowed for the thinking model"));
        let e = parse_response("{\"type\":\"plan\",\"steps\":[\"x\"]}", Role::Coder).unwrap_err();
        assert!(e.message.contains("not allowed for the coder model"));
        let e = parse_response("{\"type\":\"bogus\"}", Role::Coder).unwrap_err();
        assert!(e.message.contains("Unknown message type"));
    }

    #[test]
    fn coder_complete_becomes_report() {
        let p = parse_response("{\"type\":\"complete\",\"summary\":\"done\"}", Role::Coder).unwrap();
        assert!(matches!(p.message, AgentMessage::Report { .. }));
        let p = parse_response("{\"type\":\"complete\",\"summary\":\"done\",\"verification\":{\"build\":true,\"test\":true}}", Role::Thinking).unwrap();
        assert!(matches!(p.message, AgentMessage::Complete { ref verification, .. } if verification.build == Some(true)));
    }

    #[test]
    fn raw_newlines_and_trailing_commas_are_repaired() {
        let text = "{\"type\":\"edit\",\"path\":\"a.txt\",\"operations\":[{\"type\":\"append\",\"content\":\"line1\nline2\",}],}";
        let p = parse_response(text, Role::Coder).unwrap();
        match p.message {
            AgentMessage::Edit { operations, .. } => assert_eq!(operations[0], EditOp::Append { content: "line1\nline2".into() }),
            o => panic!("{o:?}"),
        }
    }

    #[test]
    fn extra_objects_are_counted() {
        let p = parse_response("{\"type\":\"plan\",\"steps\":[\"a\"]} {\"type\":\"plan\",\"steps\":[\"b\"]}", Role::Thinking).unwrap();
        assert_eq!(p.extra_objects, 1);
    }

    #[test]
    fn braces_inside_strings_do_not_confuse_the_scanner() {
        let p = parse_response("{\"type\":\"edit\",\"path\":\"a.rs\",\"operations\":[{\"type\":\"append\",\"content\":\"fn x() { if a { } }\"}]}", Role::Coder).unwrap();
        assert!(matches!(p.message, AgentMessage::Edit { .. }));
    }

    #[test]
    fn complete_json_detection() {
        assert!(!has_complete_json("<think>hmm {\"a\":1}"));
        assert!(!has_complete_json("{\"type\":\"plan\",\"steps\":[\"a\""));
        assert!(has_complete_json("<think>x</think>{\"type\":\"plan\",\"steps\":[\"a\"]}"));
    }

    #[test]
    fn spec_examples_validate() {
        for (s, role) in [
            ("{\"type\":\"tool_call\",\"tool\":\"read_file\",\"arguments\":{\"path\":\"src/main.cpp\"}}", Role::Coder),
            ("{\"type\":\"edit\",\"path\":\"src/main.cpp\",\"operations\":[{\"type\":\"replace\",\"old\":\"old code\",\"new\":\"new code\"}]}", Role::Coder),
            ("{\"type\":\"complete\",\"summary\":\"Implemented the requested feature.\",\"verification\":{\"build\":true,\"test\":true}}", Role::Thinking),
            ("{\"type\":\"plan\",\"steps\":[\"Inspect project structure\",\"Modify source files\",\"Build project\",\"Run tests\"]}", Role::Thinking),
        ] {
            parse_response(s, role).unwrap();
        }
    }
}
