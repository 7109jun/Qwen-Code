//! A small JSON Schema validator (the subset used by Qwen Code's tool and message schemas).
//!
//! Supported keywords: `type`, `enum`, `const`, `properties`, `required`, `additionalProperties`,
//! `items`, `minItems`, `maxItems`, `minLength`, `maxLength`, `minimum`, `maximum`, `pattern`,
//! `oneOf`, `anyOf`.

use regex::Regex;
use serde_json::Value;

/// Validates `value` against `schema`. Returns every violation found (JSON-pointer prefixed).
pub fn validate(value: &Value, schema: &Value) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    check(value, schema, "", &mut errors);
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn where_(path: &str) -> String {
    if path.is_empty() {
        "$".to_string()
    } else {
        format!("${path}")
    }
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                "integer"
            } else {
                "number"
            }
        }
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn type_matches(v: &Value, t: &str) -> bool {
    match t {
        "integer" => match v {
            Value::Number(n) => n.is_i64() || n.is_u64() || n.as_f64().is_some_and(|f| f.fract() == 0.0),
            _ => false,
        },
        "number" => v.is_number(),
        other => type_name(v) == other,
    }
}

fn check(value: &Value, schema: &Value, path: &str, errors: &mut Vec<String>) {
    let Some(obj) = schema.as_object() else {
        return; // `true` / empty schema accepts everything
    };

    if let Some(t) = obj.get("type") {
        let ok = match t {
            Value::String(s) => type_matches(value, s),
            Value::Array(list) => list.iter().filter_map(|x| x.as_str()).any(|s| type_matches(value, s)),
            _ => true,
        };
        if !ok {
            errors.push(format!("{}: expected {}, got {}", where_(path), t, type_name(value)));
            return;
        }
    }

    if let Some(Value::Array(options)) = obj.get("enum") {
        if !options.contains(value) {
            let list: Vec<String> = options.iter().map(|o| o.to_string()).collect();
            errors.push(format!("{}: must be one of [{}]", where_(path), list.join(", ")));
        }
    }
    if let Some(c) = obj.get("const") {
        if c != value {
            errors.push(format!("{}: must equal {}", where_(path), c));
        }
    }

    if let Some(Value::Array(subs)) = obj.get("oneOf") {
        let matching = subs.iter().filter(|s| validate(value, s).is_ok()).count();
        if matching != 1 {
            errors.push(format!("{}: must match exactly one alternative (matched {matching})", where_(path)));
        }
    }
    if let Some(Value::Array(subs)) = obj.get("anyOf") {
        if !subs.iter().any(|s| validate(value, s).is_ok()) {
            errors.push(format!("{}: does not match any allowed alternative", where_(path)));
        }
    }

    match value {
        Value::Object(map) => {
            if let Some(Value::Array(req)) = obj.get("required") {
                for r in req.iter().filter_map(|x| x.as_str()) {
                    if !map.contains_key(r) {
                        errors.push(format!("{}: missing required property \"{r}\"", where_(path)));
                    }
                }
            }
            let props = obj.get("properties").and_then(|p| p.as_object());
            for (k, v) in map {
                let child = format!("{path}/{k}");
                match props.and_then(|p| p.get(k)) {
                    Some(sub) => check(v, sub, &child, errors),
                    None => match obj.get("additionalProperties") {
                        Some(Value::Bool(false)) => {
                            errors.push(format!("{}: unknown property \"{k}\"", where_(path)));
                        }
                        Some(sub @ Value::Object(_)) => check(v, sub, &child, errors),
                        _ => {}
                    },
                }
            }
        }
        Value::Array(items) => {
            if let Some(min) = obj.get("minItems").and_then(|v| v.as_u64()) {
                if (items.len() as u64) < min {
                    errors.push(format!("{}: needs at least {min} items", where_(path)));
                }
            }
            if let Some(max) = obj.get("maxItems").and_then(|v| v.as_u64()) {
                if (items.len() as u64) > max {
                    errors.push(format!("{}: allows at most {max} items", where_(path)));
                }
            }
            if let Some(sub) = obj.get("items") {
                for (i, it) in items.iter().enumerate() {
                    check(it, sub, &format!("{path}/{i}"), errors);
                }
            }
        }
        Value::String(s) => {
            let len = s.chars().count() as u64;
            if let Some(min) = obj.get("minLength").and_then(|v| v.as_u64()) {
                if len < min {
                    errors.push(format!("{}: must have at least {min} characters", where_(path)));
                }
            }
            if let Some(max) = obj.get("maxLength").and_then(|v| v.as_u64()) {
                if len > max {
                    errors.push(format!("{}: must have at most {max} characters", where_(path)));
                }
            }
            if let Some(p) = obj.get("pattern").and_then(|v| v.as_str()) {
                match Regex::new(p) {
                    Ok(re) if !re.is_match(s) => errors.push(format!("{}: does not match pattern {p}", where_(path))),
                    Err(e) => errors.push(format!("{}: invalid schema pattern {p}: {e}", where_(path))),
                    _ => {}
                }
            }
        }
        Value::Number(n) => {
            if let (Some(min), Some(x)) = (obj.get("minimum").and_then(|v| v.as_f64()), n.as_f64()) {
                if x < min {
                    errors.push(format!("{}: must be >= {min}", where_(path)));
                }
            }
            if let (Some(max), Some(x)) = (obj.get("maximum").and_then(|v| v.as_f64()), n.as_f64()) {
                if x > max {
                    errors.push(format!("{}: must be <= {max}", where_(path)));
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "n": {"type": "integer", "minimum": 1, "maximum": 10},
                "mode": {"enum": ["a", "b"]},
                "list": {"type": "array", "items": {"type": "string"}, "maxItems": 2}
            },
            "required": ["path"],
            "additionalProperties": false
        })
    }

    #[test]
    fn accepts_valid() {
        assert!(validate(&json!({"path": "x", "n": 3, "mode": "a", "list": ["q"]}), &schema()).is_ok());
    }

    #[test]
    fn reports_all_problems() {
        let errs = validate(&json!({"n": 0, "mode": "z", "extra": 1, "list": [1, "a", "b"]}), &schema()).unwrap_err();
        let joined = errs.join("\n");
        assert!(joined.contains("missing required property \"path\""));
        assert!(joined.contains(">= 1"));
        assert!(joined.contains("one of"));
        assert!(joined.contains("unknown property \"extra\""));
        assert!(joined.contains("at most 2"));
        assert!(joined.contains("/list/0"));
    }

    #[test]
    fn type_mismatch_stops_descent() {
        let errs = validate(&json!("nope"), &schema()).unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(errs[0].contains("expected"));
    }

    #[test]
    fn one_of_and_const() {
        let s = json!({"oneOf": [
            {"type": "object", "properties": {"type": {"const": "a"}}, "required": ["type"]},
            {"type": "object", "properties": {"type": {"const": "b"}}, "required": ["type"]}
        ]});
        assert!(validate(&json!({"type": "a"}), &s).is_ok());
        assert!(validate(&json!({"type": "c"}), &s).is_err());
    }

    #[test]
    fn integers_and_numbers() {
        assert!(validate(&json!(3), &json!({"type": "integer"})).is_ok());
        assert!(validate(&json!(3.5), &json!({"type": "integer"})).is_err());
        assert!(validate(&json!(3.5), &json!({"type": "number"})).is_ok());
    }
}
