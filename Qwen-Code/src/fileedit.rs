//! Precise file editing: `create`, `replace`, `insert`, `delete`, `append`.
//!
//! Edits are applied in memory and written atomically. Before a file is modified its current
//! state is verified (expected hash, agent read-tracking, anchor text). If the file is not in the
//! expected state the edit fails safely and the current content is returned to the caller.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum EditOp {
    Create {
        content: String,
        #[serde(default)]
        overwrite: bool,
    },
    Replace {
        #[serde(default)]
        old: Option<String>,
        new: String,
        #[serde(default)]
        all: bool,
        #[serde(default)]
        start_line: Option<usize>,
        #[serde(default)]
        end_line: Option<usize>,
        #[serde(default)]
        expected_old: Option<String>,
    },
    Insert {
        content: String,
        #[serde(default)]
        line: Option<usize>,
        #[serde(default)]
        before: Option<String>,
        #[serde(default)]
        after: Option<String>,
    },
    Delete {
        #[serde(default)]
        old: Option<String>,
        #[serde(default)]
        start_line: Option<usize>,
        #[serde(default)]
        end_line: Option<usize>,
        #[serde(default)]
        all: bool,
        #[serde(default)]
        expected_old: Option<String>,
    },
    Append {
        content: String,
    },
}

impl EditOp {
    fn is_line_based(&self) -> bool {
        match self {
            EditOp::Replace { old, start_line, .. } => old.is_none() && start_line.is_some(),
            EditOp::Insert { line, .. } => line.is_some(),
            EditOp::Delete { old, start_line, .. } => old.is_none() && start_line.is_some(),
            _ => false,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            EditOp::Create { .. } => "create",
            EditOp::Replace { .. } => "replace",
            EditOp::Insert { .. } => "insert",
            EditOp::Delete { .. } => "delete",
            EditOp::Append { .. } => "append",
        }
    }
}

/// Remembers the state (hash) of files the agent has read or written.
#[derive(Debug, Default, Clone)]
pub struct FileTracker {
    seen: HashMap<PathBuf, String>,
}

impl FileTracker {
    pub fn record(&mut self, path: &Path, content: &[u8]) {
        self.seen.insert(path.to_path_buf(), sha256_hex(content));
    }
    pub fn get(&self, path: &Path) -> Option<&str> {
        self.seen.get(path).map(|s| s.as_str())
    }
    pub fn forget(&mut self, path: &Path) {
        self.seen.remove(path);
    }
    pub fn clear(&mut self) {
        self.seen.clear();
    }
    pub fn len(&self) -> usize {
        self.seen.len()
    }
    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct EditError {
    pub message: String,
    /// Hash of the file as it currently is (when it exists).
    pub current_hash: Option<String>,
    /// Numbered excerpt of the current content, so the caller can re-check the file.
    pub current_excerpt: Option<String>,
}

impl EditError {
    fn plain(msg: impl Into<String>) -> Self {
        Self { message: msg.into(), current_hash: None, current_excerpt: None }
    }
}

impl std::fmt::Display for EditError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)?;
        if let Some(h) = &self.current_hash {
            write!(f, "\ncurrent sha256: {h}")?;
        }
        if let Some(x) = &self.current_excerpt {
            write!(f, "\ncurrent content:\n{x}")?;
        }
        Ok(())
    }
}

impl std::error::Error for EditError {}

#[derive(Debug, Clone)]
pub struct EditReport {
    pub path: PathBuf,
    pub created: bool,
    pub ops_applied: usize,
    pub bytes_before: usize,
    pub bytes_after: usize,
    pub sha256: String,
    pub diff: String,
    pub lines_added: usize,
    pub lines_removed: usize,
}

pub struct EditRequest<'a> {
    pub path: &'a Path,
    pub ops: &'a [EditOp],
    pub expected_hash: Option<&'a str>,
}

/// Numbered excerpt (`  12| text`) of at most `max_lines` lines, optionally centred on a line.
pub fn numbered_excerpt(text: &str, center: Option<usize>, max_lines: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let total = lines.len();
    let (start, end) = match center {
        Some(c) if total > max_lines => {
            let s = c.saturating_sub(max_lines / 2).min(total.saturating_sub(max_lines));
            (s, (s + max_lines).min(total))
        }
        _ => (0, total.min(max_lines)),
    };
    let mut out = String::new();
    for (i, l) in lines[start..end].iter().enumerate() {
        out.push_str(&format!("{:>5}| {}\n", start + i + 1, l));
    }
    if end < total {
        out.push_str(&format!("... ({} more lines)\n", total - end));
    }
    out
}

fn hash_matches(expected: &str, actual: &str) -> bool {
    let e = expected.trim().trim_start_matches("sha256:").to_ascii_lowercase();
    e.len() >= 8 && actual.starts_with(&e)
}

fn split_keep(text: &str) -> Vec<&str> {
    text.split_inclusive('\n').collect()
}

fn line_number_of(text: &str, byte_pos: usize) -> usize {
    text[..byte_pos.min(text.len())].bytes().filter(|&b| b == b'\n').count() + 1
}

fn norm_lf(s: &str) -> String {
    s.replace("\r\n", "\n")
}

fn ensure_nl(mut s: String) -> String {
    if !s.is_empty() && !s.ends_with('\n') {
        s.push('\n');
    }
    s
}

fn candidates_hint(text: &str, old: &str) -> String {
    let needle = old.lines().map(|l| l.trim()).find(|l| !l.is_empty()).unwrap_or("");
    if needle.is_empty() {
        return String::new();
    }
    let hits: Vec<String> = text
        .lines()
        .enumerate()
        .filter(|(_, l)| l.trim().contains(needle))
        .take(3)
        .map(|(i, l)| format!("  line {}: {}", i + 1, l.trim_end()))
        .collect();
    if hits.is_empty() {
        format!("\nNo line contains the first line of `old` (\"{needle}\"). Re-read the file.")
    } else {
        format!("\nSimilar lines (check whitespace/indentation):\n{}", hits.join("\n"))
    }
}

fn find_all(text: &str, needle: &str) -> Vec<usize> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(p) = text[from..].find(needle) {
        out.push(from + p);
        from += p + needle.len().max(1);
        if from > text.len() {
            break;
        }
    }
    out
}

fn range_of(text: &str, start: usize, end: usize) -> Result<(usize, usize, usize), String> {
    let lines = split_keep(text);
    if lines.is_empty() {
        return Err("the file is empty; there are no lines to address".into());
    }
    if start == 0 || end < start {
        return Err(format!("invalid line range {start}..{end} (lines are 1-based and end >= start)"));
    }
    if end > lines.len() {
        return Err(format!("line range {start}..{end} is outside the file ({} lines)", lines.len()));
    }
    let byte_start: usize = lines[..start - 1].iter().map(|l| l.len()).sum();
    let byte_end: usize = byte_start + lines[start - 1..end].iter().map(|l| l.len()).sum::<usize>();
    Ok((byte_start, byte_end, lines.len()))
}

fn apply_one(text: &mut String, op: &EditOp) -> Result<String, String> {
    match op {
        EditOp::Create { .. } => Err("`create` must be the first operation".into()),
        EditOp::Replace { old, new, all, start_line, end_line, expected_old } => {
            let new = norm_lf(new);
            if let Some(old) = old {
                let old = norm_lf(old);
                if old.is_empty() {
                    return Err("replace: `old` must not be empty".into());
                }
                let hits = find_all(text, &old);
                match hits.len() {
                    0 => Err(format!("replace: `old` text not found in the file.{}", candidates_hint(text, &old))),
                    1 => {
                        text.replace_range(hits[0]..hits[0] + old.len(), &new);
                        Ok(format!("replace text at line {}", line_number_of(text, hits[0])))
                    }
                    n if *all => {
                        *text = text.replace(&old, &new);
                        Ok(format!("replace {n} occurrences"))
                    }
                    n => {
                        let lines: Vec<String> = hits.iter().take(6).map(|h| line_number_of(text, *h).to_string()).collect();
                        Err(format!(
                            "replace: `old` matches {n} places (lines {}). Add more surrounding context to make it unique, or set \"all\": true.",
                            lines.join(", ")
                        ))
                    }
                }
            } else if let Some(start) = start_line {
                let end = end_line.unwrap_or(*start);
                let (bs, be, _) = range_of(text, *start, end)?;
                if let Some(exp) = expected_old {
                    let cur = text[bs..be].trim_end_matches(['\n', '\r']).to_string();
                    if cur != norm_lf(exp).trim_end_matches(['\n', '\r']) {
                        return Err(format!("replace: lines {start}..{end} do not contain the expected text. Current text:\n{cur}"));
                    }
                }
                let mut repl = new;
                if !repl.is_empty() && !repl.ends_with('\n') && text[bs..be].ends_with('\n') {
                    repl.push('\n');
                }
                text.replace_range(bs..be, &repl);
                Ok(format!("replace lines {start}..{end}"))
            } else {
                Err("replace: give either `old` or `start_line`".into())
            }
        }
        EditOp::Insert { content, line, before, after } => {
            let content = ensure_nl(norm_lf(content));
            if let Some(l) = line {
                let lines = split_keep(text);
                if *l == 0 || *l > lines.len() + 1 {
                    return Err(format!("insert: line {l} is outside 1..={} (file has {} lines)", lines.len() + 1, lines.len()));
                }
                let pos: usize = lines[..l - 1].iter().map(|x| x.len()).sum();
                if *l == lines.len() + 1 && !text.is_empty() && !text.ends_with('\n') {
                    text.push('\n');
                    text.push_str(&content);
                } else {
                    text.insert_str(pos, &content);
                }
                Ok(format!("insert before line {l}"))
            } else if let Some(anchor) = after {
                let anchor = norm_lf(anchor);
                let hits = find_all(text, &anchor);
                if hits.len() != 1 {
                    return Err(format!("insert: `after` anchor must match exactly once (matched {})", hits.len()));
                }
                let end = hits[0] + anchor.len();
                let pos = if anchor.ends_with('\n') {
                    end
                } else {
                    match text[end..].find('\n') {
                        Some(p) => end + p + 1,
                        None => {
                            text.push('\n');
                            text.len()
                        }
                    }
                };
                text.insert_str(pos, &content);
                Ok("insert after anchor".into())
            } else if let Some(anchor) = before {
                let anchor = norm_lf(anchor);
                let hits = find_all(text, &anchor);
                if hits.len() != 1 {
                    return Err(format!("insert: `before` anchor must match exactly once (matched {})", hits.len()));
                }
                let pos = text[..hits[0]].rfind('\n').map(|p| p + 1).unwrap_or(0);
                text.insert_str(pos, &content);
                Ok("insert before anchor".into())
            } else {
                Err("insert: give `line`, `before` or `after`".into())
            }
        }
        EditOp::Delete { old, start_line, end_line, all, expected_old } => {
            if let Some(old) = old {
                let old = norm_lf(old);
                if old.is_empty() {
                    return Err("delete: `old` must not be empty".into());
                }
                let hits = find_all(text, &old);
                match hits.len() {
                    0 => Err(format!("delete: `old` text not found in the file.{}", candidates_hint(text, &old))),
                    1 => {
                        text.replace_range(hits[0]..hits[0] + old.len(), "");
                        Ok("delete text".into())
                    }
                    n if *all => {
                        *text = text.replace(&old, "");
                        Ok(format!("delete {n} occurrences"))
                    }
                    n => Err(format!("delete: `old` matches {n} places; add context or set \"all\": true")),
                }
            } else if let Some(start) = start_line {
                let end = end_line.unwrap_or(*start);
                let (bs, be, _) = range_of(text, *start, end)?;
                if let Some(exp) = expected_old {
                    let cur = text[bs..be].trim_end_matches(['\n', '\r']).to_string();
                    if cur != norm_lf(exp).trim_end_matches(['\n', '\r']) {
                        return Err(format!("delete: lines {start}..{end} do not contain the expected text. Current text:\n{cur}"));
                    }
                }
                text.replace_range(bs..be, "");
                Ok(format!("delete lines {start}..{end}"))
            } else {
                Err("delete: give either `old` or `start_line`".into())
            }
        }
        EditOp::Append { content } => {
            let content = ensure_nl(norm_lf(content));
            if !text.is_empty() && !text.ends_with('\n') {
                text.push('\n');
            }
            text.push_str(&content);
            Ok("append".into())
        }
    }
}

/// Compact diff: strips the common prefix/suffix lines and shows what changed.
pub fn summarize_diff(before: &str, after: &str, max_lines: usize) -> (String, usize, usize) {
    let a: Vec<&str> = before.lines().collect();
    let b: Vec<&str> = after.lines().collect();
    let mut p = 0;
    while p < a.len() && p < b.len() && a[p] == b[p] {
        p += 1;
    }
    let mut s = 0;
    while s < a.len() - p && s < b.len() - p && a[a.len() - 1 - s] == b[b.len() - 1 - s] {
        s += 1;
    }
    let removed = &a[p..a.len() - s];
    let added = &b[p..b.len() - s];
    if removed.is_empty() && added.is_empty() {
        return ("(no textual change)".into(), 0, 0);
    }
    let mut out = format!("@@ line {} @@\n", p + 1);
    let mut shown = 0;
    let clip = |l: &str| -> String {
        if l.chars().count() > 160 {
            format!("{}...", l.chars().take(160).collect::<String>())
        } else {
            l.to_string()
        }
    };
    for l in removed {
        if shown >= max_lines {
            break;
        }
        out.push_str(&format!("-{}\n", clip(l)));
        shown += 1;
    }
    for l in added {
        if shown >= max_lines {
            break;
        }
        out.push_str(&format!("+{}\n", clip(l)));
        shown += 1;
    }
    let total = removed.len() + added.len();
    if total > shown {
        out.push_str(&format!("... ({} more changed lines)\n", total - shown));
    }
    (out, added.len(), removed.len())
}

/// Writes a file atomically (temporary file + rename).
pub fn write_atomic(path: &Path, data: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "file".into());
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let tmp = path.with_file_name(format!(".{name}.qwen-tmp-{}-{nanos}", std::process::id()));
    let result = (|| {
        std::fs::write(&tmp, data)?;
        #[cfg(unix)]
        if let Ok(meta) = std::fs::metadata(path) {
            let _ = std::fs::set_permissions(&tmp, meta.permissions());
        }
        std::fs::rename(&tmp, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

/// Reads a UTF-8 text file.
pub fn read_text(path: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    String::from_utf8(bytes).map_err(|_| format!("{} is not valid UTF-8 text (binary file?)", path.display()))
}

/// Applies a list of operations to a file, verifying its state first.
pub fn apply_edit(req: &EditRequest, tracker: &mut FileTracker) -> Result<EditReport, EditError> {
    let path = req.path;
    if req.ops.is_empty() {
        return Err(EditError::plain("no operations given"));
    }
    let exists = path.exists();
    if exists && !path.is_file() {
        return Err(EditError::plain(format!("{} is not a regular file", path.display())));
    }

    let raw: Option<String> = if exists { Some(read_text(path).map_err(EditError::plain)?) } else { None };
    let current_hash = raw.as_ref().map(|r| sha256_hex(r.as_bytes()));

    let state_error = |msg: String, raw: &Option<String>, hash: &Option<String>| EditError {
        message: msg,
        current_hash: hash.clone(),
        current_excerpt: raw.as_ref().map(|r| numbered_excerpt(r, None, 200)),
    };

    if let (Some(exp), Some(cur)) = (req.expected_hash, current_hash.as_ref()) {
        if !hash_matches(exp, cur) {
            if let Some(r) = &raw {
                tracker.record(path, r.as_bytes());
            }
            return Err(state_error(
                format!("{} is not in the expected state (expected sha256 {exp}). It was probably modified; review the current content below.", path.display()),
                &raw,
                &current_hash,
            ));
        }
    }
    if let (Some(seen), Some(cur), Some(r)) = (tracker.get(path), current_hash.as_ref(), raw.as_ref()) {
        if seen != cur {
            tracker.record(path, r.as_bytes());
            return Err(state_error(
                format!("{} changed since the agent last read or wrote it. Review the current content below and retry.", path.display()),
                &raw,
                &current_hash,
            ));
        }
    }

    let first_is_create = matches!(req.ops[0], EditOp::Create { .. });
    if !exists && !first_is_create {
        return Err(EditError::plain(format!(
            "{} does not exist. Start with a `create` operation (or use write_file).",
            path.display()
        )));
    }
    if exists && tracker.get(path).is_none() && req.ops.iter().any(|o| o.is_line_based()) {
        return Err(state_error(
            format!(
                "{} has not been read yet, so line numbers cannot be trusted. Current content is shown below; retry with these line numbers or anchor the edit with `old` text.",
                path.display()
            ),
            &raw,
            &current_hash,
        ));
    }

    let had_crlf = raw.as_ref().is_some_and(|r| r.contains("\r\n"));
    let original_lf = raw.as_ref().map(|r| norm_lf(r)).unwrap_or_default();
    let mut text = original_lf.clone();
    let mut descriptions: Vec<String> = Vec::new();
    let mut created = !exists;

    for (i, op) in req.ops.iter().enumerate() {
        if let EditOp::Create { content, overwrite } = op {
            if i != 0 {
                return Err(EditError::plain(format!("operation {}: `create` must be the first operation", i + 1)));
            }
            if exists && !overwrite {
                return Err(state_error(
                    format!("{} already exists. Use replace/insert/append, or set \"overwrite\": true.", path.display()),
                    &raw,
                    &current_hash,
                ));
            }
            text = norm_lf(content);
            created = !exists;
            descriptions.push(if exists { "overwrite file".into() } else { "create file".into() });
            continue;
        }
        match apply_one(&mut text, op) {
            Ok(d) => descriptions.push(d),
            Err(e) => {
                return Err(state_error(
                    format!("operation {} ({}) failed: {e}\nNothing was written.", i + 1, op.label()),
                    &raw,
                    &current_hash,
                ));
            }
        }
    }

    let (diff, added, removed) = summarize_diff(&original_lf, &text, 40);
    let out_text = if had_crlf { text.replace('\n', "\r\n") } else { text };
    write_atomic(path, out_text.as_bytes()).map_err(|e| EditError::plain(format!("cannot write {}: {e}", path.display())))?;
    tracker.record(path, out_text.as_bytes());
    Ok(EditReport {
        path: path.to_path_buf(),
        created,
        ops_applied: descriptions.len(),
        bytes_before: raw.as_ref().map(|r| r.len()).unwrap_or(0),
        bytes_after: out_text.len(),
        sha256: sha256_hex(out_text.as_bytes()),
        diff,
        lines_added: added,
        lines_removed: removed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op_replace(old: &str, new: &str) -> EditOp {
        EditOp::Replace { old: Some(old.into()), new: new.into(), all: false, start_line: None, end_line: None, expected_old: None }
    }

    fn run(path: &Path, ops: Vec<EditOp>, tr: &mut FileTracker) -> Result<EditReport, EditError> {
        apply_edit(&EditRequest { path, ops: &ops, expected_hash: None }, tr)
    }

    #[test]
    fn create_replace_insert_delete_append() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("sub/a.txt");
        let mut tr = FileTracker::default();
        let r = run(&p, vec![EditOp::Create { content: "one\ntwo\nthree\n".into(), overwrite: false }], &mut tr).unwrap();
        assert!(r.created);
        run(&p, vec![op_replace("two", "2")], &mut tr).unwrap();
        run(&p, vec![EditOp::Insert { content: "inserted".into(), line: Some(1), before: None, after: None }], &mut tr).unwrap();
        run(&p, vec![EditOp::Delete { old: Some("three\n".into()), start_line: None, end_line: None, all: false, expected_old: None }], &mut tr).unwrap();
        run(&p, vec![EditOp::Append { content: "end".into() }], &mut tr).unwrap();
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "inserted\none\n2\nend\n");
    }

    #[test]
    fn ambiguous_replace_fails_and_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.txt");
        std::fs::write(&p, "x\nx\n").unwrap();
        let mut tr = FileTracker::default();
        let err = run(&p, vec![op_replace("x", "y")], &mut tr).unwrap_err();
        assert!(err.message.contains("matches 2 places"));
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "x\nx\n");
        assert!(err.current_excerpt.is_some());
    }

    #[test]
    fn multi_op_is_atomic() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.txt");
        std::fs::write(&p, "a\nb\n").unwrap();
        let mut tr = FileTracker::default();
        let err = run(&p, vec![op_replace("a", "A"), op_replace("missing", "z")], &mut tr).unwrap_err();
        assert!(err.message.contains("operation 2"));
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "a\nb\n");
    }

    #[test]
    fn expected_hash_mismatch_fails_safely() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.txt");
        std::fs::write(&p, "hello\n").unwrap();
        let mut tr = FileTracker::default();
        let ops = vec![op_replace("hello", "bye")];
        let err = apply_edit(&EditRequest { path: &p, ops: &ops, expected_hash: Some("deadbeefdeadbeef") }, &mut tr).unwrap_err();
        assert!(err.message.contains("not in the expected state"));
        assert!(err.current_excerpt.unwrap().contains("hello"));
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "hello\n");
        // correct hash works
        let good = sha256_hex(b"hello\n");
        apply_edit(&EditRequest { path: &p, ops: &ops, expected_hash: Some(&good) }, &mut tr).unwrap();
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "bye\n");
    }

    #[test]
    fn external_modification_is_detected() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.txt");
        std::fs::write(&p, "v1\n").unwrap();
        let mut tr = FileTracker::default();
        tr.record(&p, b"v1\n");
        std::fs::write(&p, "v2 changed elsewhere\n").unwrap();
        let err = run(&p, vec![op_replace("v1", "v3")], &mut tr).unwrap_err();
        assert!(err.message.contains("changed since"));
        // after seeing the current state a retry is allowed
        let err2 = run(&p, vec![op_replace("v1", "v3")], &mut tr).unwrap_err();
        assert!(err2.message.contains("not found"));
        run(&p, vec![op_replace("v2", "v3")], &mut tr).unwrap();
    }

    #[test]
    fn line_ops_require_a_read_first() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.txt");
        std::fs::write(&p, "a\nb\nc\n").unwrap();
        let mut tr = FileTracker::default();
        let del = EditOp::Delete { old: None, start_line: Some(2), end_line: Some(2), all: false, expected_old: None };
        let err = run(&p, vec![del.clone()], &mut tr).unwrap_err();
        assert!(err.message.contains("has not been read"));
        tr.record(&p, b"a\nb\nc\n");
        run(&p, vec![del], &mut tr).unwrap();
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "a\nc\n");
    }

    #[test]
    fn crlf_files_stay_crlf() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.txt");
        std::fs::write(&p, "a\r\nb\r\n").unwrap();
        let mut tr = FileTracker::default();
        run(&p, vec![op_replace("a\nb", "x\ny")], &mut tr).unwrap();
        assert_eq!(std::fs::read(&p).unwrap(), b"x\r\ny\r\n");
    }

    #[test]
    fn create_refuses_to_clobber() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.txt");
        std::fs::write(&p, "keep\n").unwrap();
        let mut tr = FileTracker::default();
        let err = run(&p, vec![EditOp::Create { content: "new".into(), overwrite: false }], &mut tr).unwrap_err();
        assert!(err.message.contains("already exists"));
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "keep\n");
    }

    #[test]
    fn insert_after_anchor_and_range_replace() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.rs");
        std::fs::write(&p, "fn a() {\n}\nfn b() {\n}\n").unwrap();
        let mut tr = FileTracker::default();
        run(&p, vec![EditOp::Insert { content: "    todo();".into(), line: None, before: None, after: Some("fn a() {".into()) }], &mut tr).unwrap();
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "fn a() {\n    todo();\n}\nfn b() {\n}\n");
        tr.record(&p, std::fs::read_to_string(&p).unwrap().as_bytes());
        run(
            &p,
            vec![EditOp::Replace { old: None, new: "    done();".into(), all: false, start_line: Some(2), end_line: Some(2), expected_old: Some("    todo();".into()) }],
            &mut tr,
        )
        .unwrap();
        assert!(std::fs::read_to_string(&p).unwrap().contains("    done();\n}"));
    }

    #[test]
    fn deserializes_from_json() {
        let ops: Vec<EditOp> = serde_json::from_str(r#"[{"type":"replace","old":"a","new":"b"},{"type":"append","content":"x"}]"#).unwrap();
        assert_eq!(ops.len(), 2);
    }
}
