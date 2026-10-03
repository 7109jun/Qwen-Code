//! Context manager.
//!
//! Large projects never fit into a prompt, so only what is needed is included:
//! * a bounded project snapshot (tree + manifests + git summary),
//! * retrieved file snippets matching the task (never whole files by default),
//! * the conversation history, which is compacted automatically when it grows too large.

use crate::config::ContextSection;
use crate::platform;
use crate::runtime::ChatMessage;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Rough token estimate: ASCII ≈ 4 chars per token, other scripts (Korean, CJK) ≈ 1 token per char.
pub fn estimate_tokens(s: &str) -> usize {
    let mut ascii = 0usize;
    let mut other = 0usize;
    for c in s.chars() {
        if c.is_ascii() {
            ascii += 1;
        } else {
            other += 1;
        }
    }
    ascii.div_ceil(4) + other
}

/// Keeps the head and tail of a long text.
pub fn truncate_middle(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let head = max_bytes * 2 / 3;
    let tail = max_bytes - head;
    let mut h = head;
    while !s.is_char_boundary(h) {
        h -= 1;
    }
    let mut t = s.len() - tail;
    while !s.is_char_boundary(t) {
        t += 1;
    }
    format!("{}\n... [{} bytes omitted] ...\n{}", &s[..h], t.saturating_sub(h), &s[t..])
}

// ---------------------------------------------------------------------------------------------
// History
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Entry {
    /// `user` or `assistant`.
    pub role: String,
    pub text: String,
    /// What the entry is: `task`, `model`, `tool`, `report`, `feedback`, ...
    pub tag: String,
}

#[derive(Debug, Clone, Default)]
pub struct History {
    entries: Vec<Entry>,
    digest: String,
    compactions: usize,
}

const DIGEST_MAX_FRACTION: f32 = 0.3;

impl History {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, role: &str, tag: &str, text: impl Into<String>) {
        self.entries.push(Entry { role: role.to_string(), text: text.into(), tag: tag.to_string() });
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.digest.is_empty()
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    pub fn compactions(&self) -> usize {
        self.compactions
    }

    pub fn digest(&self) -> &str {
        &self.digest
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.digest.clear();
        self.compactions = 0;
    }

    pub fn tokens(&self) -> usize {
        self.entries.iter().map(|e| estimate_tokens(&e.text) + 4).sum::<usize>() + estimate_tokens(&self.digest)
    }

    /// Compacts when the history uses more than `threshold * budget` tokens. Returns `true` if
    /// anything changed. Old entries become one-line digests; if the digest itself grows too large
    /// its oldest lines are dropped.
    pub fn compact(&mut self, budget: usize, threshold: f32, keep_recent: usize) -> bool {
        if (self.tokens() as f32) <= budget as f32 * threshold {
            return false;
        }
        self.force_compact(budget, keep_recent)
    }

    pub fn force_compact(&mut self, budget: usize, keep_recent: usize) -> bool {
        let mut changed = false;
        // 1. digest the old entries, oldest first, until the history fits comfortably (or only the
        //    protected recent entries remain)
        let target = (budget as f32 * 0.6) as usize;
        while self.entries.len() > keep_recent.max(1) && (self.tokens() > target || !changed) {
            let e = self.entries.remove(0);
            let line = summarize_entry(&e);
            if !line.is_empty() {
                if !self.digest.is_empty() {
                    self.digest.push('\n');
                }
                self.digest.push_str(&line);
            }
            changed = true;
            if self.entries.len() <= keep_recent.max(1) && self.tokens() <= budget {
                break;
            }
        }
        // 2. bound the digest
        let max_digest = (budget as f32 * DIGEST_MAX_FRACTION) as usize;
        while estimate_tokens(&self.digest) > max_digest {
            match self.digest.split_once('\n') {
                Some((_, rest)) => self.digest = rest.to_string(),
                None => {
                    self.digest = truncate_middle(&self.digest, max_digest * 3);
                    break;
                }
            }
            changed = true;
        }
        // 3. still too large: shrink the biggest recent entries (typically huge tool outputs)
        let mut guard = 0;
        while self.tokens() > budget && guard < 64 {
            guard += 1;
            let Some((idx, _)) = self.entries.iter().enumerate().max_by_key(|(_, e)| e.text.len()) else { break };
            let len = self.entries[idx].text.len();
            if len < 600 {
                // nothing left to shrink: drop the oldest remaining entry
                if self.entries.len() > 1 {
                    let e = self.entries.remove(0);
                    let line = summarize_entry(&e);
                    if !line.is_empty() {
                        self.digest.push('\n');
                        self.digest.push_str(&line);
                    }
                } else {
                    break;
                }
            } else {
                let new_len = (len / 2).max(500);
                self.entries[idx].text = truncate_middle(&self.entries[idx].text, new_len);
            }
            changed = true;
        }
        if changed {
            self.compactions += 1;
        }
        changed
    }

    /// Converts the history into chat messages. Consecutive messages with the same role are merged
    /// (several chat templates require alternating roles).
    pub fn messages(&self) -> Vec<ChatMessage> {
        let mut out: Vec<ChatMessage> = Vec::new();
        if !self.digest.is_empty() {
            out.push(ChatMessage::user(format!("[Earlier conversation, compacted]\n{}", self.digest)));
        }
        for e in &self.entries {
            match out.last_mut() {
                Some(last) if last.role == e.role => {
                    last.content.push_str("\n\n");
                    last.content.push_str(&e.text);
                }
                _ => out.push(ChatMessage::new(&e.role, e.text.clone())),
            }
        }
        out
    }
}

fn first_line(s: &str, max: usize) -> String {
    let l = s.lines().map(|l| l.trim()).find(|l| !l.is_empty()).unwrap_or("");
    if l.chars().count() > max {
        format!("{}…", l.chars().take(max).collect::<String>())
    } else {
        l.to_string()
    }
}

/// One-line digest of a history entry.
pub fn summarize_entry(e: &Entry) -> String {
    match e.tag.as_str() {
        "task" => format!("- user task: {}", first_line(&e.text, 200)),
        "model" => format!("- model said: {}", first_line(&e.text, 140)),
        "tool" => format!("- tool result: {}", first_line(&e.text, 140)),
        "report" => format!("- coder report: {}", first_line(&e.text, 200)),
        "feedback" => String::new(),
        _ => format!("- {}: {}", e.tag, first_line(&e.text, 140)),
    }
}

// ---------------------------------------------------------------------------------------------
// Context manager
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct Retrieved {
    pub text: String,
    /// `path (lines a-b)` for every included snippet.
    pub files: Vec<String>,
}

pub struct ContextManager {
    pub cfg: ContextSection,
    pub workspace: PathBuf,
    /// Files pinned by the user (`qwen src/main.cpp`).
    pub focus: Vec<String>,
    last_files: Vec<String>,
}

const MANIFESTS: &[&str] = &[
    "Cargo.toml", "package.json", "pyproject.toml", "go.mod", "CMakeLists.txt", "Makefile", "build.gradle", "pom.xml", "requirements.txt", "qwen.toml",
];

const STOPWORDS: &[&str] = &[
    "the", "and", "for", "with", "that", "this", "from", "into", "please", "fix", "make", "create", "add", "write", "implement", "change", "update", "remove", "delete",
    "build", "file", "files", "code", "project", "error", "errors", "using", "use", "should", "need", "want", "all", "not", "are", "can", "you", "run", "test", "tests",
    "new", "get", "set", "show", "how", "why", "what", "when", "where", "then", "also", "have", "has", "will", "would", "could", "some", "any", "now", "just", "like",
];

impl ContextManager {
    pub fn new(cfg: ContextSection, workspace: &Path) -> Self {
        Self { cfg, workspace: platform::canonicalize_lossy(workspace), focus: Vec::new(), last_files: Vec::new() }
    }

    /// Token budget for the prompt, bounded by the model's context window (leaving room for the answer).
    pub fn budget(&self, window: Option<usize>, max_new: usize) -> usize {
        let base = self.cfg.max_tokens;
        match window {
            Some(w) => base.min(w.saturating_sub(max_new + 256)).max(1024),
            None => base,
        }
    }

    pub fn last_files(&self) -> &[String] {
        &self.last_files
    }

    fn walker(&self, depth: Option<usize>) -> ignore::Walk {
        let mut b = ignore::WalkBuilder::new(&self.workspace);
        b.hidden(true).git_global(false).require_git(false).follow_links(false).max_depth(depth).sort_by_file_path(|a, b| a.cmp(b));
        b.filter_entry(|e| {
            let n = e.file_name().to_string_lossy();
            !(matches!(n.as_ref(), ".git" | "node_modules" | "__pycache__" | ".venv" | "dist" | "build" | ".qwen") || (n == "target" && e.file_type().is_some_and(|t| t.is_dir())))
        });
        b.build()
    }

    /// Directory tree (bounded) plus excerpts of the project manifests.
    pub fn project_snapshot(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("workspace: {}\n", self.workspace.display()));
        out.push_str("project structure (depth-limited, .gitignore respected):\n");
        let mut count = 0usize;
        let mut more = 0usize;
        for e in self.walker(Some(self.cfg.snapshot_depth)).flatten() {
            if e.depth() == 0 {
                continue;
            }
            if count >= self.cfg.snapshot_max_entries {
                more += 1;
                continue;
            }
            count += 1;
            let indent = "  ".repeat(e.depth() - 1);
            let name = e.file_name().to_string_lossy();
            if e.file_type().is_some_and(|t| t.is_dir()) {
                out.push_str(&format!("{indent}{name}/\n"));
            } else {
                out.push_str(&format!("{indent}{name}\n"));
            }
        }
        if more > 0 {
            out.push_str(&format!("… {more} more entries not shown (use list_directory / search_files)\n"));
        }
        for m in MANIFESTS {
            let p = self.workspace.join(m);
            if let Ok(text) = std::fs::read_to_string(&p) {
                let lines: Vec<&str> = text.lines().collect();
                out.push_str(&format!("\n--- {m} ({} lines{}) ---\n", lines.len(), if lines.len() > 30 { ", first 30 shown" } else { "" }));
                for l in lines.iter().take(30) {
                    out.push_str(l);
                    out.push('\n');
                }
            }
        }
        out
    }

    fn keywords(query: &str) -> Vec<String> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for raw in query.split(|c: char| !(c.is_alphanumeric() || matches!(c, '_' | '.' | '/' | '\\' | '-'))) {
            let w = raw.trim_matches(|c: char| matches!(c, '.' | '-' | '/' | '\\')).to_lowercase();
            if w.chars().count() < 3 || STOPWORDS.contains(&w.as_str()) || !w.is_ascii() {
                continue;
            }
            if seen.insert(w.clone()) {
                out.push(w);
            }
            if out.len() >= 10 {
                break;
            }
        }
        out
    }

    /// Finds the files and line ranges that matter for `query` and renders them as snippets.
    pub fn retrieve(&mut self, query: &str) -> Retrieved {
        let words = Self::keywords(query);
        let max_files = self.cfg.retrieval_max_files;
        let snippet = self.cfg.retrieval_snippet_lines.max(10);
        let mut scored: Vec<(f32, PathBuf, Option<usize>)> = Vec::new();

        let mut scanned = 0usize;
        if !words.is_empty() {
            for e in self.walker(None).flatten() {
                if !e.file_type().is_some_and(|t| t.is_file()) {
                    continue;
                }
                scanned += 1;
                if scanned > 4000 {
                    break;
                }
                let path = e.path().to_path_buf();
                let rel = platform::display_path(&path, &self.workspace).to_lowercase();
                let mut score = 0f32;
                let mut first_hit: Option<usize> = None;
                for w in &words {
                    if rel.contains(w.as_str()) {
                        score += 6.0;
                    }
                }
                if let Ok(meta) = e.metadata() {
                    if meta.len() > 0 && meta.len() <= self.cfg.max_file_bytes as u64 {
                        if let Ok(bytes) = std::fs::read(&path) {
                            if !bytes.iter().take(4000).any(|&b| b == 0) {
                                let text = String::from_utf8_lossy(&bytes).to_lowercase();
                                for w in &words {
                                    let n = text.matches(w.as_str()).count().min(8);
                                    score += n as f32 * 0.8;
                                }
                                if score > 0.0 {
                                    for (i, l) in text.lines().enumerate() {
                                        if words.iter().any(|w| l.contains(w.as_str())) {
                                            first_hit = Some(i);
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if score > 0.0 {
                    scored.push((score, path, first_hit));
                }
            }
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut chosen: Vec<(PathBuf, Option<usize>)> = Vec::new();
        for f in &self.focus {
            let p = platform::canonicalize_lossy(&self.workspace.join(f));
            if p.is_file() && !chosen.iter().any(|(c, _)| *c == p) {
                chosen.push((p, None));
            }
        }
        for (_, p, hit) in scored {
            if chosen.len() >= max_files + self.focus.len() {
                break;
            }
            if !chosen.iter().any(|(c, _)| *c == p) {
                chosen.push((p, hit));
            }
        }

        let mut out = Retrieved::default();
        for (p, hit) in chosen {
            let Ok(text) = std::fs::read_to_string(&p) else { continue };
            let lines: Vec<&str> = text.lines().collect();
            if lines.is_empty() {
                continue;
            }
            let (from, to) = match hit {
                Some(h) => {
                    let from = h.saturating_sub(snippet / 4);
                    (from, (from + snippet).min(lines.len()))
                }
                None => (0, snippet.min(lines.len())),
            };
            let rel = platform::display_path(&p, &self.workspace);
            out.text.push_str(&format!("### {rel} (lines {}-{} of {})\n", from + 1, to, lines.len()));
            for (i, l) in lines[from..to].iter().enumerate() {
                let shown: String = if l.chars().count() > 200 { format!("{}…", l.chars().take(200).collect::<String>()) } else { l.to_string() };
                out.text.push_str(&format!("{:>5}| {}\n", from + i + 1, shown));
            }
            out.files.push(format!("{rel} (lines {}-{})", from + 1, to));
        }
        self.last_files = out.files.clone();
        out
    }

    /// Initial context block for a task.
    pub fn task_context(&mut self, task: &str, git_summary: Option<&str>) -> String {
        let mut s = String::new();
        s.push_str("=== PROJECT CONTEXT ===\n");
        s.push_str(&self.project_snapshot());
        if let Some(g) = git_summary {
            s.push('\n');
            s.push_str(g);
        }
        let r = self.retrieve(task);
        if !r.text.is_empty() {
            s.push_str("\n=== POTENTIALLY RELEVANT FILE EXCERPTS (retrieved; use read_file for more) ===\n");
            s.push_str(&r.text);
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_estimates() {
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcdefgh"), 2);
        assert_eq!(estimate_tokens("한국어"), 3);
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn truncation_keeps_head_and_tail() {
        let s = format!("HEAD{}TAIL", "x".repeat(5000));
        let t = truncate_middle(&s, 300);
        assert!(t.starts_with("HEAD") && t.ends_with("TAIL") && t.contains("bytes omitted") && t.len() < 400);
        assert_eq!(truncate_middle("short", 300), "short");
        // multi-byte safety
        let k = "가".repeat(1000);
        let t = truncate_middle(&k, 301);
        assert!(t.contains("omitted"));
    }

    #[test]
    fn history_compacts_old_entries_and_keeps_recent() {
        let mut h = History::new();
        h.push("user", "task", "Build a REST API server in C++");
        for i in 0..30 {
            h.push("assistant", "model", format!("{{\"type\":\"tool_call\",\"tool\":\"read_file\",\"arguments\":{{\"path\":\"f{i}.cpp\"}}}}"));
            h.push("user", "tool", format!("f{i}.cpp\n{}", "line of code\n".repeat(200)));
        }
        let before = h.tokens();
        assert!(h.compact(4000, 0.85, 6));
        assert!(h.tokens() < before / 3, "{} -> {}", before, h.tokens());
        assert!(h.tokens() <= 4000);
        assert!(h.compactions() >= 1);
        let msgs = h.messages();
        assert!(msgs[0].content.contains("Earlier conversation") && msgs[0].content.contains("Build a REST API"));
        // the most recent entries survive verbatim
        assert!(h.entries().last().unwrap().text.contains("f29.cpp"));
        // untouched when small
        let mut small = History::new();
        small.push("user", "task", "hi");
        assert!(!small.compact(4000, 0.85, 6));
    }

    #[test]
    fn messages_merge_same_roles() {
        let mut h = History::new();
        h.push("user", "task", "a");
        h.push("user", "tool", "b");
        h.push("assistant", "model", "c");
        let m = h.messages();
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].content, "a\n\nb");
    }

    #[test]
    fn giant_single_entry_is_shrunk() {
        let mut h = History::new();
        h.push("user", "task", "x");
        h.push("user", "tool", "y".repeat(200_000));
        h.force_compact(2000, 6);
        assert!(h.tokens() <= 2000, "{}", h.tokens());
    }

    fn project() -> (tempfile::TempDir, ContextManager) {
        let dir = tempfile::tempdir().unwrap();
        let ws = dir.path();
        std::fs::create_dir_all(ws.join("src")).unwrap();
        std::fs::create_dir_all(ws.join("target/debug")).unwrap();
        std::fs::write(ws.join("Cargo.toml"), "[package]\nname = \"demo\"\n").unwrap();
        std::fs::write(ws.join("target/debug/junk.rs"), "parse_config junk").unwrap();
        let mut main = String::from("fn main() {\n");
        for i in 0..100 {
            main.push_str(&format!("    // filler {i}\n"));
        }
        main.push_str("    let cfg = parse_config(\"qwen.toml\");\n}\n");
        std::fs::write(ws.join("src/main.rs"), main).unwrap();
        std::fs::write(ws.join("src/config.rs"), "pub fn parse_config(p: &str) -> String { p.to_string() }\n").unwrap();
        std::fs::write(ws.join("src/other.rs"), "pub fn unrelated() {}\n").unwrap();
        std::fs::write(ws.join("blob.bin"), [0u8, 1, 2, 3]).unwrap();
        let cm = ContextManager::new(ContextSection::default(), ws);
        (dir, cm)
    }

    #[test]
    fn snapshot_lists_structure_and_manifests_but_not_build_output() {
        let (_d, cm) = project();
        let s = cm.project_snapshot();
        assert!(s.contains("src/") && s.contains("main.rs") && s.contains("name = \"demo\""));
        assert!(!s.contains("junk.rs") && !s.contains("debug/"));
    }

    #[test]
    fn retrieval_returns_snippets_not_whole_files() {
        let (_d, mut cm) = project();
        let r = cm.retrieve("fix the parse_config function");
        assert!(r.files.iter().any(|f| f.starts_with("src/config.rs")), "{:?}", r.files);
        let main = r.files.iter().find(|f| f.starts_with("src/main.rs")).expect("main.rs hit");
        assert!(!main.contains("lines 1-"), "snippet should be centred on the hit, got {main}");
        assert!(r.text.contains("parse_config(\"qwen.toml\")"));
        assert!(!r.files.iter().any(|f| f.contains("junk") || f.contains("other.rs")));
        assert!(r.text.lines().count() < 200);
    }

    #[test]
    fn focus_files_are_pinned() {
        let (_d, mut cm) = project();
        cm.focus = vec!["src/other.rs".into()];
        let r = cm.retrieve("anything about parse_config");
        assert!(r.files[0].starts_with("src/other.rs"));
    }

    #[test]
    fn korean_only_queries_do_not_crash() {
        let (_d, mut cm) = project();
        let r = cm.retrieve("컴파일 오류를 고쳐줘");
        assert!(r.files.is_empty());
        let c = cm.task_context("컴파일 오류를 고쳐줘", Some("git status:\n  clean"));
        assert!(c.contains("PROJECT CONTEXT") && c.contains("git status"));
    }

    #[test]
    fn budget_respects_the_model_window() {
        let (_d, cm) = project();
        assert_eq!(cm.budget(None, 4096), 24_000);
        assert_eq!(cm.budget(Some(8192), 2048), 8192 - 2048 - 256);
        assert_eq!(cm.budget(Some(1000), 900), 1024);
    }
}
