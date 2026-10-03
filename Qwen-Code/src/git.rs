//! Git integration. Git is invoked directly (no shell) through the platform process layer.

use crate::platform::{self, ProcessSpec};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct Git {
    root: PathBuf,
    max_output_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct GitResult {
    pub ok: bool,
    pub output: String,
}

impl Git {
    pub fn new(dir: &Path) -> Self {
        Self { root: dir.to_path_buf(), max_output_bytes: 60_000 }
    }

    pub fn with_limit(mut self, bytes: usize) -> Self {
        self.max_output_bytes = bytes;
        self
    }

    pub fn dir(&self) -> &Path {
        &self.root
    }

    pub fn is_available() -> bool {
        platform::which("git").is_some()
    }

    fn run(&self, args: &[&str], cancel: Option<&AtomicBool>) -> GitResult {
        if !Git::is_available() {
            return GitResult { ok: false, output: "git is not installed or not on PATH".into() };
        }
        let mut full: Vec<String> = vec!["-c".into(), "core.quotepath=false".into(), "-c".into(), "color.ui=never".into(), "--no-pager".into()];
        full.extend(args.iter().map(|s| s.to_string()));
        let mut spec = ProcessSpec::new("git", full);
        spec.cwd = Some(self.root.clone());
        spec.timeout = Duration::from_secs(60);
        spec.max_output_bytes = self.max_output_bytes;
        spec.env = vec![
            ("GIT_TERMINAL_PROMPT".into(), "0".into()),
            ("GIT_PAGER".into(), "cat".into()),
            ("LC_ALL".into(), "C.UTF-8".into()),
        ];
        match platform::run_process(&spec, cancel) {
            Ok(out) => {
                let ok = out.success();
                let mut text = out.stdout.trim_end().to_string();
                if !ok {
                    let err = out.stderr.trim();
                    if !err.is_empty() {
                        if !text.is_empty() {
                            text.push('\n');
                        }
                        text.push_str(err);
                    }
                    if out.timed_out {
                        text.push_str("\n(git timed out)");
                    }
                }
                GitResult { ok, output: text }
            }
            Err(e) => GitResult { ok: false, output: format!("cannot run git: {e}") },
        }
    }

    pub fn is_repo(&self) -> bool {
        let r = self.run(&["rev-parse", "--is-inside-work-tree"], None);
        r.ok && r.output.trim() == "true"
    }

    pub fn toplevel(&self) -> Option<PathBuf> {
        let r = self.run(&["rev-parse", "--show-toplevel"], None);
        if r.ok {
            Some(platform::strip_unc(PathBuf::from(r.output.trim())))
        } else {
            None
        }
    }

    pub fn current_branch(&self) -> Option<String> {
        let r = self.run(&["rev-parse", "--abbrev-ref", "HEAD"], None);
        if r.ok && !r.output.is_empty() {
            Some(r.output.trim().to_string())
        } else {
            None
        }
    }

    pub fn status(&self) -> GitResult {
        let r = self.run(&["status", "--short", "--branch", "--untracked-files=all"], None);
        if r.ok && r.output.lines().count() <= 1 {
            return GitResult { ok: true, output: format!("{}\n(working tree clean)", r.output) };
        }
        r
    }

    pub fn diff(&self, staged: bool, path: Option<&str>, stat_only: bool) -> GitResult {
        let mut args: Vec<&str> = vec!["diff"];
        if staged {
            args.push("--cached");
        }
        if stat_only {
            args.push("--stat");
        }
        if let Some(p) = path {
            args.push("--");
            args.push(p);
        }
        let r = self.run(&args, None);
        if r.ok && r.output.is_empty() {
            return GitResult { ok: true, output: "(no differences)".into() };
        }
        r
    }

    pub fn log(&self, max_count: usize, path: Option<&str>) -> GitResult {
        let n = format!("--max-count={}", max_count.clamp(1, 200));
        let mut args: Vec<&str> = vec!["log", &n, "--date=short", "--pretty=format:%h %ad %an  %s"];
        if let Some(p) = path {
            args.push("--");
            args.push(p);
        }
        let r = self.run(&args, None);
        if !r.ok && r.output.contains("does not have any commits") {
            return GitResult { ok: true, output: "(no commits yet)".into() };
        }
        r
    }

    pub fn branches(&self) -> GitResult {
        let r = self.run(&["branch", "--all", "--verbose", "--no-color"], None);
        if r.ok && r.output.is_empty() {
            let cur = self.current_branch().unwrap_or_else(|| "(none)".into());
            return GitResult { ok: true, output: format!("* {cur} (no commits yet)") };
        }
        r
    }

    pub fn valid_branch_name(name: &str) -> bool {
        !name.is_empty()
            && !name.starts_with('-')
            && !name.starts_with('/')
            && !name.ends_with('/')
            && !name.ends_with('.')
            && !name.contains("..")
            && !name.contains("//")
            && name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '/' | '-'))
    }

    pub fn create_branch(&self, name: &str, switch: bool) -> GitResult {
        if !Git::valid_branch_name(name) {
            return GitResult { ok: false, output: format!("invalid branch name \"{name}\"") };
        }
        if switch {
            self.run(&["switch", "-c", name], None)
        } else {
            self.run(&["branch", name], None)
        }
    }

    pub fn switch_branch(&self, name: &str) -> GitResult {
        if !Git::valid_branch_name(name) {
            return GitResult { ok: false, output: format!("invalid branch name \"{name}\"") };
        }
        self.run(&["switch", name], None)
    }

    /// Compact summary for the model context: branch, short status, diff stat, recent commits.
    pub fn summary(&self) -> Option<String> {
        if !Git::is_available() || !self.is_repo() {
            return None;
        }
        let mut out = String::new();
        let status = self.run(&["status", "--short", "--branch"], None);
        out.push_str("git status:\n");
        let lines: Vec<&str> = status.output.lines().collect();
        for l in lines.iter().take(40) {
            out.push_str(&format!("  {l}\n"));
        }
        if lines.len() > 40 {
            out.push_str(&format!("  … {} more entries\n", lines.len() - 40));
        }
        if lines.len() <= 1 {
            out.push_str("  (working tree clean)\n");
        }
        let stat = self.run(&["diff", "--stat", "HEAD"], None);
        if stat.ok && !stat.output.is_empty() {
            out.push_str("git diff --stat HEAD:\n");
            for l in stat.output.lines().take(25) {
                out.push_str(&format!("  {l}\n"));
            }
        }
        let log = self.log(5, None);
        if log.ok && !log.output.is_empty() {
            out.push_str("recent commits:\n");
            for l in log.output.lines() {
                out.push_str(&format!("  {l}\n"));
            }
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init_repo(dir: &Path) -> Git {
        let g = Git::new(dir);
        for args in [
            vec!["init", "-q"],
            vec!["config", "user.email", "t@example.com"],
            vec!["config", "user.name", "Tester"],
            vec!["config", "commit.gpgsign", "false"],
        ] {
            assert!(g.run(&args, None).ok, "{args:?}");
        }
        g
    }

    #[test]
    fn full_git_flow() {
        if !Git::is_available() {
            eprintln!("git not available; skipping");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let g = init_repo(dir.path());
        assert!(g.is_repo());
        std::fs::write(dir.path().join("a.txt"), "one\n").unwrap();
        assert!(g.status().output.contains("a.txt"));
        assert!(g.run(&["add", "."], None).ok);
        assert!(g.run(&["commit", "-q", "-m", "first commit"], None).ok);
        assert!(g.log(5, None).output.contains("first commit"));
        std::fs::write(dir.path().join("a.txt"), "one\ntwo\n").unwrap();
        let d = g.diff(false, None, false);
        assert!(d.output.contains("+two"));
        assert!(g.diff(false, Some("a.txt"), true).output.contains("a.txt"));
        assert!(g.diff(true, None, false).output.contains("no differences"));
        assert!(g.create_branch("feature/x", true).ok);
        assert_eq!(g.current_branch().as_deref(), Some("feature/x"));
        assert!(g.branches().output.contains("feature/x"));
        assert!(g.switch_branch("-bad").output.contains("invalid"));
        let s = g.summary().unwrap();
        assert!(s.contains("git status") && s.contains("first commit"));
    }

    #[test]
    fn non_repo_is_handled() {
        let dir = tempfile::tempdir().unwrap();
        let g = Git::new(dir.path());
        assert!(!g.is_repo());
        assert!(g.summary().is_none());
        assert!(!g.status().ok);
    }

    #[test]
    fn branch_name_validation() {
        for bad in ["", "-x", "a..b", "a b", "a;b", "/a", "a/", "x$(y)"] {
            assert!(!Git::valid_branch_name(bad), "{bad}");
        }
        for good in ["main", "feature/new-thing", "v1.2.3", "fix_1"] {
            assert!(Git::valid_branch_name(good), "{good}");
        }
    }
}
