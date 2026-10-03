//! Tool system.
//!
//! Every tool call goes through the same pipeline:
//!
//! ```text
//! Model JSON -> (already parsed) -> Schema Validation -> Permission Check -> Execution -> Tool Result
//! ```
//!
//! Model text is never executed directly. `run_command` receives a command *string* only after it
//! has been validated, parsed by the command parser and cleared by the permission policy.

use crate::config::Config;
use crate::fileedit::{self, numbered_excerpt, sha256_hex, EditOp, EditRequest, FileTracker};
use crate::git::Git;
use crate::permissions::{self, Category, PermissionManager, PermissionRequest};
use crate::platform::{self, ProcessSpec};
use crate::schema;
use crate::ui::Ui;
use regex::RegexBuilder;
use serde::Deserialize;
use serde_json::{json, Value};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Duration;

// ---------------------------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyKind {
    Build,
    Test,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Completed,
    UnknownTool,
    Schema,
    Permission,
    Execution,
}

#[derive(Debug, Clone)]
pub struct ToolOutcome {
    pub ok: bool,
    pub stage: Stage,
    /// Text returned to the model.
    pub output: String,
    /// One-line summary for the UI.
    pub summary: String,
    pub changed_files: Vec<String>,
    /// Set for build/test commands: kind and whether it succeeded.
    pub verify: Option<(VerifyKind, bool)>,
}

impl ToolOutcome {
    fn fail(stage: Stage, msg: impl Into<String>) -> Self {
        let m = msg.into();
        Self { ok: false, stage, summary: m.lines().next().unwrap_or("").to_string(), output: m, changed_files: vec![], verify: None }
    }
    fn exec_fail(msg: impl Into<String>) -> Self {
        Self::fail(Stage::Execution, msg)
    }
    fn success(output: impl Into<String>, summary: impl Into<String>) -> Self {
        Self { ok: true, stage: Stage::Completed, output: output.into(), summary: summary.into(), changed_files: vec![], verify: None }
    }
}

#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: &'static str,
    pub description: &'static str,
    pub schema: Value,
    /// Read-only tools may also be used by the Thinking model.
    pub read_only: bool,
}

pub struct ToolEnv {
    /// Canonical workspace root.
    pub workspace: PathBuf,
    pub cfg: Arc<Config>,
    pub tracker: Mutex<FileTracker>,
    pub cancel: Arc<AtomicBool>,
    pub git: Git,
}

pub struct ToolRegistry {
    env: Arc<ToolEnv>,
    specs: Vec<ToolSpec>,
}

// ---------------------------------------------------------------------------------------------
// Argument structs
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ReadFileArgs {
    path: String,
    #[serde(default)]
    start_line: Option<usize>,
    #[serde(default)]
    end_line: Option<usize>,
    #[serde(default)]
    max_lines: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct WriteFileArgs {
    path: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct EditFileArgs {
    path: String,
    operations: Vec<EditOp>,
    #[serde(default)]
    expected_hash: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DeleteFileArgs {
    path: String,
    #[serde(default)]
    recursive: bool,
}

#[derive(Debug, Deserialize)]
struct ListDirArgs {
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    depth: Option<usize>,
    #[serde(default)]
    include_hidden: bool,
    #[serde(default)]
    include_ignored: bool,
}

#[derive(Debug, Deserialize)]
struct SearchArgs {
    pattern: String,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    glob: Option<String>,
    #[serde(default)]
    regex: bool,
    #[serde(default)]
    case_sensitive: Option<bool>,
    #[serde(default)]
    context: Option<usize>,
    #[serde(default)]
    max_results: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct RunCommandArgs {
    command: String,
    #[serde(default)]
    cwd: Option<String>,
    #[serde(default)]
    timeout_secs: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct RunTestArgs {
    #[serde(default)]
    command: Option<String>,
    #[serde(default)]
    filter: Option<String>,
    #[serde(default)]
    timeout_secs: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct GitDiffArgs {
    #[serde(default)]
    staged: bool,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    stat: bool,
}

#[derive(Debug, Deserialize)]
struct GitLogArgs {
    #[serde(default)]
    max_count: Option<usize>,
    #[serde(default)]
    path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GitBranchArgs {
    #[serde(default)]
    action: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

enum Call {
    ReadFile(ReadFileArgs),
    WriteFile(WriteFileArgs),
    EditFile(EditFileArgs),
    DeleteFile(DeleteFileArgs),
    ListDirectory(ListDirArgs),
    SearchFiles(SearchArgs),
    RunCommand(RunCommandArgs),
    RunTest(RunTestArgs),
    GitStatus,
    GitDiff(GitDiffArgs),
    GitLog(GitLogArgs),
    GitBranch(GitBranchArgs),
}

fn parse_call(name: &str, args: &Value) -> Result<Call, String> {
    fn de<T: serde::de::DeserializeOwned>(v: &Value) -> Result<T, String> {
        serde_json::from_value(v.clone()).map_err(|e| format!("cannot decode arguments: {e}"))
    }
    Ok(match name {
        "read_file" => Call::ReadFile(de(args)?),
        "write_file" => Call::WriteFile(de(args)?),
        "edit_file" => Call::EditFile(de(args)?),
        "delete_file" => Call::DeleteFile(de(args)?),
        "list_directory" => Call::ListDirectory(de(args)?),
        "search_files" => Call::SearchFiles(de(args)?),
        "run_command" => Call::RunCommand(de(args)?),
        "run_test" => Call::RunTest(de(args)?),
        "git_status" => Call::GitStatus,
        "git_diff" => Call::GitDiff(de(args)?),
        "git_log" => Call::GitLog(de(args)?),
        "git_branch" => Call::GitBranch(de(args)?),
        other => return Err(format!("unknown tool \"{other}\"")),
    })
}

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

struct Resolved {
    abs: PathBuf,
    inside: bool,
}

fn human_size(n: u64) -> String {
    if n < 1024 {
        format!("{n} B")
    } else if n < 1024 * 1024 {
        format!("{:.1} KB", n as f64 / 1024.0)
    } else {
        format!("{:.1} MB", n as f64 / 1048576.0)
    }
}

fn looks_binary(bytes: &[u8]) -> bool {
    bytes.iter().take(8000).any(|&b| b == 0)
}

fn short_hash(h: &str) -> &str {
    &h[..16.min(h.len())]
}

/// Classifies build and test commands so the agent can tell whether the work was verified.
pub fn classify_verification(command: &str) -> Option<VerifyKind> {
    let c = command.to_lowercase();
    let has = |s: &str| c.contains(s);
    let test_markers = [
        "cargo test", "ctest", "pytest", "go test", "npm test", "npm run test", "yarn test", "pnpm test", "dotnet test", "make test", "make check", "mvn test",
        "gradle test", "gradlew test", "jest", "vitest", "unittest", "cmake --build . --target test", "nextest",
    ];
    if test_markers.iter().any(|m| has(m)) {
        return Some(VerifyKind::Test);
    }
    let build_markers = [
        "cargo build", "cargo check", "cargo run", "cargo clippy", "cmake --build", "ninja", "go build", "go vet", "npm run build", "yarn build", "pnpm build",
        "tsc", "dotnet build", "gradle build", "gradlew build", "mvn compile", "mvn package", "mvn install", "javac", "py_compile", "make", "msbuild", "gcc ",
        "g++ ", "clang ", "clang++ ", "cc ", "rustc ",
    ];
    if build_markers.iter().any(|m| has(m)) {
        return Some(VerifyKind::Build);
    }
    None
}

fn detect_test_command(ws: &Path) -> Option<String> {
    if ws.join("Cargo.toml").is_file() {
        return Some("cargo test".into());
    }
    if let Ok(pkg) = std::fs::read_to_string(ws.join("package.json")) {
        if pkg.contains("\"test\"") {
            return Some("npm test".into());
        }
    }
    if ws.join("go.mod").is_file() {
        return Some("go test ./...".into());
    }
    let py = if platform::is_windows() { "python" } else { "python3" };
    if ws.join("pytest.ini").is_file() || ws.join("tox.ini").is_file() || ws.join("conftest.py").is_file() || ws.join("pyproject.toml").is_file() {
        return Some(format!("{py} -m pytest"));
    }
    if ws.join("CMakeLists.txt").is_file() {
        for b in ["build", "cmake-build-debug", "out/build"] {
            if ws.join(b).join("CTestTestfile.cmake").is_file() {
                return Some(format!("ctest --test-dir {b} --output-on-failure"));
            }
        }
    }
    if let Ok(mk) = std::fs::read_to_string(ws.join("Makefile")) {
        if mk.lines().any(|l| l.starts_with("test:") || l.starts_with("check:")) {
            return Some(if mk.lines().any(|l| l.starts_with("test:")) { "make test" } else { "make check" }.into());
        }
    }
    if let Ok(rd) = std::fs::read_dir(ws) {
        for e in rd.flatten() {
            let n = e.file_name().to_string_lossy().to_string();
            if n.ends_with(".sln") || n.ends_with(".csproj") {
                return Some("dotnet test".into());
            }
        }
    }
    None
}

fn truncate_middle(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let head = max / 2;
    let tail = max - head;
    let mut h = head;
    while !s.is_char_boundary(h) {
        h -= 1;
    }
    let mut t = s.len() - tail;
    while !s.is_char_boundary(t) {
        t += 1;
    }
    format!("{}\n... [{} bytes omitted] ...\n{}", &s[..h], t - h, &s[t..])
}

// ---------------------------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------------------------

fn spec(name: &'static str, description: &'static str, read_only: bool, schema: Value) -> ToolSpec {
    ToolSpec { name, description, schema, read_only }
}

impl ToolRegistry {
    pub fn new(workspace: &Path, cfg: Arc<Config>, cancel: Arc<AtomicBool>) -> Self {
        let ws = platform::canonicalize_lossy(workspace);
        let git = Git::new(&ws).with_limit(cfg.tools.max_output_bytes);
        let env = Arc::new(ToolEnv { workspace: ws, cfg, tracker: Mutex::new(FileTracker::default()), cancel, git });
        Self { env, specs: build_specs() }
    }

    pub fn env(&self) -> &Arc<ToolEnv> {
        &self.env
    }

    pub fn specs(&self) -> &[ToolSpec] {
        &self.specs
    }

    pub fn spec(&self, name: &str) -> Option<&ToolSpec> {
        self.specs.iter().find(|s| s.name == name)
    }

    pub fn is_read_only(&self, name: &str) -> bool {
        self.spec(name).is_some_and(|s| s.read_only)
    }

    /// Forgets which files were read (used by `/reset`).
    pub fn reset(&self) {
        self.env.tracker.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }

    /// Tool documentation embedded into the system prompts.
    pub fn prompt_description(&self, read_only_only: bool) -> String {
        let mut out = String::new();
        for s in self.specs.iter().filter(|s| !read_only_only || s.read_only) {
            out.push_str(&format!("- {}: {}\n  arguments schema: {}\n", s.name, s.description, s.schema));
        }
        out
    }

    /// Step 2 of the pipeline: schema validation.
    pub fn validate_args(&self, name: &str, args: &Value) -> Result<(), String> {
        let spec = self.spec(name).ok_or_else(|| format!("unknown tool \"{name}\". Available tools: {}", self.specs.iter().map(|s| s.name).collect::<Vec<_>>().join(", ")))?;
        schema::validate(args, &spec.schema).map_err(|errs| format!("invalid arguments for {name}:\n- {}", errs.join("\n- ")))
    }

    /// Runs the full pipeline for one tool call.
    pub fn invoke(&self, name: &str, args: &Value, perms: &PermissionManager, ui: &dyn Ui) -> ToolOutcome {
        // 1. tool lookup + schema validation
        if self.spec(name).is_none() {
            return ToolOutcome::fail(Stage::UnknownTool, format!("unknown tool \"{name}\". Available tools: {}", self.specs.iter().map(|s| s.name).collect::<Vec<_>>().join(", ")));
        }
        let args = if args.is_null() { json!({}) } else { args.clone() };
        if let Err(e) = self.validate_args(name, &args) {
            return ToolOutcome::fail(Stage::Schema, e);
        }
        let call = match parse_call(name, &args) {
            Ok(c) => c,
            Err(e) => return ToolOutcome::fail(Stage::Schema, e),
        };
        // 2. permission check
        let reqs = match self.requests(&call, perms) {
            Ok(r) => r,
            Err(e) => return ToolOutcome::exec_fail(e),
        };
        if let Err(denied) = perms.authorize(&reqs, ui) {
            let who = if denied.by_user { "denied by the user" } else { "blocked by the permission policy" };
            return ToolOutcome::fail(Stage::Permission, format!("Permission denied ({who}): {}\nThe action was NOT executed. Choose a different approach; do not retry the same action.", denied.message));
        }
        // 3. execution (panics are contained)
        match catch_unwind(AssertUnwindSafe(|| self.execute(&call))) {
            Ok(o) => o,
            Err(_) => ToolOutcome::exec_fail(format!("internal error while running {name}; the tool panicked and was aborted")),
        }
    }

    // -----------------------------------------------------------------------------------------
    // Permission requests
    // -----------------------------------------------------------------------------------------

    fn resolve(&self, raw: &str) -> Result<Resolved, String> {
        let raw = raw.trim();
        if raw.is_empty() {
            return Err("path must not be empty".into());
        }
        if raw.contains('\0') {
            return Err("path contains a NUL byte".into());
        }
        let expanded: PathBuf = if raw == "~" || raw.starts_with("~/") || raw.starts_with("~\\") {
            match platform::home_dir() {
                Some(h) => h.join(raw.trim_start_matches('~').trim_start_matches(['/', '\\'])),
                None => PathBuf::from(raw),
            }
        } else {
            PathBuf::from(raw)
        };
        let abs = if expanded.is_absolute() { expanded } else { self.env.workspace.join(expanded) };
        let canon = platform::canonicalize_lossy(&abs);
        let inside = canon.starts_with(&self.env.workspace);
        Ok(Resolved { abs: canon, inside })
    }

    fn rel(&self, p: &Path) -> String {
        platform::display_path(p, &self.env.workspace)
    }

    fn path_requests(&self, cat: Category, verb: &str, raw: &str, check_risk: bool) -> Result<Vec<PermissionRequest>, String> {
        let r = self.resolve(raw)?;
        let desc = format!("{verb} {}", if r.inside { self.rel(&r.abs) } else { r.abs.display().to_string() });
        let mut main = PermissionRequest::simple(cat, desc.clone());
        if check_risk {
            if let Some((risk, why)) = permissions::path_risk(&r.abs, &self.env.workspace) {
                main = main.with_risk(risk, why);
            }
        }
        let mut v = vec![main];
        if !r.inside {
            let mut fs = PermissionRequest::simple(Category::Filesystem, desc);
            fs.reason = Some("path is outside the workspace".into());
            v.push(fs);
        }
        Ok(v)
    }

    fn requests(&self, call: &Call, perms: &PermissionManager) -> Result<Vec<PermissionRequest>, String> {
        match call {
            Call::ReadFile(a) => self.path_requests(Category::Read, "read_file", &a.path, false),
            Call::ListDirectory(a) => self.path_requests(Category::Read, "list_directory", a.path.as_deref().unwrap_or("."), false),
            Call::SearchFiles(a) => self.path_requests(Category::Read, "search_files", a.path.as_deref().unwrap_or("."), false),
            Call::WriteFile(a) => self.path_requests(Category::Write, &format!("write_file ({} bytes)", a.content.len()), &a.path, true),
            Call::EditFile(a) => self.path_requests(Category::Edit, "edit_file", &a.path, true),
            Call::DeleteFile(a) => {
                let verb = if a.recursive { "delete_file --recursive" } else { "delete_file" };
                self.path_requests(Category::Delete, verb, &a.path, true)
            }
            Call::RunCommand(a) => {
                let cwd = self.command_cwd(a.cwd.as_deref())?;
                let mut reqs = perms.requests_for_command(&a.command, &cwd.abs, &self.env.workspace);
                if !cwd.inside {
                    let mut fs = PermissionRequest::simple(Category::Filesystem, a.command.clone());
                    fs.reason = Some("working directory is outside the workspace".into());
                    reqs.push(fs);
                }
                Ok(reqs)
            }
            Call::RunTest(a) => {
                let command = self.test_command(a)?;
                Ok(perms.requests_for_command(&command, &self.env.workspace, &self.env.workspace))
            }
            Call::GitStatus => Ok(vec![PermissionRequest::simple(Category::Git, "git status")]),
            Call::GitDiff(a) => {
                if let Some(p) = &a.path {
                    self.resolve(p)?;
                }
                Ok(vec![PermissionRequest::simple(Category::Git, "git diff")])
            }
            Call::GitLog(_) => Ok(vec![PermissionRequest::simple(Category::Git, "git log")]),
            Call::GitBranch(a) => {
                let action = a.action.as_deref().unwrap_or("list");
                Ok(vec![PermissionRequest::simple(Category::Git, format!("git branch {action} {}", a.name.as_deref().unwrap_or("")).trim().to_string())])
            }
        }
    }

    fn command_cwd(&self, cwd: Option<&str>) -> Result<Resolved, String> {
        match cwd {
            None => Ok(Resolved { abs: self.env.workspace.clone(), inside: true }),
            Some(c) => {
                let r = self.resolve(c)?;
                if !r.abs.is_dir() {
                    return Err(format!("working directory {} does not exist", self.rel(&r.abs)));
                }
                Ok(r)
            }
        }
    }

    fn test_command(&self, a: &RunTestArgs) -> Result<String, String> {
        if let Some(c) = a.command.as_deref().filter(|c| !c.trim().is_empty()) {
            return Ok(c.to_string());
        }
        let mut cmd = detect_test_command(&self.env.workspace)
            .ok_or_else(|| "cannot detect how to run tests in this project; pass the \"command\" argument".to_string())?;
        if let Some(f) = a.filter.as_deref().filter(|f| !f.trim().is_empty()) {
            if !f.chars().all(|c| c.is_alphanumeric() || matches!(c, '_' | ':' | '.' | '-' | '/' | ' ')) {
                return Err("filter may only contain letters, digits and _ : . - /".into());
            }
            if cmd.starts_with("cargo test") {
                cmd = format!("cargo test {f}");
            } else if cmd.contains("pytest") {
                cmd = format!("{cmd} -k \"{f}\"");
            } else if cmd.starts_with("go test") {
                cmd = format!("go test ./... -run \"{f}\"");
            }
        }
        Ok(cmd)
    }

    // -----------------------------------------------------------------------------------------
    // Execution
    // -----------------------------------------------------------------------------------------

    fn execute(&self, call: &Call) -> ToolOutcome {
        match call {
            Call::ReadFile(a) => self.read_file(a),
            Call::WriteFile(a) => self.write_file(a),
            Call::EditFile(a) => self.edit_file(a),
            Call::DeleteFile(a) => self.delete_file(a),
            Call::ListDirectory(a) => self.list_directory(a),
            Call::SearchFiles(a) => self.search_files(a),
            Call::RunCommand(a) => {
                let cwd = match self.command_cwd(a.cwd.as_deref()) {
                    Ok(c) => c.abs,
                    Err(e) => return ToolOutcome::exec_fail(e),
                };
                let timeout = a.timeout_secs.unwrap_or(self.env.cfg.tools.command_timeout_secs).clamp(1, 3600);
                self.exec_command(&a.command, &cwd, Duration::from_secs(timeout))
            }
            Call::RunTest(a) => {
                let command = match self.test_command(a) {
                    Ok(c) => c,
                    Err(e) => return ToolOutcome::exec_fail(e),
                };
                let timeout = a.timeout_secs.unwrap_or(self.env.cfg.tools.test_timeout_secs).clamp(1, 7200);
                let mut o = self.exec_command(&command, &self.env.workspace.clone(), Duration::from_secs(timeout));
                o.output = format!("TEST RESULT: {}\n{}", if o.ok { "PASSED" } else { "FAILED" }, o.output);
                o.verify = Some((VerifyKind::Test, o.ok));
                o
            }
            Call::GitStatus => self.git_result(self.env.git.status(), "git status"),
            Call::GitDiff(a) => {
                let path = match a.path.as_deref().map(|p| self.resolve(p)) {
                    Some(Ok(r)) => Some(self.rel(&r.abs)),
                    Some(Err(e)) => return ToolOutcome::exec_fail(e),
                    None => None,
                };
                self.git_result(self.env.git.diff(a.staged, path.as_deref(), a.stat), "git diff")
            }
            Call::GitLog(a) => {
                let path = match a.path.as_deref().map(|p| self.resolve(p)) {
                    Some(Ok(r)) => Some(self.rel(&r.abs)),
                    Some(Err(e)) => return ToolOutcome::exec_fail(e),
                    None => None,
                };
                self.git_result(self.env.git.log(a.max_count.unwrap_or(20), path.as_deref()), "git log")
            }
            Call::GitBranch(a) => {
                let action = a.action.as_deref().unwrap_or("list");
                match action {
                    "list" => self.git_result(self.env.git.branches(), "git branch"),
                    "create" | "switch" => {
                        let Some(name) = a.name.as_deref() else {
                            return ToolOutcome::exec_fail(format!("git_branch {action} needs a \"name\""));
                        };
                        let r = if action == "create" { self.env.git.create_branch(name, false) } else { self.env.git.switch_branch(name) };
                        self.git_result(r, &format!("git branch {action} {name}"))
                    }
                    other => ToolOutcome::exec_fail(format!("unknown git_branch action \"{other}\" (list, create, switch)")),
                }
            }
        }
    }

    fn git_result(&self, r: crate::git::GitResult, label: &str) -> ToolOutcome {
        let text = truncate_middle(&r.output, self.env.cfg.tools.max_output_bytes);
        if r.ok {
            let first = text.lines().next().unwrap_or("").to_string();
            ToolOutcome::success(text, format!("{label}: {first}"))
        } else {
            ToolOutcome::exec_fail(format!("{label} failed: {text}"))
        }
    }

    fn track(&self, path: &Path, bytes: &[u8]) {
        self.env.tracker.lock().unwrap_or_else(|e| e.into_inner()).record(path, bytes);
    }

    fn read_file(&self, a: &ReadFileArgs) -> ToolOutcome {
        let r = match self.resolve(&a.path) {
            Ok(r) => r,
            Err(e) => return ToolOutcome::exec_fail(e),
        };
        let name = self.rel(&r.abs);
        let meta = match std::fs::metadata(&r.abs) {
            Ok(m) => m,
            Err(_) => return ToolOutcome::exec_fail(format!("file not found: {name}. Use list_directory or search_files (mode \"name\") to find the right path.")),
        };
        if meta.is_dir() {
            return ToolOutcome::exec_fail(format!("{name} is a directory; use list_directory"));
        }
        if meta.len() > 8 * 1024 * 1024 {
            return ToolOutcome::exec_fail(format!("{name} is too large to read ({}); use search_files", human_size(meta.len())));
        }
        let bytes = match std::fs::read(&r.abs) {
            Ok(b) => b,
            Err(e) => return ToolOutcome::exec_fail(format!("cannot read {name}: {e}")),
        };
        if looks_binary(&bytes) {
            return ToolOutcome::exec_fail(format!("{name} looks like a binary file ({})", human_size(meta.len())));
        }
        let text = match String::from_utf8(bytes.clone()) {
            Ok(t) => t,
            Err(_) => return ToolOutcome::exec_fail(format!("{name} is not valid UTF-8 text")),
        };
        self.track(&r.abs, &bytes);
        let hash = sha256_hex(&bytes);
        let lines: Vec<&str> = text.lines().collect();
        let total = lines.len();
        let crlf = if text.contains("\r\n") { ", CRLF line endings" } else { "" };
        if total == 0 {
            return ToolOutcome::success(format!("{name} (empty file, sha256 {})", short_hash(&hash)), format!("read {name} (empty)"));
        }
        let start = a.start_line.unwrap_or(1).max(1);
        if start > total {
            return ToolOutcome::exec_fail(format!("{name} has only {total} lines (start_line {start})"));
        }
        let cap = self.env.cfg.tools.max_read_lines.max(10);
        let want = a.max_lines.unwrap_or(cap).min(cap);
        let mut end = a.end_line.unwrap_or(total).min(total);
        if end < start {
            return ToolOutcome::exec_fail(format!("end_line {end} is before start_line {start}"));
        }
        let mut truncated = false;
        if end - start + 1 > want {
            end = start + want - 1;
            truncated = true;
        }
        let mut out = format!("{name} ({total} lines, {}, sha256 {}{crlf})\n", human_size(meta.len()), short_hash(&hash));
        for (i, l) in lines[start - 1..end].iter().enumerate() {
            out.push_str(&format!("{:>5}| {}\n", start + i, l));
        }
        if truncated || end < total {
            out.push_str(&format!("[showing lines {start}-{end} of {total}; use start_line={} to continue]\n", end + 1));
        }
        ToolOutcome::success(out, format!("read {name} lines {start}-{end} of {total}"))
    }

    fn write_file(&self, a: &WriteFileArgs) -> ToolOutcome {
        let r = match self.resolve(&a.path) {
            Ok(r) => r,
            Err(e) => return ToolOutcome::exec_fail(e),
        };
        let name = self.rel(&r.abs);
        if r.abs.is_dir() {
            return ToolOutcome::exec_fail(format!("{name} is a directory"));
        }
        let exists = r.abs.exists();
        let known = self.env.tracker.lock().unwrap_or_else(|e| e.into_inner()).get(&r.abs).is_some();
        if exists && !known {
            return ToolOutcome::exec_fail(format!(
                "{name} already exists and has not been read in this session, so the current state is unknown. Read it with read_file first (or modify it with edit_file). Nothing was written."
            ));
        }
        let ops = [EditOp::Create { content: a.content.clone(), overwrite: true }];
        let mut tracker = self.env.tracker.lock().unwrap_or_else(|e| e.into_inner());
        match fileedit::apply_edit(&EditRequest { path: &r.abs, ops: &ops, expected_hash: None }, &mut tracker) {
            Ok(rep) => {
                let lines = a.content.lines().count();
                let mut o = if rep.created {
                    ToolOutcome::success(format!("created {name} ({lines} lines, {} bytes, sha256 {})", rep.bytes_after, short_hash(&rep.sha256)), format!("created {name} ({lines} lines)"))
                } else {
                    ToolOutcome::success(
                        format!("overwrote {name} (+{} -{} lines, sha256 {})\n{}", rep.lines_added, rep.lines_removed, short_hash(&rep.sha256), rep.diff),
                        format!("overwrote {name} (+{} -{})", rep.lines_added, rep.lines_removed),
                    )
                };
                o.changed_files.push(name);
                o
            }
            Err(e) => ToolOutcome::exec_fail(e.to_string()),
        }
    }

    fn edit_file(&self, a: &EditFileArgs) -> ToolOutcome {
        let r = match self.resolve(&a.path) {
            Ok(r) => r,
            Err(e) => return ToolOutcome::exec_fail(e),
        };
        let name = self.rel(&r.abs);
        let mut tracker = self.env.tracker.lock().unwrap_or_else(|e| e.into_inner());
        match fileedit::apply_edit(&EditRequest { path: &r.abs, ops: &a.operations, expected_hash: a.expected_hash.as_deref() }, &mut tracker) {
            Ok(rep) => {
                let verb = if rep.created { "created" } else { "edited" };
                let mut o = ToolOutcome::success(
                    format!("{verb} {name}: {} operation(s), +{} -{} lines, sha256 {}\n{}", rep.ops_applied, rep.lines_added, rep.lines_removed, short_hash(&rep.sha256), rep.diff),
                    format!("{verb} {name} (+{} -{})", rep.lines_added, rep.lines_removed),
                );
                o.changed_files.push(name);
                o
            }
            Err(e) => ToolOutcome::exec_fail(format!("edit of {name} failed: {e}")),
        }
    }

    fn delete_file(&self, a: &DeleteFileArgs) -> ToolOutcome {
        let r = match self.resolve(&a.path) {
            Ok(r) => r,
            Err(e) => return ToolOutcome::exec_fail(e),
        };
        let name = self.rel(&r.abs);
        let meta = match std::fs::symlink_metadata(&r.abs) {
            Ok(m) => m,
            Err(_) => return ToolOutcome::exec_fail(format!("{name} does not exist")),
        };
        let result = if meta.is_dir() && !meta.file_type().is_symlink() {
            if a.recursive {
                std::fs::remove_dir_all(&r.abs)
            } else {
                std::fs::remove_dir(&r.abs)
            }
        } else {
            std::fs::remove_file(&r.abs)
        };
        match result {
            Ok(()) => {
                self.env.tracker.lock().unwrap_or_else(|e| e.into_inner()).forget(&r.abs);
                let mut o = ToolOutcome::success(format!("deleted {name}"), format!("deleted {name}"));
                o.changed_files.push(name);
                o
            }
            Err(e) => {
                let hint = if meta.is_dir() && !a.recursive { " (directory is not empty? set \"recursive\": true)" } else { "" };
                ToolOutcome::exec_fail(format!("cannot delete {name}: {e}{hint}"))
            }
        }
    }

    fn walker(&self, root: &Path, depth: Option<usize>, hidden: bool, ignored: bool) -> ignore::Walk {
        let mut b = ignore::WalkBuilder::new(root);
        b.hidden(!hidden)
            .git_ignore(!ignored)
            .git_global(false)
            .git_exclude(!ignored)
            .ignore(!ignored)
            .parents(!ignored)
            .require_git(false)
            .follow_links(false)
            .max_depth(depth)
            .sort_by_file_path(|a, b| a.cmp(b));
        let skip_build_dirs = !ignored;
        b.filter_entry(move |e| {
            let name = e.file_name().to_string_lossy();
            if name == ".git" {
                return false;
            }
            if skip_build_dirs && e.file_type().is_some_and(|t| t.is_dir()) {
                if matches!(name.as_ref(), "node_modules" | "__pycache__" | ".venv") {
                    return false;
                }
                if name == "target" && e.path().parent().is_some_and(|p| p.join("Cargo.toml").is_file()) {
                    return false;
                }
            }
            true
        });
        b.build()
    }

    fn list_directory(&self, a: &ListDirArgs) -> ToolOutcome {
        let r = match self.resolve(a.path.as_deref().unwrap_or(".")) {
            Ok(r) => r,
            Err(e) => return ToolOutcome::exec_fail(e),
        };
        let name = self.rel(&r.abs);
        if !r.abs.is_dir() {
            return ToolOutcome::exec_fail(if r.abs.exists() { format!("{name} is a file; use read_file") } else { format!("directory not found: {name}") });
        }
        let depth = a.depth.unwrap_or(1).clamp(1, 8);
        let cap = 300usize;
        let mut lines: Vec<String> = Vec::new();
        let mut count = 0usize;
        let mut more = false;
        for entry in self.walker(&r.abs, Some(depth), a.include_hidden, a.include_ignored).flatten() {
            if entry.depth() == 0 {
                continue;
            }
            if count >= cap {
                more = true;
                break;
            }
            count += 1;
            let indent = "  ".repeat(entry.depth() - 1);
            let fname = entry.file_name().to_string_lossy().to_string();
            let is_dir = entry.file_type().is_some_and(|t| t.is_dir());
            if is_dir {
                lines.push(format!("{indent}{fname}/"));
            } else {
                let size = entry.metadata().map(|m| human_size(m.len())).unwrap_or_default();
                lines.push(format!("{indent}{fname}  ({size})"));
            }
        }
        let mut out = format!("{}/ ({} entries{})\n", if name == "." { "." } else { &name }, count, if more { ", list truncated" } else { "" });
        out.push_str(&lines.join("\n"));
        if lines.is_empty() {
            out.push_str("(empty)");
        }
        out.push('\n');
        ToolOutcome::success(out, format!("listed {name} ({count} entries)"))
    }

    fn search_files(&self, a: &SearchArgs) -> ToolOutcome {
        let r = match self.resolve(a.path.as_deref().unwrap_or(".")) {
            Ok(r) => r,
            Err(e) => return ToolOutcome::exec_fail(e),
        };
        let mode = a.mode.as_deref().unwrap_or("content");
        if mode != "content" && mode != "name" {
            return ToolOutcome::exec_fail("mode must be \"content\" or \"name\"");
        }
        let max_results = a.max_results.unwrap_or(self.env.cfg.tools.max_search_results).clamp(1, 1000);
        let pat = if a.regex { a.pattern.clone() } else { regex::escape(&a.pattern) };
        let sensitive = a.case_sensitive.unwrap_or_else(|| a.pattern.chars().any(|c| c.is_uppercase()));
        let re = match RegexBuilder::new(&pat).case_insensitive(!sensitive).size_limit(10 * 1024 * 1024).build() {
            Ok(re) => re,
            Err(e) => return ToolOutcome::exec_fail(format!("invalid regular expression: {e}")),
        };
        let glob = match a.glob.as_deref().filter(|g| !g.is_empty()) {
            Some(g) => match globset::Glob::new(g) {
                Ok(g) => Some(g.compile_matcher()),
                Err(e) => return ToolOutcome::exec_fail(format!("invalid glob: {e}")),
            },
            None => None,
        };
        let ctx = a.context.unwrap_or(0).min(5);
        let max_file = self.env.cfg.context.max_file_bytes as u64;

        let files: Vec<PathBuf> = if r.abs.is_file() {
            vec![r.abs.clone()]
        } else {
            self.walker(&r.abs, None, false, false)
                .flatten()
                .filter(|e| e.file_type().is_some_and(|t| t.is_file()))
                .map(|e| e.into_path())
                .collect()
        };

        let mut out: Vec<String> = Vec::new();
        let mut hits = 0usize;
        let mut files_hit = 0usize;
        let mut capped = false;
        'files: for f in &files {
            let rel = self.rel(f);
            if let Some(g) = &glob {
                let base = f.file_name().map(PathBuf::from).unwrap_or_default();
                if !g.is_match(&rel) && !g.is_match(&base) {
                    continue;
                }
            }
            if mode == "name" {
                if re.is_match(&rel) {
                    hits += 1;
                    out.push(rel);
                    if hits >= max_results {
                        capped = true;
                        break;
                    }
                }
                continue;
            }
            let Ok(meta) = std::fs::metadata(f) else { continue };
            if meta.len() > max_file {
                continue;
            }
            let Ok(bytes) = std::fs::read(f) else { continue };
            if looks_binary(&bytes) {
                continue;
            }
            let text = String::from_utf8_lossy(&bytes);
            let lines: Vec<&str> = text.lines().collect();
            let mut file_hit = false;
            let mut last_printed: Option<usize> = None;
            for (i, l) in lines.iter().enumerate() {
                if re.is_match(l) {
                    if !file_hit {
                        file_hit = true;
                        files_hit += 1;
                    }
                    hits += 1;
                    let from = i.saturating_sub(ctx);
                    let to = (i + ctx).min(lines.len() - 1);
                    if ctx > 0 && last_printed.is_some_and(|p| from > p + 1) {
                        out.push("--".into());
                    }
                    for (j, line) in lines.iter().enumerate().take(to + 1).skip(from) {
                        if last_printed.is_some_and(|p| j <= p) {
                            continue;
                        }
                        let sep = if j == i { ':' } else { '-' };
                        let shown: String = if line.chars().count() > 300 { format!("{}…", line.chars().take(300).collect::<String>()) } else { line.to_string() };
                        out.push(format!("{rel}{sep}{}{sep} {shown}", j + 1));
                        last_printed = Some(j);
                    }
                    if hits >= max_results {
                        capped = true;
                        break 'files;
                    }
                }
            }
        }
        if hits == 0 {
            return ToolOutcome::success(format!("no matches for \"{}\" ({} files searched)", a.pattern, files.len()), "no matches".to_string());
        }
        let mut text = out.join("\n");
        text.push('\n');
        if capped {
            text.push_str(&format!("[stopped after {max_results} matches; narrow the search]\n"));
        }
        let summary = if mode == "name" { format!("{hits} file(s) match") } else { format!("{hits} match(es) in {files_hit} file(s)") };
        ToolOutcome::success(text, summary)
    }

    fn exec_command(&self, command: &str, cwd: &Path, timeout: Duration) -> ToolOutcome {
        let cfg = &self.env.cfg;
        let analysis = permissions::analyze_command(command, cwd, &self.env.workspace);
        let direct = analysis.parse_error.is_none()
            && !analysis.needs_shell
            && analysis.segments.len() == 1
            && analysis.segments[0].redirects.is_empty()
            && !analysis.segments[0].argv.is_empty()
            && !analysis.segments[0].argv[0].contains('=');

        let env_vars = vec![
            ("NO_COLOR".to_string(), "1".to_string()),
            ("PAGER".to_string(), "cat".to_string()),
            ("GIT_PAGER".to_string(), "cat".to_string()),
            ("GIT_TERMINAL_PROMPT".to_string(), "0".to_string()),
            ("DEBIAN_FRONTEND".to_string(), "noninteractive".to_string()),
        ];
        let shell = platform::resolve_shell(&cfg.tools.shell);
        let shell_spec = |cmd: &str| {
            let mut args = shell.args.clone();
            args.push(cmd.to_string());
            let mut s = ProcessSpec::new(shell.program.clone(), args);
            s.cwd = Some(cwd.to_path_buf());
            s.env = env_vars.clone();
            s.timeout = timeout;
            s.max_output_bytes = cfg.tools.max_output_bytes;
            s
        };

        let started_direct = direct;
        let mut result = if direct {
            let seg = &analysis.segments[0];
            let program = platform::which(&seg.argv[0]).map(|p| p.to_string_lossy().to_string()).unwrap_or_else(|| seg.argv[0].clone());
            let mut spec = ProcessSpec::new(program, seg.argv[1..].to_vec());
            spec.cwd = Some(cwd.to_path_buf());
            spec.env = env_vars.clone();
            spec.timeout = timeout;
            spec.max_output_bytes = cfg.tools.max_output_bytes;
            platform::run_process(&spec, Some(&self.env.cancel))
        } else {
            platform::run_process(&shell_spec(command), Some(&self.env.cancel))
        };
        if started_direct {
            if let Err(e) = &result {
                if e.kind() == std::io::ErrorKind::NotFound {
                    result = platform::run_process(&shell_spec(command), Some(&self.env.cancel));
                }
            }
        }
        let out = match result {
            Ok(o) => o,
            Err(e) => return ToolOutcome::exec_fail(format!("cannot start the command: {e}")),
        };

        let mut text = format!("$ {command}\n");
        if out.timed_out {
            text.push_str(&format!("TIMED OUT after {}s (the process tree was killed)\n", timeout.as_secs()));
        } else if out.cancelled {
            text.push_str("CANCELLED by the user (the process tree was killed)\n");
        } else {
            text.push_str(&format!("exit code: {} ({:.1}s)\n", out.code.map(|c| c.to_string()).unwrap_or_else(|| "none (terminated by signal)".into()), out.duration.as_secs_f32()));
        }
        if !out.stdout.trim().is_empty() {
            text.push_str("--- stdout ---\n");
            text.push_str(out.stdout.trim_end());
            text.push('\n');
        }
        if !out.stderr.trim().is_empty() {
            text.push_str("--- stderr ---\n");
            text.push_str(out.stderr.trim_end());
            text.push('\n');
        }
        let ok = out.success();
        let first_line = if ok {
            format!("exit 0 in {:.1}s", out.duration.as_secs_f32())
        } else if out.timed_out {
            "timed out".to_string()
        } else if out.cancelled {
            "cancelled".to_string()
        } else {
            format!("exit {}", out.code.map(|c| c.to_string()).unwrap_or_else(|| "?".into()))
        };
        let mut o = if ok { ToolOutcome::success(text, format!("{command} → {first_line}")) } else { ToolOutcome::exec_fail(text) };
        if !ok {
            o.summary = format!("{command} → {first_line}");
        }
        if let Some(kind) = classify_verification(command) {
            o.verify = Some((kind, ok));
        }
        o
    }
}

fn build_specs() -> Vec<ToolSpec> {
    let path = json!({"type": "string", "minLength": 1});
    vec![
        spec(
            "read_file",
            "Read a text file with line numbers. Returns the file's sha256 prefix. Use start_line/end_line for large files.",
            true,
            json!({"type": "object", "properties": {
                "path": path, "start_line": {"type": "integer", "minimum": 1}, "end_line": {"type": "integer", "minimum": 1}, "max_lines": {"type": "integer", "minimum": 1}
            }, "required": ["path"], "additionalProperties": false}),
        ),
        spec(
            "write_file",
            "Create a file, or overwrite a file that was read earlier in this session. Prefer edit_file for changes to existing files.",
            false,
            json!({"type": "object", "properties": {"path": path, "content": {"type": "string"}}, "required": ["path", "content"], "additionalProperties": false}),
        ),
        spec(
            "edit_file",
            "Precisely modify a file with operations create/replace/insert/delete/append (all-or-nothing). `replace` needs `old` text that matches exactly once (or line range start_line/end_line); `insert` needs line, before or after; optional expected_hash guards against concurrent changes.",
            false,
            json!({"type": "object", "properties": {
                "path": path,
                "operations": {"type": "array", "minItems": 1, "maxItems": 50, "items": crate::protocol::edit_op_schema()},
                "expected_hash": {"type": "string"}
            }, "required": ["path", "operations"], "additionalProperties": false}),
        ),
        spec(
            "delete_file",
            "Delete a file or (with recursive=true) a directory.",
            false,
            json!({"type": "object", "properties": {"path": path, "recursive": {"type": "boolean"}}, "required": ["path"], "additionalProperties": false}),
        ),
        spec(
            "list_directory",
            "List a directory (default: workspace root). depth 1-8. Respects .gitignore unless include_ignored.",
            true,
            json!({"type": "object", "properties": {
                "path": {"type": "string"}, "depth": {"type": "integer", "minimum": 1, "maximum": 8},
                "include_hidden": {"type": "boolean"}, "include_ignored": {"type": "boolean"}
            }, "additionalProperties": false}),
        ),
        spec(
            "search_files",
            "Search file contents (mode \"content\", default) or file names (mode \"name\"). Literal text unless regex=true; smart-case unless case_sensitive is set.",
            true,
            json!({"type": "object", "properties": {
                "pattern": {"type": "string", "minLength": 1},
                "mode": {"enum": ["content", "name"]},
                "path": {"type": "string"}, "glob": {"type": "string"}, "regex": {"type": "boolean"},
                "case_sensitive": {"type": "boolean"}, "context": {"type": "integer", "minimum": 0, "maximum": 5},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 1000}
            }, "required": ["pattern"], "additionalProperties": false}),
        ),
        spec(
            "run_command",
            "Run a command in the workspace (build, run, lint, ...). Returns exit code, stdout and stderr. Dangerous commands are blocked or need user confirmation.",
            false,
            json!({"type": "object", "properties": {
                "command": {"type": "string", "minLength": 1}, "cwd": {"type": "string"}, "timeout_secs": {"type": "integer", "minimum": 1, "maximum": 3600}
            }, "required": ["command"], "additionalProperties": false}),
        ),
        spec(
            "run_test",
            "Run the project's tests (auto-detected: cargo, npm, go, pytest, ctest, make, dotnet) or a given test command.",
            false,
            json!({"type": "object", "properties": {
                "command": {"type": "string"}, "filter": {"type": "string"}, "timeout_secs": {"type": "integer", "minimum": 1, "maximum": 7200}
            }, "additionalProperties": false}),
        ),
        spec("git_status", "Show the Git status (branch and changed files).", true, json!({"type": "object", "properties": {}, "additionalProperties": false})),
        spec(
            "git_diff",
            "Show the Git diff of the working tree (or staged changes).",
            true,
            json!({"type": "object", "properties": {"staged": {"type": "boolean"}, "path": {"type": "string"}, "stat": {"type": "boolean"}}, "additionalProperties": false}),
        ),
        spec(
            "git_log",
            "Show recent commits.",
            true,
            json!({"type": "object", "properties": {"max_count": {"type": "integer", "minimum": 1, "maximum": 200}, "path": {"type": "string"}}, "additionalProperties": false}),
        ),
        spec(
            "git_branch",
            "List branches (default), create a branch (action \"create\") or switch branches (action \"switch\").",
            false,
            json!({"type": "object", "properties": {"action": {"enum": ["list", "create", "switch"]}, "name": {"type": "string", "minLength": 1}}, "additionalProperties": false}),
        ),
    ]
}

/// Short description of a tool call for the UI.
pub fn describe_call(tool: &str, args: &Value) -> String {
    let get = |k: &str| args.get(k).and_then(|v| v.as_str()).unwrap_or("");
    match tool {
        "run_command" => get("command").to_string(),
        "read_file" | "write_file" | "edit_file" | "delete_file" => get("path").to_string(),
        "search_files" => format!("\"{}\"", get("pattern")),
        "list_directory" => {
            let p = get("path");
            if p.is_empty() { ".".into() } else { p.to_string() }
        }
        _ => crate::protocol::compact_json(args, 100),
    }
}

/// Excerpt helper re-exported for the agent (numbered lines of a file).
pub fn excerpt(text: &str, max_lines: usize) -> String {
    numbered_excerpt(text, None, max_lines)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, PermissionsSection};
    use crate::ui::BufferUi;

    struct Fx {
        _dir: tempfile::TempDir,
        reg: ToolRegistry,
        perms: PermissionManager,
        ui: BufferUi,
        ws: PathBuf,
    }

    fn fx_with(answers: Vec<bool>) -> Fx {
        let dir = tempfile::tempdir().unwrap();
        let cfg = Arc::new(Config::default());
        let reg = ToolRegistry::new(dir.path(), cfg, Arc::new(AtomicBool::new(false)));
        let ws = reg.env().workspace.clone();
        Fx { _dir: dir, reg, perms: PermissionManager::new(&PermissionsSection::default()), ui: BufferUi::with_answers(answers), ws }
    }

    fn fx() -> Fx {
        fx_with(vec![])
    }

    impl Fx {
        fn call(&self, tool: &str, args: Value) -> ToolOutcome {
            self.reg.invoke(tool, &args, &self.perms, &self.ui)
        }
    }

    #[test]
    fn unknown_tool_and_bad_schema_are_rejected_before_execution() {
        let f = fx();
        let o = f.call("format_disk", json!({}));
        assert!(!o.ok && o.stage == Stage::UnknownTool);
        let o = f.call("read_file", json!({}));
        assert!(!o.ok && o.stage == Stage::Schema, "{}", o.output);
        assert!(o.output.contains("missing required property \"path\""));
        let o = f.call("read_file", json!({"path": "a", "bogus": 1}));
        assert_eq!(o.stage, Stage::Schema);
        let o = f.call("run_command", json!({"command": 42}));
        assert_eq!(o.stage, Stage::Schema);
    }

    #[test]
    fn write_read_edit_roundtrip() {
        let f = fx();
        let o = f.call("write_file", json!({"path": "src/hello.txt", "content": "alpha\nbeta\ngamma\n"}));
        assert!(o.ok, "{}", o.output);
        assert!(f.ws.join("src/hello.txt").is_file());
        let o = f.call("read_file", json!({"path": "src/hello.txt"}));
        assert!(o.ok && o.output.contains("    2| beta") && o.output.contains("3 lines"), "{}", o.output);
        let o = f.call("edit_file", json!({"path": "src/hello.txt", "operations": [{"type": "replace", "old": "beta", "new": "BETA"}]}));
        assert!(o.ok, "{}", o.output);
        assert!(o.output.contains("+BETA"));
        assert_eq!(std::fs::read_to_string(f.ws.join("src/hello.txt")).unwrap(), "alpha\nBETA\ngamma\n");
        assert_eq!(o.changed_files, vec!["src/hello.txt".to_string()]);
    }

    #[test]
    fn overwrite_requires_reading_first() {
        let f = fx();
        std::fs::write(f.ws.join("x.txt"), "old\n").unwrap();
        let o = f.call("write_file", json!({"path": "x.txt", "content": "new\n"}));
        assert!(!o.ok && o.output.contains("has not been read"));
        assert_eq!(std::fs::read_to_string(f.ws.join("x.txt")).unwrap(), "old\n");
        f.call("read_file", json!({"path": "x.txt"}));
        let o = f.call("write_file", json!({"path": "x.txt", "content": "new\n"}));
        assert!(o.ok, "{}", o.output);
    }

    #[test]
    fn edit_detects_external_changes() {
        let f = fx();
        std::fs::write(f.ws.join("a.txt"), "one\n").unwrap();
        f.call("read_file", json!({"path": "a.txt"}));
        std::fs::write(f.ws.join("a.txt"), "changed behind our back\n").unwrap();
        let o = f.call("edit_file", json!({"path": "a.txt", "operations": [{"type": "replace", "old": "one", "new": "two"}]}));
        assert!(!o.ok);
        assert!(o.output.contains("changed since") && o.output.contains("changed behind our back"), "{}", o.output);
    }

    #[test]
    fn list_and_search() {
        let f = fx();
        std::fs::create_dir_all(f.ws.join("src")).unwrap();
        std::fs::write(f.ws.join("src/main.rs"), "fn main() {\n    println!(\"hi\");\n}\n").unwrap();
        std::fs::write(f.ws.join("src/lib.rs"), "pub fn helper() {}\n").unwrap();
        std::fs::write(f.ws.join("README.md"), "# Title\n").unwrap();
        let o = f.call("list_directory", json!({"depth": 2}));
        assert!(o.ok && o.output.contains("src/") && o.output.contains("main.rs") && o.output.contains("README.md"), "{}", o.output);
        let o = f.call("search_files", json!({"pattern": "println", "context": 1}));
        assert!(o.ok && o.output.contains("src/main.rs:2:") && o.output.contains("src/main.rs-1-"), "{}", o.output);
        let o = f.call("search_files", json!({"pattern": "\\.rs$", "mode": "name", "regex": true}));
        assert!(o.ok && o.output.contains("src/lib.rs") && !o.output.contains("README"), "{}", o.output);
        let o = f.call("search_files", json!({"pattern": "nothing_like_this"}));
        assert!(o.ok && o.output.contains("no matches"));
        let o = f.call("search_files", json!({"pattern": "(", "regex": true}));
        assert!(!o.ok && o.output.contains("invalid regular expression"));
    }

    #[test]
    fn run_command_allowed_and_failing() {
        let f = fx();
        let o = f.call("run_command", json!({"command": "echo hello-from-qwen"}));
        assert!(o.ok && o.output.contains("hello-from-qwen") && o.output.contains("exit code: 0"), "{}", o.output);
        let o = f.call("run_command", json!({"command": "sh -c 'exit 3'"}));
        assert!(!o.ok && o.output.contains("exit code: 3"), "{}", o.output);
        assert!(f.ui.prompts().is_empty(), "ordinary commands must not prompt");
        let o = f.call("run_command", json!({"command": "echo a | tr a b"}));
        assert!(o.ok && o.output.contains('b'), "{}", o.output);
    }

    #[test]
    fn permission_deny_ask_allow() {
        // deny: system destructive is blocked without prompting
        let f = fx_with(vec![true]);
        let o = f.call("run_command", json!({"command": "rm -rf /"}));
        assert!(!o.ok && o.stage == Stage::Permission && o.output.contains("NOT executed"), "{}", o.output);
        assert!(f.ui.prompts().is_empty());
        // ask -> N
        std::fs::write(f.ws.join("victim.txt"), "x").unwrap();
        let f2 = fx_with(vec![false]);
        std::fs::write(f2.ws.join("victim.txt"), "x").unwrap();
        let o = f2.call("delete_file", json!({"path": "victim.txt"}));
        assert!(!o.ok && o.stage == Stage::Permission && o.output.contains("denied by the user"));
        assert!(f2.ws.join("victim.txt").exists());
        assert_eq!(f2.ui.prompts().len(), 1);
        // ask -> Y
        let f3 = fx_with(vec![true]);
        std::fs::write(f3.ws.join("victim.txt"), "x").unwrap();
        let o = f3.call("delete_file", json!({"path": "victim.txt"}));
        assert!(o.ok, "{}", o.output);
        assert!(!f3.ws.join("victim.txt").exists());
    }

    #[test]
    fn paths_outside_the_workspace_need_confirmation() {
        let f = fx();
        let outside = tempfile::tempdir().unwrap();
        let target = outside.path().join("o.txt");
        let o = f.call("write_file", json!({"path": target.to_string_lossy(), "content": "x"}));
        assert!(!o.ok && o.stage == Stage::Permission);
        assert!(!target.exists());
        assert_eq!(f.ui.prompts().len(), 1);
        let f = fx_with(vec![true]);
        let o = f.call("write_file", json!({"path": target.to_string_lossy(), "content": "x"}));
        assert!(o.ok, "{}", o.output);
        assert!(target.exists());
        // `..` escapes are normalised
        let o = f.call("read_file", json!({"path": "../../../../../../etc/hostname"}));
        assert_eq!(f.ui.prompts().len(), 2, "second outside access prompts again; answers exhausted -> denied");
        assert!(!o.ok);
    }

    #[test]
    fn deleting_the_workspace_root_is_never_silent() {
        let f = fx_with(vec![false]);
        let o = f.call("delete_file", json!({"path": ".", "recursive": true}));
        assert!(!o.ok);
        assert!(f.ws.exists());
    }

    #[test]
    fn command_timeout_kills_the_process() {
        let f = fx();
        let o = f.call("run_command", json!({"command": "sleep 5", "timeout_secs": 1}));
        assert!(!o.ok && o.output.contains("TIMED OUT"), "{}", o.output);
    }

    #[test]
    fn git_tools_work_in_a_repo_and_fail_gracefully_outside() {
        if !Git::is_available() {
            return;
        }
        let f = fx();
        let o = f.call("git_status", json!({}));
        assert!(!o.ok, "not a repo");
        assert!(f.ws.exists());
        assert!(f.call("run_command", json!({"command": "git init -q"})).ok);
        std::fs::write(f.ws.join("a.txt"), "x\n").unwrap();
        let o = f.call("git_status", json!({}));
        assert!(o.ok && o.output.contains("a.txt"), "{}", o.output);
        let o = f.call("git_diff", json!({}));
        assert!(o.ok);
        let o = f.call("git_log", json!({"max_count": 3}));
        assert!(o.ok, "{}", o.output);
        let o = f.call("git_branch", json!({}));
        assert!(o.ok, "{}", o.output);
        let o = f.call("git_branch", json!({"action": "create", "name": "-bad"}));
        assert!(!o.ok);
    }

    #[test]
    fn verification_classification() {
        assert_eq!(classify_verification("cargo build --release"), Some(VerifyKind::Build));
        assert_eq!(classify_verification("cargo test"), Some(VerifyKind::Test));
        assert_eq!(classify_verification("cmake --build build"), Some(VerifyKind::Build));
        assert_eq!(classify_verification("ctest --output-on-failure"), Some(VerifyKind::Test));
        assert_eq!(classify_verification("ls -la"), None);
    }

    #[test]
    fn run_test_detects_the_project_type() {
        let f = fx();
        let o = f.call("run_test", json!({}));
        assert!(!o.ok && o.output.contains("cannot detect"));
        std::fs::write(f.ws.join("Makefile"), "test:\n\t@echo ran-tests\n").unwrap();
        if platform::which("make").is_some() {
            let o = f.call("run_test", json!({}));
            assert!(o.ok && o.output.contains("TEST RESULT: PASSED") && o.output.contains("ran-tests"), "{}", o.output);
            assert_eq!(o.verify, Some((VerifyKind::Test, true)));
        }
    }
}
