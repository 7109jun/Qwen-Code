//! The agent loop.
//!
//! ```text
//! User -> Thinking (analysis / plan) -> Coder (tool calls) -> tool results -> Thinking (review)
//!      -> Coder -> ... -> Thinking says `complete`
//! ```
//!
//! The loop only ends when the Thinking model declares the task complete (and its verification
//! claims hold up), when it needs input from the user, when it is blocked, on cancellation, on an
//! unrecoverable model failure, or when `max_iterations` is reached.

use crate::config::Config;
use crate::context::{estimate_tokens, truncate_middle, ContextManager, History};
use crate::models::{ModelRole, ModelSet};
use crate::permissions::PermissionManager;
use crate::platform;
use crate::prompts::{self, PromptEnv};
use crate::protocol::{self, AgentMessage, CompleteStatus, Parsed, Role, Verification};
use crate::runtime::{ChatMessage, GenOutput, RuntimeError};
use crate::tools::{self, ToolOutcome, ToolRegistry, VerifyKind};
use crate::ui::{Event, Ui};
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Thinking model supervises the Coder (default).
    Auto,
    /// Talk to the Thinking model only (analysis, review; read-only tools).
    Thinking,
    /// Talk to the Coder directly, without Thinking supervision.
    Coder,
}

impl Mode {
    pub fn parse(s: &str) -> Option<Mode> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" | "both" | "agent" => Some(Mode::Auto),
            "thinking" | "think" => Some(Mode::Thinking),
            "coder" | "code" => Some(Mode::Coder),
            _ => None,
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            Mode::Auto => "auto (Thinking + Coder)",
            Mode::Thinking => "thinking",
            Mode::Coder => "coder",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Status {
    Completed,
    NeedsInput,
    Blocked,
    MaxIterations,
    Failed(String),
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct TaskOutcome {
    pub status: Status,
    pub summary: String,
    /// Verification as observed by the agent (not merely claimed by the model).
    pub verification: Verification,
    pub iterations: usize,
    pub tool_calls: usize,
    pub files_changed: Vec<String>,
}

impl TaskOutcome {
    pub fn is_success(&self) -> bool {
        self.status == Status::Completed
    }
}

#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub model_calls: usize,
    pub tool_calls: usize,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub tasks: usize,
}

#[derive(Debug, Clone)]
enum AgentError {
    Cancelled,
    Model(String),
    InvalidOutput(String),
    ContextOverflow(String),
}

impl AgentError {
    fn message(&self) -> String {
        match self {
            AgentError::Cancelled => "cancelled".into(),
            AgentError::Model(m) => m.clone(),
            AgentError::InvalidOutput(m) => format!("the model kept producing invalid output: {m}"),
            AgentError::ContextOverflow(m) => format!("the prompt does not fit into the model's context window even after compaction: {m}"),
        }
    }
}

#[derive(Default)]
struct TaskState {
    files_changed: BTreeSet<String>,
    tool_calls: usize,
    seq: usize,
    last_mod_seq: usize,
    last_build: Option<(usize, bool)>,
    last_test: Option<(usize, bool)>,
    pushbacks: usize,
    iterations: usize,
}

impl TaskState {
    fn observed(&self) -> Verification {
        let fresh = |v: Option<(usize, bool)>| v.filter(|(s, _)| *s >= self.last_mod_seq).map(|(_, ok)| ok);
        Verification { build: fresh(self.last_build), test: fresh(self.last_test), notes: None }
    }
}

struct LogEntry {
    tool: String,
    args: String,
    ok: bool,
    summary: String,
    output: String,
}

struct CoderResult {
    summary: String,
    status: &'static str,
    log: Vec<LogEntry>,
}

pub struct Agent {
    pub cfg: Arc<Config>,
    pub models: ModelSet,
    pub tools: ToolRegistry,
    pub perms: PermissionManager,
    pub ctx: ContextManager,
    pub ui: Arc<dyn Ui>,
    pub cancel: Arc<AtomicBool>,
    pub max_iterations: usize,
    pub stats: Stats,
    thinking_hist: History,
    sys_thinking: String,
    sys_coder: String,
    workspace: PathBuf,
    is_repo: bool,
}

impl Agent {
    pub fn new(cfg: Arc<Config>, workspace: &Path, models: ModelSet, perms: PermissionManager, ui: Arc<dyn Ui>, cancel: Arc<AtomicBool>) -> Agent {
        let tools = ToolRegistry::new(workspace, cfg.clone(), cancel.clone());
        let ws = tools.env().workspace.clone();
        let is_repo = crate::git::Git::new(&ws).is_repo();
        let ctx = ContextManager::new(cfg.context.clone(), &ws);
        let (os, shell) = prompts::env_description();
        let ws_str = ws.to_string_lossy().to_string();
        let all = tools.prompt_description(false);
        let ro = tools.prompt_description(true);
        let env = PromptEnv { workspace: &ws_str, os: &os, shell: &shell, is_git_repo: is_repo, tools_all: &all, tools_read_only: &ro };
        let sys_thinking = prompts::thinking_system(&env);
        let sys_coder = prompts::coder_system(&env);
        let max_iterations = cfg.qwen.max_iterations;
        Agent { cfg, models, tools, perms, ctx, ui, cancel, max_iterations, stats: Stats::default(), thinking_hist: History::new(), sys_thinking, sys_coder, workspace: ws, is_repo }
    }

    pub fn workspace(&self) -> &Path {
        &self.workspace
    }

    pub fn is_git_repo(&self) -> bool {
        self.is_repo
    }

    /// Forgets the conversation (`/reset`).
    pub fn reset(&mut self) {
        self.thinking_hist.clear();
        self.tools.reset();
        self.ctx.focus.clear();
    }

    pub fn history_tokens(&self) -> usize {
        self.thinking_hist.tokens()
    }

    pub fn history_len(&self) -> usize {
        self.thinking_hist.len()
    }

    pub fn compactions(&self) -> usize {
        self.thinking_hist.compactions()
    }

    pub fn context_budget(&self) -> usize {
        let m = self.models.thinking.as_ref();
        self.ctx.budget(m.context_window(), m.max_new_tokens())
    }

    fn warn(&self, s: impl Into<String>) {
        self.ui.emit(Event::Warn(s.into()));
    }

    // ---------------------------------------------------------------------------------------
    // Public entry point
    // ---------------------------------------------------------------------------------------

    pub fn run_task(&mut self, task: &str, mode: Mode) -> TaskOutcome {
        self.cancel.store(false, Ordering::Relaxed);
        self.stats.tasks += 1;
        let mut hist = std::mem::take(&mut self.thinking_hist);
        let mut st = TaskState::default();
        let outcome = match mode {
            Mode::Coder => self.run_coder_only(task, &mut st),
            _ => self.run_supervised(task, mode, &mut hist, &mut st),
        };
        self.thinking_hist = hist;
        self.report(&outcome);
        outcome
    }

    fn report(&self, o: &TaskOutcome) {
        let mut text = o.summary.trim().to_string();
        if !o.files_changed.is_empty() {
            text.push_str(&format!("\n\nFiles changed: {}", o.files_changed.join(", ")));
        }
        let mark = |v: Option<bool>| match v {
            Some(true) => "passed",
            Some(false) => "FAILED",
            None => "not run",
        };
        if o.verification.build.is_some() || o.verification.test.is_some() || !o.files_changed.is_empty() {
            text.push_str(&format!("\nVerification (observed): build {}, tests {}", mark(o.verification.build), mark(o.verification.test)));
        }
        match &o.status {
            Status::Completed | Status::NeedsInput | Status::Blocked => self.ui.emit(Event::Message(text)),
            Status::MaxIterations => self.ui.emit(Event::Warn(format!("stopped: max_iterations ({}) reached\n{text}", self.max_iterations))),
            Status::Failed(e) => self.ui.emit(Event::Error(format!("task failed: {e}"))),
            Status::Cancelled => self.ui.emit(Event::Warn("task cancelled".into())),
        }
    }

    fn outcome(&self, st: &TaskState, status: Status, summary: String) -> TaskOutcome {
        TaskOutcome {
            status,
            summary,
            verification: st.observed(),
            iterations: st.iterations,
            tool_calls: st.tool_calls,
            files_changed: st.files_changed.iter().cloned().collect(),
        }
    }

    fn fail(&self, st: &TaskState, e: AgentError) -> TaskOutcome {
        match e {
            AgentError::Cancelled => self.outcome(st, Status::Cancelled, "Cancelled by the user.".into()),
            other => {
                let m = other.message();
                self.outcome(st, Status::Failed(m.clone()), format!("The task could not be completed: {m}"))
            }
        }
    }

    // ---------------------------------------------------------------------------------------
    // Model calls (retries, compaction, protocol repair)
    // ---------------------------------------------------------------------------------------

    fn generate_with_retries(&self, model: &dyn ModelRole, msgs: &[ChatMessage]) -> Result<GenOutput, RuntimeError> {
        let retries = self.cfg.agent.model_retries;
        let mut attempt = 0usize;
        loop {
            match model.generate(msgs, &self.cancel) {
                Ok(o) => return Ok(o),
                Err(RuntimeError::Connection(m)) if attempt < retries => {
                    attempt += 1;
                    self.warn(format!("{} model connection problem ({m}); retry {attempt}/{retries}", model.role()));
                    let wait = Duration::from_millis(self.cfg.agent.retry_backoff_ms.saturating_mul(1 << (attempt - 1).min(6)));
                    let start = std::time::Instant::now();
                    while start.elapsed() < wait {
                        if self.cancel.load(Ordering::Relaxed) {
                            return Err(RuntimeError::Cancelled);
                        }
                        std::thread::sleep(Duration::from_millis(20));
                    }
                }
                Err(e) => return Err(e),
            }
        }
    }

    fn call_model(&mut self, role: Role, hist: &mut History) -> Result<Parsed, AgentError> {
        let models = self.models.clone();
        let model: &dyn ModelRole = models.by_role(role.name());
        let system = if role == Role::Thinking { self.sys_thinking.clone() } else { self.sys_coder.clone() };
        let sys_tokens = estimate_tokens(&system);
        let mut overflow = 0usize;
        let mut repairs = 0usize;
        loop {
            if self.cancel.load(Ordering::Relaxed) {
                return Err(AgentError::Cancelled);
            }
            let budget = self.ctx.budget(model.context_window(), model.max_new_tokens()).saturating_sub(sys_tokens).max(512);
            let shrink = 1.0 - 0.3 * overflow as f32;
            let eff = (budget as f32 * shrink) as usize;
            if hist.compact(eff, self.cfg.context.compact_threshold, self.cfg.context.keep_recent) {
                self.ui.emit(Event::Info(format!("(context compacted: history now ≈ {} tokens)", hist.tokens())));
            }
            let mut msgs = vec![ChatMessage::system(system.clone())];
            msgs.extend(hist.messages());

            let out = match self.generate_with_retries(model, &msgs) {
                Ok(o) => o,
                Err(RuntimeError::Cancelled) => return Err(AgentError::Cancelled),
                Err(RuntimeError::ContextOverflow(m)) => {
                    overflow += 1;
                    if overflow > 2 {
                        return Err(AgentError::ContextOverflow(m));
                    }
                    self.warn(format!("context window exceeded; compacting more aggressively ({overflow}/2)"));
                    hist.force_compact(((budget as f32) * (1.0 - 0.3 * overflow as f32)) as usize, 2);
                    continue;
                }
                Err(e) => return Err(AgentError::Model(format!("{} model: {e}", role.name()))),
            };
            self.stats.model_calls += 1;
            self.stats.prompt_tokens += out.prompt_tokens;
            self.stats.completion_tokens += out.completion_tokens;

            match protocol::parse_response(&out.text, role) {
                Ok(p) => {
                    if !p.reasoning.is_empty() {
                        self.ui.emit(Event::Reasoning(p.reasoning.clone()));
                    }
                    if p.extra_objects > 0 {
                        self.warn(format!("{} model sent {} extra JSON object(s); only the first was used", role.name(), p.extra_objects));
                    }
                    let mut text = serde_json::to_string(&p.raw).unwrap_or_default();
                    if p.extra_objects > 0 {
                        text.push_str("\n(note: only your first JSON object was processed; send one object per reply)");
                    }
                    hist.push("assistant", "model", text);
                    return Ok(p);
                }
                Err(e) => {
                    repairs += 1;
                    let first = e.message.lines().next().unwrap_or("").to_string();
                    if repairs > self.cfg.agent.max_repairs {
                        return Err(AgentError::InvalidOutput(first));
                    }
                    self.warn(format!("invalid {} output ({first}); asking the model to repair it ({repairs}/{})", role.name(), self.cfg.agent.max_repairs));
                    let (_, answer, _) = protocol::split_reasoning(&out.text);
                    hist.push("assistant", "model", truncate_middle(&answer, 1200));
                    let mut msg = prompts::format_repair_request(&e.message);
                    if out.finish_reason == "length" {
                        msg.push_str("\nYour reply was cut off by the token limit. Make it shorter: split the work into smaller steps or edits.");
                    }
                    hist.push("user", "feedback", msg);
                }
            }
        }
    }

    // ---------------------------------------------------------------------------------------
    // Tools
    // ---------------------------------------------------------------------------------------

    fn run_tool(&mut self, role: &str, tool: &str, args: &Value, st: &mut TaskState) -> ToolOutcome {
        self.ui.emit(Event::ToolCall { role: role.into(), tool: tool.into(), summary: tools::describe_call(tool, args) });
        let o = self.tools.invoke(tool, args, &self.perms, self.ui.as_ref());
        st.tool_calls += 1;
        self.stats.tool_calls += 1;
        self.ui.emit(Event::ToolResult { tool: tool.into(), ok: o.ok, summary: o.summary.clone() });
        if o.ok && matches!(tool, "edit_file" | "write_file") {
            if let Some(pos) = o.output.find("@@") {
                self.ui.emit(Event::Diff(o.output[pos..].trim_end().to_string()));
            }
        }
        if !o.changed_files.is_empty() {
            st.seq += 1;
            st.last_mod_seq = st.seq;
            st.files_changed.extend(o.changed_files.iter().cloned());
        }
        if let Some((kind, ok)) = o.verify {
            st.seq += 1;
            match kind {
                VerifyKind::Build => st.last_build = Some((st.seq, ok)),
                VerifyKind::Test => {
                    st.last_test = Some((st.seq, ok));
                    // a passing test run implies the project builds; a failing one proves nothing about the build
                    if ok {
                        st.last_build = Some((st.seq, true));
                    }
                }
            }
        }
        o
    }

    // ---------------------------------------------------------------------------------------
    // Coder
    // ---------------------------------------------------------------------------------------

    fn run_coder(&mut self, instruction: &str, files: &[String], acceptance: &[String], st: &mut TaskState) -> Result<CoderResult, AgentError> {
        let mut hist = History::new();
        let mut first = format!("INSTRUCTION FROM THE THINKING MODEL:\n{instruction}\n");
        if !files.is_empty() {
            first.push_str(&format!("\nFiles of interest: {}\n", files.join(", ")));
        }
        if !acceptance.is_empty() {
            first.push_str("\nAcceptance criteria:\n");
            for a in acceptance {
                first.push_str(&format!("- {a}\n"));
            }
        }
        let saved_focus = self.ctx.focus.clone();
        for f in files {
            if !self.ctx.focus.contains(f) {
                self.ctx.focus.push(f.clone());
            }
        }
        let context = self.ctx.task_context(instruction, None);
        self.ctx.focus = saved_focus;
        first.push_str(&format!("\n{}", truncate_middle(&context, 9000)));
        hist.push("user", "task", first);
        self.run_coder_loop(&mut hist, st)
    }

    fn run_coder_loop(&mut self, hist: &mut History, st: &mut TaskState) -> Result<CoderResult, AgentError> {
        let max_steps = self.cfg.agent.max_coder_steps;
        let max_out = self.cfg.context.max_tool_output_bytes;
        let mut log: Vec<LogEntry> = Vec::new();
        let mut repeats: HashMap<String, usize> = HashMap::new();
        for step in 1..=max_steps {
            if self.cancel.load(Ordering::Relaxed) {
                return Err(AgentError::Cancelled);
            }
            self.ui.emit(Event::Phase { role: "coder".into(), text: format!("step {step}/{max_steps}") });
            let parsed = self.call_model(Role::Coder, hist)?;
            let (tool, args) = match parsed.message {
                AgentMessage::Report { summary, status, .. } => {
                    return Ok(CoderResult { summary, status: if status.as_deref() == Some("blocked") { "blocked" } else { "done" }, log });
                }
                AgentMessage::ToolCall { tool, arguments } => (tool, arguments),
                AgentMessage::Edit { path, expected_hash, .. } => {
                    let mut a = json!({"path": path, "operations": parsed.raw.get("operations").cloned().unwrap_or(Value::Null)});
                    if let Some(h) = expected_hash {
                        a["expected_hash"] = json!(h);
                    }
                    ("edit_file".to_string(), a)
                }
                _ => {
                    hist.push("user", "feedback", prompts::format_repair_request("That message type is not allowed for the Coder."));
                    continue;
                }
            };
            let key = format!("{tool}{}", args);
            let o = self.run_tool("coder", &tool, &args, st);
            let n = repeats.entry(key).and_modify(|c| *c += 1).or_insert(1);
            let mut text = prompts::format_tool_result(&tool, o.ok, &truncate_middle(&o.output, max_out));
            let mut stuck = false;
            if !o.ok && *n >= 3 {
                text.push_str(&format!("\nNOTE: you have now made this exact call {n} times with the same failure. Change your approach, or send a \"report\" explaining what blocks you."));
                stuck = *n >= 5;
            }
            hist.push("user", "tool", text);
            log.push(LogEntry { tool: tool.clone(), args: tools::describe_call(&tool, &args), ok: o.ok, summary: o.summary.clone(), output: o.output.clone() });
            if stuck {
                return Ok(CoderResult { summary: "Stopped: the Coder repeated the same failing action five times.".into(), status: "stuck", log });
            }
        }
        Ok(CoderResult { summary: format!("Step limit reached ({max_steps} tool steps) before the Coder sent its report."), status: "step_limit", log })
    }

    fn format_report(&self, r: &CoderResult, st: &TaskState) -> String {
        let max_out = self.cfg.context.max_tool_output_bytes;
        let mut s = format!("CODER REPORT (status: {})\n{}\n", r.status, r.summary.trim());
        if !st.files_changed.is_empty() {
            s.push_str(&format!("\nFiles changed so far in this task: {}\n", st.files_changed.iter().cloned().collect::<Vec<_>>().join(", ")));
        }
        s.push_str("\nTool log:\n");
        for (i, e) in r.log.iter().enumerate() {
            s.push_str(&format!("{:>3}. {} {} → {} ({})\n", i + 1, e.tool, truncate_middle(&e.args, 100), if e.ok { "ok" } else { "FAILED" }, truncate_middle(&e.summary, 100)));
        }
        if r.log.is_empty() {
            s.push_str("  (no tool calls)\n");
        }
        // full output of the last failures and of the last build/test run, so the Thinking model can analyse them
        let mut shown = 0;
        for e in r.log.iter().rev().filter(|e| !e.ok).take(2) {
            s.push_str(&format!("\nOutput of failed {} {}:\n{}\n", e.tool, e.args, truncate_middle(&e.output, max_out / 2)));
            shown += 1;
        }
        if let Some(e) = r.log.iter().rev().find(|e| e.ok && matches!(e.tool.as_str(), "run_command" | "run_test") && tools::classify_verification(&e.args).is_some()) {
            if shown == 0 {
                s.push_str(&format!("\nOutput of the last build/test run ({}):\n{}\n", e.args, truncate_middle(&e.output, max_out / 3)));
            }
        }
        let obs = st.observed();
        let m = |v: Option<bool>| match v {
            Some(true) => "passed",
            Some(false) => "failed",
            None => "not run since the last change",
        };
        s.push_str(&format!("\nObserved verification: build {}, tests {}\n", m(obs.build), m(obs.test)));
        if self.is_repo {
            let d = crate::git::Git::new(&self.workspace).diff(false, None, true);
            if d.ok && !d.output.contains("no differences") {
                s.push_str(&format!("\ngit diff --stat:\n{}\n", truncate_middle(&d.output, 1500)));
            }
        }
        s
    }

    fn run_coder_only(&mut self, task: &str, st: &mut TaskState) -> TaskOutcome {
        self.ui.emit(Event::Phase { role: "coder".into(), text: "working directly (no Thinking supervision)".into() });
        let mut hist = History::new();
        let context = self.ctx.task_context(task, None);
        hist.push("user", "task", format!("USER REQUEST:\n{task}\n\nWork on it directly with your tools, then send a \"report\".\n\n{}", truncate_middle(&context, 9000)));
        match self.run_coder_loop(&mut hist, st) {
            Ok(r) => {
                let status = if r.status == "blocked" { Status::Blocked } else if r.status == "done" { Status::Completed } else { Status::Failed(r.summary.clone()) };
                self.outcome(st, status, r.summary)
            }
            Err(e) => self.fail(st, e),
        }
    }

    // ---------------------------------------------------------------------------------------
    // Thinking + supervision loop
    // ---------------------------------------------------------------------------------------

    fn run_supervised(&mut self, task: &str, mode: Mode, th: &mut History, st: &mut TaskState) -> TaskOutcome {
        let git = if self.is_repo { crate::git::Git::new(&self.workspace).summary() } else { None };
        let mut first = String::new();
        if th.is_empty() {
            first.push_str(&format!("USER REQUEST:\n{task}\n\n"));
            first.push_str(&self.ctx.task_context(task, git.as_deref()));
        } else {
            first.push_str(&format!("NEW USER REQUEST (follow-up in the same session):\n{task}\n\n"));
            if let Some(g) = &git {
                first.push_str(g);
                first.push('\n');
            }
            let r = self.ctx.retrieve(task);
            if !r.text.is_empty() {
                first.push_str(&format!("=== POTENTIALLY RELEVANT FILE EXCERPTS ===\n{}", r.text));
            }
        }
        if mode == Mode::Thinking {
            first.push_str("\nMODE: thinking-only. Answer the user yourself using read-only tools if needed, then send a \"complete\" message whose summary is your answer. Do not send instructions to the Coder.\n");
        }
        th.push("user", "task", truncate_middle(&first, 24_000));

        let max_iter = self.max_iterations;
        let mut last_report = String::new();
        loop {
            if self.cancel.load(Ordering::Relaxed) {
                return self.fail(st, AgentError::Cancelled);
            }
            if st.iterations >= max_iter {
                let summary = format!("Stopped after {max_iter} Thinking/Coder iterations without a completed result.{}", if last_report.is_empty() { String::new() } else { format!("\nLast Coder report:\n{}", truncate_middle(&last_report, 1500)) });
                th.push("user", "feedback", "The iteration limit was reached and the task was stopped.");
                return self.outcome(st, Status::MaxIterations, summary);
            }
            st.iterations += 1;
            self.ui.emit(Event::Phase { role: "thinking".into(), text: format!("iteration {}/{max_iter}", st.iterations) });

            let decision = match self.thinking_turn(th, st, mode) {
                Ok(d) => d,
                Err(e) => return self.fail(st, e),
            };
            match decision {
                Decision::Complete { summary, verification, status } => {
                    if status == CompleteStatus::Done {
                        if let Some(msg) = self.check_claims(&verification, st) {
                            st.pushbacks += 1;
                            if st.pushbacks <= 2 {
                                self.warn("the Thinking model claimed verification that was not observed; asking it to verify");
                                th.push("user", "feedback", msg);
                                st.iterations -= 1;
                                continue;
                            }
                        }
                    }
                    let s = match status {
                        CompleteStatus::Done => Status::Completed,
                        CompleteStatus::NeedsInput => Status::NeedsInput,
                        CompleteStatus::Blocked => Status::Blocked,
                    };
                    return self.outcome(st, s, summary);
                }
                Decision::Instruction { instruction, files, acceptance } => {
                    self.ui.emit(Event::Phase { role: "coder".into(), text: truncate_middle(instruction.lines().next().unwrap_or(""), 120) });
                    let result = match self.run_coder(&instruction, &files, &acceptance, st) {
                        Ok(r) => r,
                        Err(e) => return self.fail(st, e),
                    };
                    let report = self.format_report(&result, st);
                    last_report = report.clone();
                    th.push("user", "report", truncate_middle(&report, 14_000));
                }
            }
        }
    }

    fn check_claims(&self, v: &Verification, st: &TaskState) -> Option<String> {
        let obs = st.observed();
        let mut problems = Vec::new();
        if v.build == Some(true) && obs.build != Some(true) {
            problems.push("\"build\": true, but no successful build or test run happened after the last file change");
        }
        if v.test == Some(true) && obs.test != Some(true) {
            problems.push("\"test\": true, but no successful test run happened after the last file change");
        }
        if problems.is_empty() {
            return None;
        }
        Some(format!(
            "Your \"complete\" claims verification that I could not observe: {}.\nInstruct the Coder to run the build/tests now (or set the claim to false/null if they do not exist), then complete again.",
            problems.join("; ")
        ))
    }

    fn thinking_turn(&mut self, th: &mut History, st: &mut TaskState, mode: Mode) -> Result<Decision, AgentError> {
        let max_steps = self.cfg.agent.max_thinking_steps;
        let max_out = self.cfg.context.max_tool_output_bytes;
        let mut steps = 0usize;
        loop {
            if self.cancel.load(Ordering::Relaxed) {
                return Err(AgentError::Cancelled);
            }
            steps += 1;
            let parsed = self.call_model(Role::Thinking, th)?;
            match parsed.message {
                AgentMessage::Plan { steps: plan, .. } => {
                    self.ui.emit(Event::Plan(plan));
                    th.push("user", "feedback", "Plan recorded. Continue: inspect with a read-only tool, send an instruction to the Coder, or complete.");
                }
                AgentMessage::ToolCall { tool, arguments } => {
                    if !self.tools.is_read_only(&tool) {
                        let msg = if self.tools.spec(&tool).is_some() {
                            format!("The tool \"{tool}\" is not available to the Thinking model (read-only tools only). Delegate it to the Coder with an \"instruction\".")
                        } else {
                            format!("Unknown tool \"{tool}\".")
                        };
                        th.push("user", "feedback", msg);
                    } else {
                        let o = self.run_tool("thinking", &tool, &arguments, st);
                        th.push("user", "tool", prompts::format_tool_result(&tool, o.ok, &truncate_middle(&o.output, max_out)));
                    }
                }
                AgentMessage::Instruction { instruction, files, acceptance } => {
                    if mode == Mode::Thinking {
                        th.push("user", "feedback", "Instructions are disabled in thinking-only mode. Answer the user directly with a \"complete\" message.");
                    } else {
                        return Ok(Decision::Instruction { instruction, files, acceptance });
                    }
                }
                AgentMessage::Complete { summary, verification, status } => return Ok(Decision::Complete { summary, verification, status }),
                _ => {
                    th.push("user", "feedback", prompts::format_repair_request("That message type is not allowed for the Thinking model."));
                }
            }
            if steps >= max_steps {
                // the Thinking model keeps inspecting; force a decision
                if steps == max_steps {
                    th.push("user", "feedback", "You have used your inspection steps for this turn. Now send an \"instruction\" for the Coder or a \"complete\" message.");
                } else if steps >= max_steps + 2 {
                    return Err(AgentError::InvalidOutput("the Thinking model did not make a decision (instruction or complete)".into()));
                }
            }
        }
    }
}

enum Decision {
    Instruction { instruction: String, files: Vec<String>, acceptance: Vec<String> },
    Complete { summary: String, verification: Verification, status: CompleteStatus },
}

/// Convenience used by the CLI: a one-line description of the platform and workspace.
pub fn banner_info(workspace: &Path) -> String {
    format!("{} · {}", platform::os_name(), workspace.display())
}
