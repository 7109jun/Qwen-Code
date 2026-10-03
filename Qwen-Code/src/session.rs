//! A session ties the agent to a user interface: slash commands, modes and diagnostics.
//! It is UI-agnostic; all output goes through the [`Ui`] trait.

use crate::agent::{Agent, Mode, Status, TaskOutcome};
use crate::config::{Config, Policy};
use crate::git::Git;
use crate::models::{ModelRole, ModelSet};
use crate::permissions::{Category, PermissionManager};
use crate::platform;
use crate::ui::{Event, Ui};
use crate::{APP_NAME, VERSION};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    Continue,
    Exit,
    /// Clear the screen / log view.
    Clear,
}

pub struct Session {
    pub agent: Agent,
    pub mode: Mode,
    pub cfg: Arc<Config>,
    pub config_path: Option<PathBuf>,
    pub base_dir: PathBuf,
    pub ui: Arc<dyn Ui>,
    pub last: Option<TaskOutcome>,
    /// `true` while a task is running (used by the Ctrl-C handler).
    pub busy: Arc<AtomicBool>,
}

pub const HELP_TEXT: &str = "\
Qwen Code - AI coding agent. Type a request in plain language (English, Korean, ...), or a command:

  /help                 show this help
  /models               show the Thinking and Coder models
  /model [coder|thinking|auto]
                        show or set who answers: auto = Thinking supervises the Coder (default),
                        thinking = Thinking model only (analysis, read-only), coder = Coder directly
  /context              show context usage (tokens, files in context, compaction)
  /tools                list the tools the agent can call
  /status               show workspace, Git, models, runtime and statistics
  /diff                 show the current Git diff
  /reset                forget the conversation and file state
  /clear                clear the screen
  /permissions          show permission policies
                        /permissions set <category> <allow|ask|deny>
                        /permissions command <destructive|system_destructive|unknown> <allow|ask|deny>
                        /permissions yes <on|off>     (auto-answer ask prompts; deny stays enforced)
  /config [show]        show the active configuration (qwen.toml)
  /exit                 quit (also /quit, Ctrl-D)

Press Ctrl-C (REPL) or Esc (TUI) to cancel a running task.";

impl Session {
    pub fn new(cfg: Arc<Config>, config_path: Option<PathBuf>, workspace: &Path, base_dir: &Path, ui: Arc<dyn Ui>, cancel: Arc<AtomicBool>, auto_yes: bool) -> Result<Session, String> {
        let models = ModelSet::from_config(&cfg, base_dir)?;
        Ok(Self::with_models(cfg, config_path, workspace, base_dir, models, ui, cancel, auto_yes))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_models(cfg: Arc<Config>, config_path: Option<PathBuf>, workspace: &Path, base_dir: &Path, models: ModelSet, ui: Arc<dyn Ui>, cancel: Arc<AtomicBool>, auto_yes: bool) -> Session {
        let mut perms = PermissionManager::new(&cfg.permissions);
        perms.set_auto_approve(auto_yes);
        let agent = Agent::new(cfg.clone(), workspace, models, perms, ui.clone(), cancel);
        Session { agent, mode: Mode::Auto, cfg, config_path, base_dir: base_dir.to_path_buf(), ui, last: None, busy: Arc::new(AtomicBool::new(false)) }
    }

    fn info(&self, s: impl Into<String>) {
        self.ui.emit(Event::Info(s.into()));
    }

    fn err(&self, s: impl Into<String>) {
        self.ui.emit(Event::Error(s.into()));
    }

    pub fn banner(&self) -> String {
        format!("{APP_NAME}\n")
    }

    /// Handles one line typed by the user: a slash command or a task for the agent.
    pub fn handle_line(&mut self, line: &str) -> Flow {
        let line = line.trim();
        if line.is_empty() {
            return Flow::Continue;
        }
        if let Some(cmd) = line.strip_prefix('/') {
            // `/path/to/file` is a task, not a command
            let name = cmd.split_whitespace().next().unwrap_or("");
            if !name.contains('/') && !name.contains('\\') && !name.contains('.') {
                return self.command(cmd);
            }
        }
        self.run_task(line);
        Flow::Continue
    }

    pub fn run_task(&mut self, task: &str) -> TaskOutcome {
        self.busy.store(true, Ordering::Relaxed);
        let out = self.agent.run_task(task, self.mode);
        self.busy.store(false, Ordering::Relaxed);
        self.last = Some(out.clone());
        out
    }

    fn command(&mut self, cmd: &str) -> Flow {
        let mut parts = cmd.split_whitespace();
        let name = parts.next().unwrap_or("").to_ascii_lowercase();
        let args: Vec<&str> = parts.collect();
        match name.as_str() {
            "help" | "?" | "h" => self.info(HELP_TEXT),
            "models" => self.cmd_models(),
            "model" => self.cmd_model(&args),
            "context" => self.cmd_context(),
            "tools" => self.cmd_tools(),
            "status" => self.cmd_status(),
            "diff" => self.cmd_diff(),
            "reset" => {
                self.agent.reset();
                self.last = None;
                self.info("Conversation and file state reset.");
            }
            "clear" => return Flow::Clear,
            "permissions" | "perm" => self.cmd_permissions(&args),
            "config" => self.cmd_config(&args),
            "exit" | "quit" | "q" => return Flow::Exit,
            other => self.err(format!("unknown command /{other}. Type /help for the list of commands.")),
        }
        Flow::Continue
    }

    fn cmd_models(&self) {
        let mut out = String::from("Models:\n");
        for m in [self.agent.models.thinking.as_ref() as &dyn ModelRole, self.agent.models.coder.as_ref() as &dyn ModelRole] {
            let i = m.info();
            out.push_str(&format!(
                "  {:<9} {} / {}  [{}]\n            {}\n",
                i.role,
                i.provider,
                i.name,
                if i.loaded { "loaded" } else { "not loaded yet" },
                i.description
            ));
        }
        out.push_str(&format!("Mode: {}  (change with /model coder | thinking | auto)", self.mode.name()));
        self.info(out);
    }

    fn cmd_model(&mut self, args: &[&str]) {
        match args.first() {
            None => self.info(format!("Current mode: {}\nUse /model coder, /model thinking or /model auto.", self.mode.name())),
            Some(a) => match Mode::parse(a) {
                Some(m) => {
                    self.mode = m;
                    self.info(format!("Mode set to {}.", m.name()));
                }
                None => self.err(format!("unknown model role \"{a}\". Use: coder, thinking or auto.")),
            },
        }
    }

    fn cmd_context(&self) {
        let used = self.agent.history_tokens();
        let budget = self.agent.context_budget();
        let mut out = format!(
            "Context (Thinking conversation):\n  history: {} entries, ≈ {} tokens of a {} token budget ({}%)\n  compactions so far: {}\n",
            self.agent.history_len(),
            used,
            budget,
            used * 100 / budget.max(1),
            self.agent.compactions()
        );
        let files = self.agent.ctx.last_files();
        if files.is_empty() {
            out.push_str("  retrieved files: none yet\n");
        } else {
            out.push_str("  retrieved files:\n");
            for f in files {
                out.push_str(&format!("    {f}\n"));
            }
        }
        if !self.agent.ctx.focus.is_empty() {
            out.push_str(&format!("  pinned files: {}\n", self.agent.ctx.focus.join(", ")));
        }
        out.push_str(&format!(
            "  settings: max_tokens={} compact_threshold={} keep_recent={} max_tool_output_bytes={}",
            self.cfg.context.max_tokens, self.cfg.context.compact_threshold, self.cfg.context.keep_recent, self.cfg.context.max_tool_output_bytes
        ));
        self.info(out);
    }

    fn cmd_tools(&self) {
        let mut out = String::from("Tools (model JSON → schema validation → permission check → execution):\n");
        for s in self.agent.tools.specs() {
            out.push_str(&format!("  {:<15} {}{}\n", s.name, s.description, if s.read_only { "  [read-only; also usable by Thinking]" } else { "" }));
        }
        self.info(out.trim_end().to_string());
    }

    fn cmd_status(&self) {
        let ws = self.agent.workspace().to_path_buf();
        let mut out = format!("{APP_NAME} {VERSION}\n  platform:  {}\n  workspace: {}\n", platform::os_name(), ws.display());
        out.push_str(&format!("  config:    {}\n", self.config_path.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "built-in defaults".into())));
        if self.agent.is_git_repo() {
            let g = Git::new(&ws);
            out.push_str(&format!("  git:       branch {}\n", g.current_branch().unwrap_or_else(|| "?".into())));
        } else {
            out.push_str("  git:       not a Git repository\n");
        }
        out.push_str(&format!("  mode:      {}\n  max_iterations: {}\n", self.mode.name(), self.agent.max_iterations));
        for m in [self.agent.models.thinking.as_ref() as &dyn ModelRole, self.agent.models.coder.as_ref() as &dyn ModelRole] {
            let i = m.info();
            out.push_str(&format!("  {:<9} {}/{} ({}, {})\n", i.role, i.provider, i.name, i.runtime_kind, if i.loaded { "loaded" } else { "not loaded" }));
        }
        let lib = platform::find_onnxruntime_lib(self.cfg.runtime.onnx.dylib_path.as_deref(), &self.base_dir);
        out.push_str(&format!(
            "  onnxruntime: {} (device {})\n",
            lib.map(|p| p.display().to_string()).unwrap_or_else(|| "shared library not found (only needed for runtime = \"onnx\")".into()),
            self.cfg.runtime.onnx.device
        ));
        let st = &self.agent.stats;
        out.push_str(&format!(
            "  session:   {} task(s), {} model call(s), {} tool call(s), ≈{} prompt / {} completion tokens\n",
            st.tasks, st.model_calls, st.tool_calls, st.prompt_tokens, st.completion_tokens
        ));
        if let Some(l) = &self.last {
            out.push_str(&format!("  last task: {:?} ({} iteration(s), {} tool call(s))", l.status, l.iterations, l.tool_calls));
        } else {
            out.push_str("  last task: none");
        }
        self.info(out);
    }

    fn cmd_diff(&self) {
        if !self.agent.is_git_repo() {
            self.err("this workspace is not a Git repository");
            return;
        }
        let g = Git::new(self.agent.workspace());
        let stat = g.diff(false, None, true);
        let staged = g.diff(true, None, true);
        let full = g.diff(false, None, false);
        if !full.ok {
            self.err(format!("git diff failed: {}", full.output));
            return;
        }
        let mut shown = false;
        if staged.ok && !staged.output.contains("no differences") {
            self.info(format!("Staged changes:\n{}", staged.output));
            shown = true;
        }
        if full.output.contains("no differences") {
            if !shown {
                self.info("No unstaged changes (working tree matches the index).");
            }
            return;
        }
        self.info(format!("Unstaged changes:\n{}", stat.output));
        let lines: Vec<&str> = full.output.lines().collect();
        let max = 400;
        let mut text = lines.iter().take(max).cloned().collect::<Vec<_>>().join("\n");
        if lines.len() > max {
            text.push_str(&format!("\n… {} more lines (run `git diff` for everything)", lines.len() - max));
        }
        self.ui.emit(Event::Diff(text));
    }

    fn cmd_permissions(&mut self, args: &[&str]) {
        if args.is_empty() {
            let mut out = String::from("Permission policies (allow = run, ask = Allow [Y/N] prompt, deny = never):\n");
            for l in self.agent.perms.describe() {
                out.push_str(&format!("  {l}\n"));
            }
            out.push_str("Change: /permissions set <category> <allow|ask|deny>   |   /permissions command <kind> <policy>   |   /permissions yes <on|off>\nPersistent changes belong in qwen.toml ([permissions]).");
            self.info(out);
            return;
        }
        let parse_policy = |s: &str| s.parse::<Policy>().map_err(|e| e.to_string());
        match (args[0], args.get(1), args.get(2)) {
            ("set", Some(cat), Some(pol)) => match (Category::parse(cat), parse_policy(pol)) {
                (Some(c), Ok(p)) => {
                    self.agent.perms.set_policy(c, p);
                    self.info(format!("{} → {}", c.name(), p.as_str()));
                }
                (None, _) if *cat == "default" => match parse_policy(pol) {
                    Ok(p) => {
                        self.agent.perms.set_default(p);
                        self.info(format!("default → {}", p.as_str()));
                    }
                    Err(e) => self.err(e),
                },
                (None, _) => self.err(format!("unknown category \"{cat}\" (read, write, edit, delete, execute, network, git, process, filesystem, default)")),
                (_, Err(e)) => self.err(e),
            },
            ("command", Some(kind), Some(pol)) => match parse_policy(pol) {
                Ok(p) => {
                    if self.agent.perms.set_command_policy(kind, p) {
                        self.info(format!("commands.{kind} → {}", p.as_str()));
                    } else {
                        self.err(format!("unknown command class \"{kind}\" (destructive, system_destructive, unknown)"));
                    }
                }
                Err(e) => self.err(e),
            },
            ("yes", Some(v), None) => match *v {
                "on" => {
                    self.agent.perms.set_auto_approve(true);
                    self.info("Auto-approve is ON: `ask` prompts are answered with Y automatically (deny rules are still enforced).");
                }
                "off" => {
                    self.agent.perms.set_auto_approve(false);
                    self.info("Auto-approve is OFF.");
                }
                _ => self.err("usage: /permissions yes <on|off>"),
            },
            _ => self.err("usage: /permissions | /permissions set <category> <policy> | /permissions command <kind> <policy> | /permissions yes <on|off>"),
        }
    }

    fn cmd_config(&self, args: &[&str]) {
        if args.first() == Some(&"show") {
            match self.cfg.to_toml() {
                Ok(t) => self.info(t.trim_end().to_string()),
                Err(e) => self.err(e.to_string()),
            }
            return;
        }
        let c = &self.cfg;
        let mut out = format!("Configuration: {}\n", self.config_path.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "built-in defaults (no qwen.toml found; create one with `qwen --init`)".into()));
        out.push_str(&format!("  [qwen]        max_iterations = {}\n", c.qwen.max_iterations));
        for role in ["thinking", "coder"] {
            let m = c.resolve_model(role, &self.base_dir);
            out.push_str(&format!("  [models.{role}] provider = {}, name = {}, runtime = {}, path = {}\n", m.cfg.provider, m.cfg.name, m.cfg.runtime, m.path.display()));
        }
        out.push_str(&format!("  [runtime.onnx] device = {}, dylib_path = {}\n", c.runtime.onnx.device, c.runtime.onnx.dylib_path.as_deref().unwrap_or("(auto)")));
        out.push_str(&format!("  [context]     max_tokens = {}, compact_threshold = {}\n", c.context.max_tokens, c.context.compact_threshold));
        out.push_str(&format!("  [agent]       max_coder_steps = {}, max_thinking_steps = {}\n", c.agent.max_coder_steps, c.agent.max_thinking_steps));
        out.push_str(&format!("  [tools]       shell = {}, command_timeout_secs = {}\n", c.tools.shell, c.tools.command_timeout_secs));
        out.push_str(&format!("  [cli]         ui = {}\n", c.cli.ui));
        out.push_str("Use `/config show` for the full TOML and `/permissions` for the permission policies.");
        self.info(out);
    }

    /// `qwen --check`: initialises providers and runtimes and reports problems. Returns `true` if everything works.
    pub fn doctor(&self) -> bool {
        let mut ok = true;
        self.info(format!("{APP_NAME} {VERSION} — {} — workspace {}", platform::os_name(), self.agent.workspace().display()));
        self.info(format!("config: {}", self.config_path.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "built-in defaults".into())));
        self.info(format!("git: {}", if Git::is_available() { if self.agent.is_git_repo() { "available, repository detected" } else { "available, not a repository" } } else { "NOT installed" }));
        for m in [self.agent.models.thinking.as_ref() as &dyn ModelRole, self.agent.models.coder.as_ref() as &dyn ModelRole] {
            let i = m.info();
            self.info(format!("{} model: provider {} / {} ({})", i.role, i.provider, i.name, i.runtime_kind));
            match m.load() {
                Ok(d) => self.info(format!("  OK: {d}")),
                Err(e) => {
                    ok = false;
                    self.err(format!("  {} model cannot be used yet: {e}", i.role));
                }
            }
        }
        ok
    }
}

/// One-line result of a task for exit codes in one-shot mode.
pub fn exit_code(out: &TaskOutcome) -> i32 {
    match out.status {
        Status::Completed => 0,
        Status::NeedsInput | Status::Blocked => 3,
        Status::Cancelled => 130,
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{CoderModel, ModelHandle, ThinkingModel};
    use crate::runtime::MockRuntime;
    use crate::ui::BufferUi;

    fn session(thinking: Vec<String>, coder: Vec<String>) -> (Session, Arc<BufferUi>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let cfg = Arc::new(Config::default());
        let models = ModelSet::new(
            Arc::new(ThinkingModel(ModelHandle::from_runtime("thinking", "qwen", "think-x", Arc::new(MockRuntime::scripted("t", thinking))))),
            Arc::new(CoderModel(ModelHandle::from_runtime("coder", "qwen", "coder-x", Arc::new(MockRuntime::scripted("c", coder))))),
        );
        let ui = Arc::new(BufferUi::new());
        let s = Session::with_models(cfg, None, dir.path(), dir.path(), models, ui.clone(), Arc::new(AtomicBool::new(false)), false);
        (s, ui, dir)
    }

    fn infos(ui: &BufferUi) -> String {
        ui.events().iter().filter_map(|e| if let Event::Info(s) = e { Some(s.clone()) } else { None }).collect::<Vec<_>>().join("\n")
    }

    #[test]
    fn all_documented_commands_exist() {
        let (mut s, ui, _d) = session(vec![], vec![]);
        for c in ["/help", "/models", "/model", "/context", "/tools", "/status", "/permissions", "/config", "/config show", "/reset"] {
            assert_eq!(s.handle_line(c), Flow::Continue, "{c}");
        }
        assert!(!ui.events().iter().any(|e| matches!(e, Event::Error(_))), "{}", ui.text());
        assert_eq!(s.handle_line("/clear"), Flow::Clear);
        assert_eq!(s.handle_line("/exit"), Flow::Exit);
        assert_eq!(s.handle_line("/quit"), Flow::Exit);
        s.handle_line("/bogus");
        assert!(ui.text().contains("unknown command /bogus"));
    }

    #[test]
    fn help_lists_every_command() {
        for c in ["/help", "/models", "/model", "/context", "/tools", "/status", "/diff", "/reset", "/clear", "/permissions", "/config", "/exit"] {
            assert!(HELP_TEXT.contains(c), "{c}");
        }
    }

    #[test]
    fn model_command_switches_mode() {
        let (mut s, ui, _d) = session(vec![], vec![]);
        assert_eq!(s.mode, Mode::Auto);
        s.handle_line("/model coder");
        assert_eq!(s.mode, Mode::Coder);
        s.handle_line("/model thinking");
        assert_eq!(s.mode, Mode::Thinking);
        s.handle_line("/model nonsense");
        assert!(ui.text().contains("unknown model role"));
        s.handle_line("/models");
        assert!(infos(&ui).contains("think-x") && infos(&ui).contains("coder-x"));
    }

    #[test]
    fn permissions_can_be_changed_at_runtime() {
        let (mut s, ui, _d) = session(vec![], vec![]);
        s.handle_line("/permissions");
        assert!(infos(&ui).contains("delete") && infos(&ui).contains("ask"));
        s.handle_line("/permissions set delete allow");
        assert_eq!(s.agent.perms.policy_for(Category::Delete), Policy::Allow);
        s.handle_line("/permissions set network deny");
        assert_eq!(s.agent.perms.policy_for(Category::Network), Policy::Deny);
        s.handle_line("/permissions command destructive deny");
        assert_eq!(s.agent.perms.command_policies().destructive, Policy::Deny);
        s.handle_line("/permissions yes on");
        assert!(s.agent.perms.auto_approve());
        s.handle_line("/permissions set bogus allow");
        s.handle_line("/permissions set delete maybe");
        assert_eq!(ui.events().iter().filter(|e| matches!(e, Event::Error(_))).count(), 2);
    }

    #[test]
    fn plain_text_runs_a_task_and_slash_paths_are_tasks() {
        let (mut s, ui, _d) = session(
            vec![r#"{"type":"complete","summary":"Hello there","status":"done"}"#.into(), r#"{"type":"complete","summary":"second"}"#.into()],
            vec![],
        );
        s.handle_line("say hello");
        assert!(s.last.as_ref().unwrap().is_success());
        assert!(ui.events().iter().any(|e| matches!(e, Event::Message(m) if m.contains("Hello there"))));
        // "/src/main.rs is broken" is a request, not a command
        s.handle_line("/src/main.rs is broken");
        assert_eq!(s.agent.stats.tasks, 2);
        assert!(!ui.text().contains("unknown command"));
    }

    #[test]
    fn status_context_and_diff_work_outside_git() {
        let (mut s, ui, _d) = session(vec![], vec![]);
        s.handle_line("/status");
        s.handle_line("/context");
        s.handle_line("/diff");
        let t = infos(&ui);
        assert!(t.contains("workspace:") && t.contains("not a Git repository") && t.contains("history: 0 entries"), "{t}");
        assert!(ui.text().contains("not a Git repository"));
    }

    #[test]
    fn exit_codes() {
        let mk = |status| TaskOutcome { status, summary: String::new(), verification: Default::default(), iterations: 0, tool_calls: 0, files_changed: vec![] };
        assert_eq!(exit_code(&mk(Status::Completed)), 0);
        assert_eq!(exit_code(&mk(Status::Failed("x".into()))), 1);
        assert_eq!(exit_code(&mk(Status::MaxIterations)), 1);
        assert_eq!(exit_code(&mk(Status::NeedsInput)), 3);
    }
}
