//! Command line interface: `qwen`, `qwen "task"`, `qwen .`, `qwen src/main.cpp`.

use crate::config::{Config, CONFIG_FILE_NAME, DEFAULT_CONFIG_TOML};
use crate::platform;
use crate::session::{exit_code, Flow, Session};
use crate::ui::{PlainUi, Ui};
use crate::APP_NAME;
use clap::Parser;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Parser, Debug, Clone, Default)]
#[command(
    name = "qwen",
    version,
    about = "Qwen Code - an AI coding agent for your terminal (Thinking model + Coder model)",
    after_help = "EXAMPLES:\n  qwen                         interactive session in the current directory\n  qwen \"fix the compile errors\"  run one task and exit\n  qwen .                       interactive session for the current project\n  qwen src/main.cpp            interactive session focused on a file\n  qwen --check                 verify configuration, providers and model runtimes\n  qwen --init                  write a default qwen.toml here"
)]
pub struct Cli {
    /// A task to run once (quote it), or a path (`.`, a directory or a file) to open an interactive session for.
    #[arg(value_name = "TASK_OR_PATH")]
    pub input: Vec<String>,

    /// Configuration file (TOML). Default: ./qwen.toml, then the user config directory.
    #[arg(short = 'c', long, value_name = "FILE")]
    pub config: Option<PathBuf>,

    /// Run as if started in this directory.
    #[arg(short = 'C', long, value_name = "DIR")]
    pub cwd: Option<PathBuf>,

    /// Use the plain line-based interface instead of the full-screen TUI.
    #[arg(long, conflicts_with = "tui")]
    pub repl: bool,

    /// Force the full-screen TUI.
    #[arg(long)]
    pub tui: bool,

    /// Answer every `ask` permission prompt with Y automatically (deny rules still apply).
    #[arg(short = 'y', long)]
    pub yes: bool,

    /// Override [qwen].max_iterations.
    #[arg(long, value_name = "N")]
    pub max_iterations: Option<usize>,

    /// Start in this mode: auto (Thinking supervises Coder), thinking, or coder.
    #[arg(long, value_name = "MODE", value_parser = ["auto", "thinking", "coder"])]
    pub mode: Option<String>,

    /// Initialise providers and model runtimes, report problems and exit.
    #[arg(long)]
    pub check: bool,

    /// Write a default qwen.toml into the current directory and exit.
    #[arg(long)]
    pub init: bool,

    /// Print the effective configuration as TOML and exit.
    #[arg(long)]
    pub print_config: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Target {
    pub workspace: PathBuf,
    pub focus: Vec<String>,
    pub task: Option<String>,
}

/// Decides what the positional arguments mean.
pub fn classify_input(args: &[String], cwd: &Path) -> Result<Target, String> {
    let base = Target { workspace: cwd.to_path_buf(), focus: vec![], task: None };
    if args.is_empty() {
        return Ok(base);
    }
    if args.len() == 1 && !args[0].chars().any(char::is_whitespace) {
        let arg = &args[0];
        let p = Path::new(arg);
        let abs = if p.is_absolute() { p.to_path_buf() } else { cwd.join(p) };
        if abs.is_dir() {
            return Ok(Target { workspace: platform::canonicalize_lossy(&abs), ..base });
        }
        if abs.is_file() {
            let abs = platform::canonicalize_lossy(&abs);
            let ws = platform::canonicalize_lossy(cwd);
            let rel = platform::display_path(&abs, &ws);
            return Ok(Target { workspace: ws, focus: vec![rel], task: None });
        }
        if arg.contains('/') || arg.contains('\\') {
            return Err(format!("path not found: {arg}"));
        }
    }
    Ok(Target { task: Some(args.join(" ")), ..base })
}

fn print_err(msg: &str) {
    eprintln!("error: {msg}");
}

/// Runs the program and returns the process exit code.
pub fn run(cli: Cli) -> i32 {
    platform::init_console();
    if platform::current_os() == platform::Os::Other {
        print_err("this platform is not supported (Windows and Linux only)");
        return 2;
    }
    let cwd = match &cli.cwd {
        Some(c) => platform::canonicalize_lossy(c),
        None => std::env::current_dir().map(platform::strip_unc).unwrap_or_else(|_| PathBuf::from(".")),
    };
    if !cwd.is_dir() {
        print_err(&format!("{} is not a directory", cwd.display()));
        return 2;
    }

    if cli.init {
        let target = cwd.join(CONFIG_FILE_NAME);
        if target.exists() {
            print_err(&format!("{} already exists", target.display()));
            return 1;
        }
        return match std::fs::write(&target, DEFAULT_CONFIG_TOML) {
            Ok(()) => {
                println!("Created {}", target.display());
                0
            }
            Err(e) => {
                print_err(&format!("cannot write {}: {e}", target.display()));
                1
            }
        };
    }

    let target = match classify_input(&cli.input, &cwd) {
        Ok(t) => t,
        Err(e) => {
            print_err(&e);
            return 2;
        }
    };
    let (mut cfg, config_path) = match Config::load(cli.config.as_deref(), &target.workspace) {
        Ok(v) => v,
        Err(e) => {
            print_err(&format!("{e:#}"));
            return 2;
        }
    };
    if let Some(n) = cli.max_iterations {
        if n == 0 {
            print_err("--max-iterations must be at least 1");
            return 2;
        }
        cfg.qwen.max_iterations = n;
    }
    if cli.print_config {
        return match cfg.to_toml() {
            Ok(t) => {
                println!("{t}");
                0
            }
            Err(e) => {
                print_err(&e.to_string());
                1
            }
        };
    }
    let base_dir = config_path.as_ref().and_then(|p| p.parent().map(|d| d.to_path_buf())).map(|d| if d.as_os_str().is_empty() { target.workspace.clone() } else { d }).unwrap_or_else(|| target.workspace.clone());
    let cfg = Arc::new(cfg);
    let cancel = Arc::new(AtomicBool::new(false));

    let one_shot = target.task.is_some();
    let interactive_tty = platform::is_stdin_tty() && platform::is_stdout_tty();
    let use_tui = !one_shot && !cli.check && !cli.repl && interactive_tty && (cli.tui || cfg.cli.ui != "repl");
    let use_tui = if cli.tui && !interactive_tty && !one_shot { false } else { use_tui };

    let mode = cli.mode.as_deref().and_then(crate::agent::Mode::parse);
    let build = {
        let cfg = cfg.clone();
        let config_path = config_path.clone();
        let ws = target.workspace.clone();
        let base_dir = base_dir.clone();
        let cancel = cancel.clone();
        let focus = target.focus.clone();
        let yes = cli.yes;
        move |ui: Arc<dyn Ui>| -> Result<Session, String> {
            let mut s = Session::new(cfg, config_path, &ws, &base_dir, ui, cancel, yes)?;
            s.agent.ctx.focus = focus;
            if let Some(m) = mode {
                s.mode = m;
            }
            Ok(s)
        }
    };

    if use_tui {
        return crate::tui::run(build, target.workspace.clone());
    }

    let ui: Arc<dyn Ui> = Arc::new(PlainUi::new(cfg.cli.color, cfg.cli.show_thinking));
    let mut session = match build(ui) {
        Ok(s) => s,
        Err(e) => {
            print_err(&e);
            return 2;
        }
    };
    install_ctrlc(session.busy.clone(), cancel);

    if cli.check {
        return if session.doctor() { 0 } else { 1 };
    }
    if let Some(task) = target.task {
        let out = session.run_task(&task);
        return exit_code(&out);
    }
    run_repl(&mut session, &cfg.cli.prompt)
}

fn install_ctrlc(busy: Arc<AtomicBool>, cancel: Arc<AtomicBool>) {
    let _ = ctrlc::set_handler(move || {
        if busy.load(Ordering::Relaxed) {
            cancel.store(true, Ordering::Relaxed);
        } else {
            println!();
            std::process::exit(130);
        }
    });
}

/// Plain line-based interactive loop.
pub fn run_repl(session: &mut Session, prompt: &str) -> i32 {
    println!("{APP_NAME}\n");
    println!("{}  ·  mode: {}  ·  /help for commands\n", crate::agent::banner_info(session.agent.workspace()), session.mode.name());
    let stdin = std::io::stdin();
    loop {
        print!("{prompt}");
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) | Err(_) => {
                println!();
                break;
            }
            Ok(_) => {}
        }
        match session.handle_line(&line) {
            Flow::Exit => break,
            Flow::Clear => {
                if platform::is_stdout_tty() {
                    print!("\x1b[2J\x1b[H");
                    let _ = std::io::stdout().flush();
                }
            }
            Flow::Continue => {}
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn classifies_arguments() {
        let dir = tempfile::tempdir().unwrap();
        let cwd = platform::canonicalize_lossy(dir.path());
        std::fs::create_dir_all(cwd.join("src")).unwrap();
        std::fs::write(cwd.join("src/main.cpp"), "int main(){}").unwrap();

        assert_eq!(classify_input(&[], &cwd).unwrap(), Target { workspace: cwd.clone(), focus: vec![], task: None });
        // `qwen .`
        let t = classify_input(&s(&["."]), &cwd).unwrap();
        assert_eq!((t.workspace.clone(), t.task.clone()), (platform::canonicalize_lossy(&cwd), None));
        // `qwen src/main.cpp`
        let t = classify_input(&s(&["src/main.cpp"]), &cwd).unwrap();
        assert_eq!(t.focus, vec!["src/main.cpp".to_string()]);
        assert_eq!(t.workspace, cwd);
        // `qwen src` (a directory) opens that directory as the workspace
        let t = classify_input(&s(&["src"]), &cwd).unwrap();
        assert_eq!(t.workspace, cwd.join("src"));
        // `qwen "fix the compile errors"`
        let t = classify_input(&s(&["fix the compile errors"]), &cwd).unwrap();
        assert_eq!(t.task.as_deref(), Some("fix the compile errors"));
        // several words without quotes are joined
        let t = classify_input(&s(&["fix", "the", "bug"]), &cwd).unwrap();
        assert_eq!(t.task.as_deref(), Some("fix the bug"));
        // Korean
        let t = classify_input(&s(&["컴파일 오류 고쳐줘"]), &cwd).unwrap();
        assert_eq!(t.task.as_deref(), Some("컴파일 오류 고쳐줘"));
        // a single word that is no path is a task; a missing path is an error
        assert_eq!(classify_input(&s(&["refactor"]), &cwd).unwrap().task.as_deref(), Some("refactor"));
        assert!(classify_input(&s(&["src/missing.cpp"]), &cwd).unwrap_err().contains("path not found"));
    }

    #[test]
    fn clap_parses_the_documented_forms() {
        let c = Cli::try_parse_from(["qwen", "fix the compile errors"]).unwrap();
        assert_eq!(c.input, vec!["fix the compile errors"]);
        let c = Cli::try_parse_from(["qwen", "-y", "--max-iterations", "5", "--mode", "coder", "."]).unwrap();
        assert!(c.yes && c.max_iterations == Some(5) && c.mode.as_deref() == Some("coder"));
        assert!(Cli::try_parse_from(["qwen", "--mode", "bogus"]).is_err());
        assert!(Cli::try_parse_from(["qwen", "--repl", "--tui"]).is_err());
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }
}
