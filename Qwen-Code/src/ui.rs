//! UI abstraction. The agent talks to the user only through the [`Ui`] trait, so the same core runs
//! with the plain REPL, the TUI, or silently in tests.

use std::io::{BufRead, Write};
use std::sync::Mutex;

#[derive(Debug, Clone)]
pub enum Event {
    Info(String),
    Warn(String),
    Error(String),
    /// A new phase of the agent loop started (`thinking`, `coder`, ...).
    Phase { role: String, text: String },
    /// Raw reasoning text of a model (hidden unless `cli.show_thinking`).
    Reasoning(String),
    Plan(Vec<String>),
    ToolCall { role: String, tool: String, summary: String },
    ToolResult { tool: String, ok: bool, summary: String },
    /// Final answer / report for the user.
    Message(String),
    Diff(String),
}

#[derive(Debug, Clone)]
pub struct ConfirmRequest {
    /// The command / action the agent wants to execute.
    pub command: String,
    pub reason: Option<String>,
}

pub trait Ui: Send + Sync {
    fn emit(&self, event: Event);
    /// Asks the user `Allow [Y/N]`. Must return `false` when no answer can be obtained.
    fn confirm(&self, request: &ConfirmRequest) -> bool;
}

/// Parses the answer to an `Allow [Y/N]` prompt. `None` means "not understood".
pub fn parse_yes_no(answer: &str) -> Option<bool> {
    match answer.trim().to_lowercase().as_str() {
        "y" | "yes" | "ㅛ" | "예" | "네" | "응" | "ㅇㅇ" | "허용" => Some(true),
        "n" | "no" | "ㅜ" | "아니" | "아니오" | "아니요" | "노" | "거부" | "" => Some(false),
        _ => None,
    }
}

/// The exact confirmation text required by the specification.
pub fn confirm_prompt_text(req: &ConfirmRequest) -> String {
    format!("Qwen Code wants to execute:\n\n{}\n\nAllow [Y/N]: ", req.command)
}

// ---------------------------------------------------------------------------------------------
// Plain terminal UI
// ---------------------------------------------------------------------------------------------

pub struct PlainUi {
    color: bool,
    show_reasoning: bool,
    lock: Mutex<()>,
}

impl PlainUi {
    pub fn new(color: bool, show_reasoning: bool) -> Self {
        let color = color && std::env::var_os("NO_COLOR").is_none() && crate::platform::is_stdout_tty();
        Self { color, show_reasoning, lock: Mutex::new(()) }
    }

    fn paint(&self, code: &str, s: &str) -> String {
        if self.color {
            format!("\x1b[{code}m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }
}

fn clip(s: &str, max: usize) -> String {
    let one_line = s.replace('\n', " ");
    if one_line.chars().count() > max {
        format!("{}…", one_line.chars().take(max).collect::<String>())
    } else {
        one_line
    }
}

impl Ui for PlainUi {
    fn emit(&self, event: Event) {
        let _g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        let mut out = std::io::stdout().lock();
        match event {
            Event::Info(s) => {
                let _ = writeln!(out, "{s}");
            }
            Event::Warn(s) => {
                let _ = writeln!(out, "{}", self.paint("33", &format!("warning: {s}")));
            }
            Event::Error(s) => {
                let _ = writeln!(out, "{}", self.paint("31", &format!("error: {s}")));
            }
            Event::Phase { role, text } => {
                let _ = writeln!(out, "{}", self.paint("1;36", &format!("[{role}] {text}")));
            }
            Event::Reasoning(s) => {
                if self.show_reasoning {
                    let _ = writeln!(out, "{}", self.paint("2", &s));
                }
            }
            Event::Plan(steps) => {
                let _ = writeln!(out, "{}", self.paint("1", "Plan:"));
                for (i, s) in steps.iter().enumerate() {
                    let _ = writeln!(out, "  {}. {}", i + 1, s);
                }
            }
            Event::ToolCall { role, tool, summary } => {
                let _ = writeln!(out, "{}", self.paint("34", &format!("  → {role}: {tool} {}", clip(&summary, 160))));
            }
            Event::ToolResult { tool, ok, summary } => {
                let mark = if ok { self.paint("32", "✓") } else { self.paint("31", "✗") };
                let _ = writeln!(out, "  {mark} {tool}: {}", clip(&summary, 200));
            }
            Event::Message(s) => {
                let _ = writeln!(out, "\n{s}\n");
            }
            Event::Diff(s) => {
                for line in s.lines() {
                    let painted = if line.starts_with('+') && !line.starts_with("+++") {
                        self.paint("32", line)
                    } else if line.starts_with('-') && !line.starts_with("---") {
                        self.paint("31", line)
                    } else if line.starts_with("@@") {
                        self.paint("36", line)
                    } else {
                        line.to_string()
                    };
                    let _ = writeln!(out, "{painted}");
                }
            }
        }
        let _ = out.flush();
    }

    fn confirm(&self, request: &ConfirmRequest) -> bool {
        let _g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        let stdin = std::io::stdin();
        for attempt in 0..3 {
            {
                let mut out = std::io::stdout().lock();
                if attempt == 0 {
                    if let Some(r) = &request.reason {
                        let _ = writeln!(out, "{}", self.paint("33", &format!("! {r}")));
                    }
                    let _ = write!(out, "{}", confirm_prompt_text(request));
                } else {
                    let _ = write!(out, "Please answer Y or N: ");
                }
                let _ = out.flush();
            }
            let mut line = String::new();
            match stdin.lock().read_line(&mut line) {
                Ok(0) | Err(_) => {
                    println!();
                    return false;
                }
                Ok(_) => {}
            }
            if let Some(v) = parse_yes_no(&line) {
                return v;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------------------------
// Buffer UI (tests, scripted use)
// ---------------------------------------------------------------------------------------------

#[derive(Default)]
pub struct BufferUi {
    pub events: Mutex<Vec<Event>>,
    pub answers: Mutex<Vec<bool>>,
    pub prompts: Mutex<Vec<ConfirmRequest>>,
}

impl BufferUi {
    pub fn new() -> Self {
        Self::default()
    }

    /// Answers are consumed from the front; once exhausted the answer is `false`.
    pub fn with_answers(answers: Vec<bool>) -> Self {
        Self { answers: Mutex::new(answers), ..Self::default() }
    }

    pub fn events(&self) -> Vec<Event> {
        self.events.lock().unwrap().clone()
    }

    pub fn prompts(&self) -> Vec<ConfirmRequest> {
        self.prompts.lock().unwrap().clone()
    }

    pub fn text(&self) -> String {
        self.events().iter().map(|e| format!("{e:?}")).collect::<Vec<_>>().join("\n")
    }
}

impl Ui for BufferUi {
    fn emit(&self, event: Event) {
        self.events.lock().unwrap().push(event);
    }

    fn confirm(&self, request: &ConfirmRequest) -> bool {
        self.prompts.lock().unwrap().push(request.clone());
        let mut a = self.answers.lock().unwrap();
        if a.is_empty() {
            false
        } else {
            a.remove(0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yes_no_parsing() {
        assert_eq!(parse_yes_no("Y\n"), Some(true));
        assert_eq!(parse_yes_no("yes"), Some(true));
        assert_eq!(parse_yes_no("예"), Some(true));
        assert_eq!(parse_yes_no("N"), Some(false));
        assert_eq!(parse_yes_no(""), Some(false));
        assert_eq!(parse_yes_no("maybe"), None);
    }

    #[test]
    fn prompt_has_the_required_format() {
        let t = confirm_prompt_text(&ConfirmRequest { command: "rm -rf build".into(), reason: None });
        assert_eq!(t, "Qwen Code wants to execute:\n\nrm -rf build\n\nAllow [Y/N]: ");
    }
}
