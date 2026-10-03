//! Full-screen terminal UI (ratatui). The session runs on a worker thread; the UI thread only
//! draws and handles keys, so it stays responsive while models and tools work.

use crate::agent::Mode;
use crate::session::{Flow, Session};
use crate::ui::{ConfirmRequest, Event, Ui};
use crate::APP_NAME;
use ratatui::crossterm::event::{self, Event as CtEvent, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

// ---------------------------------------------------------------------------------------------
// Messages between the worker and the UI thread
// ---------------------------------------------------------------------------------------------

pub enum Msg {
    Ev(Event),
    Confirm { request: ConfirmRequest, reply: Sender<bool> },
    /// The session finished handling a line.
    Done(Flow),
    Ready { mode: Mode },
    Fatal(String),
}

pub enum Cmd {
    Line(String),
    Quit,
}

struct TuiUi {
    tx: Mutex<Sender<Msg>>,
}

impl Ui for TuiUi {
    fn emit(&self, event: Event) {
        let _ = self.tx.lock().unwrap_or_else(|e| e.into_inner()).send(Msg::Ev(event));
    }

    fn confirm(&self, request: &ConfirmRequest) -> bool {
        let (reply, answer) = mpsc::channel();
        if self.tx.lock().unwrap_or_else(|e| e.into_inner()).send(Msg::Confirm { request: request.clone(), reply }).is_err() {
            return false;
        }
        answer.recv().unwrap_or(false)
    }
}

// ---------------------------------------------------------------------------------------------
// Application state (testable without a terminal)
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    None,
    Submit(String),
    Cancel,
    Quit,
    Answer(bool),
}

#[derive(Clone)]
struct LogLine {
    style: Style,
    text: String,
}

pub struct App {
    pub workspace: String,
    log: Vec<LogLine>,
    pub input: String,
    /// Cursor position in characters.
    pub cursor: usize,
    history: Vec<String>,
    history_pos: Option<usize>,
    pub scroll: usize,
    pub busy: bool,
    pub mode: String,
    pub confirm: Option<ConfirmRequest>,
    pub status: String,
    pub quit: bool,
}

fn style_fg(c: Color) -> Style {
    Style::default().fg(c)
}

impl App {
    pub fn new(workspace: &str) -> Self {
        let mut app = Self {
            workspace: workspace.to_string(),
            log: Vec::new(),
            input: String::new(),
            cursor: 0,
            history: Vec::new(),
            history_pos: None,
            scroll: 0,
            busy: false,
            mode: Mode::Auto.name().to_string(),
            confirm: None,
            status: String::new(),
            quit: false,
        };
        app.push(Style::default().add_modifier(Modifier::BOLD), "Type a request (English, Korean, ...) or /help. Esc cancels a running task, Ctrl-C quits when idle.");
        app
    }

    pub fn push(&mut self, style: Style, text: &str) {
        for l in text.lines() {
            self.log.push(LogLine { style, text: l.to_string() });
        }
        if text.is_empty() {
            self.log.push(LogLine { style, text: String::new() });
        }
        if self.log.len() > 5000 {
            let drop = self.log.len() - 4000;
            self.log.drain(..drop);
        }
        self.scroll = 0;
    }

    pub fn clear_log(&mut self) {
        self.log.clear();
        self.scroll = 0;
    }

    pub fn log_text(&self) -> String {
        self.log.iter().map(|l| l.text.clone()).collect::<Vec<_>>().join("\n")
    }

    pub fn on_event(&mut self, e: Event) {
        match e {
            Event::Info(s) => self.push(Style::default(), &s),
            Event::Warn(s) => self.push(style_fg(Color::Yellow), &format!("warning: {s}")),
            Event::Error(s) => self.push(style_fg(Color::Red), &format!("error: {s}")),
            Event::Phase { role, text } => {
                self.status = format!("{role}: {text}");
                self.push(style_fg(Color::Cyan).add_modifier(Modifier::BOLD), &format!("[{role}] {text}"));
            }
            Event::Reasoning(s) => self.push(style_fg(Color::DarkGray), &s),
            Event::Plan(steps) => {
                self.push(Style::default().add_modifier(Modifier::BOLD), "Plan:");
                for (i, s) in steps.iter().enumerate() {
                    self.push(Style::default(), &format!("  {}. {s}", i + 1));
                }
            }
            Event::ToolCall { role, tool, summary } => self.push(style_fg(Color::Blue), &format!("  → {role}: {tool} {summary}")),
            Event::ToolResult { tool, ok, summary } => {
                let (mark, c) = if ok { ("✓", Color::Green) } else { ("✗", Color::Red) };
                self.push(style_fg(c), &format!("  {mark} {tool}: {summary}"));
            }
            Event::Message(s) => {
                self.push(Style::default(), "");
                self.push(Style::default().add_modifier(Modifier::BOLD), &s);
                self.push(Style::default(), "");
            }
            Event::Diff(s) => {
                for l in s.lines() {
                    let st = if l.starts_with('+') && !l.starts_with("+++") {
                        style_fg(Color::Green)
                    } else if l.starts_with('-') && !l.starts_with("---") {
                        style_fg(Color::Red)
                    } else if l.starts_with("@@") {
                        style_fg(Color::Cyan)
                    } else {
                        Style::default()
                    };
                    self.push(st, l);
                }
            }
        }
    }

    pub fn on_msg(&mut self, m: Msg) {
        match m {
            Msg::Ev(e) => self.on_event(e),
            Msg::Confirm { request, reply } => {
                // The reply channel is parked in the caller; here we only show the prompt.
                self.push(style_fg(Color::Yellow), &format!("Qwen Code wants to execute:\n\n{}\n\nAllow [Y/N]:", request.command));
                self.confirm = Some(request);
                drop(reply);
            }
            Msg::Done(flow) => {
                self.busy = false;
                self.status.clear();
                match flow {
                    Flow::Exit => self.quit = true,
                    Flow::Clear => self.clear_log(),
                    Flow::Continue => {}
                }
            }
            Msg::Ready { mode } => self.mode = mode.name().to_string(),
            Msg::Fatal(s) => {
                self.push(style_fg(Color::Red), &format!("error: {s}"));
                self.quit = true;
            }
        }
    }

    fn byte_index(&self, chars: usize) -> usize {
        self.input.char_indices().nth(chars).map(|(i, _)| i).unwrap_or(self.input.len())
    }

    pub fn on_key(&mut self, key: KeyEvent) -> Action {
        if key.kind != KeyEventKind::Press {
            return Action::None;
        }
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        if self.confirm.is_some() {
            return match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Char('ㅛ') => {
                    self.confirm = None;
                    self.push(style_fg(Color::Green), "Y");
                    Action::Answer(true)
                }
                KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Char('ㅜ') | KeyCode::Esc | KeyCode::Enter => {
                    self.confirm = None;
                    self.push(style_fg(Color::Red), "N");
                    Action::Answer(false)
                }
                KeyCode::Char('c') if ctrl => {
                    self.confirm = None;
                    Action::Answer(false)
                }
                _ => Action::None,
            };
        }
        match key.code {
            KeyCode::Char('c') if ctrl => {
                if self.busy {
                    Action::Cancel
                } else {
                    Action::Quit
                }
            }
            KeyCode::Char('d') if ctrl => {
                if self.input.is_empty() && !self.busy {
                    Action::Quit
                } else {
                    Action::None
                }
            }
            KeyCode::Char('u') if ctrl => {
                self.input.clear();
                self.cursor = 0;
                Action::None
            }
            KeyCode::Esc => {
                if self.busy {
                    Action::Cancel
                } else {
                    self.input.clear();
                    self.cursor = 0;
                    Action::None
                }
            }
            KeyCode::Enter => {
                if self.busy {
                    return Action::None;
                }
                let line = std::mem::take(&mut self.input);
                self.cursor = 0;
                self.history_pos = None;
                if line.trim().is_empty() {
                    return Action::None;
                }
                self.history.push(line.clone());
                self.push(style_fg(Color::Magenta).add_modifier(Modifier::BOLD), &format!("> {line}"));
                self.busy = true;
                Action::Submit(line)
            }
            KeyCode::Char(c) if !ctrl => {
                let at = self.byte_index(self.cursor);
                self.input.insert(at, c);
                self.cursor += 1;
                Action::None
            }
            KeyCode::Backspace => {
                if self.cursor > 0 {
                    let at = self.byte_index(self.cursor - 1);
                    self.input.remove(at);
                    self.cursor -= 1;
                }
                Action::None
            }
            KeyCode::Delete => {
                if self.cursor < self.input.chars().count() {
                    let at = self.byte_index(self.cursor);
                    self.input.remove(at);
                }
                Action::None
            }
            KeyCode::Left => {
                self.cursor = self.cursor.saturating_sub(1);
                Action::None
            }
            KeyCode::Right => {
                self.cursor = (self.cursor + 1).min(self.input.chars().count());
                Action::None
            }
            KeyCode::Home => {
                self.cursor = 0;
                Action::None
            }
            KeyCode::End => {
                self.cursor = self.input.chars().count();
                Action::None
            }
            KeyCode::Up => {
                if !self.history.is_empty() {
                    let pos = match self.history_pos {
                        None => self.history.len() - 1,
                        Some(p) => p.saturating_sub(1),
                    };
                    self.history_pos = Some(pos);
                    self.input = self.history[pos].clone();
                    self.cursor = self.input.chars().count();
                }
                Action::None
            }
            KeyCode::Down => {
                if let Some(p) = self.history_pos {
                    if p + 1 < self.history.len() {
                        self.history_pos = Some(p + 1);
                        self.input = self.history[p + 1].clone();
                    } else {
                        self.history_pos = None;
                        self.input.clear();
                    }
                    self.cursor = self.input.chars().count();
                }
                Action::None
            }
            KeyCode::PageUp => {
                self.scroll = self.scroll.saturating_add(10);
                Action::None
            }
            KeyCode::PageDown => {
                self.scroll = self.scroll.saturating_sub(10);
                Action::None
            }
            _ => Action::None,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------------------------

/// Hard-wraps `text` to `width` display columns (CJK characters count as two columns).
pub fn wrap_to_width(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    if text.is_empty() {
        return vec![String::new()];
    }
    let mut rows = Vec::new();
    let mut cur = String::new();
    let mut w = 0;
    for c in text.chars() {
        let cw = UnicodeWidthChar::width(c).unwrap_or(0);
        if w + cw > width && !cur.is_empty() {
            rows.push(std::mem::take(&mut cur));
            w = 0;
        }
        cur.push(c);
        w += cw;
    }
    rows.push(cur);
    rows
}

pub fn draw(f: &mut Frame, app: &App) {
    let area = f.area();
    let confirm_h = if app.confirm.is_some() { 6 } else { 0 };
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(3), Constraint::Length(confirm_h), Constraint::Length(1), Constraint::Length(3)])
        .split(area);

    let header = Line::from(vec![
        Span::styled(format!(" {APP_NAME} "), Style::default().fg(Color::Black).bg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw(format!("  {}", app.workspace)),
    ]);
    f.render_widget(Paragraph::new(header), chunks[0]);

    // log (bottom-anchored, scrollable)
    let log_area = chunks[1];
    let width = log_area.width.max(1) as usize;
    let mut rows: Vec<Line> = Vec::new();
    for l in &app.log {
        for r in wrap_to_width(&l.text, width) {
            rows.push(Line::from(Span::styled(r, l.style)));
        }
    }
    let height = log_area.height as usize;
    let max_scroll = rows.len().saturating_sub(height);
    let scroll = app.scroll.min(max_scroll);
    let end = rows.len() - scroll;
    let start = end.saturating_sub(height);
    f.render_widget(Paragraph::new(rows[start..end].to_vec()), log_area);

    if let Some(req) = &app.confirm {
        let mut lines = vec![Line::from(Span::styled("Qwen Code wants to execute:", Style::default().add_modifier(Modifier::BOLD))), Line::from("")];
        lines.push(Line::from(Span::styled(req.command.clone(), style_fg(Color::Yellow))));
        if let Some(r) = &req.reason {
            lines.push(Line::from(Span::styled(format!("({r})"), style_fg(Color::DarkGray))));
        }
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled("Allow [Y/N]:", Style::default().add_modifier(Modifier::BOLD))));
        f.render_widget(Paragraph::new(lines).block(Block::default().borders(Borders::TOP | Borders::BOTTOM).border_style(style_fg(Color::Yellow))), chunks[2]);
    }

    let status = if app.busy {
        format!(" ● working — {}   (Esc cancels)", if app.status.is_empty() { "…" } else { &app.status })
    } else {
        format!(" mode: {}   ·   /help  ·  PgUp/PgDn scroll", app.mode)
    };
    f.render_widget(Paragraph::new(Span::styled(status, style_fg(Color::DarkGray))), chunks[3]);

    let input_block = Block::default().borders(Borders::ALL).title(if app.busy { " working " } else { " > " });
    let inner = input_block.inner(chunks[4]);
    let shown_w = inner.width.saturating_sub(1) as usize;
    // keep the cursor visible: show the tail of the input if it is too wide
    let before_cursor: String = app.input.chars().take(app.cursor).collect();
    let mut start_char = 0usize;
    while UnicodeWidthStr::width(before_cursor.chars().skip(start_char).collect::<String>().as_str()) > shown_w && start_char < app.cursor {
        start_char += 1;
    }
    let visible: String = app.input.chars().skip(start_char).collect();
    f.render_widget(Paragraph::new(visible).block(input_block), chunks[4]);
    if app.confirm.is_none() {
        let cursor_w = UnicodeWidthStr::width(before_cursor.chars().skip(start_char).collect::<String>().as_str()) as u16;
        f.set_cursor_position((inner.x + cursor_w.min(inner.width.saturating_sub(1)), inner.y));
    }
    let _: Rect = area;
}

// ---------------------------------------------------------------------------------------------
// Event loop
// ---------------------------------------------------------------------------------------------

type Builder = Box<dyn FnOnce(Arc<dyn Ui>) -> Result<Session, String> + Send>;

/// Runs the TUI until the user quits. Returns the process exit code.
pub fn run<F>(build: F, workspace: PathBuf) -> i32
where
    F: FnOnce(Arc<dyn Ui>) -> Result<Session, String> + Send + 'static,
{
    run_boxed(Box::new(build), workspace)
}

fn run_boxed(build: Builder, workspace: PathBuf) -> i32 {
    let (tx, rx): (Sender<Msg>, Receiver<Msg>) = mpsc::channel();
    let (cmd_tx, cmd_rx): (Sender<Cmd>, Receiver<Cmd>) = mpsc::channel();
    let cancel_slot: Arc<Mutex<Option<Arc<AtomicBool>>>> = Arc::new(Mutex::new(None));
    let cancel_slot_w = cancel_slot.clone();
    let tx_w = tx.clone();
    let ui: Arc<dyn Ui> = Arc::new(TuiUi { tx: Mutex::new(tx.clone()) });

    let worker = std::thread::spawn(move || {
        let mut session = match build(ui) {
            Ok(s) => s,
            Err(e) => {
                let _ = tx_w.send(Msg::Fatal(e));
                return;
            }
        };
        *cancel_slot_w.lock().unwrap_or_else(|e| e.into_inner()) = Some(session.agent.cancel.clone());
        let _ = tx_w.send(Msg::Ready { mode: session.mode });
        while let Ok(cmd) = cmd_rx.recv() {
            match cmd {
                Cmd::Quit => break,
                Cmd::Line(l) => {
                    let flow = session.handle_line(&l);
                    let _ = tx_w.send(Msg::Ready { mode: session.mode });
                    let _ = tx_w.send(Msg::Done(flow));
                    if flow == Flow::Exit {
                        break;
                    }
                }
            }
        }
    });

    let mut terminal = ratatui::init();
    let mut app = App::new(&workspace.display().to_string());
    let mut pending_reply: Option<Sender<bool>> = None;
    let mut code = 0;
    'outer: loop {
        if terminal.draw(|f| draw(f, &app)).is_err() {
            code = 1;
            break;
        }
        // 1. messages from the worker
        loop {
            match rx.try_recv() {
                Ok(Msg::Confirm { request, reply }) => {
                    pending_reply = Some(reply);
                    app.push(style_fg(Color::Yellow), &format!("Qwen Code wants to execute:\n\n{}\n\nAllow [Y/N]:", request.command));
                    app.confirm = Some(request);
                }
                Ok(m) => app.on_msg(m),
                Err(_) => break,
            }
        }
        if app.quit {
            if app.status.is_empty() && !app.log_text().contains("error:") {
                break;
            }
            // show a fatal error for a moment before leaving
            let _ = terminal.draw(|f| draw(f, &app));
            std::thread::sleep(Duration::from_millis(1500));
            code = 1;
            break;
        }
        // 2. keys
        if event::poll(Duration::from_millis(40)).unwrap_or(false) {
            if let Ok(CtEvent::Key(k)) = event::read() {
                match app.on_key(k) {
                    Action::None => {}
                    Action::Submit(line) => {
                        let _ = cmd_tx.send(Cmd::Line(line));
                    }
                    Action::Cancel => {
                        if let Some(c) = cancel_slot.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
                            c.store(true, Ordering::Relaxed);
                        }
                        app.push(style_fg(Color::Yellow), "cancelling…");
                    }
                    Action::Quit => break 'outer,
                    Action::Answer(v) => {
                        if let Some(r) = pending_reply.take() {
                            let _ = r.send(v);
                        }
                    }
                }
            }
        }
    }
    ratatui::restore();
    if let Some(r) = pending_reply.take() {
        let _ = r.send(false);
    }
    if let Some(c) = cancel_slot.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
        c.store(true, Ordering::Relaxed);
    }
    let _ = cmd_tx.send(Cmd::Quit);
    drop(cmd_tx);
    // do not wait for a model call that may take minutes
    drop(worker);
    code
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::crossterm::event::KeyEventState;
    use ratatui::Terminal;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent { code, modifiers: KeyModifiers::NONE, kind: KeyEventKind::Press, state: KeyEventState::NONE }
    }

    fn ctrl(c: char) -> KeyEvent {
        KeyEvent { code: KeyCode::Char(c), modifiers: KeyModifiers::CONTROL, kind: KeyEventKind::Press, state: KeyEventState::NONE }
    }

    fn render(app: &App, w: u16, h: u16) -> String {
        let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
        t.draw(|f| draw(f, app)).unwrap();
        let buf = t.backend().buffer().clone();
        (0..h)
            .map(|y| {
                let mut line = String::new();
                let mut x = 0;
                while x < w {
                    let sym = buf[(x, y)].symbol().to_string();
                    // a double-width glyph occupies two cells; the second one is blank filler
                    x += unicode_width::UnicodeWidthStr::width(sym.as_str()).max(1) as u16;
                    line.push_str(&sym);
                }
                line
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn renders_title_log_status_and_input() {
        let mut app = App::new("/work/project");
        app.on_event(Event::Phase { role: "thinking".into(), text: "iteration 1/30".into() });
        app.on_event(Event::ToolResult { tool: "read_file".into(), ok: true, summary: "read src/main.rs".into() });
        for c in "hello".chars() {
            app.on_key(key(KeyCode::Char(c)));
        }
        let screen = render(&app, 80, 20);
        assert!(screen.contains("Qwen Code") && screen.contains("/work/project"), "{screen}");
        assert!(screen.contains("[thinking] iteration 1/30") && screen.contains("✓ read_file"), "{screen}");
        assert!(screen.contains("hello") && screen.contains("mode: auto"), "{screen}");
    }

    #[test]
    fn typing_editing_history_and_korean_input() {
        let mut app = App::new("/w");
        for c in "안녕 qwen".chars() {
            app.on_key(key(KeyCode::Char(c)));
        }
        assert_eq!(app.input, "안녕 qwen");
        app.on_key(key(KeyCode::Left));
        app.on_key(key(KeyCode::Backspace));
        assert_eq!(app.input, "안녕 qwn");
        app.on_key(key(KeyCode::Home));
        app.on_key(key(KeyCode::Delete));
        assert_eq!(app.input, "녕 qwn");
        let a = app.on_key(key(KeyCode::Enter));
        assert_eq!(a, Action::Submit("녕 qwn".into()));
        assert!(app.busy && app.input.is_empty());
        // Enter while busy does nothing
        app.on_key(key(KeyCode::Char('x')));
        assert_eq!(app.on_key(key(KeyCode::Enter)), Action::None);
        app.on_msg(Msg::Done(Flow::Continue));
        assert!(!app.busy);
        // history recall
        app.input.clear();
        app.cursor = 0;
        app.on_key(key(KeyCode::Up));
        assert_eq!(app.input, "녕 qwn");
        let screen = render(&app, 60, 12);
        assert!(screen.contains("녕 qwn"));
    }

    #[test]
    fn confirm_modal_uses_the_required_format_and_keys() {
        let mut app = App::new("/w");
        let (reply, _rx) = mpsc::channel();
        app.on_msg(Msg::Confirm { request: ConfirmRequest { command: "rm -rf build".into(), reason: Some("destructive".into()) }, reply });
        let screen = render(&app, 70, 24);
        assert!(screen.contains("Qwen Code wants to execute:") && screen.contains("rm -rf build") && screen.contains("Allow [Y/N]:"), "{screen}");
        assert_eq!(app.on_key(key(KeyCode::Char('x'))), Action::None, "other keys are ignored while asking");
        assert_eq!(app.on_key(key(KeyCode::Char('y'))), Action::Answer(true));
        assert!(app.confirm.is_none());
        let (reply, _rx) = mpsc::channel();
        app.on_msg(Msg::Confirm { request: ConfirmRequest { command: "x".into(), reason: None }, reply });
        assert_eq!(app.on_key(key(KeyCode::Char('N'))), Action::Answer(false));
        let (reply, _rx) = mpsc::channel();
        app.on_msg(Msg::Confirm { request: ConfirmRequest { command: "x".into(), reason: None }, reply });
        assert_eq!(app.on_key(key(KeyCode::Esc)), Action::Answer(false));
    }

    #[test]
    fn cancel_quit_and_scroll() {
        let mut app = App::new("/w");
        assert_eq!(app.on_key(ctrl('c')), Action::Quit);
        app.busy = true;
        assert_eq!(app.on_key(ctrl('c')), Action::Cancel);
        assert_eq!(app.on_key(key(KeyCode::Esc)), Action::Cancel);
        app.busy = false;
        for i in 0..100 {
            app.on_event(Event::Info(format!("line {i}")));
        }
        let bottom = render(&app, 40, 12);
        assert!(bottom.contains("line 99"));
        app.on_key(key(KeyCode::PageUp));
        let up = render(&app, 40, 12);
        assert!(!up.contains("line 99") && up.contains("line 89"), "{up}");
        app.on_key(key(KeyCode::PageDown));
        assert!(render(&app, 40, 12).contains("line 99"));
        // key releases are ignored (Windows reports press and release)
        let mut rel = key(KeyCode::Char('a'));
        rel.kind = KeyEventKind::Release;
        app.on_key(rel);
        assert!(app.input.is_empty());
    }

    #[test]
    fn wrapping_counts_double_width_characters() {
        assert_eq!(wrap_to_width("abcdef", 4), vec!["abcd", "ef"]);
        assert_eq!(wrap_to_width("한국어입니다", 6), vec!["한국어", "입니다"]);
        assert_eq!(wrap_to_width("", 10), vec![""]);
    }

    #[test]
    fn exit_and_clear_flows() {
        let mut app = App::new("/w");
        app.on_event(Event::Info("something".into()));
        app.on_msg(Msg::Done(Flow::Clear));
        assert!(app.log_text().is_empty());
        app.on_msg(Msg::Done(Flow::Exit));
        assert!(app.quit);
    }
}
