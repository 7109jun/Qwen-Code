//! Operating-system abstraction layer (Windows and Linux).
//!
//! Everything that differs between the supported platforms lives here: shells, process
//! handling, path normalisation, terminal detection and shared-library discovery.

use std::collections::VecDeque;
use std::io::{IsTerminal, Read};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Os {
    Windows,
    Linux,
    Other,
}

pub fn current_os() -> Os {
    if cfg!(windows) {
        Os::Windows
    } else if cfg!(target_os = "linux") {
        Os::Linux
    } else {
        Os::Other
    }
}

pub fn os_name() -> &'static str {
    match current_os() {
        Os::Windows => "Windows",
        Os::Linux => "Linux",
        Os::Other => "Unsupported OS",
    }
}

pub fn is_windows() -> bool {
    cfg!(windows)
}

// ---------------------------------------------------------------------------------------------
// Directories and paths
// ---------------------------------------------------------------------------------------------

pub fn home_dir() -> Option<PathBuf> {
    if cfg!(windows) {
        std::env::var_os("USERPROFILE").map(PathBuf::from)
    } else {
        std::env::var_os("HOME").map(PathBuf::from)
    }
}

/// User-level configuration directory (`~/.config/qwen` or `%APPDATA%\qwen`).
pub fn config_dir() -> Option<PathBuf> {
    if cfg!(windows) {
        std::env::var_os("APPDATA").map(|p| PathBuf::from(p).join("qwen"))
    } else if let Some(x) = std::env::var_os("XDG_CONFIG_HOME") {
        Some(PathBuf::from(x).join("qwen"))
    } else {
        home_dir().map(|h| h.join(".config").join("qwen"))
    }
}

/// Removes the Windows verbatim prefix (`\\?\`) produced by `canonicalize`.
pub fn strip_unc(p: PathBuf) -> PathBuf {
    #[cfg(windows)]
    {
        let s = p.to_string_lossy().to_string();
        if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
            return PathBuf::from(format!(r"\\{rest}"));
        }
        if let Some(rest) = s.strip_prefix(r"\\?\") {
            return PathBuf::from(rest);
        }
    }
    p
}

/// Lexical normalisation: removes `.` and resolves `..` without touching the filesystem.
pub fn normalize_lexical(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    out.push("..");
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    if out.as_os_str().is_empty() {
        out.push(".");
    }
    out
}

/// Canonicalises the longest existing ancestor and appends the remaining components.
/// Works for paths that do not exist yet.
pub fn canonicalize_lossy(p: &Path) -> PathBuf {
    let norm = normalize_lexical(p);
    if let Ok(c) = norm.canonicalize() {
        return strip_unc(c);
    }
    let mut rest: Vec<std::ffi::OsString> = Vec::new();
    let mut cur = norm.clone();
    loop {
        if let Ok(c) = cur.canonicalize() {
            let mut base = strip_unc(c);
            for r in rest.iter().rev() {
                base.push(r);
            }
            return base;
        }
        match (cur.file_name().map(|s| s.to_os_string()), cur.parent().map(|p| p.to_path_buf())) {
            (Some(name), Some(parent)) => {
                rest.push(name);
                cur = parent;
                if cur.as_os_str().is_empty() {
                    cur = PathBuf::from(".");
                }
            }
            _ => return norm,
        }
    }
}

/// Human/model friendly path: forward slashes, relative to `base` when possible.
pub fn display_path(p: &Path, base: &Path) -> String {
    let shown = p.strip_prefix(base).unwrap_or(p);
    let s = shown.to_string_lossy().replace('\\', "/");
    if s.is_empty() {
        ".".to_string()
    } else {
        s
    }
}

pub fn is_stdin_tty() -> bool {
    std::io::stdin().is_terminal()
}

pub fn is_stdout_tty() -> bool {
    std::io::stdout().is_terminal()
}

/// Makes the console use UTF-8 (needed for Korean input/output on classic Windows consoles).
pub fn init_console() {
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::System::Console::{SetConsoleCP, SetConsoleOutputCP};
        SetConsoleOutputCP(65001);
        SetConsoleCP(65001);
    }
}

// ---------------------------------------------------------------------------------------------
// Programs and shells
// ---------------------------------------------------------------------------------------------

pub fn which(program: &str) -> Option<PathBuf> {
    let p = Path::new(program);
    if p.components().count() > 1 {
        return if p.is_file() { Some(p.to_path_buf()) } else { None };
    }
    let path = std::env::var_os("PATH")?;
    let exts: Vec<String> = if cfg!(windows) {
        std::env::var("PATHEXT")
            .unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".into())
            .split(';')
            .map(|s| s.to_ascii_lowercase())
            .collect()
    } else {
        vec![String::new()]
    };
    for dir in std::env::split_paths(&path) {
        for ext in &exts {
            let cand = if ext.is_empty() { dir.join(program) } else { dir.join(format!("{program}{ext}")) };
            if cand.is_file() {
                return Some(cand);
            }
        }
        if cfg!(windows) && dir.join(program).is_file() {
            return Some(dir.join(program));
        }
    }
    None
}

#[derive(Debug, Clone)]
pub struct ShellSpec {
    pub name: String,
    pub program: String,
    /// Arguments that precede the command string.
    pub args: Vec<String>,
}

/// Picks the shell used for commands that need shell syntax (pipes, `&&`, redirects, ...).
pub fn resolve_shell(preference: &str) -> ShellSpec {
    let pref = preference.to_ascii_lowercase();
    let pick = |name: &str| -> Option<ShellSpec> {
        match name {
            "powershell" => which("pwsh")
                .or_else(|| which("powershell"))
                .map(|p| ShellSpec {
                    name: "powershell".into(),
                    program: p.to_string_lossy().to_string(),
                    args: vec!["-NoProfile".into(), "-NonInteractive".into(), "-Command".into()],
                }),
            "cmd" => Some(ShellSpec { name: "cmd".into(), program: "cmd".into(), args: vec!["/d".into(), "/s".into(), "/c".into()] }),
            "bash" => which("bash").map(|p| ShellSpec { name: "bash".into(), program: p.to_string_lossy().to_string(), args: vec!["-c".into()] }),
            "sh" => Some(ShellSpec { name: "sh".into(), program: "sh".into(), args: vec!["-c".into()] }),
            _ => None,
        }
    };
    if pref != "auto" {
        if let Some(s) = pick(&pref) {
            return s;
        }
    }
    if cfg!(windows) {
        pick("powershell").or_else(|| pick("cmd")).expect("cmd is always available on Windows")
    } else {
        pick("bash").or_else(|| pick("sh")).expect("sh is always available")
    }
}

// ---------------------------------------------------------------------------------------------
// Process execution
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ProcessSpec {
    pub program: String,
    pub args: Vec<String>,
    pub cwd: Option<PathBuf>,
    pub env: Vec<(String, String)>,
    pub timeout: Duration,
    /// Maximum bytes kept per stream (head and tail are kept, the middle is dropped).
    pub max_output_bytes: usize,
}

impl ProcessSpec {
    pub fn new(program: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            program: program.into(),
            args,
            cwd: None,
            env: Vec::new(),
            timeout: Duration::from_secs(120),
            max_output_bytes: 20_000,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ProcessOutput {
    pub code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub timed_out: bool,
    pub cancelled: bool,
    pub duration: Duration,
    pub truncated: bool,
}

impl ProcessOutput {
    pub fn success(&self) -> bool {
        self.code == Some(0) && !self.timed_out && !self.cancelled
    }
}

struct Captured {
    text: String,
    truncated: bool,
}

fn capture_stream<R: Read + Send + 'static>(mut r: R, cap: usize) -> mpsc::Receiver<Captured> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let head_cap = cap / 2;
        let tail_cap = cap - head_cap;
        let mut head: Vec<u8> = Vec::new();
        let mut tail: VecDeque<u8> = VecDeque::new();
        let mut omitted: usize = 0;
        let mut buf = [0u8; 8192];
        loop {
            let n = match r.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => n,
            };
            let mut chunk = &buf[..n];
            if head.len() < head_cap {
                let take = (head_cap - head.len()).min(chunk.len());
                head.extend_from_slice(&chunk[..take]);
                chunk = &chunk[take..];
            }
            for &b in chunk {
                if tail.len() == tail_cap {
                    tail.pop_front();
                    omitted += 1;
                }
                tail.push_back(b);
            }
        }
        let mut bytes = head;
        let truncated = omitted > 0;
        if truncated {
            bytes.extend_from_slice(format!("\n... [{omitted} bytes omitted] ...\n").as_bytes());
        }
        bytes.extend(tail);
        let _ = tx.send(Captured { text: String::from_utf8_lossy(&bytes).into_owned(), truncated });
    });
    rx
}

/// Kills a process and all of its children.
pub fn kill_process_tree(child: &mut std::process::Child) {
    #[cfg(unix)]
    {
        let pid = child.id() as i32;
        unsafe {
            libc::kill(-pid, libc::SIGKILL);
        }
        let _ = child.kill();
    }
    #[cfg(windows)]
    {
        let _ = Command::new("taskkill")
            .args(["/PID", &child.id().to_string(), "/T", "/F"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        let _ = child.kill();
    }
}

/// Runs a program (no shell involved), capturing output, enforcing a timeout and honouring `cancel`.
pub fn run_process(spec: &ProcessSpec, cancel: Option<&AtomicBool>) -> std::io::Result<ProcessOutput> {
    let mut cmd = Command::new(&spec.program);
    cmd.args(&spec.args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(cwd) = &spec.cwd {
        cmd.current_dir(cwd);
    }
    for (k, v) in &spec.env {
        cmd.env(k, v);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let start = Instant::now();
    let mut child = cmd.spawn()?;
    let out_rx = capture_stream(child.stdout.take().expect("piped stdout"), spec.max_output_bytes);
    let err_rx = capture_stream(child.stderr.take().expect("piped stderr"), spec.max_output_bytes);

    let mut timed_out = false;
    let mut cancelled = false;
    let status = loop {
        match child.try_wait()? {
            Some(s) => break Some(s),
            None => {
                if start.elapsed() >= spec.timeout {
                    timed_out = true;
                    kill_process_tree(&mut child);
                    break child.wait().ok();
                }
                if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                    cancelled = true;
                    kill_process_tree(&mut child);
                    break child.wait().ok();
                }
                std::thread::sleep(Duration::from_millis(15));
            }
        }
    };
    let grace = Duration::from_secs(3);
    let out = out_rx.recv_timeout(grace).unwrap_or(Captured { text: String::new(), truncated: false });
    let err = err_rx.recv_timeout(grace).unwrap_or(Captured { text: String::new(), truncated: false });
    Ok(ProcessOutput {
        code: status.and_then(|s| s.code()),
        stdout: out.text,
        stderr: err.text,
        timed_out,
        cancelled,
        duration: start.elapsed(),
        truncated: out.truncated || err.truncated,
    })
}

// ---------------------------------------------------------------------------------------------
// ONNX Runtime shared library discovery
// ---------------------------------------------------------------------------------------------

pub fn onnxruntime_lib_file_names() -> &'static [&'static str] {
    if cfg!(windows) {
        &["onnxruntime.dll"]
    } else {
        &["libonnxruntime.so"]
    }
}

fn find_in_dir(dir: &Path) -> Option<PathBuf> {
    for n in onnxruntime_lib_file_names() {
        let p = dir.join(n);
        if p.is_file() {
            return Some(p);
        }
    }
    if !cfg!(windows) {
        // versioned names such as libonnxruntime.so.1.30.0
        let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("libonnxruntime.so."))
            })
            .collect();
        found.sort();
        return found.pop();
    }
    None
}

/// Finds the ONNX Runtime shared library.
/// Order: explicit config path, `$ORT_DYLIB_PATH`, executable dir, `lib/` and `models/` dirs.
pub fn find_onnxruntime_lib(explicit: Option<&str>, base: &Path) -> Option<PathBuf> {
    if let Some(e) = explicit.filter(|s| !s.is_empty()) {
        let p = PathBuf::from(e);
        let p = if p.is_absolute() { p } else { base.join(p) };
        if p.is_file() {
            return Some(p);
        }
        if p.is_dir() {
            if let Some(f) = find_in_dir(&p) {
                return Some(f);
            }
        }
        return None;
    }
    if let Some(env) = std::env::var_os("ORT_DYLIB_PATH").filter(|s| !s.is_empty()) {
        let p = PathBuf::from(env);
        if p.is_file() {
            return Some(p);
        }
    }
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(d) = exe.parent() {
            dirs.push(d.to_path_buf());
            dirs.push(d.join("lib"));
        }
    }
    dirs.push(base.to_path_buf());
    dirs.push(base.join("lib"));
    dirs.push(base.join("models"));
    if !cfg!(windows) {
        for d in ["/usr/local/lib", "/usr/lib", "/usr/lib/x86_64-linux-gnu", "/opt/onnxruntime/lib"] {
            dirs.push(PathBuf::from(d));
        }
    }
    dirs.iter().find_map(|d| find_in_dir(d))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_removes_dots() {
        assert_eq!(normalize_lexical(Path::new("a/./b/../c")), PathBuf::from("a/c"));
        assert_eq!(normalize_lexical(Path::new("../x")), PathBuf::from("../x"));
    }

    #[test]
    fn display_path_uses_forward_slashes() {
        let base = Path::new("/w");
        assert_eq!(display_path(Path::new("/w/src/main.rs"), base), "src/main.rs");
    }

    #[test]
    fn run_process_captures_output_and_times_out() {
        let (prog, args) = if cfg!(windows) {
            ("cmd", vec!["/c".to_string(), "echo hello".to_string()])
        } else {
            ("sh", vec!["-c".to_string(), "echo hello".to_string()])
        };
        let out = run_process(&ProcessSpec::new(prog, args), None).unwrap();
        assert!(out.success());
        assert!(out.stdout.contains("hello"));

        if !cfg!(windows) {
            let mut spec = ProcessSpec::new("sh", vec!["-c".into(), "sleep 5".into()]);
            spec.timeout = Duration::from_millis(200);
            let out = run_process(&spec, None).unwrap();
            assert!(out.timed_out);
        }
    }

    #[test]
    fn output_is_truncated_in_the_middle() {
        if cfg!(windows) {
            return;
        }
        let mut spec = ProcessSpec::new("sh", vec!["-c".into(), "seq 1 20000".into()]);
        spec.max_output_bytes = 400;
        let out = run_process(&spec, None).unwrap();
        assert!(out.truncated);
        assert!(out.stdout.starts_with("1\n"));
        assert!(out.stdout.trim_end().ends_with("20000"));
        assert!(out.stdout.contains("bytes omitted"));
    }
}
