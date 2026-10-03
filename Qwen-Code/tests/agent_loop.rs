//! End-to-end tests of the agent loop: scripted Thinking/Coder models drive the real tools,
//! real permission checks and a real `cargo` toolchain.

use qwen_code::agent::{Agent, Mode, Status};
use qwen_code::config::Config;
use qwen_code::models::{CoderModel, ModelHandle, ModelSet, ThinkingModel};
use qwen_code::permissions::PermissionManager;
use qwen_code::runtime::{ChatMessage, MockRuntime, RuntimeError};
use qwen_code::ui::{BufferUi, Event};
use serde_json::{json, Value};
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

fn j(v: Value) -> String {
    v.to_string()
}

fn tool(name: &str, args: Value) -> String {
    j(json!({"type": "tool_call", "tool": name, "arguments": args}))
}

fn report(summary: &str) -> String {
    j(json!({"type": "report", "summary": summary}))
}

fn complete(summary: &str, build: Option<bool>, test: Option<bool>) -> String {
    j(json!({"type": "complete", "summary": summary, "verification": {"build": build, "test": test}}))
}

fn instruction(text: &str) -> String {
    j(json!({"type": "instruction", "instruction": text}))
}

struct Rig {
    agent: Agent,
    ui: Arc<BufferUi>,
    thinking: Arc<MockRuntime>,
    coder: Arc<MockRuntime>,
    dir: tempfile::TempDir,
}

fn rig_with(thinking: MockRuntime, coder: MockRuntime, answers: Vec<bool>, tweak: impl FnOnce(&mut Config)) -> Rig {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = Config::default();
    cfg.agent.retry_backoff_ms = 1;
    tweak(&mut cfg);
    let cfg = Arc::new(cfg);
    let thinking = Arc::new(thinking);
    let coder = Arc::new(coder);
    let models = ModelSet::new(
        Arc::new(ThinkingModel(ModelHandle::from_runtime("thinking", "qwen", "think-mock", thinking.clone()))),
        Arc::new(CoderModel(ModelHandle::from_runtime("coder", "qwen", "coder-mock", coder.clone()))),
    );
    let ui = Arc::new(BufferUi::with_answers(answers));
    let perms = PermissionManager::new(&cfg.permissions);
    let agent = Agent::new(cfg, dir.path(), models, perms, ui.clone(), Arc::new(AtomicBool::new(false)));
    Rig { agent, ui, thinking, coder, dir }
}

fn rig(thinking: Vec<String>, coder: Vec<String>) -> Rig {
    rig_with(MockRuntime::scripted("thinking", thinking), MockRuntime::scripted("coder", coder), vec![], |_| {})
}

fn read(dir: &Path, rel: &str) -> String {
    std::fs::read_to_string(dir.join(rel)).unwrap_or_else(|_| panic!("missing {rel}"))
}

fn last_user_text(msgs: &[ChatMessage]) -> String {
    msgs.iter().rev().find(|m| m.role == "user").map(|m| m.content.clone()).unwrap_or_default()
}

const CARGO_TOML: &str = "[package]\nname = \"hello\"\nversion = \"0.1.0\"\nedition = \"2021\"\n";

fn have_cargo() -> bool {
    qwen_code::platform::which("cargo").is_some()
}

#[test]
fn hello_world_project_is_created_modified_built_and_run() {
    if !have_cargo() {
        eprintln!("cargo not found; skipping");
        return;
    }
    let mut r = rig(
        vec![
            j(json!({"type": "plan", "steps": ["Create the project", "Modify the greeting", "Build and run"]})),
            instruction("Create a Rust hello-world project (Cargo.toml + src/main.rs), change the greeting to 'Hello from Qwen Code!', build and run it."),
            complete("Hello-world project created, modified, built and run.", Some(true), None),
        ],
        vec![
            tool("write_file", json!({"path": "Cargo.toml", "content": CARGO_TOML})),
            j(json!({"type": "edit", "path": "src/main.rs", "operations": [{"type": "create", "content": "fn main() {\n    println!(\"Hello, world!\");\n}\n"}]})),
            tool("run_command", json!({"command": "cargo build"})),
            j(json!({"type": "edit", "path": "src/main.rs", "operations": [{"type": "replace", "old": "Hello, world!", "new": "Hello from Qwen Code!"}]})),
            tool("run_command", json!({"command": "cargo build"})),
            tool("run_command", json!({"command": "cargo run -q"})),
            report("Created and modified the project; the build succeeded and the program prints the new greeting."),
        ],
    );
    let out = r.agent.run_task("create a hello world program and change its greeting", Mode::Auto);
    assert_eq!(out.status, Status::Completed, "{}", out.summary);
    assert_eq!(out.verification.build, Some(true));
    assert!(out.files_changed.contains(&"Cargo.toml".to_string()) && out.files_changed.contains(&"src/main.rs".to_string()));
    assert!(read(r.dir.path(), "src/main.rs").contains("Hello from Qwen Code!"));
    assert_eq!(out.iterations, 2, "one instruction round plus the final review");
    // both roles were used repeatedly: the loop really iterated
    assert!(r.thinking.call_count() >= 3 && r.coder.call_count() == 7);
    {
        // the program output reached the Coder
        let coder_calls = r.coder.calls.lock().unwrap();
        assert!(last_user_text(coder_calls.last().unwrap()).contains("Hello from Qwen Code!"), "cargo run output must be returned to the model");
        // the Thinking model received the Coder's report
        let think_calls = r.thinking.calls.lock().unwrap();
        assert!(last_user_text(think_calls.last().unwrap()).contains("CODER REPORT"));
    }
    assert!(r.ui.prompts().is_empty(), "ordinary development commands must not ask for permission");
}

#[test]
fn deliberate_compile_error_is_analysed_by_thinking_and_fixed_by_coder() {
    if !have_cargo() {
        return;
    }
    let seen_by_thinking: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(vec![]));
    let seen = seen_by_thinking.clone();
    let mut n = 0;
    let thinking = MockRuntime::from_fn(
        "thinking",
        Box::new(move |msgs: &[ChatMessage]| {
            n += 1;
            seen.lock().unwrap().push(last_user_text(msgs));
            Ok(match n {
                1 => instruction("Create a hello-world Rust project and build it. Introduce nothing else."),
                2 => {
                    // The first report contains the compiler error; analyse it and give a precise fix.
                    assert!(last_user_text(msgs).contains("error[E0308]"), "the compile error must reach the Thinking model:\n{}", last_user_text(msgs));
                    instruction("The build failed with E0308 (mismatched types) in src/main.rs: `let x: i32 = \"text\";` assigns a string to an i32. Change the literal to 42, then rebuild with `cargo build`.")
                }
                _ => complete("Fixed the type error; the project builds.", Some(true), None),
            })
        }),
    );
    let coder = MockRuntime::scripted(
        "coder",
        vec![
            tool("write_file", json!({"path": "Cargo.toml", "content": CARGO_TOML})),
            j(json!({"type": "edit", "path": "src/main.rs", "operations": [{"type": "create", "content": "fn main() {\n    let x: i32 = \"text\";\n    println!(\"{}\", x);\n}\n"}]})),
            tool("run_command", json!({"command": "cargo build"})),
            report("Project created but the build FAILED."),
            // second round (fix)
            tool("read_file", json!({"path": "src/main.rs"})),
            j(json!({"type": "edit", "path": "src/main.rs", "operations": [{"type": "replace", "old": "\"text\"", "new": "42"}]})),
            tool("run_command", json!({"command": "cargo build"})),
            report("Replaced the literal with 42; cargo build succeeded."),
        ],
    );
    let mut r = rig_with(thinking, coder, vec![], |_| {});
    let out = r.agent.run_task("make a hello world project", Mode::Auto);
    assert_eq!(out.status, Status::Completed, "{}", out.summary);
    assert!(read(r.dir.path(), "src/main.rs").contains("let x: i32 = 42;"));
    assert_eq!(out.verification.build, Some(true), "the final build was observed to pass");
    // The deliberate error was really detected from a real compiler run.
    let seen = seen_by_thinking.lock().unwrap();
    assert!(seen.iter().any(|s| s.contains("error[E0308]") && s.contains("FAILED")));
    assert!(r.dir.path().join("target/debug").exists());
}

#[test]
fn unverified_build_claims_are_pushed_back() {
    if !have_cargo() {
        return;
    }
    let mut r = rig(
        vec![
            instruction("Create the hello project files."),
            // claims a successful build although nothing was built
            complete("All done.", Some(true), None),
            // after the push-back it asks for a build
            instruction("Run `cargo build` now."),
            complete("Built.", Some(true), None),
        ],
        vec![
            tool("write_file", json!({"path": "Cargo.toml", "content": CARGO_TOML})),
            j(json!({"type": "edit", "path": "src/main.rs", "operations": [{"type": "create", "content": "fn main() {}\n"}]})),
            report("Files created."),
            tool("run_command", json!({"command": "cargo build"})),
            report("Build ok."),
        ],
    );
    let out = r.agent.run_task("create hello", Mode::Auto);
    assert_eq!(out.status, Status::Completed, "{}", out.summary);
    assert_eq!(out.verification.build, Some(true));
    let think_calls = r.thinking.calls.lock().unwrap();
    assert!(last_user_text(&think_calls[2]).contains("could not observe"), "the false claim must be challenged");
    assert!(r.ui.text().contains("claimed verification that was not observed"));
}

#[test]
fn max_iterations_stops_the_loop() {
    let mut r = rig_with(
        MockRuntime::from_fn("thinking", Box::new(|_| Ok(instruction("keep going")))),
        MockRuntime::from_fn("coder", Box::new(|_| Ok(report("did a bit")))),
        vec![],
        |c| c.qwen.max_iterations = 3,
    );
    let out = r.agent.run_task("never ending task", Mode::Auto);
    assert_eq!(out.status, Status::MaxIterations);
    assert_eq!(out.iterations, 3);
    assert_eq!(r.coder.call_count(), 3);
    assert_eq!(r.thinking.call_count(), 3);
}

#[test]
fn invalid_json_is_repaired_and_plain_text_is_never_executed() {
    let marker = tempfile::tempdir().unwrap();
    let evil = marker.path().join("pwned.txt");
    let evil_cmd = format!("touch {}", evil.display());
    let mut r = rig(
        vec![instruction("do something"), complete("ok", None, None)],
        vec![
            // plain text that looks like a shell command: must not be executed
            evil_cmd.clone(),
            // broken JSON
            "{\"type\": \"tool_call\", \"tool\": ".to_string(),
            // wrong schema
            "{\"type\":\"tool_call\"}".to_string(),
            tool("list_directory", json!({})),
            report("listed"),
        ],
    );
    let out = r.agent.run_task("x", Mode::Auto);
    assert_eq!(out.status, Status::Completed, "{}", out.summary);
    assert!(!evil.exists(), "model plain text was executed!");
    let coder_calls = r.coder.calls.lock().unwrap();
    assert!(last_user_text(&coder_calls[1]).contains("No JSON object"), "feedback must explain the problem");
    assert!(last_user_text(&coder_calls[2]).contains("malformed"));
    assert!(last_user_text(&coder_calls[3]).contains("missing required property"));
}

#[test]
fn model_that_never_produces_json_fails_the_task_without_crashing() {
    let mut r = rig_with(
        MockRuntime::from_fn("thinking", Box::new(|_| Ok("I think we should just rm -rf everything".into()))),
        MockRuntime::scripted("coder", vec![]),
        vec![],
        |c| c.agent.max_repairs = 2,
    );
    let out = r.agent.run_task("x", Mode::Auto);
    assert!(matches!(out.status, Status::Failed(ref m) if m.contains("invalid output")), "{:?}", out.status);
    assert_eq!(r.thinking.call_count(), 3);
}

#[test]
fn permission_ask_yes_and_no_reach_the_agent() {
    let build = |answers: Vec<bool>| {
        rig_with(
            MockRuntime::scripted("thinking", vec![instruction("delete old.txt"), complete("done", None, None)]),
            MockRuntime::scripted("coder", vec![tool("delete_file", json!({"path": "old.txt"})), report("done")]),
            answers,
            |_| {},
        )
    };
    // N: file survives and the Coder is told it was denied
    let mut r = build(vec![false]);
    std::fs::write(r.dir.path().join("old.txt"), "x").unwrap();
    let out = r.agent.run_task("remove old.txt", Mode::Auto);
    assert_eq!(out.status, Status::Completed);
    assert!(r.dir.path().join("old.txt").exists());
    assert_eq!(r.ui.prompts().len(), 1);
    assert!(r.ui.prompts()[0].command.contains("delete_file"));
    assert!(last_user_text(&r.coder.calls.lock().unwrap()[1]).contains("denied by the user"));
    // Y: file is deleted
    let mut r = build(vec![true]);
    std::fs::write(r.dir.path().join("old.txt"), "x").unwrap();
    let out = r.agent.run_task("remove old.txt", Mode::Auto);
    assert_eq!(out.status, Status::Completed);
    assert!(!r.dir.path().join("old.txt").exists());
}

#[test]
fn permission_deny_blocks_system_destructive_commands() {
    let mut r = rig(
        vec![instruction("wipe"), complete("could not", None, None)],
        vec![tool("run_command", json!({"command": "rm -rf /"})), report("blocked")],
    );
    let out = r.agent.run_task("wipe the machine", Mode::Auto);
    assert_eq!(out.status, Status::Completed);
    assert!(r.ui.prompts().is_empty(), "deny must not even prompt");
    let coder_calls = r.coder.calls.lock().unwrap();
    let t = last_user_text(&coder_calls[1]);
    assert!(t.contains("blocked by the permission policy") && t.contains("NOT executed"), "{t}");
}

#[test]
fn connection_failures_are_retried_and_context_overflow_is_compacted() {
    let mut fails = 2;
    let mut overflow = true;
    let coder = MockRuntime::from_fn(
        "coder",
        Box::new(move |_| {
            if fails > 0 {
                fails -= 1;
                return Err(RuntimeError::Connection("connection reset".into()));
            }
            if overflow {
                overflow = false;
                return Err(RuntimeError::ContextOverflow("too long".into()));
            }
            Ok(report("recovered"))
        }),
    );
    let mut r = rig_with(MockRuntime::scripted("thinking", vec![instruction("go"), complete("fine", None, None)]), coder, vec![], |_| {});
    let out = r.agent.run_task("x", Mode::Auto);
    assert_eq!(out.status, Status::Completed, "{}", out.summary);
    let text = r.ui.text();
    assert!(text.contains("connection problem") && text.contains("retry 2/2") && text.contains("context window exceeded"), "{text}");
}

#[test]
fn unreachable_model_fails_the_task_but_not_the_process() {
    let mut r = rig_with(
        MockRuntime::from_fn("thinking", Box::new(|_| Err(RuntimeError::Connection("no route to host".into())))),
        MockRuntime::scripted("coder", vec![]),
        vec![],
        |_| {},
    );
    let out = r.agent.run_task("x", Mode::Auto);
    assert!(matches!(out.status, Status::Failed(ref m) if m.contains("no route to host")));
    assert_eq!(r.thinking.call_count(), 3, "1 try + 2 retries");
    // the agent is still usable afterwards
    let out2 = r.agent.run_task("again", Mode::Auto);
    assert!(matches!(out2.status, Status::Failed(_)));
}

#[test]
fn thinking_uses_read_only_tools_but_cannot_modify() {
    let mut r = rig(
        vec![
            tool("read_file", json!({"path": "notes.txt"})),
            tool("write_file", json!({"path": "hack.txt", "content": "nope"})),
            complete("The file says: hello notes", None, None),
        ],
        vec![],
    );
    std::fs::write(r.dir.path().join("notes.txt"), "hello notes\n").unwrap();
    let out = r.agent.run_task("what does notes.txt say?", Mode::Auto);
    assert_eq!(out.status, Status::Completed);
    assert!(!r.dir.path().join("hack.txt").exists());
    let calls = r.thinking.calls.lock().unwrap();
    assert!(last_user_text(&calls[1]).contains("hello notes"));
    assert!(last_user_text(&calls[2]).contains("not available to the Thinking model"));
    assert_eq!(r.coder.call_count(), 0);
}

#[test]
fn coder_only_mode_skips_thinking() {
    let mut r = rig(vec![], vec![j(json!({"type": "edit", "path": "a.txt", "operations": [{"type": "create", "content": "hi\n"}]})), report("created a.txt")]);
    let out = r.agent.run_task("create a.txt", Mode::Coder);
    assert_eq!(out.status, Status::Completed);
    assert_eq!(read(r.dir.path(), "a.txt"), "hi\n");
    assert_eq!(r.thinking.call_count(), 0);
}

#[test]
fn thinking_only_mode_answers_without_the_coder() {
    let mut r = rig(vec![instruction("do it"), complete("The answer is 42.", None, None)], vec![]);
    let out = r.agent.run_task("what is the answer?", Mode::Thinking);
    assert_eq!(out.status, Status::Completed);
    assert!(out.summary.contains("42"));
    assert_eq!(r.coder.call_count(), 0);
    assert!(last_user_text(&r.thinking.calls.lock().unwrap()[1]).contains("disabled in thinking-only mode"));
}

#[test]
fn repeated_identical_failures_are_flagged_and_stop_the_coder() {
    let mut r = rig(
        vec![instruction("read the missing file"), complete("gave up", None, None)],
        std::iter::repeat_with(|| tool("read_file", json!({"path": "nope.txt"}))).take(8).collect(),
    );
    let out = r.agent.run_task("x", Mode::Auto);
    assert_eq!(out.status, Status::Completed);
    assert_eq!(r.coder.call_count(), 5, "stops after the 5th identical failure");
    assert!(last_user_text(&r.coder.calls.lock().unwrap()[3]).contains("NOTE: you have now made this exact call"));
    assert!(last_user_text(&r.thinking.calls.lock().unwrap()[1]).contains("status: stuck"));
}

#[test]
fn follow_up_tasks_keep_the_conversation_and_reset_clears_it() {
    let mut r = rig(vec![complete("first", None, None), complete("second", None, None), complete("third", None, None)], vec![]);
    r.agent.run_task("first task", Mode::Auto);
    r.agent.run_task("second task", Mode::Auto);
    {
        let calls = r.thinking.calls.lock().unwrap();
        let all: String = calls[1].iter().map(|m| m.content.clone()).collect::<Vec<_>>().join("\n");
        assert!(all.contains("first task") && all.contains("NEW USER REQUEST"), "history must carry over");
    }
    r.agent.reset();
    assert_eq!(r.agent.history_len(), 0);
    r.agent.run_task("third task", Mode::Auto);
    let calls = r.thinking.calls.lock().unwrap();
    let all: String = calls[2].iter().map(|m| m.content.clone()).collect::<Vec<_>>().join("\n");
    assert!(!all.contains("first task"));
}

#[test]
fn cancellation_stops_the_task() {
    let cancel = Arc::new(AtomicBool::new(false));
    let c2 = cancel.clone();
    // simulates Ctrl-C arriving while the Thinking model is generating
    let thinking = Arc::new(MockRuntime::from_fn(
        "thinking",
        Box::new(move |_| {
            c2.store(true, std::sync::atomic::Ordering::Relaxed);
            Ok(instruction("work"))
        }),
    ));
    let coder = Arc::new(MockRuntime::scripted("coder", vec![report("x")]));
    let dir = tempfile::tempdir().unwrap();
    let cfg = Arc::new(Config::default());
    let models = ModelSet::new(
        Arc::new(ThinkingModel(ModelHandle::from_runtime("thinking", "qwen", "t", thinking.clone()))),
        Arc::new(CoderModel(ModelHandle::from_runtime("coder", "qwen", "c", coder.clone()))),
    );
    let ui = Arc::new(BufferUi::new());
    let mut agent = Agent::new(cfg.clone(), dir.path(), models, PermissionManager::new(&cfg.permissions), ui.clone(), cancel);
    let out = agent.run_task("x", Mode::Auto);
    assert_eq!(out.status, Status::Cancelled);
    assert_eq!(coder.call_count(), 0, "the Coder must not run after cancellation");
    assert!(ui.text().contains("task cancelled"));
}

#[test]
fn ui_receives_the_expected_event_sequence() {
    let mut r = rig(
        vec![j(json!({"type": "plan", "steps": ["one", "two"]})), instruction("write a file"), complete("done", None, None)],
        vec![j(json!({"type": "edit", "path": "x.txt", "operations": [{"type": "create", "content": "a\n"}]})), report("wrote x.txt")],
    );
    r.agent.run_task("write x.txt", Mode::Auto);
    let ev = r.ui.events();
    assert!(ev.iter().any(|e| matches!(e, Event::Plan(p) if p.len() == 2)));
    assert!(ev.iter().any(|e| matches!(e, Event::Phase { role, .. } if role == "thinking")));
    assert!(ev.iter().any(|e| matches!(e, Event::Phase { role, .. } if role == "coder")));
    assert!(ev.iter().any(|e| matches!(e, Event::ToolCall { tool, .. } if tool == "edit_file")));
    assert!(ev.iter().any(|e| matches!(e, Event::Diff(_))));
    assert!(ev.iter().any(|e| matches!(e, Event::Message(m) if m.contains("done") && m.contains("x.txt"))));
}
