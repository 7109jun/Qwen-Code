# Qwen Code

A Claude-Code-style AI coding agent for the terminal (`qwen`), written in Rust for Linux and Windows.
Two models cooperate in a real iterative loop: a **Thinking** model (analysis, planning, review, error
analysis) and a **Coder** model (code, files, refactoring, tests, fixing build errors). The loop ends only
when the Thinking model declares the task complete (guarded by `max_iterations`).

## Quick start

```
cargo build --release
./target/release/qwen --init          # writes qwen.toml
./target/release/qwen --check         # verifies config, providers, model runtimes
./target/release/qwen                 # TUI session in the current directory
./target/release/qwen "task"          # one task, then exit
./target/release/qwen .               # session for this project
./target/release/qwen src/main.rs     # session focused on a file
./target/release/qwen --repl          # plain line interface
```

### Models: ONNX or Qwen API

* **Local ONNX** (`runtime = "onnx"`): put `model.onnx` + `tokenizer.json` in `models/thinking` and
  `models/coder` (see `models/README.md`, convert with `tools/convert_to_onnx.py`). ONNX Runtime is
  loaded dynamically: set `[runtime.onnx].dylib_path` or `ORT_DYLIB_PATH`. Devices: `cpu` (default),
  `cuda`, `directml`.
* **Qwen API** (`runtime = "api"`): DashScope native streaming API, no model files needed.
  `export DASHSCOPE_API_KEY=...` and see the commented example in `qwen.toml`.

## Agent protocol and tools

Models answer with JSON (`tool_call`, `edit`, `plan`, `instruction`, `report`, `complete`). Every call goes
through parser → schema validation → permission check → execution → result. Plain model text is never
executed. Tools: `read_file write_file edit_file delete_file list_directory search_files run_command
run_test git_status git_diff git_log git_branch`. File edits (`create replace insert delete append`)
verify the current file state first and fail safely if it changed.

## Permissions

Policies `allow` / `ask` / `deny` per category and per command class in `qwen.toml`. Commands are
tokenised and analysed (quotes, pipes, `$()`, `sudo`, `bash -c`, `cmd /c`, `powershell -Command`...), not
substring-matched. Ordinary development commands run without prompts; risky ones ask:

```
Qwen Code wants to execute:

<command>

Allow [Y/N]:
```

`N` returns a denial to the agent. `-y` auto-answers `ask` but never overrides `deny`.

## Commands

`/help /models /model /context /tools /status /diff /reset /clear /permissions /config /exit`

## Tests

```
cargo test                                     # unit + agent-loop integration tests (real cargo builds)
python3 tools/make_tiny_model.py /tmp/tiny     # real ONNX Runtime test:
ORT_DYLIB_PATH=/path/to/libonnxruntime.so QWEN_TEST_ONNX_DIR=/tmp/tiny cargo test --test onnx_real
python3 tools/mock_qwen_api.py scenario.json 18765   # scripted Qwen API for end-to-end runs of the binary
```

CI (`.github/workflows/build.yml`) builds and tests on Ubuntu and Windows and runs the ONNX test on Linux.
