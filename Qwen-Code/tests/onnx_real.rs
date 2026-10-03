//! Exercises the real ONNX Runtime path with a tiny generated model.
//!
//! Skipped unless `ORT_DYLIB_PATH` (ONNX Runtime shared library) and `QWEN_TEST_ONNX_DIR`
//! (output of `python3 tools/make_tiny_model.py <dir>`) are set; CI sets both.

use qwen_code::onnx::{OnnxParams, OnnxRuntime};
use qwen_code::runtime::{ChatMessage, ChatTemplate, GenOptions, Runtime};
use std::path::PathBuf;

fn params(device: &str) -> Option<OnnxParams> {
    let dir = PathBuf::from(std::env::var_os("QWEN_TEST_ONNX_DIR")?);
    std::env::var_os("ORT_DYLIB_PATH")?;
    Some(OnnxParams {
        path: dir.clone(),
        device: device.into(),
        device_id: 0,
        intra_threads: 1,
        dylib: None,
        base_dir: dir,
        template: ChatTemplate::ChatMl { think_prefill: false },
        context_window: Some(4096),
        label: "tiny".into(),
    })
}

#[test]
fn onnx_runtime_loads_and_generates() {
    let Some(p) = params("cpu") else {
        eprintln!("skipped: ORT_DYLIB_PATH / QWEN_TEST_ONNX_DIR not set");
        return;
    };
    let rt = OnnxRuntime::load(&p).expect("load tiny model");
    assert!(!rt.has_kv_cache());
    let opts = GenOptions { max_new_tokens: 7, temperature: 0.0, stop_on_json: false, seed: Some(1), ..GenOptions::default() };
    let out = rt.generate(&[ChatMessage { role: "user".into(), content: "안녕 hello".into() }], &opts).expect("generate");
    assert_eq!(out.text, "abcabca", "{out:?}");
    assert!(rt.count_tokens("abc") >= 3);
}

#[test]
fn unavailable_execution_provider_is_a_clean_error() {
    let Some(p) = params("cuda") else { return };
    // No CUDA on CI / most machines: must be an Err (or succeed where CUDA exists), never a crash.
    let _ = OnnxRuntime::load(&p);
}

#[test]
fn missing_model_is_a_clean_error() {
    let Some(mut p) = params("cpu") else { return };
    p.path = p.path.join("does-not-exist");
    assert!(OnnxRuntime::load(&p).is_err());
}
