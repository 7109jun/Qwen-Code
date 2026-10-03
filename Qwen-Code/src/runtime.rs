//! Model runtime layer: how text is generated for a list of chat messages.
//!
//! ```text
//! ModelProvider -> Runtime -> { OnnxRuntime (src/onnx.rs) | QwenApiRuntime | MockRuntime }
//! ```

use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    /// `system`, `user` or `assistant`.
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: &str, content: impl Into<String>) -> Self {
        Self { role: role.to_string(), content: content.into() }
    }
    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }
}

#[derive(Debug, Clone)]
pub struct GenOptions {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    /// Stop as soon as a complete JSON answer has been generated.
    pub stop_on_json: bool,
    pub seed: Option<u64>,
    pub cancel: Option<Arc<AtomicBool>>,
}

impl Default for GenOptions {
    fn default() -> Self {
        Self { max_new_tokens: 4096, temperature: 0.2, top_p: 0.95, top_k: 20, repetition_penalty: 1.0, stop_on_json: true, seed: None, cancel: None }
    }
}

impl GenOptions {
    pub fn cancelled(&self) -> bool {
        self.cancel.as_ref().is_some_and(|c| c.load(Ordering::Relaxed))
    }
}

#[derive(Debug, Clone, Default)]
pub struct GenOutput {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: String,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum RuntimeError {
    #[error("context window exceeded: {0}")]
    ContextOverflow(String),
    #[error("model connection failure: {0}")]
    Connection(String),
    #[error("model failure: {0}")]
    Failed(String),
    #[error("generation cancelled")]
    Cancelled,
}

impl RuntimeError {
    pub fn is_retryable(&self) -> bool {
        matches!(self, RuntimeError::Connection(_))
    }
}

pub trait Runtime: Send + Sync {
    fn generate(&self, messages: &[ChatMessage], opts: &GenOptions) -> Result<GenOutput, RuntimeError>;
    /// One-line description (runtime kind, device, location).
    fn describe(&self) -> String;
    fn context_window(&self) -> Option<usize> {
        None
    }
    /// Loads the model now (for lazily initialised runtimes). Used by `--check` and `/status`.
    fn load(&self) -> Result<String, String> {
        Ok(self.describe())
    }
    fn is_loaded(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------------------------
// Chat templates
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    /// Qwen (ChatML).
    ChatMl { think_prefill: bool },
    /// Kimi K2.
    Kimi,
    /// GLM 4.x.
    Glm,
}

impl ChatTemplate {
    /// Renders the prompt string that is tokenised for ONNX models.
    pub fn render(&self, messages: &[ChatMessage]) -> String {
        let mut s = String::new();
        match self {
            ChatTemplate::ChatMl { think_prefill } => {
                for m in messages {
                    s.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", m.role, m.content));
                }
                s.push_str("<|im_start|>assistant\n");
                if *think_prefill {
                    s.push_str("<think>\n");
                }
            }
            ChatTemplate::Kimi => {
                for m in messages {
                    let (tag, name) = match m.role.as_str() {
                        "system" => ("<|im_system|>", "system"),
                        "assistant" => ("<|im_assistant|>", "assistant"),
                        _ => ("<|im_user|>", "user"),
                    };
                    s.push_str(&format!("{tag}{name}<|im_middle|>{}<|im_end|>", m.content));
                }
                s.push_str("<|im_assistant|>assistant<|im_middle|>");
            }
            ChatTemplate::Glm => {
                s.push_str("[gMASK]<sop>");
                for m in messages {
                    s.push_str(&format!("<|{}|>\n{}", m.role, m.content));
                }
                s.push_str("<|assistant|>\n");
            }
        }
        s
    }

    pub fn stop_strings(&self) -> &'static [&'static str] {
        match self {
            ChatTemplate::ChatMl { .. } => &["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
            ChatTemplate::Kimi => &["<|im_end|>", "<|im_user|>"],
            ChatTemplate::Glm => &["<|user|>", "<|endoftext|>", "<|observation|>"],
        }
    }

    pub fn prefills_think(&self) -> bool {
        matches!(self, ChatTemplate::ChatMl { think_prefill: true })
    }
}

// ---------------------------------------------------------------------------------------------
// Qwen API runtime (Alibaba Cloud Model Studio / DashScope native text-generation API)
// ---------------------------------------------------------------------------------------------

/// Default base URL of the Qwen (DashScope) API, international endpoint.
pub const QWEN_API_BASE: &str = "https://dashscope-intl.aliyuncs.com/api/v1";
/// Environment variable that holds the Qwen API key.
pub const QWEN_API_KEY_ENV: &str = "DASHSCOPE_API_KEY";

/// True for `http(s)://localhost`, `127.x.x.x` and `[::1]` URLs.
pub fn is_loopback_url(url: &str) -> bool {
    let rest = url.split("://").nth(1).unwrap_or(url);
    let authority = rest.split('/').next().unwrap_or("");
    let host = if authority.starts_with('[') {
        authority.split(']').next().unwrap_or("").trim_start_matches('[')
    } else {
        authority.split(':').next().unwrap_or("")
    };
    host.eq_ignore_ascii_case("localhost") || host == "::1" || host.starts_with("127.")
}

/// Talks to the native Qwen API (`POST {base}/services/aigc/text-generation/generation`) with
/// server-sent-event streaming, which every Qwen model (including the thinking-only ones) supports.
pub struct QwenApiRuntime {
    pub label: String,
    pub base_url: String,
    pub model: String,
    pub api_key: Option<String>,
    pub json_mode: bool,
    pub timeout: Duration,
    pub context_window: Option<usize>,
    agent: ureq::Agent,
}

impl QwenApiRuntime {
    pub fn new(label: &str, base_url: &str, model: &str, api_key: Option<String>, json_mode: bool, timeout_secs: u64, context_window: Option<usize>) -> Self {
        let timeout = Duration::from_secs(timeout_secs.max(5));
        // `timeout_read` bounds the silence between two streamed chunks, not the whole answer.
        // ureq does not honour NO_PROXY, so loopback endpoints (local gateways) bypass the proxy explicitly.
        let loopback = is_loopback_url(base_url);
        let agent = ureq::AgentBuilder::new().timeout_connect(Duration::from_secs(15)).timeout_read(timeout).timeout_write(Duration::from_secs(30)).try_proxy_from_env(!loopback).build();
        Self {
            label: label.to_string(),
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            api_key: api_key.filter(|k| !k.is_empty()),
            json_mode,
            timeout,
            context_window,
            agent,
        }
    }

    fn endpoint(&self) -> String {
        format!("{}/services/aigc/text-generation/generation", self.base_url)
    }

    fn classify_http_error(code: u16, body: &str) -> RuntimeError {
        let lower = body.to_lowercase();
        let short: String = body.chars().take(400).collect();
        if (lower.contains("context") && (lower.contains("length") || lower.contains("window") || lower.contains("maximum")))
            || lower.contains("too many tokens")
            || lower.contains("input is too long")
            || lower.contains("range of input length")
            || lower.contains("input length")
        {
            return RuntimeError::ContextOverflow(short);
        }
        if code == 408 || code == 429 || code >= 500 || lower.contains("throttling") || lower.contains("limit_requests") {
            return RuntimeError::Connection(format!("HTTP {code}: {short}"));
        }
        if code == 401 || code == 403 {
            return RuntimeError::Failed(format!("HTTP {code}: authentication failed (check the Qwen API key in ${QWEN_API_KEY_ENV}). {short}"));
        }
        RuntimeError::Failed(format!("HTTP {code}: {short}"))
    }
}

impl Runtime for QwenApiRuntime {
    fn generate(&self, messages: &[ChatMessage], opts: &GenOptions) -> Result<GenOutput, RuntimeError> {
        use std::io::BufRead;
        if opts.cancelled() {
            return Err(RuntimeError::Cancelled);
        }
        if self.api_key.is_none() {
            return Err(RuntimeError::Failed(format!("no Qwen API key: set the {QWEN_API_KEY_ENV} environment variable (or api_key_env / api_key in qwen.toml)")));
        }
        let msgs: Vec<Value> = messages.iter().map(|m| json!({"role": m.role, "content": m.content})).collect();
        let mut params = json!({
            "result_format": "message",
            "incremental_output": true,
            "max_tokens": opts.max_new_tokens,
            "temperature": opts.temperature,
            "top_p": opts.top_p,
        });
        if opts.top_k > 0 {
            params["top_k"] = json!(opts.top_k);
        }
        if opts.repetition_penalty > 1.0 {
            params["repetition_penalty"] = json!(opts.repetition_penalty);
        }
        if let Some(seed) = opts.seed {
            params["seed"] = json!(seed % 2_147_483_647);
        }
        if self.json_mode {
            params["response_format"] = json!({"type": "json_object"});
        }
        let body = json!({"model": self.model, "input": {"messages": msgs}, "parameters": params});
        let mut req = self.agent.post(&self.endpoint()).set("Content-Type", "application/json").set("Accept", "text/event-stream").set("X-DashScope-SSE", "enable");
        if let Some(k) = &self.api_key {
            req = req.set("Authorization", &format!("Bearer {k}"));
        }
        let resp = match req.send_json(body) {
            Ok(r) => r,
            Err(ureq::Error::Status(code, r)) => {
                let text = r.into_string().unwrap_or_default();
                return Err(Self::classify_http_error(code, &text));
            }
            Err(ureq::Error::Transport(t)) => return Err(RuntimeError::Connection(format!("{}: {t}", self.endpoint()))),
        };

        let mut content = String::new();
        let mut reasoning = String::new();
        let mut finish = String::new();
        let (mut prompt_tokens, mut completion_tokens) = (0usize, 0usize);
        let reader = std::io::BufReader::new(resp.into_reader());
        for line in reader.lines() {
            if opts.cancelled() {
                return Err(RuntimeError::Cancelled);
            }
            let line = line.map_err(|e| RuntimeError::Connection(format!("the stream was interrupted: {e}")))?;
            let Some(data) = line.strip_prefix("data:") else { continue };
            let data = data.trim();
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            let v: Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if let Some(code) = v.get("code").and_then(|c| c.as_str()).filter(|c| !c.is_empty()) {
                let msg = v.get("message").and_then(|m| m.as_str()).unwrap_or("");
                return Err(Self::classify_http_error(400, &format!("{code}: {msg}")));
            }
            if let Some(u) = v.get("usage") {
                prompt_tokens = u.get("input_tokens").and_then(|x| x.as_u64()).unwrap_or(prompt_tokens as u64) as usize;
                completion_tokens = u.get("output_tokens").and_then(|x| x.as_u64()).unwrap_or(completion_tokens as u64) as usize;
            }
            let Some(choice) = v.get("output").and_then(|o| o.get("choices")).and_then(|c| c.get(0)) else { continue };
            if let Some(m) = choice.get("message") {
                if let Some(t) = m.get("content").and_then(|c| c.as_str()) {
                    content.push_str(t);
                }
                if let Some(t) = m.get("reasoning_content").and_then(|c| c.as_str()) {
                    reasoning.push_str(t);
                }
            }
            if let Some(f) = choice.get("finish_reason").and_then(|f| f.as_str()) {
                if f != "null" && !f.is_empty() {
                    finish = f.to_string();
                }
            }
        }
        if content.is_empty() && reasoning.is_empty() && finish.is_empty() {
            return Err(RuntimeError::Connection("the Qwen API stream ended without any content".into()));
        }
        let text = if !reasoning.is_empty() && !content.contains("</think>") { format!("<think>\n{reasoning}\n</think>\n{content}") } else { content };
        Ok(GenOutput { text, prompt_tokens, completion_tokens, finish_reason: finish })
    }

    fn describe(&self) -> String {
        format!("qwen api {} model {} ({})", self.base_url, self.model, self.label)
    }

    fn load(&self) -> Result<String, String> {
        if self.api_key.is_none() {
            return Err(format!("no Qwen API key: set the {QWEN_API_KEY_ENV} environment variable (or api_key_env / api_key in qwen.toml)"));
        }
        Ok(self.describe())
    }

    fn context_window(&self) -> Option<usize> {
        self.context_window
    }
}

// ---------------------------------------------------------------------------------------------
// Lazy runtime: the model is loaded on first use; load failures are not cached.
// ---------------------------------------------------------------------------------------------

pub type RuntimeFactory = Box<dyn Fn() -> Result<Arc<dyn Runtime>, String> + Send + Sync>;

pub struct LazyRuntime {
    label: String,
    factory: RuntimeFactory,
    inner: Mutex<Option<Arc<dyn Runtime>>>,
    window: Option<usize>,
}

impl LazyRuntime {
    pub fn new(label: impl Into<String>, window: Option<usize>, factory: RuntimeFactory) -> Self {
        Self { label: label.into(), factory, inner: Mutex::new(None), window }
    }

    fn get(&self) -> Result<Arc<dyn Runtime>, String> {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(r) = g.as_ref() {
            return Ok(r.clone());
        }
        let r = (self.factory)()?;
        *g = Some(r.clone());
        Ok(r)
    }
}

impl Runtime for LazyRuntime {
    fn generate(&self, messages: &[ChatMessage], opts: &GenOptions) -> Result<GenOutput, RuntimeError> {
        let r = self.get().map_err(RuntimeError::Failed)?;
        r.generate(messages, opts)
    }

    fn describe(&self) -> String {
        let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        match g.as_ref() {
            Some(r) => r.describe(),
            None => format!("{} (not loaded yet)", self.label),
        }
    }

    fn context_window(&self) -> Option<usize> {
        let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.as_ref().and_then(|r| r.context_window()).or(self.window)
    }

    fn load(&self) -> Result<String, String> {
        let r = self.get()?;
        r.load()
    }

    fn is_loaded(&self) -> bool {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).is_some()
    }
}

// ---------------------------------------------------------------------------------------------
// Scripted runtime (tests and offline demos of the agent loop)
// ---------------------------------------------------------------------------------------------

pub type MockFn = Box<dyn FnMut(&[ChatMessage]) -> Result<String, RuntimeError> + Send>;

pub struct MockRuntime {
    name: String,
    responder: Mutex<MockFn>,
    pub calls: Arc<Mutex<Vec<Vec<ChatMessage>>>>,
}

impl MockRuntime {
    /// Returns the given responses in order; fails once they are exhausted.
    pub fn scripted(name: &str, responses: Vec<String>) -> Self {
        let mut it = responses.into_iter();
        Self::from_fn(name, Box::new(move |_| it.next().ok_or_else(|| RuntimeError::Failed("mock model: script exhausted".into()))))
    }

    pub fn from_fn(name: &str, f: MockFn) -> Self {
        Self { name: name.to_string(), responder: Mutex::new(f), calls: Arc::new(Mutex::new(Vec::new())) }
    }

    pub fn call_count(&self) -> usize {
        self.calls.lock().unwrap().len()
    }
}

impl Runtime for MockRuntime {
    fn generate(&self, messages: &[ChatMessage], opts: &GenOptions) -> Result<GenOutput, RuntimeError> {
        if opts.cancelled() {
            return Err(RuntimeError::Cancelled);
        }
        self.calls.lock().unwrap().push(messages.to_vec());
        let text = (self.responder.lock().unwrap())(messages)?;
        let prompt_tokens = messages.iter().map(|m| m.content.len() / 4).sum();
        Ok(GenOutput { completion_tokens: text.len() / 4, text, prompt_tokens, finish_reason: "stop".into() })
    }

    fn describe(&self) -> String {
        format!("mock runtime ({})", self.name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    #[test]
    fn chatml_rendering_and_think_prefill() {
        let msgs = [ChatMessage::system("sys"), ChatMessage::user("hi")];
        let t = ChatTemplate::ChatMl { think_prefill: false }.render(&msgs);
        assert_eq!(t, "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
        let t = ChatTemplate::ChatMl { think_prefill: true }.render(&msgs);
        assert!(t.ends_with("<|im_start|>assistant\n<think>\n"));
        assert!(ChatTemplate::Kimi.render(&msgs).contains("<|im_user|>user<|im_middle|>hi<|im_end|>"));
        assert!(ChatTemplate::Glm.render(&msgs).starts_with("[gMASK]<sop><|system|>\nsys<|user|>\nhi<|assistant|>"));
    }

    /// Minimal one-shot HTTP server returning a canned response.
    fn serve_once(status: &str, content_type: &str, body: &str) -> (String, std::thread::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let status = status.to_string();
        let body = body.to_string();
        let ct = content_type.to_string();
        let h = std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let mut buf = vec![0u8; 65536];
            let mut got = Vec::new();
            loop {
                let n = s.read(&mut buf).unwrap();
                got.extend_from_slice(&buf[..n]);
                let text = String::from_utf8_lossy(&got).to_string();
                if let Some(pos) = text.find("\r\n\r\n") {
                    let len: usize = text[..pos].lines().find_map(|l| l.to_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse().unwrap())).unwrap_or(0);
                    if got.len() >= pos + 4 + len {
                        break;
                    }
                }
                if n == 0 {
                    break;
                }
            }
            let resp = format!("HTTP/1.1 {status}\r\nContent-Type: {ct}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body);
            s.write_all(resp.as_bytes()).unwrap();
            String::from_utf8_lossy(&got).to_string()
        });
        (format!("http://{addr}/api/v1"), h)
    }

    #[test]
    fn qwen_api_streams_and_merges_reasoning() {
        let sse = "id:1\nevent:result\n:HTTP_STATUS/200\ndata:{\"output\":{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"\",\"reasoning_content\":\"because \"},\"finish_reason\":\"null\"}]},\"usage\":{\"input_tokens\":11,\"output_tokens\":1}}\n\n\
                   id:2\nevent:result\n:HTTP_STATUS/200\ndata:{\"output\":{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"{\\\"type\\\":\\\"plan\\\",\",\"reasoning_content\":\"it works\"},\"finish_reason\":\"null\"}]},\"usage\":{\"input_tokens\":11,\"output_tokens\":4}}\n\n\
                   id:3\nevent:result\n:HTTP_STATUS/200\ndata:{\"output\":{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"\\\"steps\\\":[\\\"x\\\"]}\"},\"finish_reason\":\"stop\"}]},\"usage\":{\"input_tokens\":11,\"output_tokens\":9}}\n\n";
        let (base, h) = serve_once("200 OK", "text/event-stream", sse);
        let rt = QwenApiRuntime::new("test", &base, "qwen3-coder-next", Some("sekret".into()), false, 10, None);
        let out = rt.generate(&[ChatMessage::system("s"), ChatMessage::user("hello")], &GenOptions { seed: Some(7), ..Default::default() }).unwrap();
        assert_eq!(out.text, "<think>\nbecause it works\n</think>\n{\"type\":\"plan\",\"steps\":[\"x\"]}");
        assert_eq!((out.prompt_tokens, out.completion_tokens, out.finish_reason.as_str()), (11, 9, "stop"));
        let request = h.join().unwrap();
        assert!(request.starts_with("POST /api/v1/services/aigc/text-generation/generation"), "{request}");
        let lower = request.to_lowercase();
        assert!(lower.contains("authorization: bearer sekret") && lower.contains("x-dashscope-sse: enable"));
        assert!(request.contains("\"model\":\"qwen3-coder-next\"") && request.contains("\"result_format\":\"message\"") && request.contains("\"incremental_output\":true"));
        assert!(request.contains("\"input\":{\"messages\""));
        assert!(!request.contains("response_format\":{"), "json mode is off by default");
    }

    #[test]
    fn qwen_api_requires_a_key_and_classifies_errors() {
        let rt = QwenApiRuntime::new("t", "http://127.0.0.1:1/api/v1", "m", None, false, 5, None);
        let e = rt.generate(&[ChatMessage::user("x")], &GenOptions::default()).unwrap_err();
        assert!(matches!(e, RuntimeError::Failed(ref m) if m.contains("DASHSCOPE_API_KEY")));

        let (base, _h) = serve_once("429 Too Many Requests", "application/json", r#"{"code":"Throttling.RateQuota","message":"slow down"}"#);
        let rt = QwenApiRuntime::new("t", &base, "m", Some("k".into()), false, 10, None);
        let e = rt.generate(&[ChatMessage::user("x")], &GenOptions::default()).unwrap_err();
        assert!(matches!(e, RuntimeError::Connection(_)) && e.is_retryable());

        let (base, _h) = serve_once("400 Bad Request", "application/json", r#"{"code":"InvalidParameter","message":"Range of input length should be [1, 98304]"}"#);
        let rt = QwenApiRuntime::new("t", &base, "m", Some("k".into()), false, 10, None);
        assert!(matches!(rt.generate(&[ChatMessage::user("x")], &GenOptions::default()).unwrap_err(), RuntimeError::ContextOverflow(_)));

        let (base, _h) = serve_once("401 Unauthorized", "application/json", r#"{"code":"InvalidApiKey","message":"Invalid API-key provided."}"#);
        let rt = QwenApiRuntime::new("t", &base, "m", Some("k".into()), false, 10, None);
        let e = rt.generate(&[ChatMessage::user("x")], &GenOptions::default()).unwrap_err();
        assert!(matches!(e, RuntimeError::Failed(ref m) if m.contains("authentication")));

        // an error event inside a 200 stream
        let (base, _h) = serve_once("200 OK", "text/event-stream", "id:1\nevent:error\ndata:{\"code\":\"InvalidParameter\",\"message\":\"bad things\"}\n\n");
        let rt = QwenApiRuntime::new("t", &base, "m", Some("k".into()), false, 10, None);
        assert!(matches!(rt.generate(&[ChatMessage::user("x")], &GenOptions::default()).unwrap_err(), RuntimeError::Failed(_)));

        // nothing listening
        let rt = QwenApiRuntime::new("t", "http://127.0.0.1:1/api/v1", "m", Some("k".into()), false, 5, None);
        assert!(matches!(rt.generate(&[ChatMessage::user("x")], &GenOptions::default()).unwrap_err(), RuntimeError::Connection(_)));
    }

    #[test]
    fn lazy_runtime_retries_after_a_failed_load() {
        let attempts = Arc::new(Mutex::new(0));
        let a2 = attempts.clone();
        let lazy = LazyRuntime::new(
            "lazy",
            None,
            Box::new(move || {
                let mut n = a2.lock().unwrap();
                *n += 1;
                if *n == 1 {
                    Err("model files missing".to_string())
                } else {
                    Ok(Arc::new(MockRuntime::scripted("m", vec!["ok".into()])) as Arc<dyn Runtime>)
                }
            }),
        );
        assert!(!lazy.is_loaded());
        let e = lazy.generate(&[ChatMessage::user("x")], &GenOptions::default()).unwrap_err();
        assert!(e.to_string().contains("model files missing"));
        assert_eq!(lazy.generate(&[ChatMessage::user("x")], &GenOptions::default()).unwrap().text, "ok");
        assert!(lazy.is_loaded());
    }

    #[test]
    fn mock_runtime_scripts_and_records() {
        let m = MockRuntime::scripted("t", vec!["a".into(), "b".into()]);
        assert_eq!(m.generate(&[ChatMessage::user("1")], &GenOptions::default()).unwrap().text, "a");
        assert_eq!(m.generate(&[ChatMessage::user("2")], &GenOptions::default()).unwrap().text, "b");
        assert!(m.generate(&[ChatMessage::user("3")], &GenOptions::default()).is_err());
        assert_eq!(m.call_count(), 3);
    }
}
