//! Model provider abstraction.
//!
//! ```text
//! ModelProvider
//!  ├── QwenProvider   (ONNX Runtime or the Qwen API)
//!  ├── KimiProvider   (ONNX Runtime; API not available yet)
//!  └── GLMProvider    (ONNX Runtime; API not available yet)
//!
//! ThinkingProvider / CoderProvider  = the two roles of the agent loop
//! ```
//!
//! Model names are never hard-coded: they come from `qwen.toml`.

use crate::config::{Config, ModelConfig, OnnxSection, ResolvedModel};
use crate::onnx::{OnnxParams, OnnxRuntime};
use crate::runtime::{ChatMessage, ChatTemplate, GenOptions, GenOutput, LazyRuntime, QwenApiRuntime, Runtime, RuntimeError, QWEN_API_BASE, QWEN_API_KEY_ENV};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

pub struct ProviderContext {
    pub onnx: OnnxSection,
    /// Directory against which relative paths are resolved.
    pub base_dir: PathBuf,
    pub provider_api_base: Option<String>,
    pub provider_api_key_env: Option<String>,
}

pub trait ModelProvider: Send + Sync {
    fn id(&self) -> &'static str;
    fn display_name(&self) -> &'static str;
    /// Prompt format of the provider's models.
    fn template(&self, role: &str) -> ChatTemplate;
    /// Creates the runtime for a model. Heavy initialisation (ONNX loading) is deferred to first use.
    fn create_runtime(&self, m: &ResolvedModel, ctx: &ProviderContext) -> Result<Arc<dyn Runtime>, String>;
}

fn onnx_runtime(provider: &dyn ModelProvider, m: &ResolvedModel, ctx: &ProviderContext) -> Arc<dyn Runtime> {
    let device = m.cfg.device.clone().unwrap_or_else(|| ctx.onnx.device.clone());
    let params_template = provider.template(&m.role);
    let path = m.path.clone();
    let dylib = ctx.onnx.dylib_path.clone();
    let base = ctx.base_dir.clone();
    let device_id = ctx.onnx.device_id;
    let threads = ctx.onnx.intra_threads;
    let window = m.cfg.context_window;
    let label2 = format!("{} {}", provider.id(), m.cfg.name);
    Arc::new(LazyRuntime::new(
        format!("onnx {} ({device})", path.display()),
        window,
        Box::new(move || {
            let rt = OnnxRuntime::load(&OnnxParams {
                path: path.clone(),
                device: device.clone(),
                device_id,
                intra_threads: threads,
                dylib: dylib.clone(),
                base_dir: base.clone(),
                template: params_template,
                context_window: window,
                label: label2.clone(),
            })?;
            Ok(Arc::new(rt) as Arc<dyn Runtime>)
        }),
    ))
}

fn resolve_api_key(cfg: &ModelConfig, ctx: &ProviderContext, default_env: &str) -> Option<String> {
    if let Some(k) = cfg.api_key.as_deref().filter(|k| !k.is_empty()) {
        return Some(k.to_string());
    }
    for env in [cfg.api_key_env.as_deref(), ctx.provider_api_key_env.as_deref(), Some(default_env)].into_iter().flatten() {
        if let Ok(v) = std::env::var(env) {
            if !v.trim().is_empty() {
                return Some(v.trim().to_string());
            }
        }
    }
    None
}

pub struct QwenProvider;

impl ModelProvider for QwenProvider {
    fn id(&self) -> &'static str {
        "qwen"
    }
    fn display_name(&self) -> &'static str {
        "Qwen"
    }
    fn template(&self, role: &str) -> ChatTemplate {
        // Qwen3 "Thinking" checkpoints open the assistant turn with <think> in their chat template.
        ChatTemplate::ChatMl { think_prefill: role == "thinking" }
    }
    fn create_runtime(&self, m: &ResolvedModel, ctx: &ProviderContext) -> Result<Arc<dyn Runtime>, String> {
        match m.cfg.runtime.as_str() {
            "onnx" => Ok(onnx_runtime(self, m, ctx)),
            "api" => {
                if m.cfg.name.trim().is_empty() {
                    return Err(format!("models.{}.name must be set for the Qwen API", m.role));
                }
                let base = m.cfg.api_base.clone().or_else(|| ctx.provider_api_base.clone()).unwrap_or_else(|| QWEN_API_BASE.to_string());
                let key = resolve_api_key(&m.cfg, ctx, QWEN_API_KEY_ENV);
                Ok(Arc::new(QwenApiRuntime::new(&m.role, &base, &m.cfg.name, key, m.cfg.json_mode, m.cfg.timeout_secs, m.cfg.context_window)))
            }
            other => Err(format!("unsupported runtime \"{other}\"")),
        }
    }
}

pub struct KimiProvider;

impl ModelProvider for KimiProvider {
    fn id(&self) -> &'static str {
        "kimi"
    }
    fn display_name(&self) -> &'static str {
        "Kimi"
    }
    fn template(&self, _role: &str) -> ChatTemplate {
        ChatTemplate::Kimi
    }
    fn create_runtime(&self, m: &ResolvedModel, ctx: &ProviderContext) -> Result<Arc<dyn Runtime>, String> {
        match m.cfg.runtime.as_str() {
            "onnx" => Ok(onnx_runtime(self, m, ctx)),
            _ => Err("the Kimi API is not implemented yet; use runtime = \"onnx\" for Kimi models (or provider = \"qwen\")".into()),
        }
    }
}

pub struct GlmProvider;

impl ModelProvider for GlmProvider {
    fn id(&self) -> &'static str {
        "glm"
    }
    fn display_name(&self) -> &'static str {
        "GLM"
    }
    fn template(&self, _role: &str) -> ChatTemplate {
        ChatTemplate::Glm
    }
    fn create_runtime(&self, m: &ResolvedModel, ctx: &ProviderContext) -> Result<Arc<dyn Runtime>, String> {
        match m.cfg.runtime.as_str() {
            "onnx" => Ok(onnx_runtime(self, m, ctx)),
            _ => Err("the GLM API is not implemented yet; use runtime = \"onnx\" for GLM models (or provider = \"qwen\")".into()),
        }
    }
}

pub struct ProviderRegistry {
    providers: Vec<Box<dyn ModelProvider>>,
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self { providers: vec![Box::new(QwenProvider), Box::new(KimiProvider), Box::new(GlmProvider)] }
    }
}

impl ProviderRegistry {
    pub fn get(&self, id: &str) -> Option<&dyn ModelProvider> {
        self.providers.iter().find(|p| p.id() == id.to_ascii_lowercase()).map(|b| b.as_ref())
    }

    pub fn ids(&self) -> Vec<&'static str> {
        self.providers.iter().map(|p| p.id()).collect()
    }
}

// ---------------------------------------------------------------------------------------------
// Role handles
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub role: String,
    pub provider: String,
    pub name: String,
    pub runtime_kind: String,
    pub description: String,
    pub loaded: bool,
}

/// What the agent loop needs from a model, independent of provider and runtime.
pub trait ModelRole: Send + Sync {
    fn role(&self) -> &str;
    fn info(&self) -> ModelInfo;
    fn generate(&self, messages: &[ChatMessage], cancel: &Arc<AtomicBool>) -> Result<GenOutput, RuntimeError>;
    fn context_window(&self) -> Option<usize>;
    fn max_new_tokens(&self) -> usize;
    fn load(&self) -> Result<String, String>;
}

/// The Thinking role: analysis, planning, review.
pub trait ThinkingProvider: ModelRole {}
/// The Coder role: code generation and tool use.
pub trait CoderProvider: ModelRole {}

pub struct ModelHandle {
    role: String,
    provider: String,
    name: String,
    runtime_kind: String,
    runtime: Arc<dyn Runtime>,
    opts: GenOptions,
    window: Option<usize>,
}

impl ModelHandle {
    pub fn from_runtime(role: &str, provider: &str, name: &str, runtime: Arc<dyn Runtime>) -> Self {
        let opts = default_options(role);
        Self { role: role.into(), provider: provider.into(), name: name.into(), runtime_kind: "custom".into(), runtime, opts, window: None }
    }

    pub fn with_options(mut self, opts: GenOptions) -> Self {
        self.opts = opts;
        self
    }
}

fn default_options(role: &str) -> GenOptions {
    if role == "thinking" {
        GenOptions { max_new_tokens: 8192, temperature: 0.6, top_p: 0.95, top_k: 20, ..GenOptions::default() }
    } else {
        GenOptions { max_new_tokens: 4096, temperature: 0.2, top_p: 0.9, top_k: 20, ..GenOptions::default() }
    }
}

impl ModelRole for ModelHandle {
    fn role(&self) -> &str {
        &self.role
    }

    fn info(&self) -> ModelInfo {
        ModelInfo {
            role: self.role.clone(),
            provider: self.provider.clone(),
            name: self.name.clone(),
            runtime_kind: self.runtime_kind.clone(),
            description: self.runtime.describe(),
            loaded: self.runtime.is_loaded(),
        }
    }

    fn generate(&self, messages: &[ChatMessage], cancel: &Arc<AtomicBool>) -> Result<GenOutput, RuntimeError> {
        let mut opts = self.opts.clone();
        opts.cancel = Some(cancel.clone());
        self.runtime.generate(messages, &opts)
    }

    fn context_window(&self) -> Option<usize> {
        self.runtime.context_window().or(self.window)
    }

    fn max_new_tokens(&self) -> usize {
        self.opts.max_new_tokens
    }

    fn load(&self) -> Result<String, String> {
        self.runtime.load()
    }
}

macro_rules! role_wrapper {
    ($name:ident, $trait_:ident) => {
        pub struct $name(pub ModelHandle);
        impl ModelRole for $name {
            fn role(&self) -> &str {
                self.0.role()
            }
            fn info(&self) -> ModelInfo {
                self.0.info()
            }
            fn generate(&self, messages: &[ChatMessage], cancel: &Arc<AtomicBool>) -> Result<GenOutput, RuntimeError> {
                self.0.generate(messages, cancel)
            }
            fn context_window(&self) -> Option<usize> {
                self.0.context_window()
            }
            fn max_new_tokens(&self) -> usize {
                self.0.max_new_tokens()
            }
            fn load(&self) -> Result<String, String> {
                self.0.load()
            }
        }
        impl $trait_ for $name {}
    };
}

role_wrapper!(ThinkingModel, ThinkingProvider);
role_wrapper!(CoderModel, CoderProvider);

#[derive(Clone)]
pub struct ModelSet {
    pub thinking: Arc<dyn ThinkingProvider>,
    pub coder: Arc<dyn CoderProvider>,
}

impl ModelSet {
    pub fn new(thinking: Arc<dyn ThinkingProvider>, coder: Arc<dyn CoderProvider>) -> Self {
        Self { thinking, coder }
    }

    /// Builds both role models from the configuration (no model is loaded yet).
    pub fn from_config(cfg: &Config, base_dir: &Path) -> Result<ModelSet, String> {
        let reg = ProviderRegistry::default();
        let build = |role: &str| -> Result<ModelHandle, String> {
            let m = cfg.resolve_model(role, base_dir);
            let provider = reg.get(&m.cfg.provider).ok_or_else(|| format!("models.{role}.provider = \"{}\" is unknown (available: {})", m.cfg.provider, reg.ids().join(", ")))?;
            let pcfg = cfg.providers.get(provider.id());
            if pcfg.is_some_and(|p| !p.enabled) {
                return Err(format!("provider \"{}\" is disabled in [providers.{}] but used by models.{role}", provider.id(), provider.id()));
            }
            let ctx = ProviderContext {
                onnx: cfg.runtime.onnx.clone(),
                base_dir: base_dir.to_path_buf(),
                provider_api_base: pcfg.and_then(|p| p.api_base.clone()),
                provider_api_key_env: pcfg.and_then(|p| p.api_key_env.clone()),
            };
            let runtime = provider.create_runtime(&m, &ctx).map_err(|e| format!("models.{role}: {e}"))?;
            let mut opts = default_options(role);
            if let Some(v) = m.cfg.max_new_tokens {
                opts.max_new_tokens = v;
            }
            if let Some(v) = m.cfg.temperature {
                opts.temperature = v;
            }
            if let Some(v) = m.cfg.top_p {
                opts.top_p = v;
            }
            if let Some(v) = m.cfg.top_k {
                opts.top_k = v;
            }
            if let Some(v) = m.cfg.repetition_penalty {
                opts.repetition_penalty = v;
            }
            let mut h = ModelHandle::from_runtime(role, provider.id(), &m.cfg.name, runtime).with_options(opts);
            h.runtime_kind = m.cfg.runtime.clone();
            h.window = m.cfg.context_window;
            Ok(h)
        };
        Ok(ModelSet { thinking: Arc::new(ThinkingModel(build("thinking")?)), coder: Arc::new(CoderModel(build("coder")?)) })
    }

    pub fn by_role(&self, role: &str) -> &dyn ModelRole {
        if role == "thinking" {
            self.thinking.as_ref()
        } else {
            self.coder.as_ref()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::MockRuntime;

    #[test]
    fn default_config_builds_two_lazy_onnx_models() {
        let cfg = Config::from_toml_str(crate::config::DEFAULT_CONFIG_TOML).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let set = ModelSet::from_config(&cfg, dir.path()).unwrap();
        let t = set.thinking.info();
        let c = set.coder.info();
        assert_eq!((t.role.as_str(), t.provider.as_str(), t.name.as_str(), t.runtime_kind.as_str()), ("thinking", "qwen", "qwen3-235b-a22b-thinking-2507", "onnx"));
        assert_eq!((c.provider.as_str(), c.name.as_str()), ("qwen", "qwen3-coder-next"));
        assert!(!t.loaded && t.description.contains("not loaded"));
        // loading fails with a helpful message because no model files exist yet — without panicking
        let err = set.coder.load().unwrap_err();
        assert!(err.contains("does not exist") || err.contains("models"), "{err}");
        let r = set.coder.generate(&[ChatMessage::user("hi")], &Arc::new(AtomicBool::new(false)));
        assert!(matches!(r, Err(RuntimeError::Failed(_))));
    }

    #[test]
    fn qwen_api_runtime_from_config() {
        let toml = r#"
[models.thinking]
provider = "qwen"
name = "qwen3-235b-a22b-thinking-2507"
runtime = "api"
api_key = "k"
[models.coder]
provider = "qwen"
name = "qwen3-coder-next"
runtime = "api"
api_base = "http://127.0.0.1:9/api/v1"
api_key_env = "QWEN_TEST_KEY_THAT_DOES_NOT_EXIST"
"#;
        let cfg = Config::from_toml_str(toml).unwrap();
        let set = ModelSet::from_config(&cfg, Path::new(".")).unwrap();
        assert!(set.thinking.info().description.contains("dashscope-intl.aliyuncs.com"));
        assert!(set.coder.info().description.contains("127.0.0.1:9"));
        // no key available -> a clear error, not a panic
        let e = set.coder.generate(&[ChatMessage::user("x")], &Arc::new(AtomicBool::new(false))).unwrap_err();
        assert!(e.to_string().contains("DASHSCOPE_API_KEY"));
    }

    #[test]
    fn provider_selection_and_errors() {
        let reg = ProviderRegistry::default();
        assert_eq!(reg.ids(), vec!["qwen", "kimi", "glm"]);
        assert!(reg.get("KIMI").is_some() && reg.get("openai").is_none());

        let cfg = Config::from_toml_str("[models.thinking]\nprovider = \"openai\"\nname = \"x\"\n").unwrap();
        assert!(ModelSet::from_config(&cfg, Path::new(".")).err().unwrap().contains("unknown"));

        let cfg = Config::from_toml_str("[providers.qwen]\nenabled = false\n").unwrap();
        assert!(ModelSet::from_config(&cfg, Path::new(".")).err().unwrap().contains("disabled"));

        let cfg = Config::from_toml_str("[models.coder]\nprovider = \"kimi\"\nname = \"kimi-k2\"\nruntime = \"api\"\n").unwrap();
        assert!(ModelSet::from_config(&cfg, Path::new(".")).err().unwrap().contains("Kimi API is not implemented"));

        // Kimi and GLM work through ONNX with their own prompt formats
        let cfg = Config::from_toml_str("[models.coder]\nprovider = \"glm\"\nname = \"glm-4.6\"\nruntime = \"onnx\"\npath = \"./models/glm\"\n").unwrap();
        let set = ModelSet::from_config(&cfg, Path::new(".")).unwrap();
        assert_eq!(set.coder.info().provider, "glm");
        assert_eq!(GlmProvider.template("coder"), ChatTemplate::Glm);
        assert_eq!(QwenProvider.template("thinking"), ChatTemplate::ChatMl { think_prefill: true });
        assert_eq!(QwenProvider.template("coder"), ChatTemplate::ChatMl { think_prefill: false });
    }

    #[test]
    fn handles_pass_generation_options_and_cancel() {
        let rt = Arc::new(MockRuntime::scripted("m", vec!["hello".into()]));
        let h = ModelHandle::from_runtime("coder", "qwen", "x", rt.clone());
        let cancel = Arc::new(AtomicBool::new(false));
        assert_eq!(h.generate(&[ChatMessage::user("q")], &cancel).unwrap().text, "hello");
        cancel.store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(matches!(h.generate(&[ChatMessage::user("q")], &cancel), Err(RuntimeError::Cancelled)));
        assert_eq!(h.max_new_tokens(), 4096);
    }
}
