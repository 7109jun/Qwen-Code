//! TOML configuration (`qwen.toml`). TOML is the only supported configuration format.

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The default configuration file, embedded in the binary.
pub const DEFAULT_CONFIG_TOML: &str = include_str!("../qwen.toml");

/// Name of the configuration file searched in the workspace.
pub const CONFIG_FILE_NAME: &str = "qwen.toml";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Policy {
    Allow,
    Deny,
    Ask,
}

impl Policy {
    pub fn as_str(&self) -> &'static str {
        match self {
            Policy::Allow => "allow",
            Policy::Deny => "deny",
            Policy::Ask => "ask",
        }
    }

    /// Strictness order: deny > ask > allow.
    pub fn strictest(self, other: Policy) -> Policy {
        fn rank(p: Policy) -> u8 {
            match p {
                Policy::Allow => 0,
                Policy::Ask => 1,
                Policy::Deny => 2,
            }
        }
        if rank(other) > rank(self) {
            other
        } else {
            self
        }
    }
}

impl std::str::FromStr for Policy {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "allow" => Ok(Policy::Allow),
            "deny" => Ok(Policy::Deny),
            "ask" => Ok(Policy::Ask),
            other => bail!("invalid policy '{other}' (expected allow, deny or ask)"),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Config {
    pub qwen: QwenSection,
    pub models: ModelsSection,
    pub providers: BTreeMap<String, ProviderConfig>,
    pub runtime: RuntimeSection,
    pub context: ContextSection,
    pub agent: AgentSection,
    pub permissions: PermissionsSection,
    pub tools: ToolsSection,
    pub cli: CliSection,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct QwenSection {
    pub name: String,
    pub max_iterations: usize,
}

impl Default for QwenSection {
    fn default() -> Self {
        Self { name: "Qwen Code".into(), max_iterations: 30 }
    }
}

/// A model may be given either as a plain name (`thinking = "qwen3-..."`) or as a full table
/// (`[models.thinking]`). TOML does not allow both forms for the same key in one file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
#[allow(clippy::large_enum_variant)]
pub enum ModelSpec {
    Name(String),
    Full(ModelConfig),
}

impl ModelSpec {
    pub fn resolve(&self, role: &str, default_runtime: &str) -> ModelConfig {
        match self {
            ModelSpec::Full(c) => c.clone(),
            ModelSpec::Name(n) => ModelConfig {
                name: n.clone(),
                runtime: default_runtime.to_string(),
                path: format!("./models/{role}"),
                ..ModelConfig::default()
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ModelsSection {
    pub thinking: ModelSpec,
    pub coder: ModelSpec,
}

impl Default for ModelsSection {
    fn default() -> Self {
        Self {
            thinking: ModelSpec::Full(ModelConfig {
                provider: "qwen".into(),
                name: "qwen3-235b-a22b-thinking-2507".into(),
                runtime: "onnx".into(),
                path: "./models/thinking".into(),
                ..ModelConfig::default()
            }),
            coder: ModelSpec::Full(ModelConfig {
                provider: "qwen".into(),
                name: "qwen3-coder-next".into(),
                runtime: "onnx".into(),
                path: "./models/coder".into(),
                ..ModelConfig::default()
            }),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ModelConfig {
    /// Provider id: `qwen`, `kimi` or `glm`.
    pub provider: String,
    /// Model name (sent to API runtimes; informational for ONNX).
    pub name: String,
    /// `onnx` (local ONNX Runtime) or `api` (Qwen API / DashScope).
    pub runtime: String,
    /// ONNX model directory (or `.onnx` file), relative to the config file.
    pub path: String,
    /// Qwen API base URL (default: the international DashScope endpoint).
    pub api_base: Option<String>,
    /// Name of the environment variable that holds the API key.
    pub api_key_env: Option<String>,
    /// Literal API key (discouraged; prefer `api_key_env`).
    pub api_key: Option<String>,
    /// ONNX device override: `cpu`, `cuda` or `directml`.
    pub device: Option<String>,
    pub max_new_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f32>,
    /// Total context window in tokens.
    pub context_window: Option<usize>,
    /// Ask the Qwen API for a JSON object response (`response_format`); leave off for thinking models.
    pub json_mode: bool,
    /// HTTP timeout for API runtimes.
    pub timeout_secs: u64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            provider: "qwen".into(),
            name: String::new(),
            runtime: "onnx".into(),
            path: String::new(),
            api_base: None,
            api_key_env: None,
            api_key: None,
            device: None,
            max_new_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
            context_window: None,
            json_mode: false,
            timeout_secs: 300,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ProviderConfig {
    pub enabled: bool,
    pub api_base: Option<String>,
    pub api_key_env: Option<String>,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self { enabled: true, api_base: None, api_key_env: None }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct RuntimeSection {
    pub onnx: OnnxSection,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct OnnxSection {
    /// Path to the ONNX Runtime shared library (`libonnxruntime.so` / `onnxruntime.dll`).
    pub dylib_path: Option<String>,
    /// Default device: `cpu`, `cuda` or `directml`.
    pub device: String,
    pub device_id: i32,
    /// Intra-op threads (0 = ONNX Runtime default).
    pub intra_threads: usize,
}

impl Default for OnnxSection {
    fn default() -> Self {
        Self { dylib_path: None, device: "cpu".into(), device_id: 0, intra_threads: 0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ContextSection {
    /// Maximum prompt budget (tokens) unless the model context window is smaller.
    pub max_tokens: usize,
    /// Fraction of the budget at which history is compacted.
    pub compact_threshold: f32,
    /// Number of most recent history entries that are never compacted.
    pub keep_recent: usize,
    pub max_tool_output_bytes: usize,
    pub max_file_bytes: usize,
    pub snapshot_max_entries: usize,
    pub snapshot_depth: usize,
    pub retrieval_max_files: usize,
    pub retrieval_snippet_lines: usize,
}

impl Default for ContextSection {
    fn default() -> Self {
        Self {
            max_tokens: 24_000,
            compact_threshold: 0.85,
            keep_recent: 6,
            max_tool_output_bytes: 12_000,
            max_file_bytes: 400_000,
            snapshot_max_entries: 150,
            snapshot_depth: 3,
            retrieval_max_files: 5,
            retrieval_snippet_lines: 60,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct AgentSection {
    pub max_coder_steps: usize,
    pub max_thinking_steps: usize,
    pub max_repairs: usize,
    pub model_retries: usize,
    pub retry_backoff_ms: u64,
}

impl Default for AgentSection {
    fn default() -> Self {
        Self {
            max_coder_steps: 16,
            max_thinking_steps: 10,
            max_repairs: 3,
            model_retries: 2,
            retry_backoff_ms: 500,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PermissionsSection {
    pub read: Policy,
    pub write: Policy,
    pub edit: Policy,
    pub delete: Policy,
    pub execute: Policy,
    pub network: Policy,
    pub git: Policy,
    pub process: Policy,
    /// Access to paths outside the workspace.
    pub filesystem: Policy,
    /// Fallback for anything without an explicit rule.
    pub default: Policy,
    pub commands: CommandPolicies,
}

impl Default for PermissionsSection {
    fn default() -> Self {
        Self {
            read: Policy::Allow,
            write: Policy::Allow,
            edit: Policy::Allow,
            delete: Policy::Ask,
            execute: Policy::Allow,
            network: Policy::Allow,
            git: Policy::Allow,
            process: Policy::Allow,
            filesystem: Policy::Ask,
            default: Policy::Ask,
            commands: CommandPolicies::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct CommandPolicies {
    /// Commands that can seriously destroy data (recursive deletes, hard resets, ...).
    pub destructive: Policy,
    /// Commands that can damage the whole system (format disk, shutdown, ...).
    pub system_destructive: Policy,
    /// Programs that are not recognised as ordinary development tools.
    pub unknown: Policy,
    /// Extra glob patterns matched against each command segment.
    pub allow: Vec<String>,
    pub ask: Vec<String>,
    pub deny: Vec<String>,
}

impl Default for CommandPolicies {
    fn default() -> Self {
        Self {
            destructive: Policy::Ask,
            system_destructive: Policy::Deny,
            unknown: Policy::Ask,
            allow: Vec::new(),
            ask: Vec::new(),
            deny: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ToolsSection {
    pub command_timeout_secs: u64,
    pub test_timeout_secs: u64,
    pub max_output_bytes: usize,
    pub max_search_results: usize,
    pub max_read_lines: usize,
    /// `auto`, `sh`, `bash`, `powershell` or `cmd`.
    pub shell: String,
}

impl Default for ToolsSection {
    fn default() -> Self {
        Self {
            command_timeout_secs: 120,
            test_timeout_secs: 600,
            max_output_bytes: 20_000,
            max_search_results: 100,
            max_read_lines: 2000,
            shell: "auto".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct CliSection {
    /// `auto`, `tui` or `repl`.
    pub ui: String,
    pub color: bool,
    pub show_thinking: bool,
    pub prompt: String,
    pub history_size: usize,
}

impl Default for CliSection {
    fn default() -> Self {
        Self { ui: "auto".into(), color: true, show_thinking: false, prompt: "> ".into(), history_size: 500 }
    }
}

/// A model configuration with all defaults applied for its role.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    pub role: String,
    pub cfg: ModelConfig,
    /// Absolute (config-relative) path of the ONNX model.
    pub path: PathBuf,
}

impl Config {
    pub fn from_toml_str(s: &str) -> Result<Config> {
        let cfg: Config = toml::from_str(s).map_err(|e| anyhow!("invalid TOML configuration: {e}"))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn to_toml(&self) -> Result<String> {
        toml::to_string_pretty(self).map_err(|e| anyhow!("cannot serialize configuration: {e}"))
    }

    /// Search order: explicit path, `./qwen.toml` in the workspace, the user config dir, built-in defaults.
    pub fn load(explicit: Option<&Path>, workspace: &Path) -> Result<(Config, Option<PathBuf>)> {
        if let Some(p) = explicit {
            let text = std::fs::read_to_string(p).with_context(|| format!("cannot read config file {}", p.display()))?;
            let cfg = Config::from_toml_str(&text).with_context(|| format!("in {}", p.display()))?;
            return Ok((cfg, Some(p.to_path_buf())));
        }
        let mut candidates = vec![workspace.join(CONFIG_FILE_NAME)];
        if let Some(dir) = crate::platform::config_dir() {
            candidates.push(dir.join(CONFIG_FILE_NAME));
        }
        for c in candidates {
            if c.is_file() {
                let text = std::fs::read_to_string(&c).with_context(|| format!("cannot read config file {}", c.display()))?;
                let cfg = Config::from_toml_str(&text).with_context(|| format!("in {}", c.display()))?;
                return Ok((cfg, Some(c)));
            }
        }
        Ok((Config::from_toml_str(DEFAULT_CONFIG_TOML)?, None))
    }

    pub fn validate(&self) -> Result<()> {
        if self.qwen.max_iterations == 0 {
            bail!("qwen.max_iterations must be at least 1");
        }
        if self.context.max_tokens < 1024 {
            bail!("context.max_tokens must be at least 1024");
        }
        if !(0.3..=0.99).contains(&self.context.compact_threshold) {
            bail!("context.compact_threshold must be between 0.3 and 0.99");
        }
        for (label, spec) in [("thinking", &self.models.thinking), ("coder", &self.models.coder)] {
            let m = spec.resolve(label, "onnx");
            match m.runtime.as_str() {
                "onnx" | "api" => {}
                other => bail!("models.{label}.runtime = \"{other}\" is not supported (use \"onnx\" or \"api\")"),
            }
            if let Some(d) = &m.device {
                check_device(d).with_context(|| format!("models.{label}.device"))?;
            }
            if m.temperature.is_some_and(|t| !(0.0..=2.0).contains(&t)) {
                bail!("models.{label}.temperature must be between 0 and 2");
            }
            if m.top_p.is_some_and(|t| !(0.0..=1.0).contains(&t)) {
                bail!("models.{label}.top_p must be between 0 and 1");
            }
        }
        check_device(&self.runtime.onnx.device).context("runtime.onnx.device")?;
        match self.tools.shell.as_str() {
            "auto" | "sh" | "bash" | "powershell" | "cmd" => {}
            other => bail!("tools.shell = \"{other}\" is not supported (auto, sh, bash, powershell, cmd)"),
        }
        match self.cli.ui.as_str() {
            "auto" | "tui" | "repl" => {}
            other => bail!("cli.ui = \"{other}\" is not supported (auto, tui, repl)"),
        }
        Ok(())
    }

    /// Resolve a role's model configuration. Relative model paths are resolved against `base_dir`.
    pub fn resolve_model(&self, role: &str, base_dir: &Path) -> ResolvedModel {
        let spec = if role == "thinking" { &self.models.thinking } else { &self.models.coder };
        let cfg = spec.resolve(role, "onnx");
        let raw = if cfg.path.is_empty() { format!("./models/{role}") } else { cfg.path.clone() };
        let p = PathBuf::from(&raw);
        let path = if p.is_absolute() { p } else { base_dir.join(p) };
        ResolvedModel { role: role.to_string(), cfg, path }
    }
}

fn check_device(d: &str) -> Result<()> {
    match d {
        "cpu" | "cuda" | "directml" => Ok(()),
        other => bail!("unsupported device \"{other}\" (cpu, cuda, directml)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_default_config_parses_and_matches_the_spec() {
        let c = Config::from_toml_str(DEFAULT_CONFIG_TOML).unwrap();
        assert_eq!(c.qwen.name, "Qwen Code");
        assert_eq!(c.qwen.max_iterations, 30);
        let t = c.resolve_model("thinking", Path::new("/base"));
        assert_eq!((t.cfg.name.as_str(), t.cfg.runtime.as_str()), ("qwen3-235b-a22b-thinking-2507", "onnx"));
        assert!(t.path.ends_with("models/thinking"));
        let k = c.resolve_model("coder", Path::new("/base"));
        assert_eq!(k.cfg.name, "qwen3-coder-next");
        assert_eq!(c.permissions.read, Policy::Allow);
        assert_eq!(c.permissions.delete, Policy::Ask);
        assert_eq!(c.permissions.commands.system_destructive, Policy::Deny);
        assert_eq!(c.permissions.commands.destructive, Policy::Ask);
    }

    #[test]
    fn plain_string_models_are_accepted() {
        let c = Config::from_toml_str("[models]\nthinking = \"my-think\"\ncoder = \"my-coder\"\n[providers.qwen]\nenabled = true\n").unwrap();
        let t = c.resolve_model("thinking", Path::new("/b"));
        assert_eq!((t.cfg.name.as_str(), t.cfg.runtime.as_str()), ("my-think", "onnx"));
        assert!(t.path.ends_with("models/thinking"));
    }

    #[test]
    fn validation_reports_problems() {
        for (toml, needle) in [
            ("[qwen]\nmax_iterations = 0\n", "max_iterations"),
            ("[models.coder]\nruntime = \"tensorrt\"\n", "not supported"),
            ("[runtime.onnx]\ndevice = \"tpu\"\n", "unsupported device"),
            ("[permissions]\ndelete = \"maybe\"\n", "invalid TOML"),
            ("[cli]\nui = \"gui\"\n", "cli.ui"),
        ] {
            let e = format!("{:#}", Config::from_toml_str(toml).unwrap_err());
            assert!(e.contains(needle), "{toml} -> {e}");
        }
    }

    #[test]
    fn config_round_trips_through_toml() {
        let c = Config::default();
        let again = Config::from_toml_str(&c.to_toml().unwrap()).unwrap();
        assert_eq!(c, again);
    }

    #[test]
    fn load_prefers_explicit_then_workspace_then_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let (c, p) = Config::load(None, dir.path()).unwrap();
        assert!(p.is_none() && c.qwen.max_iterations == 30);
        std::fs::write(dir.path().join("qwen.toml"), "[qwen]\nmax_iterations = 7\n").unwrap();
        let (c, p) = Config::load(None, dir.path()).unwrap();
        assert_eq!(c.qwen.max_iterations, 7);
        assert!(p.unwrap().ends_with("qwen.toml"));
        let other = dir.path().join("other.toml");
        std::fs::write(&other, "[qwen]\nmax_iterations = 9\n").unwrap();
        assert_eq!(Config::load(Some(&other), dir.path()).unwrap().0.qwen.max_iterations, 9);
        assert!(Config::load(Some(&dir.path().join("missing.toml")), dir.path()).is_err());
    }

    #[test]
    fn policy_strictness() {
        assert_eq!(Policy::Allow.strictest(Policy::Ask), Policy::Ask);
        assert_eq!(Policy::Deny.strictest(Policy::Ask), Policy::Deny);
        assert_eq!(Policy::Ask.strictest(Policy::Allow), Policy::Ask);
        assert_eq!("DENY".parse::<Policy>().unwrap(), Policy::Deny);
        assert!("x".parse::<Policy>().is_err());
    }
}
