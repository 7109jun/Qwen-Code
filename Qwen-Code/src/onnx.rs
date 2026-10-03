//! ONNX Runtime model execution.
//!
//! Loads a causal language model exported to ONNX (for example with `optimum-cli export onnx` or
//! `onnxruntime-genai`'s model builder) together with its `tokenizer.json`, and generates text with
//! a plain autoregressive loop. Both layouts are supported:
//!
//! * decoder **with** KV cache (`past_key_values.N.key/value` inputs, `present.N.key/value` outputs),
//! * decoder **without** cache (the whole sequence is fed on every step).
//!
//! The ONNX Runtime shared library is loaded dynamically (`load-dynamic`), so the binary starts
//! without it and reports a clear error when it is missing.

use crate::platform;
use crate::protocol;
use crate::runtime::{ChatMessage, ChatTemplate, GenOptions, GenOutput, Runtime, RuntimeError};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{Session, SessionInputValue};
use ort::value::{Tensor, TensorElementType, ValueType};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tokenizers::Tokenizer;

// ---------------------------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------------------------

static ORT_LIB: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Loads the ONNX Runtime shared library once. Failures are not cached, so the user can fix the
/// path in `qwen.toml` and retry without restarting.
pub fn init_ort(dylib: Option<&str>, base: &Path) -> Result<PathBuf, String> {
    let mut g = ORT_LIB.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(p) = g.as_ref() {
        return Ok(p.clone());
    }
    let lib = platform::find_onnxruntime_lib(dylib, base).ok_or_else(|| {
        let names = platform::onnxruntime_lib_file_names().join(" / ");
        let hint = match dylib.filter(|s| !s.is_empty()) {
            Some(p) => format!("[runtime.onnx].dylib_path = \"{p}\" does not exist"),
            None => format!("searched $ORT_DYLIB_PATH, the executable directory, ./lib and ./models for {names}"),
        };
        format!(
            "ONNX Runtime shared library not found ({hint}). Install ONNX Runtime (https://github.com/microsoft/onnxruntime/releases, or `pip install onnxruntime` and point dylib_path at its capi/{names}) and set [runtime.onnx].dylib_path or the ORT_DYLIB_PATH environment variable."
        )
    })?;
    let builder = ort::init_from(&lib).map_err(|e| format!("cannot load ONNX Runtime from {}: {e}", lib.display()))?;
    builder.with_name("qwen-code").commit();
    *g = Some(lib.clone());
    Ok(lib)
}

// ---------------------------------------------------------------------------------------------
// Model files
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ModelFiles {
    pub onnx: PathBuf,
    pub tokenizer: PathBuf,
    pub generation_config: Option<PathBuf>,
    pub config: Option<PathBuf>,
}

const ONNX_NAMES: &[&str] = &["model.onnx", "decoder_model_merged.onnx", "model_quantized.onnx", "model_q4.onnx", "model_q4f16.onnx", "model_fp16.onnx", "decoder_model.onnx"];

fn onnx_files_in(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|rd| rd.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "onnx")).collect())
        .unwrap_or_default();
    v.sort();
    v
}

/// Finds `model.onnx` + `tokenizer.json` (+ optional configs) for a configured model path.
pub fn locate_model_files(path: &Path) -> Result<ModelFiles, String> {
    let layout_help = "Expected a directory containing model.onnx (plus its external weight files) and tokenizer.json. See tools/convert_to_onnx.py to convert a Hugging Face model.";
    if !path.exists() {
        return Err(format!("model path {} does not exist. {layout_help} Alternatively set runtime = \"api\" for this model in qwen.toml.", path.display()));
    }
    let onnx: PathBuf = if path.is_file() {
        path.to_path_buf()
    } else {
        let mut chosen: Option<PathBuf> = None;
        for dir in [path.to_path_buf(), path.join("onnx")] {
            for n in ONNX_NAMES {
                let c = dir.join(n);
                if c.is_file() {
                    chosen = Some(c);
                    break;
                }
            }
            if chosen.is_some() {
                break;
            }
        }
        if chosen.is_none() {
            let all: Vec<PathBuf> = onnx_files_in(path).into_iter().chain(onnx_files_in(&path.join("onnx"))).collect();
            match all.len() {
                0 => return Err(format!("no .onnx file found in {}. {layout_help}", path.display())),
                1 => chosen = all.into_iter().next(),
                _ => {
                    let names: Vec<String> = all.iter().map(|p| p.file_name().unwrap_or_default().to_string_lossy().to_string()).collect();
                    return Err(format!("several .onnx files found in {} ({}); point the path at the one to use", path.display(), names.join(", ")));
                }
            }
        }
        chosen.expect("chosen is set")
    };
    let model_dir = onnx.parent().unwrap_or(Path::new(".")).to_path_buf();
    let parent_dir = model_dir.parent().map(|p| p.to_path_buf());
    let find = |name: &str| -> Option<PathBuf> {
        let a = model_dir.join(name);
        if a.is_file() {
            return Some(a);
        }
        if model_dir.file_name().is_some_and(|n| n == "onnx") {
            if let Some(p) = &parent_dir {
                let b = p.join(name);
                if b.is_file() {
                    return Some(b);
                }
            }
        }
        if path.is_dir() {
            let c = path.join(name);
            if c.is_file() {
                return Some(c);
            }
        }
        None
    };
    let tokenizer = find("tokenizer.json").ok_or_else(|| format!("tokenizer.json not found next to {}. {layout_help}", onnx.display()))?;
    Ok(ModelFiles { onnx, tokenizer, generation_config: find("generation_config.json"), config: find("config.json") })
}

fn read_json(p: &Option<PathBuf>) -> Value {
    p.as_ref().and_then(|p| std::fs::read_to_string(p).ok()).and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(Value::Null)
}

fn ids_from(v: &Value) -> Vec<i64> {
    match v {
        Value::Number(n) => n.as_i64().into_iter().collect(),
        Value::Array(a) => a.iter().filter_map(|x| x.as_i64()).collect(),
        _ => vec![],
    }
}

// ---------------------------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------------------------

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: Option<u64>) -> Self {
        let s = seed.unwrap_or_else(|| {
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0x9E3779B97F4A7C15) ^ (std::process::id() as u64) << 32
        });
        Rng(s | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    /// Uniform in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

/// Picks the next token from raw logits.
pub fn sample_token(logits: &mut [f32], history: &[i64], opts: &GenOptions, rng: &mut Rng) -> usize {
    for l in logits.iter_mut() {
        if l.is_nan() {
            *l = f32::NEG_INFINITY;
        }
    }
    if opts.repetition_penalty > 1.0 {
        let mut seen: HashSet<usize> = HashSet::new();
        for &t in history.iter().rev().take(512) {
            if t >= 0 && (t as usize) < logits.len() && seen.insert(t as usize) {
                let l = &mut logits[t as usize];
                *l = if *l > 0.0 { *l / opts.repetition_penalty } else { *l * opts.repetition_penalty };
            }
        }
    }
    let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|(i, _)| i).unwrap_or(0);
    if opts.temperature <= 0.01 {
        return argmax(logits);
    }
    let inv_t = 1.0 / opts.temperature;
    let mut cand: Vec<(usize, f32)> = logits.iter().enumerate().filter(|(_, l)| l.is_finite()).map(|(i, &l)| (i, l * inv_t)).collect();
    if cand.is_empty() {
        return argmax(logits);
    }
    let k = if opts.top_k == 0 { cand.len() } else { opts.top_k.min(cand.len()) };
    if k < cand.len() {
        cand.select_nth_unstable_by(k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        cand.truncate(k);
    }
    cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let max = cand[0].1;
    let mut probs: Vec<f32> = cand.iter().map(|(_, l)| (l - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }
    if opts.top_p < 1.0 {
        let mut cum = 0.0;
        let mut cut = probs.len();
        for (i, p) in probs.iter().enumerate() {
            cum += p;
            if cum >= opts.top_p {
                cut = i + 1;
                break;
            }
        }
        probs.truncate(cut);
        cand.truncate(cut);
        let s: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= s;
        }
    }
    let r = rng.next_f32();
    let mut acc = 0.0;
    for (i, p) in probs.iter().enumerate() {
        acc += p;
        if r < acc {
            return cand[i].0;
        }
    }
    cand[cand.len() - 1].0
}

/// Cuts `text` at the first stop string.
pub fn cut_at_stop(text: &str, stops: &[&str]) -> (String, bool) {
    let mut cut: Option<usize> = None;
    for s in stops {
        if let Some(p) = text.find(s) {
            cut = Some(cut.map_or(p, |c| c.min(p)));
        }
    }
    match cut {
        Some(p) => (text[..p].to_string(), true),
        None => (text.to_string(), false),
    }
}

// ---------------------------------------------------------------------------------------------
// Runtime
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct PastInput {
    name: String,
    ty: TensorElementType,
    dims: Vec<i64>,
}

#[derive(Debug, Clone)]
struct KvSpec {
    past: Vec<PastInput>,
    /// present output name -> past input name
    carry: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
struct IoSpec {
    input_ids: (String, TensorElementType),
    attention_mask: Option<(String, TensorElementType)>,
    position_ids: Option<(String, TensorElementType)>,
    token_type_ids: Option<(String, TensorElementType)>,
    use_cache_branch: Option<String>,
    kv: Option<KvSpec>,
    logits_output: String,
}

pub struct OnnxParams {
    pub path: PathBuf,
    pub device: String,
    pub device_id: i32,
    pub intra_threads: usize,
    pub dylib: Option<String>,
    pub base_dir: PathBuf,
    pub template: ChatTemplate,
    pub context_window: Option<usize>,
    pub label: String,
}

pub struct OnnxRuntime {
    label: String,
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    template: ChatTemplate,
    io: IoSpec,
    eos_ids: HashSet<i64>,
    window: usize,
    files: ModelFiles,
    device: String,
    notes: Vec<String>,
}

fn ort_err(ctx: &str, e: impl std::fmt::Display) -> RuntimeError {
    RuntimeError::Failed(format!("{ctx}: {e}"))
}

fn inspect_io(session: &Session) -> Result<IoSpec, String> {
    let mut input_ids = None;
    let mut attention_mask = None;
    let mut position_ids = None;
    let mut token_type_ids = None;
    let mut use_cache_branch = None;
    let mut past: Vec<PastInput> = Vec::new();
    let mut other: Vec<String> = Vec::new();
    for inp in session.inputs() {
        let name = inp.name().to_string();
        let (ty, dims) = match inp.dtype() {
            ValueType::Tensor { ty, shape, .. } => (*ty, shape.to_vec()),
            _ => return Err(format!("model input \"{name}\" is not a tensor")),
        };
        match name.as_str() {
            "input_ids" => input_ids = Some((name, ty)),
            "attention_mask" => attention_mask = Some((name, ty)),
            "position_ids" => position_ids = Some((name, ty)),
            "token_type_ids" => token_type_ids = Some((name, ty)),
            "use_cache_branch" => use_cache_branch = Some(name),
            n if n.starts_with("past_key_values") || n.starts_with("past.") || n.starts_with("past_") => past.push(PastInput { name, ty, dims }),
            _ => other.push(name),
        }
    }
    let input_ids = input_ids.ok_or_else(|| "the model has no \"input_ids\" input; only causal language models are supported".to_string())?;
    if !other.is_empty() {
        return Err(format!("the model has unsupported inputs: {}", other.join(", ")));
    }
    let outputs: Vec<String> = session.outputs().iter().map(|o| o.name().to_string()).collect();
    let logits_output = outputs.iter().find(|n| *n == "logits").or_else(|| outputs.iter().find(|n| n.contains("logits"))).or_else(|| outputs.first()).cloned().ok_or("the model has no outputs")?;

    let kv = if past.is_empty() {
        None
    } else {
        let suffix = |n: &str| n.split_once('.').map(|(_, r)| r.to_string()).unwrap_or_else(|| n.to_string());
        let mut carry = Vec::new();
        for p in &past {
            let want = suffix(&p.name);
            let out = outputs.iter().find(|o| *o != &logits_output && (o.starts_with("present") || o.starts_with("new_")) && suffix(o) == want);
            match out {
                Some(o) => carry.push((o.clone(), p.name.clone())),
                None => return Err(format!("cannot find the \"present\" output matching KV-cache input \"{}\"", p.name)),
            }
        }
        Some(KvSpec { past, carry })
    };
    Ok(IoSpec { input_ids, attention_mask, position_ids, token_type_ids, use_cache_branch, kv, logits_output })
}

fn int_input(ty: TensorElementType, dims: [usize; 2], data: &[i64]) -> Result<SessionInputValue<'static>, RuntimeError> {
    let shape = vec![dims[0] as i64, dims[1] as i64];
    match ty {
        TensorElementType::Int32 => {
            let v: Vec<i32> = data.iter().map(|&x| x as i32).collect();
            Tensor::<i32>::from_array((shape, v)).map(Into::into).map_err(|e| ort_err("cannot create an input tensor", e))
        }
        _ => Tensor::<i64>::from_array((shape, data.to_vec())).map(Into::into).map_err(|e| ort_err("cannot create an input tensor", e)),
    }
}

fn zero_past(p: &PastInput) -> Result<SessionInputValue<'static>, RuntimeError> {
    let mut dims: Vec<i64> = p.dims.clone();
    for (i, d) in dims.iter_mut().enumerate() {
        if *d < 0 {
            *d = if i == 0 { 1 } else { 0 };
        }
    }
    let n: usize = dims.iter().map(|&d| d.max(0) as usize).product();
    let r = match p.ty {
        TensorElementType::Float16 => Tensor::<half::f16>::from_array((dims, vec![half::f16::ZERO; n])).map(Into::into),
        TensorElementType::Bfloat16 => Tensor::<half::bf16>::from_array((dims, vec![half::bf16::ZERO; n])).map(Into::into),
        TensorElementType::Float32 => Tensor::<f32>::from_array((dims, vec![0f32; n])).map(Into::into),
        other => return Err(RuntimeError::Failed(format!("unsupported KV-cache element type {other:?}"))),
    };
    r.map_err(|e| ort_err("cannot create an empty KV cache", e))
}

fn extract_last_logits(value: &ort::value::DynValue) -> Result<Vec<f32>, RuntimeError> {
    fn last_row<T: Copy>(shape: &[i64], data: &[T]) -> Result<Vec<T>, RuntimeError> {
        let vocab = *shape.last().ok_or_else(|| RuntimeError::Failed("logits tensor has no dimensions".into()))? as usize;
        if vocab == 0 || data.len() < vocab {
            return Err(RuntimeError::Failed(format!("unexpected logits shape {shape:?}")));
        }
        Ok(data[data.len() - vocab..].to_vec())
    }
    if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
        return last_row(shape, data);
    }
    if let Ok((shape, data)) = value.try_extract_tensor::<half::f16>() {
        return last_row(shape, data).map(|v| v.into_iter().map(|x| x.to_f32()).collect());
    }
    if let Ok((shape, data)) = value.try_extract_tensor::<half::bf16>() {
        return last_row(shape, data).map(|v| v.into_iter().map(|x| x.to_f32()).collect());
    }
    Err(RuntimeError::Failed("the logits output has an unsupported element type".into()))
}

impl OnnxRuntime {
    pub fn load(p: &OnnxParams) -> Result<OnnxRuntime, String> {
        let files = locate_model_files(&p.path)?;
        init_ort(p.dylib.as_deref(), &p.base_dir)?;
        let tokenizer = Tokenizer::from_file(&files.tokenizer).map_err(|e| format!("cannot load {}: {e}", files.tokenizer.display()))?;

        let mut notes: Vec<String> = Vec::new();
        let build = |device: &str| -> Result<Session, String> {
            let mut b = Session::builder().map_err(|e| e.to_string())?;
            b = b.with_optimization_level(GraphOptimizationLevel::Level3).map_err(|e| e.to_string())?;
            if p.intra_threads > 0 {
                b = b.with_intra_threads(p.intra_threads).map_err(|e| e.to_string())?;
            }
            match device {
                "cuda" => {
                    b = b.with_execution_providers([ort::ep::CUDA::default().with_device_id(p.device_id).build().error_on_failure()]).map_err(|e| e.to_string())?;
                }
                "directml" => {
                    b = b.with_execution_providers([ort::ep::DirectML::default().with_device_id(p.device_id).build().error_on_failure()]).map_err(|e| e.to_string())?;
                }
                _ => {}
            }
            b.commit_from_file(&files.onnx).map_err(|e| e.to_string())
        };
        let mut device_used = p.device.clone();
        let session = match build(&p.device) {
            Ok(s) => s,
            Err(e) if p.device != "cpu" => {
                notes.push(format!("{} execution provider unavailable ({e}); fell back to CPU", p.device));
                device_used = "cpu".into();
                build("cpu").map_err(|e2| format!("cannot load {}: {e2}", files.onnx.display()))?
            }
            Err(e) => return Err(format!("cannot load {}: {e}", files.onnx.display())),
        };
        let io = inspect_io(&session)?;

        let gen = read_json(&files.generation_config);
        let cfg = read_json(&files.config);
        let mut eos: HashSet<i64> = HashSet::new();
        eos.extend(ids_from(&gen["eos_token_id"]));
        if eos.is_empty() {
            eos.extend(ids_from(&cfg["eos_token_id"]));
        }
        for tok in ["<|im_end|>", "<|endoftext|>", "<|end▁of▁sentence|>", "</s>", "<|eot_id|>"] {
            if let Some(id) = tokenizer.token_to_id(tok) {
                if matches!(p.template, ChatTemplate::ChatMl { .. }) || tok == "<|endoftext|>" || tok == "</s>" {
                    eos.insert(id as i64);
                }
            }
        }
        let window = p.context_window.or_else(|| cfg["max_position_embeddings"].as_u64().map(|v| v as usize)).unwrap_or(32768);
        Ok(OnnxRuntime { label: p.label.clone(), session: Mutex::new(session), tokenizer, template: p.template, io, eos_ids: eos, window, files, device: device_used, notes })
    }

    pub fn has_kv_cache(&self) -> bool {
        self.io.kv.is_some()
    }

    pub fn warnings(&self) -> &[String] {
        &self.notes
    }

    pub fn count_tokens(&self, text: &str) -> usize {
        self.tokenizer.encode(text, false).map(|e| e.len()).unwrap_or(text.len() / 3)
    }

    fn decode(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids, false).unwrap_or_default()
    }

    /// Runs one forward pass. Returns the last-position logits and (for KV models) the new cache.
    fn step(
        &self,
        session: &mut Session,
        tokens: &[i64],
        past_len: usize,
        cache: &mut Vec<(String, ort::value::DynValue)>,
    ) -> Result<Vec<f32>, RuntimeError> {
        let n = tokens.len();
        let total = past_len + n;
        let mut inputs: Vec<(String, SessionInputValue<'static>)> = Vec::new();
        inputs.push((self.io.input_ids.0.clone(), int_input(self.io.input_ids.1, [1, n], tokens)?));
        if let Some((name, ty)) = &self.io.attention_mask {
            inputs.push((name.clone(), int_input(*ty, [1, total], &vec![1i64; total])?));
        }
        if let Some((name, ty)) = &self.io.position_ids {
            let pos: Vec<i64> = (past_len..total).map(|x| x as i64).collect();
            inputs.push((name.clone(), int_input(*ty, [1, n], &pos)?));
        }
        if let Some((name, ty)) = &self.io.token_type_ids {
            inputs.push((name.clone(), int_input(*ty, [1, n], &vec![0i64; n])?));
        }
        if let Some(name) = &self.io.use_cache_branch {
            let flag = Tensor::<bool>::from_array((vec![1i64], vec![past_len > 0])).map_err(|e| ort_err("cannot create use_cache_branch", e))?;
            inputs.push((name.clone(), flag.into()));
        }
        if let Some(kv) = &self.io.kv {
            if cache.is_empty() {
                for p in &kv.past {
                    inputs.push((p.name.clone(), zero_past(p)?));
                }
            } else {
                for (name, v) in cache.drain(..) {
                    inputs.push((name, v.into()));
                }
            }
        }
        let mut outputs = session.run(inputs).map_err(|e| {
            let m = e.to_string();
            if m.to_lowercase().contains("out of memory") || m.to_lowercase().contains("bad_alloc") {
                RuntimeError::Failed(format!("ONNX Runtime ran out of memory: {m}"))
            } else {
                ort_err("ONNX Runtime inference failed", m)
            }
        })?;
        let logits = {
            let v = outputs.get(self.io.logits_output.as_str()).ok_or_else(|| RuntimeError::Failed(format!("output \"{}\" missing", self.io.logits_output)))?;
            extract_last_logits(v)?
        };
        if let Some(kv) = &self.io.kv {
            for (present, past_name) in &kv.carry {
                let v = outputs.remove(present.as_str()).ok_or_else(|| RuntimeError::Failed(format!("output \"{present}\" missing")))?;
                cache.push((past_name.clone(), v));
            }
        }
        Ok(logits)
    }
}

impl Runtime for OnnxRuntime {
    fn generate(&self, messages: &[ChatMessage], opts: &GenOptions) -> Result<GenOutput, RuntimeError> {
        if opts.cancelled() {
            return Err(RuntimeError::Cancelled);
        }
        let prompt = self.template.render(messages);
        let enc = self.tokenizer.encode(prompt.as_str(), false).map_err(|e| ort_err("tokenization failed", e))?;
        let prompt_ids: Vec<i64> = enc.get_ids().iter().map(|&i| i as i64).collect();
        if prompt_ids.is_empty() {
            return Err(RuntimeError::Failed("the prompt is empty after tokenization".into()));
        }
        let reserve = 64.min(self.window / 8);
        if prompt_ids.len() + reserve >= self.window {
            return Err(RuntimeError::ContextOverflow(format!("prompt has {} tokens but the context window is {}", prompt_ids.len(), self.window)));
        }
        let max_new = opts.max_new_tokens.min(self.window - prompt_ids.len()).max(1);

        let think_prefilled = self.template.prefills_think();
        let stops = self.template.stop_strings();
        let mut rng = Rng::new(opts.seed);
        let mut session = self.session.lock().unwrap_or_else(|e| e.into_inner());
        let mut cache: Vec<(String, ort::value::DynValue)> = Vec::new();
        let mut generated: Vec<u32> = Vec::new();
        let mut history: Vec<i64> = prompt_ids.iter().rev().take(256).rev().cloned().collect();
        let mut all_tokens: Vec<i64> = prompt_ids.clone();
        let mut past_len = 0usize;
        let mut finish = "length".to_string();
        let mut text = String::new();

        for step_no in 0..max_new {
            if opts.cancelled() {
                return Err(RuntimeError::Cancelled);
            }
            let (feed, pl): (&[i64], usize) = if self.io.kv.is_some() {
                if step_no == 0 {
                    (&all_tokens[..], 0)
                } else {
                    (&all_tokens[all_tokens.len() - 1..], past_len)
                }
            } else {
                (&all_tokens[..], 0)
            };
            let mut logits = self.step(&mut session, feed, pl, &mut cache)?;
            past_len = pl + feed.len();
            let next = sample_token(&mut logits, &history, opts, &mut rng) as i64;
            if self.eos_ids.contains(&next) {
                finish = "stop".into();
                break;
            }
            generated.push(next as u32);
            all_tokens.push(next);
            history.push(next);
            if generated.len() % 8 == 0 || generated.len() == max_new {
                text = self.decode(&generated);
                let (cut, hit) = cut_at_stop(&text, stops);
                if hit {
                    text = cut;
                    finish = "stop".into();
                    break;
                }
                if opts.stop_on_json {
                    let probe = if think_prefilled { format!("<think>\n{text}") } else { text.clone() };
                    if protocol::has_complete_json(&probe) {
                        finish = "stop".into();
                        break;
                    }
                }
            }
        }
        if !generated.is_empty() && finish != "stop" || text.is_empty() {
            text = self.decode(&generated);
        }
        let (mut text, _) = cut_at_stop(&text, stops);
        if think_prefilled {
            text = format!("<think>\n{text}");
        }
        Ok(GenOutput { text, prompt_tokens: prompt_ids.len(), completion_tokens: generated.len(), finish_reason: finish })
    }

    fn describe(&self) -> String {
        let kv = match &self.io.kv {
            Some(k) => format!("KV cache, {} tensors", k.past.len()),
            None => "no KV cache".to_string(),
        };
        let mut s = format!("onnx {} ({}, {}, window {})", self.files.onnx.display(), self.device, kv, self.window);
        for n in &self.notes {
            s.push_str(&format!("; note: {n}"));
        }
        let _ = &self.label;
        s
    }

    fn context_window(&self) -> Option<usize> {
        Some(self.window)
    }
}

/// Environment-independent helper used by tests and `--check`.
pub fn describe_io_support() -> HashMap<&'static str, &'static str> {
    HashMap::from([("inputs", "input_ids, attention_mask, position_ids, token_type_ids, use_cache_branch, past_key_values.*"), ("outputs", "logits, present.*")])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_and_top_k_sampling() {
        let opts = GenOptions { temperature: 0.0, ..Default::default() };
        let mut rng = Rng::new(Some(1));
        assert_eq!(sample_token(&mut [0.1, 3.0, 0.2], &[], &opts, &mut rng), 1);
        // top_k = 1 behaves greedily even with temperature
        let opts = GenOptions { temperature: 1.0, top_k: 1, top_p: 1.0, ..Default::default() };
        for _ in 0..20 {
            assert_eq!(sample_token(&mut [0.1, 0.2, 5.0, 0.3], &[], &opts, &mut rng), 2);
        }
        // sampling stays inside the top-k set
        let opts = GenOptions { temperature: 1.0, top_k: 2, top_p: 1.0, ..Default::default() };
        for _ in 0..200 {
            let t = sample_token(&mut [5.0, 4.9, -9.0, -9.0], &[], &opts, &mut rng);
            assert!(t < 2);
        }
    }

    #[test]
    fn repetition_penalty_changes_the_winner() {
        let opts = GenOptions { temperature: 0.0, repetition_penalty: 2.0, ..Default::default() };
        let mut rng = Rng::new(Some(1));
        assert_eq!(sample_token(&mut [4.0, 3.0], &[0], &opts, &mut rng), 1);
        let nan = sample_token(&mut [f32::NAN, 1.0], &[], &opts, &mut rng);
        assert_eq!(nan, 1);
    }

    #[test]
    fn stop_strings_cut_the_text() {
        assert_eq!(cut_at_stop("abc<|im_end|>def", &["<|im_end|>"]), ("abc".to_string(), true));
        assert_eq!(cut_at_stop("abc", &["<|im_end|>"]), ("abc".to_string(), false));
    }

    #[test]
    fn locating_model_files() {
        let dir = tempfile::tempdir().unwrap();
        let e = locate_model_files(&dir.path().join("missing")).unwrap_err();
        assert!(e.contains("does not exist") && e.contains("runtime = \"api\""));
        let e = locate_model_files(dir.path()).unwrap_err();
        assert!(e.contains("no .onnx file"));
        std::fs::write(dir.path().join("model.onnx"), b"x").unwrap();
        let e = locate_model_files(dir.path()).unwrap_err();
        assert!(e.contains("tokenizer.json"));
        std::fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
        std::fs::write(dir.path().join("generation_config.json"), b"{}").unwrap();
        let f = locate_model_files(dir.path()).unwrap();
        assert!(f.onnx.ends_with("model.onnx") && f.generation_config.is_some() && f.config.is_none());
        // HF layout with onnx/ subdirectory and tokenizer in the parent
        let d2 = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(d2.path().join("onnx")).unwrap();
        std::fs::write(d2.path().join("onnx/model_quantized.onnx"), b"x").unwrap();
        std::fs::write(d2.path().join("tokenizer.json"), b"{}").unwrap();
        let f = locate_model_files(d2.path()).unwrap();
        assert!(f.onnx.ends_with("model_quantized.onnx"));
    }

    #[test]
    fn missing_runtime_library_gives_a_helpful_error() {
        if std::env::var_os("ORT_DYLIB_PATH").is_some() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let r = init_ort(Some(dir.path().join("nope.so").to_str().unwrap()), dir.path());
        let e = r.unwrap_err();
        assert!(e.contains("not found") && e.contains("dylib_path"), "{e}");
    }

    #[test]
    fn eos_id_parsing() {
        assert_eq!(ids_from(&serde_json::json!(5)), vec![5]);
        assert_eq!(ids_from(&serde_json::json!([1, 2])), vec![1, 2]);
        assert!(ids_from(&Value::Null).is_empty());
    }
}
