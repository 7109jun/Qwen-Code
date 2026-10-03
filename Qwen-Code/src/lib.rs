//! Qwen Code — an AI coding agent CLI driven by a Thinking model and a Coder model.
//!
//! ```text
//! cli / tui -> session -> agent loop -> models (providers) -> runtime (ONNX | Qwen API)
//!                              |
//!                              +-> tools (JSON -> schema -> permission -> execute)
//!                              +-> context manager
//! ```

pub mod agent;
pub mod cli;
pub mod config;
pub mod context;
pub mod fileedit;
pub mod git;
pub mod models;
pub mod onnx;
pub mod permissions;
pub mod platform;
pub mod prompts;
pub mod protocol;
pub mod runtime;
pub mod schema;
pub mod session;
pub mod tools;
pub mod tui;
pub mod ui;

/// Program display name.
pub const APP_NAME: &str = "Qwen Code";
/// Version string.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
