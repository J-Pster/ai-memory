//! `ai-memory import-instructions` — print the agent-driven ingestion
//! playbook to stdout.
//!
//! ## Why this exists
//!
//! The deterministic `ai-memory import` writes raw 1:1 pages. The
//! *second pass* — pruning, classifying (`kind`), cleaning, re-homing by
//! kind, and de-duplicating — is best done by whatever coding agent the
//! user already runs, following `docs/ai-ingestion-playbook.md`.
//!
//! This subcommand emits that playbook so an agent can fetch it with a
//! single command (no need to know the repo layout or hunt for the file).
//! The doc is embedded at compile time, so it ships inside the binary.

use anyhow::Result;

/// The agent-driven ingestion playbook, embedded from the repo's `docs/`
/// at compile time so it travels with the binary.
const INGESTION_PLAYBOOK: &str = include_str!("../../../../docs/ai-ingestion-playbook.md");

/// Run the `import-instructions` subcommand: print the playbook verbatim.
///
/// # Errors
/// Infallible today; returns `Result` for signature parity with the other
/// subcommands and forward-compatibility (e.g. a future `--output` flag).
pub fn run() -> Result<()> {
    print!("{INGESTION_PLAYBOOK}");
    Ok(())
}
