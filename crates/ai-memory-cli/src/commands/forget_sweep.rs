//! `ai-memory forget-sweep` — run the M8 retention sweep manually.

use ai_memory_consolidate::run_sweep;
use ai_memory_store::Store;
use anyhow::{Context, Result};

use crate::cli::ForgetSweepArgs;
use crate::config::Config;

/// Run the `forget-sweep` subcommand.
///
/// # Errors
/// Returns an error if the store cannot be opened or the sweep fails.
pub async fn run(config: &Config, args: ForgetSweepArgs) -> Result<()> {
    let store = Store::open(&config.data_dir)
        .with_context(|| format!("opening store at {}", config.data_dir.display()))?;
    let ws = store.writer.get_or_create_workspace("default").await?;
    let proj = store
        .writer
        .get_or_create_project(ws, "scratch", None)
        .await?;
    let report = run_sweep(
        &store.reader,
        &store.writer,
        ws,
        proj,
        &config.decay,
        args.dry_run,
    )
    .await?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
