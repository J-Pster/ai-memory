//! `ai-memory status` — report runtime config and persisted counts.
//!
//! Thin HTTP client. Calls `GET /admin/status` on the configured
//! server; renders the response as human text or JSON. Never opens
//! the store directly — the server is the source of truth.

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::cli::StatusArgs;
use crate::config::Config;
use crate::http_client::{ServerEndpoint, get_json};

/// Server-shaped response. Mirrors `ai_memory_mcp::admin::StatusReport`.
#[derive(Debug, Deserialize, Serialize)]
struct Report {
    /// Server binary version.
    version: String,
    /// Server-side data directory path.
    data_dir: String,
    /// Server bind address.
    bind: String,
    /// Server-side SQLite path.
    db_path: String,
    /// Lifetime counts.
    counts: Counts,
    /// Derived-index diagnostics.
    #[serde(default)]
    derived: Derived,
}

#[derive(Debug, Deserialize, Serialize)]
struct Counts {
    pages_latest: u64,
    pages_all: u64,
    sessions: u64,
    observations: u64,
}

#[derive(Debug, Default, Deserialize, Serialize)]
struct Derived {
    pages_rows: u64,
    pages_fts_rows: u64,
    observations_rows: u64,
    observations_fts_rows: u64,
    latest_pages_missing_embeddings: u64,
    embedding_rows: u64,
    embedding_triples: Vec<EmbeddingTriple>,
    links_from_latest_pages: u64,
    unresolved_links_from_latest_pages: u64,
    stale_links_from_latest_pages: u64,
}

#[derive(Debug, Deserialize, Serialize)]
struct EmbeddingTriple {
    provider: String,
    model: String,
    dim: u32,
    count: u64,
}

/// Run the `status` subcommand.
///
/// # Errors
/// Returns an error if the server is unreachable, returns non-2xx, or
/// the response can't be parsed.
pub async fn run(config: &Config, args: StatusArgs) -> Result<()> {
    let ep = ServerEndpoint::from_config(config);
    let report: Report = get_json(&ep, "/admin/status", &[]).await?;

    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "version": report.version,
                "data_dir": report.data_dir,
                "bind": report.bind,
                "db_path": report.db_path,
                "counts": {
                    "pages_latest": report.counts.pages_latest,
                    "pages_all": report.counts.pages_all,
                    "sessions": report.counts.sessions,
                    "observations": report.counts.observations,
                },
                "derived": report.derived,
                "client": { "server_url": ep.url, "auth": ep.auth_token.is_some() },
            }))?
        );
    } else {
        println!("ai-memory {} (server)", report.version);
        println!("  server:       {}", ep.url);
        println!("  data-dir:     {}", report.data_dir);
        println!("  db:           {}", report.db_path);
        println!("  bind:         {}", report.bind);
        println!(
            "  pages:        {} (all versions: {})",
            report.counts.pages_latest, report.counts.pages_all
        );
        println!("  sessions:     {}", report.counts.sessions);
        println!("  observations: {}", report.counts.observations);
        println!(
            "  fts:          pages {}/{}; observations {}/{}",
            report.derived.pages_fts_rows,
            report.derived.pages_rows,
            report.derived.observations_fts_rows,
            report.derived.observations_rows
        );
        println!(
            "  embeddings:   {} rows; {} latest pages missing",
            report.derived.embedding_rows, report.derived.latest_pages_missing_embeddings
        );
        println!(
            "  links:        {} latest-page links (unresolved: {}, stale: {})",
            report.derived.links_from_latest_pages,
            report.derived.unresolved_links_from_latest_pages,
            report.derived.stale_links_from_latest_pages
        );
    }
    Ok(())
}
