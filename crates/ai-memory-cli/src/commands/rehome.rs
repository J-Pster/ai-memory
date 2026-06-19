//! `ai-memory rehome` — re-home classified pages into their native kind
//! folders and rewrite the links to moved pages.
//!
//! Thin HTTP client: it POSTs `/admin/rehome-by-kind` on the running server
//! (where the deterministic move + link-rewrite lives) and prints the
//! report. Shared with `import --rehome` via [`run_rehome`].

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::cli::RehomeArgs;
use crate::config::Config;
use crate::http_client::{ServerEndpoint, post_json};

/// Request body for `POST /admin/rehome-by-kind`.
#[derive(Serialize)]
struct RehomeBody {
    workspace: String,
    project: String,
    dry_run: bool,
}

/// One move in the report.
#[derive(Deserialize)]
struct RehomeMoveReport {
    from: String,
    to: String,
}

/// One skipped page in the report.
#[derive(Deserialize)]
struct RehomeSkipReport {
    path: String,
    reason: String,
}

/// Response body for `POST /admin/rehome-by-kind`.
#[derive(Deserialize)]
struct RehomeResponseBody {
    dry_run: bool,
    pages_considered: usize,
    pages_moved: usize,
    links_rewritten: usize,
    moves: Vec<RehomeMoveReport>,
    skipped: Vec<RehomeSkipReport>,
}

/// Run the `rehome` subcommand.
///
/// # Errors
/// Bails when the project cannot be resolved or the server request fails.
pub async fn run(config: &Config, args: RehomeArgs) -> Result<()> {
    let project = super::resolve_project_name(config, args.project.as_deref())?;
    let endpoint = ServerEndpoint::from_config(config);
    run_rehome(&endpoint, &args.workspace, &project, args.dry_run).await
}

/// POST `/admin/rehome-by-kind` and print the report. Shared by the
/// `rehome` subcommand and `import --rehome`.
///
/// # Errors
/// Bails when the request fails.
pub async fn run_rehome(
    endpoint: &ServerEndpoint,
    workspace: &str,
    project: &str,
    dry_run: bool,
) -> Result<()> {
    let report: RehomeResponseBody = post_json(
        endpoint,
        "/admin/rehome-by-kind",
        &RehomeBody {
            workspace: workspace.to_string(),
            project: project.to_string(),
            dry_run,
        },
    )
    .await
    .context("running the rehome-by-kind pass")?;

    print_report(&report, workspace, project);
    Ok(())
}

/// Print the rehome report — a dry-run plan or the live result.
fn print_report(report: &RehomeResponseBody, workspace: &str, project: &str) {
    let verb = if report.dry_run {
        "would move"
    } else {
        "moved"
    };
    println!(
        "\nRehome ({}): {} {} of {} page{} in {}/{}, rewriting {} link{}",
        if report.dry_run { "dry-run" } else { "live" },
        verb,
        report.pages_moved,
        report.pages_considered,
        if report.pages_considered == 1 {
            ""
        } else {
            "s"
        },
        workspace,
        project,
        report.links_rewritten,
        if report.links_rewritten == 1 { "" } else { "s" },
    );
    for m in &report.moves {
        println!("  - {}  ->  {}", m.from, m.to);
    }
    if !report.skipped.is_empty() {
        println!(
            "\n⚠ {} page{} skipped (conflict):",
            report.skipped.len(),
            if report.skipped.len() == 1 { "" } else { "s" }
        );
        for s in &report.skipped {
            println!("  - {}  ({})", s.path, s.reason);
        }
    }
    if report.dry_run {
        println!("\n(dry-run -- nothing written to the server)");
    }
}
