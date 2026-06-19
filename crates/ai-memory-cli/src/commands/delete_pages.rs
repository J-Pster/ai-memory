//! `ai-memory delete-pages`, bulk-delete wiki pages via the server.
//!
//! Sends a `POST /admin/delete-pages` request to the running server,
//! which removes every targeted page and commits the wiki git repo ONCE
//! at the end (vs one commit per page in `delete-page`). The deletion set
//! is the union of an explicit `--paths-file` list and every LATEST page
//! under `--prefix`. The server resolves `(workspace, project)` the same
//! way the single-page delete does, so the bulk delete can never silently
//! land in the wrong workspace.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::cli::DeletePagesArgs;
use crate::config::Config;
use crate::http_client::{ServerEndpoint, post_json};

#[derive(Serialize)]
struct DeletePagesBody {
    workspace: String,
    project: String,
    paths: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    path_prefix: Option<String>,
}

#[derive(Deserialize)]
struct DeletePagesResponseBody {
    deleted: usize,
    #[serde(default)]
    not_found: Vec<String>,
}

/// Run the `delete-pages` subcommand.
///
/// # Errors
/// Returns an error when neither `--prefix` nor `--paths-file` is given,
/// when the paths file cannot be read, or when the POST to
/// `/admin/delete-pages` fails (network, scope resolution, admission
/// webhook rejection, or filesystem error on the server).
pub async fn run(config: &Config, args: DeletePagesArgs) -> Result<()> {
    // At least one selector is required: a prefix, an explicit list, or
    // both. Without either there is nothing to delete, fail loud rather
    // than silently POSTing an empty set.
    if args.prefix.is_none() && args.paths_file.is_none() {
        bail!("provide at least one of --prefix or --paths-file");
    }

    // Read the explicit path list (if any): newline-separated, blank lines
    // ignored, each entry trimmed.
    let paths: Vec<String> = match &args.paths_file {
        Some(file) => {
            let contents = std::fs::read_to_string(file)
                .with_context(|| format!("reading paths file {}", file.display()))?;
            contents
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(str::to_string)
                .collect()
        }
        None => Vec::new(),
    };

    let path_prefix = args.prefix.as_deref().map(str::trim).and_then(|p| {
        if p.is_empty() {
            None
        } else {
            Some(p.to_string())
        }
    });

    let project = super::resolve_project_name(config, args.project.as_deref())?;

    let endpoint = ServerEndpoint::from_config(config);
    let resp: DeletePagesResponseBody = post_json(
        &endpoint,
        "/admin/delete-pages",
        &DeletePagesBody {
            workspace: args.workspace.clone(),
            project: project.clone(),
            paths,
            path_prefix,
        },
    )
    .await
    .context("bulk-deleting pages via server")?;

    println!(
        "deleted {} pages under {}/{}",
        resp.deleted, args.workspace, project
    );
    if !resp.not_found.is_empty() {
        println!("{} path(s) not found:", resp.not_found.len());
        for path in &resp.not_found {
            println!("  {path}");
        }
    }
    Ok(())
}
