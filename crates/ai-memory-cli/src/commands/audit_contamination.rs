//! `ai-memory audit-contamination` — structural cross-project contamination audit.
//!
//! Thin HTTP client. Calls `GET /admin/audit-contamination` on the configured
//! server and renders the report as human text or JSON. Read-only — the server
//! only reports, never mutates; remediation is a separate operator step.

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::cli::AuditContaminationArgs;
use crate::config::Config;
use crate::http_client::{ServerEndpoint, get_json};

/// Server-shaped response. Mirrors `ai_memory_store::ContaminationReport`.
#[derive(Debug, Default, Deserialize, Serialize)]
struct Report {
    #[serde(default)]
    summary: Summary,
    #[serde(default)]
    findings: Vec<Finding>,
}

#[derive(Debug, Default, Deserialize, Serialize)]
struct Summary {
    #[serde(default)]
    sessions_misbucketed: usize,
    #[serde(default)]
    observations_drifted: usize,
}

#[derive(Debug, Deserialize, Serialize)]
struct Finding {
    check: String,
    confidence: String,
    entity_kind: String,
    entity_id: String,
    landed_workspace: String,
    landed_project: String,
    #[serde(default)]
    expected_project: Option<String>,
    #[serde(default)]
    cwd: Option<String>,
    #[serde(default)]
    session_id: Option<String>,
}

/// Run the `audit-contamination` subcommand.
///
/// # Errors
/// Returns an error if the server is unreachable, returns non-2xx, or the
/// response can't be parsed.
pub async fn run(config: &Config, args: AuditContaminationArgs) -> Result<()> {
    let ep = ServerEndpoint::from_config(config);
    // Both workspace+project scope the audit to one landed bucket; either alone
    // is ignored (a partial scope is meaningless) — audit everything instead.
    let query: Vec<(&str, &str)> = match (args.workspace.as_deref(), args.project.as_deref()) {
        (Some(ws), Some(proj)) => vec![("workspace", ws), ("project", proj)],
        _ => Vec::new(),
    };
    let report: Report = get_json(&ep, "/admin/audit-contamination", &query).await?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    if report.findings.is_empty() {
        println!("No structural contamination found (sessions + observations consistent).");
        return Ok(());
    }
    println!(
        "Contamination audit: {} session(s) mis-bucketed, {} observation(s) drifted.",
        report.summary.sessions_misbucketed, report.summary.observations_drifted
    );
    for f in &report.findings {
        let expected = f.expected_project.as_deref().unwrap_or("?");
        match f.check.as_str() {
            "session_wrong_bucket" => println!(
                "  [{}] session {} landed in {}/{} but its cwd ({}) resolves to project {}",
                f.confidence,
                f.entity_id,
                f.landed_workspace,
                f.landed_project,
                f.cwd.as_deref().unwrap_or("?"),
                expected,
            ),
            "observation_session_drift" => println!(
                "  [{}] observation {} in {}/{} but its session ({}) is in project {}",
                f.confidence,
                f.entity_id,
                f.landed_workspace,
                f.landed_project,
                f.session_id.as_deref().unwrap_or("?"),
                expected,
            ),
            other => println!(
                "  [{}] {} {} in {}/{}",
                f.confidence, other, f.entity_id, f.landed_workspace, f.landed_project
            ),
        }
    }
    Ok(())
}
