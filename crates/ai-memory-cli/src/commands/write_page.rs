//! `ai-memory write-page` — testing helper for the M1-C wiki write path.
//!
//! Once hooks (M3) and consolidation (M7) are wired, real callers will
//! flow through those, not this command. It stays around as a manual
//! escape hatch and to exercise the chain in development.

use std::io::Read;

use ai_memory_core::{PagePath, Tier};
use ai_memory_store::Store;
use ai_memory_wiki::{Wiki, WritePageRequest};
use anyhow::{Context, Result};
use serde_json::{Map, Value};

use crate::cli::WritePageArgs;
use crate::config::Config;

/// Run the `write-page` subcommand.
///
/// # Errors
/// Returns an error if the store cannot be opened, the path is invalid,
/// the tier is unknown, the body cannot be read, or the underlying
/// `Wiki::write_page` call fails.
pub async fn run(config: &Config, args: WritePageArgs) -> Result<()> {
    let body = if args.body == "-" {
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf)?;
        buf
    } else {
        args.body
    };
    let tier: Tier = args.tier.parse()?;
    let path = PagePath::new(args.path.clone())?;

    let store = Store::open(&config.data_dir)
        .with_context(|| format!("opening store at {}", config.data_dir.display()))?;
    let ws = store
        .writer
        .get_or_create_workspace(args.workspace.clone())
        .await?;
    let proj = store
        .writer
        .get_or_create_project(ws, args.project.clone(), None)
        .await?;

    let wiki = Wiki::new(&config.data_dir, store.writer.clone())?;

    let mut fm = Map::new();
    if let Some(title) = args.title {
        fm.insert("title".into(), Value::String(title));
    }
    if !args.tag.is_empty() {
        fm.insert(
            "tags".into(),
            Value::Array(args.tag.into_iter().map(Value::String).collect()),
        );
    }
    if args.pinned {
        fm.insert("pinned".into(), Value::Bool(true));
    }
    let frontmatter = if fm.is_empty() {
        Value::Null
    } else {
        Value::Object(fm)
    };

    let id = wiki
        .write_page(WritePageRequest {
            workspace_id: ws,
            project_id: proj,
            path: path.clone(),
            frontmatter,
            body,
            tier,
            pinned: args.pinned,
            title: None,
        })
        .await?;

    println!(
        "wrote {} ({}) under workspace={} project={}",
        path, id, args.workspace, args.project
    );
    Ok(())
}
