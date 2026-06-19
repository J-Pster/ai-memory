//! `ai-memory import`, import a Claude Code "dual-store" memory setup.
//!
//! Thin HTTP client, like every other state-touching subcommand. It
//! reads the sources locally (a `@modelcontextprotocol/server-memory`
//! `memory.jsonl` file and/or an `mcp-server-qdrant` collection scrolled
//! over HTTP), builds wiki pages deterministically via
//! `ai_memory_consolidate::build_import_pages` (no LLM), then POSTs each
//! page to `POST /admin/write-page` on the running server. Under
//! `--dry-run` it prints the planned pages and writes nothing, mirroring
//! `bootstrap --dry-run`.
//!
//! Mapping rules live in `docs/import-claude-memory.md`. This module only
//! does IO + orchestration; the pure source→page transform lives in
//! `ai-memory-consolidate`.

use std::path::Path;

use ai_memory_consolidate::{
    GraphEntity, GraphRelation, ImportedPage, OmcWikiPage, QdrantPoint, build_import_pages,
    build_omc_wiki_pages,
};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::cli::ImportArgs;
use crate::config::Config;
use crate::http_client::{ServerEndpoint, post_json};

/// The `WritePageBody` shape the server's `POST /admin/write-page`
/// expects. Mirrors the struct in `write_page.rs` (kept private there);
/// imports always send `kind: None` and `tier: "semantic"`.
#[derive(Serialize)]
struct WritePageBody {
    workspace: String,
    project: String,
    path: String,
    body: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    kind: Option<String>,
    tier: String,
    tags: Vec<String>,
    pinned: bool,
}

/// Minimal response shape from `POST /admin/write-page`.
#[derive(Deserialize)]
struct WritePageResponseBody {
    page_id: String,
    path: String,
}

/// Request body for `POST /admin/import-normalize`. Mirrors the server's
/// `ImportNormalizeRequest`; the CLI sends the resolved scope + the
/// source's path prefix and (optionally) a page cap.
#[derive(Serialize)]
struct ImportNormalizeBody {
    workspace: String,
    project: String,
    path_prefixes: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    limit: Option<usize>,
    dry_run: bool,
    /// Per-batch input-token budget. Omitted → the server default.
    #[serde(skip_serializing_if = "Option::is_none")]
    max_input_tokens: Option<usize>,
}

/// One reclassified page in the normalize report.
#[derive(Deserialize)]
struct NormalizedPageReport {
    path: String,
    kind: String,
    tier: String,
}

/// Response body for `POST /admin/import-normalize`.
#[derive(Deserialize)]
struct ImportNormalizeResponseBody {
    pages_considered: usize,
    batches: usize,
    estimated_input_tokens: usize,
    pages_updated: Vec<NormalizedPageReport>,
    /// Paths of pages whose batch failed every retry and was skipped.
    /// Older servers omit this; `#[serde(default)]` keeps the CLI forward-
    /// and backward-compatible.
    #[serde(default)]
    pages_failed: Vec<String>,
    dry_run: bool,
}

/// Run the `import` subcommand.
///
/// Dispatches on `--source`: `claude-memory` imports a Claude Code
/// dual-store setup; `omc-wiki` imports an oh-my-claudecode Karpathy
/// wiki. Either way the sources are read locally, mapped to wiki pages,
/// then printed (`--dry-run`) or POSTed to the server.
///
/// # Errors
/// Bails on an unknown `--source`, when a required source argument is
/// missing, when a source cannot be read, or when a page write fails.
pub async fn run(config: &Config, args: ImportArgs) -> Result<()> {
    let project = super::resolve_project_name(config, args.project.as_deref())?;
    let endpoint = ServerEndpoint::from_config(config);

    // The source determines both the deterministic importer AND the path
    // prefix the normalize pass scopes to (the importer's output root).
    let path_prefix = source_path_prefix(&args.source)?;

    info!(
        server = %endpoint.url,
        workspace = %args.workspace,
        project = %project,
        source = %args.source,
        "import target",
    );

    // `--normalize-only`: skip the import/source step entirely; just run
    // the normalize pass over the scope. The source flags are not
    // required in this mode.
    if args.normalize_only {
        run_normalize(
            &endpoint,
            &args.workspace,
            &project,
            path_prefix,
            args.normalize_limit,
            args.normalize_max_tokens,
            args.dry_run,
        )
        .await?;
        if args.rehome {
            super::rehome::run_rehome(&endpoint, &args.workspace, &project, args.dry_run).await?;
        }
        return Ok(());
    }

    let pages = match args.source.as_str() {
        "claude-memory" => collect_claude_memory_pages(&args).await?,
        "omc-wiki" => collect_omc_wiki_pages(&args)?,
        // Unreachable: source_path_prefix already validated the value.
        other => bail!(
            "unknown --source `{other}`; supported sources are `claude-memory` and `omc-wiki`"
        ),
    };

    write_pages(&endpoint, &args.workspace, &project, &pages, args.dry_run).await?;

    // `--normalize`: after the deterministic import, run the LLM
    // normalize pass over what was just imported (this project + the
    // source's path prefix).
    if args.normalize {
        run_normalize(
            &endpoint,
            &args.workspace,
            &project,
            path_prefix,
            args.normalize_limit,
            args.normalize_max_tokens,
            args.dry_run,
        )
        .await?;
    }

    // `--rehome`: after import (+ optional normalize), move each classified
    // page into its native kind folder and rewrite the links. Deterministic.
    if args.rehome {
        super::rehome::run_rehome(&endpoint, &args.workspace, &project, args.dry_run).await?;
    }

    Ok(())
}

/// The wiki path prefix each `--source` writes under, also the scope the
/// normalize pass operates on. `claude-memory` → `imported/`,
/// `omc-wiki` → `omc/`.
///
/// # Errors
/// Bails on an unknown source value.
fn source_path_prefix(source: &str) -> Result<&'static str> {
    match source {
        "claude-memory" => Ok("imported/"),
        "omc-wiki" => Ok("omc/"),
        other => bail!(
            "unknown --source `{other}`; supported sources are `claude-memory` and `omc-wiki`"
        ),
    }
}

/// POST `/admin/import-normalize` for one path prefix and print the
/// returned report. Shared by `--normalize` and `--normalize-only`.
///
/// # Errors
/// Bails when the request fails (e.g. the server has no LLM provider
/// configured, which returns a clear "set AI_MEMORY_LLM_PROVIDER" error).
async fn run_normalize(
    endpoint: &ServerEndpoint,
    workspace: &str,
    project: &str,
    path_prefix: &str,
    limit: Option<usize>,
    max_input_tokens: Option<usize>,
    dry_run: bool,
) -> Result<()> {
    let report: ImportNormalizeResponseBody = post_json(
        endpoint,
        "/admin/import-normalize",
        &ImportNormalizeBody {
            workspace: workspace.to_string(),
            project: project.to_string(),
            path_prefixes: vec![path_prefix.to_string()],
            limit,
            dry_run,
            max_input_tokens,
        },
    )
    .await
    .context("running the import normalize pass")?;

    print_normalize_report(&report, workspace, project);
    Ok(())
}

/// Print the normalize report, a dry-run plan or the live result.
fn print_normalize_report(report: &ImportNormalizeResponseBody, workspace: &str, project: &str) {
    let verb = if report.dry_run {
        "would normalize"
    } else {
        "normalized"
    };
    println!(
        "\nNormalize ({}): {} {} page{} in {}/{} across {} batch{} (~{} input tokens)",
        if report.dry_run { "dry-run" } else { "live" },
        verb,
        report.pages_considered,
        if report.pages_considered == 1 {
            ""
        } else {
            "s"
        },
        workspace,
        project,
        report.batches,
        if report.batches == 1 { "" } else { "es" },
        report.estimated_input_tokens,
    );
    for page in &report.pages_updated {
        // Dry-run lists the paths that WOULD be normalized; the new
        // kind/tier only exist after the (skipped) LLM call, so the
        // server leaves them empty and we print the path alone.
        if page.kind.is_empty() && page.tier.is_empty() {
            println!("  - {}", page.path);
        } else {
            println!("  - {}  kind={} tier={}", page.path, page.kind, page.tier);
        }
    }
    if !report.pages_failed.is_empty() {
        println!(
            "\n⚠ {} page{} failed (re-run to retry):",
            report.pages_failed.len(),
            if report.pages_failed.len() == 1 {
                ""
            } else {
                "s"
            }
        );
        for path in &report.pages_failed {
            println!("  - {path}");
        }
    }
    if report.dry_run {
        println!("\n(dry-run -- no LLM call, nothing written to the server)");
    }
}

/// Collect the Claude Code dual-store sources (the `memory.jsonl` graph
/// and/or a Qdrant collection) and map them to wiki pages.
///
/// # Errors
/// Bails when neither source is given, or when the graph file / Qdrant
/// collection cannot be read, or when the transform yields no pages.
async fn collect_claude_memory_pages(args: &ImportArgs) -> Result<Vec<ImportedPage>> {
    if args.memory_graph_file.is_none() && args.qdrant_url.is_none() {
        bail!(
            "nothing to import: pass at least one of --memory-graph-file <memory.jsonl> \
             or --qdrant-url <http://host:6333>"
        );
    }

    // ---- collect sources locally ----------------------------------
    let (entities, relations) = match &args.memory_graph_file {
        Some(path) => parse_memory_graph(path)
            .with_context(|| format!("parsing memory graph file {}", path.display()))?,
        None => (Vec::new(), Vec::new()),
    };
    let points = match &args.qdrant_url {
        Some(url) => scroll_qdrant(url, &args.qdrant_collection)
            .await
            .with_context(|| {
                format!(
                    "scrolling Qdrant collection `{}` at {url}",
                    args.qdrant_collection
                )
            })?,
        None => Vec::new(),
    };
    info!(
        entities = entities.len(),
        relations = relations.len(),
        points = points.len(),
        "collected dual-store sources",
    );

    // ---- pure transform -------------------------------------------
    let pages = build_import_pages(&entities, &relations, &points, args.pinned);
    if pages.is_empty() {
        bail!(
            "no pages produced: the memory graph had no entities and Qdrant returned no points. \
             Check --memory-graph-file points at a non-empty memory.jsonl and/or the Qdrant \
             collection name is correct."
        );
    }
    Ok(pages)
}

/// Collect an oh-my-claudecode Karpathy wiki (a flat dir of `*.md` pages
/// with YAML frontmatter) and map it to wiki pages. The `index.md`
/// manifest is skipped; unparsable pages are warned about and skipped
/// rather than aborting the whole import.
///
/// # Errors
/// Bails when `--omc-wiki-dir` is missing, when the directory cannot be
/// read, or when no importable page is produced.
fn collect_omc_wiki_pages(args: &ImportArgs) -> Result<Vec<ImportedPage>> {
    let dir = args.omc_wiki_dir.as_deref().ok_or_else(|| {
        anyhow::anyhow!("--source omc-wiki requires --omc-wiki-dir <path-to-omc-wiki>")
    })?;

    let mut sources = read_omc_wiki_dir(dir)
        .with_context(|| format!("reading OMC wiki dir {}", dir.display()))?;
    if !args.include_session_logs {
        let before = sources.len();
        sources.retain(|p| !is_session_log_page(p));
        let skipped = before - sources.len();
        if skipped > 0 {
            info!(
                skipped,
                "skipped OMC auto-capture session-log pages \
                 (pass --include-session-logs to import them)"
            );
        }
    }
    info!(pages = sources.len(), "collected OMC wiki pages");

    let pages = build_omc_wiki_pages(&sources, args.pinned);
    if pages.is_empty() {
        bail!(
            "no pages produced: {} contained no importable *.md pages (index.md is skipped). \
             Check --omc-wiki-dir points at an OMC wiki directory.",
            dir.display()
        );
    }
    Ok(pages)
}

/// Whether an OMC wiki page is an auto-capture "session log", transient
/// per-session capture noise (oh-my-claudecode writes one page per session)
/// that should not flood the imported wiki. Matches a `session-log-*`
/// filename (case-insensitive, `.md` suffix stripped) or a `Session Log …`
/// frontmatter title. Pure, no IO.
fn is_session_log_page(page: &OmcWikiPage) -> bool {
    let stem = page.filename.strip_suffix(".md").unwrap_or(&page.filename);
    if stem.to_ascii_lowercase().starts_with("session-log") {
        return true;
    }
    page.title.as_deref().is_some_and(|t| {
        t.trim_start()
            .to_ascii_lowercase()
            .starts_with("session log")
    })
}

/// Write the mapped pages to the server, or print them under `--dry-run`.
/// Shared by every `--source` arm so the POST / dry-run tail lives in one
/// place.
///
/// # Errors
/// Bails when a page write fails.
async fn write_pages(
    endpoint: &ServerEndpoint,
    workspace: &str,
    project: &str,
    pages: &[ImportedPage],
    dry_run: bool,
) -> Result<()> {
    // ---- dry-run: print the plan, write nothing -------------------
    if dry_run {
        print_dry_run(pages, workspace, project);
        return Ok(());
    }

    // ---- POST each page -------------------------------------------
    for page in pages {
        let resp: WritePageResponseBody = post_json(
            endpoint,
            "/admin/write-page",
            &WritePageBody {
                workspace: workspace.to_string(),
                project: project.to_string(),
                path: page.path.clone(),
                body: page.body.clone(),
                title: Some(page.title.clone()),
                kind: None,
                tier: page.tier.clone(),
                tags: page.tags.clone(),
                pinned: page.pinned,
            },
        )
        .await
        .with_context(|| format!("writing imported page {}", page.path))?;
        let short_id = &resp.page_id[..resp.page_id.len().min(8)];
        println!("✓ wrote {} (page_id={})", resp.path, short_id);
    }

    println!(
        "\nImported {} page{} into {}/{}",
        pages.len(),
        if pages.len() == 1 { "" } else { "s" },
        workspace,
        project
    );
    Ok(())
}

/// Read every `*.md` page in an OMC wiki directory into [`OmcWikiPage`]s.
/// The flat dir's `index.md` manifest is skipped; a page whose
/// frontmatter cannot be parsed is warned about and skipped (one bad page
/// must not sink the whole import). Entries are sorted by filename so the
/// import order is deterministic regardless of directory iteration order.
fn read_omc_wiki_dir(dir: &Path) -> Result<Vec<OmcWikiPage>> {
    let mut entries: Vec<std::path::PathBuf> = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let entry = entry.with_context(|| format!("reading a dir entry in {}", dir.display()))?;
        let path = entry.path();
        let is_md = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("md"));
        if !is_md || !path.is_file() {
            continue;
        }
        let filename = match path.file_name().and_then(|s| s.to_str()) {
            Some(name) => name,
            None => continue,
        };
        if filename.eq_ignore_ascii_case("index.md") {
            continue;
        }
        entries.push(path);
    }
    entries.sort();

    let mut pages = Vec::with_capacity(entries.len());
    for path in &entries {
        // Safe: only `*.md` files with a UTF-8 name reached `entries`.
        let filename = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) => {
                eprintln!("warning: skipping unreadable OMC wiki page {filename}: {e}");
                continue;
            }
        };
        match parse_omc_wiki_page(&filename, &text) {
            Ok(page) => pages.push(page),
            Err(e) => {
                eprintln!("warning: skipping unparsable OMC wiki page {filename}: {e}");
            }
        }
    }
    Ok(pages)
}

/// Frontmatter fields we read off an OMC wiki page. Everything is
/// optional (`#[serde(default)]`) so a page missing fields still parses;
/// unknown keys (`created`, `updated`, `sources`, `confidence`,
/// `schemaVersion`, …) are ignored.
#[derive(Deserialize, Default)]
struct OmcFrontmatter {
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    category: Option<String>,
    #[serde(default)]
    links: Vec<String>,
}

/// Split an OMC wiki page's text into frontmatter + body and build an
/// [`OmcWikiPage`].
///
/// If the file starts with a `---` fence line, the YAML between it and the
/// next line that is exactly `---` is parsed as frontmatter and the body
/// is everything after that closing fence. With no leading fence the whole
/// file is the body (no frontmatter, empty tags/links).
///
/// # Errors
/// Returns an error only when the frontmatter block is present but is not
/// valid YAML; a missing/absent frontmatter block is not an error.
fn parse_omc_wiki_page(filename: &str, text: &str) -> Result<OmcWikiPage> {
    let (frontmatter, body) = match split_frontmatter(text) {
        Some((yaml, body)) => {
            let fm: OmcFrontmatter =
                serde_yaml::from_str(yaml).context("parsing YAML frontmatter")?;
            (fm, body.to_string())
        }
        None => (OmcFrontmatter::default(), text.to_string()),
    };
    Ok(OmcWikiPage {
        filename: filename.to_string(),
        title: frontmatter.title,
        tags: frontmatter.tags,
        category: frontmatter.category,
        links: frontmatter.links,
        body,
    })
}

/// Split leading YAML frontmatter from a markdown body.
///
/// Returns `Some((yaml, body))` when the text starts with a `---` fence
/// (the first line is exactly `---`) and a later line is exactly `---`:
/// `yaml` is the text between the fences, `body` is everything after the
/// closing fence's line terminator. Returns `None` when there is no
/// leading fence (the whole input is the body) or the opening fence is
/// never closed.
fn split_frontmatter(text: &str) -> Option<(&str, &str)> {
    // The opening fence must be the very first line.
    let after_open = text
        .strip_prefix("---\n")
        .or_else(|| text.strip_prefix("---\r\n"))?;
    let open_len = text.len() - after_open.len();

    // Walk lines after the opening fence looking for a closing `---`.
    let mut cursor = open_len;
    for line in after_open.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(['\n', '\r']);
        if trimmed == "---" {
            let yaml = &text[open_len..cursor];
            let body = &text[cursor + line.len()..];
            return Some((yaml, body));
        }
        cursor += line.len();
    }
    None
}

/// Parse a `memory.jsonl` knowledge-graph dump into entities and
/// relations. Each non-blank line is a JSON object tagged by `type`
/// (`entity` or `relation`); unknown / malformed lines are skipped with a
/// warning rather than aborting the whole import.
fn parse_memory_graph(path: &Path) -> Result<(Vec<GraphEntity>, Vec<GraphRelation>)> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut entities = Vec::new();
    let mut relations = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match serde_json::from_str::<GraphLine>(trimmed) {
            Ok(GraphLine::Entity {
                name,
                entity_type,
                observations,
            }) => entities.push(GraphEntity {
                name,
                entity_type,
                observations,
            }),
            Ok(GraphLine::Relation {
                from,
                to,
                relation_type,
            }) => relations.push(GraphRelation {
                from,
                to,
                relation_type,
            }),
            Err(e) => {
                // One bad line shouldn't sink the whole migration; the
                // source store may have hand-edited or partial records.
                eprintln!(
                    "warning: skipping unparsable memory-graph line {} in {}: {e}",
                    lineno + 1,
                    path.display()
                );
            }
        }
    }
    Ok((entities, relations))
}

/// One line of `memory.jsonl`, discriminated by the `type` field. Fields
/// default so a record missing `observations` (or other optionals) still
/// parses; `entityType` is mapped to `entity_type`.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum GraphLine {
    Entity {
        name: String,
        #[serde(rename = "entityType", default)]
        entity_type: String,
        #[serde(default)]
        observations: Vec<String>,
    },
    Relation {
        from: String,
        to: String,
        #[serde(rename = "relationType", default)]
        relation_type: String,
    },
}

/// Scroll an `mcp-server-qdrant` collection, following `next_page_offset`
/// until exhausted, and flatten each point's payload into a
/// [`QdrantPoint`]. Points whose payload has no `metadata.entityName` are
/// skipped (they cannot bridge to a graph entity nor stand alone).
async fn scroll_qdrant(base_url: &str, collection: &str) -> Result<Vec<QdrantPoint>> {
    let client = reqwest::Client::new();
    let url = format!(
        "{}/collections/{collection}/points/scroll",
        base_url.trim_end_matches('/')
    );
    let mut points = Vec::new();
    let mut offset: Option<serde_json::Value> = None;
    loop {
        let mut body = serde_json::json!({
            "limit": 256,
            "with_payload": true,
            "with_vector": false,
        });
        if let Some(off) = &offset {
            body["offset"] = off.clone();
        }
        let resp = client
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("POST {url}"))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            bail!("Qdrant returned {status} for scroll on `{collection}`: {text}");
        }
        let page: QdrantScrollResponse = resp
            .json()
            .await
            .with_context(|| format!("parsing Qdrant scroll response from {url}"))?;
        for raw in page.result.points {
            if let Some(point) = raw.into_point() {
                points.push(point);
            }
        }
        match page.result.next_page_offset {
            Some(next) if !next.is_null() => offset = Some(next),
            _ => break,
        }
    }
    Ok(points)
}

/// Top-level Qdrant scroll response: `{ "result": { points, next_page_offset } }`.
#[derive(Deserialize)]
struct QdrantScrollResponse {
    result: QdrantScrollResult,
}

/// The `result` object of a scroll: a page of points plus the cursor for
/// the next page (`null` when the scan is complete).
#[derive(Deserialize)]
struct QdrantScrollResult {
    #[serde(default)]
    points: Vec<QdrantRawPoint>,
    #[serde(default)]
    next_page_offset: Option<serde_json::Value>,
}

/// Deserialize a field that may be absent OR explicitly `null` into its
/// `Default`. Plain `#[serde(default)]` only covers an absent key; a key
/// present with a `null` value still tries to deserialize `null` into the
/// target type and fails. Real `mcp-server-qdrant` collections carry points
/// with `payload.metadata = null` (and occasionally a null `document`), so
/// every optional sub-object on the scroll path goes through this helper.
fn null_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Default + Deserialize<'de>,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

/// A raw scrolled point: `{ "id": ..., "payload": { document, metadata } }`.
#[derive(Deserialize)]
struct QdrantRawPoint {
    #[serde(default, deserialize_with = "null_default")]
    payload: QdrantPayload,
}

/// The payload an `mcp-server-qdrant` point carries.
#[derive(Deserialize, Default)]
struct QdrantPayload {
    #[serde(default, deserialize_with = "null_default")]
    document: String,
    #[serde(default, deserialize_with = "null_default")]
    metadata: QdrantMetadata,
}

/// The `metadata` sub-object of a Qdrant payload. `entityName` is the
/// bridge to a Memory Graph entity; the rest is surfaced into the page
/// metadata header.
#[derive(Deserialize, Default)]
struct QdrantMetadata {
    #[serde(rename = "entityName", default)]
    entity_name: Option<String>,
    #[serde(default)]
    project: Option<String>,
    #[serde(default)]
    topic: Option<String>,
    #[serde(default)]
    date: Option<String>,
    #[serde(default)]
    repos: Vec<String>,
}

impl QdrantRawPoint {
    /// Flatten a raw point into a [`QdrantPoint`], or `None` when it has
    /// no `metadata.entityName` (nothing to bridge or title it by).
    fn into_point(self) -> Option<QdrantPoint> {
        let meta = self.payload.metadata;
        let entity_name = meta.entity_name.filter(|s| !s.is_empty())?;
        Some(QdrantPoint {
            entity_name,
            document: self.payload.document,
            project: meta.project,
            topic: meta.topic,
            date: meta.date,
            repos: meta.repos,
        })
    }
}

/// Print the planned pages without writing anything, path, title, and a
/// summary of which sections each page carries. Mirrors the
/// `bootstrap --dry-run` human report.
fn print_dry_run(pages: &[ImportedPage], workspace: &str, project: &str) {
    println!(
        "\nDry-run: would import {} page{} into {}/{}\n",
        pages.len(),
        if pages.len() == 1 { "" } else { "s" },
        workspace,
        project
    );
    for page in pages {
        let mut sections = Vec::new();
        if page.body.contains("## Observations") {
            sections.push("Observations");
        }
        if page.body.contains("## Summary") {
            sections.push("Summary");
        }
        if page.body.contains("## Related") {
            sections.push("Related");
        }
        let section_summary = if sections.is_empty() {
            "(title + type only)".to_string()
        } else {
            sections.join(", ")
        };
        let tags = if page.tags.is_empty() {
            String::new()
        } else {
            format!("  tags=[{}]", page.tags.join(", "))
        };
        println!(
            "  - {}  \"{}\"  [{}]{}",
            page.path, page.title, section_summary, tags
        );
    }
    println!("\n(dry-run -- nothing written to the server)");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page(filename: &str, title: Option<&str>) -> OmcWikiPage {
        OmcWikiPage {
            filename: filename.to_string(),
            title: title.map(str::to_string),
            tags: Vec::new(),
            category: None,
            links: Vec::new(),
            body: String::new(),
        }
    }

    #[test]
    fn session_log_detected_by_filename() {
        assert!(is_session_log_page(&page(
            "session-log-2026-06-19.md",
            None
        )));
        assert!(is_session_log_page(&page("Session-Log-Abc.md", None)));
        assert!(is_session_log_page(&page("session-log", None)));
    }

    #[test]
    fn session_log_detected_by_title() {
        assert!(is_session_log_page(&page(
            "abc.md",
            Some("Session Log 2026-06-19")
        )));
        assert!(is_session_log_page(&page(
            "abc.md",
            Some("  session log x")
        )));
    }

    #[test]
    fn real_pages_are_not_session_logs() {
        assert!(!is_session_log_page(&page(
            "admin-dashboard-overview.md",
            Some("Admin Dashboard Overview")
        )));
        // "session" alone (not the session-log prefix) must not match.
        assert!(!is_session_log_page(&page(
            "user-session-management.md",
            Some("User Session Management")
        )));
    }
}
