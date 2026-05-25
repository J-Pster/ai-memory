//! [`Wiki`] — the only correct write path for the markdown source-of-truth.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ai_memory_core::{NewPage, PageId, PagePath, ProjectId, Sanitizer, Tier, WorkspaceId};
use ai_memory_llm::Embedder;
use ai_memory_store::{WriterHandle, f32_vec_to_bytes};

use crate::atomic;
use crate::error::WikiResult;
use crate::git::GitAdapter;
use crate::markdown::{Markdown, derive_title, emit, extract_links, parse};

/// Wiki filesystem handle.
///
/// Owns the path of the wiki root (`<data_dir>/wiki/`) and a cloneable
/// [`WriterHandle`] so that every public mutation writes the markdown
/// file *and* sends a `WriteCmd::UpsertPage` to the store in a single
/// call — no background-task indexing-after-return (basic-memory #763
/// lesson).
///
/// ## On-disk layout
///
/// Pages are stored at `<wiki_root>/<workspace_id>/<project_id>/<page-path>`.
/// Each of `workspace_id` and `project_id` is a UUID string. This layout is
/// the single canonical namespace; all path construction must go through
/// [`Wiki::project_root`] or [`Wiki::abs_path`] — never hand-rolled joins.
#[derive(Clone)]
pub struct Wiki {
    root: PathBuf,
    writer: WriterHandle,
    git: GitAdapter,
    embedder: Option<Arc<dyn Embedder>>,
    /// Privacy strip applied to every page body before persistence.
    /// Defence-in-depth: any caller path (LLM consolidation, manual
    /// write-page CLI, agent-supplied tool input) still gets scrubbed
    /// at the wiki boundary even if upstream forgot.
    sanitizer: Sanitizer,
}

impl Wiki {
    /// Construct a wiki handle rooted at `<data_dir>/wiki/`. Creates the
    /// directory if absent and initialises a git repo inside it.
    ///
    /// # Errors
    /// Returns [`WikiError::Io`] if the wiki root or git repo cannot be
    /// created.
    pub fn new(data_dir: &Path, writer: WriterHandle) -> WikiResult<Self> {
        let root = data_dir.join("wiki");
        std::fs::create_dir_all(&root)?;
        let git = GitAdapter::open_or_init(&root)?;
        Ok(Self {
            root,
            writer,
            git,
            embedder: None,
            sanitizer: Sanitizer::builtin(),
        })
    }

    /// Replace the default built-in-only sanitizer with one carrying
    /// the operator's `[sanitize].extra_patterns` + `allowlist`.
    #[must_use]
    pub fn with_sanitizer(mut self, sanitizer: Sanitizer) -> Self {
        self.sanitizer = sanitizer;
        self
    }

    /// Attach an embedder. When set, `write_page` computes + stores an
    /// embedding for the new version synchronously. `apply_batch` keeps
    /// the SQL/file fan-out atomic and leaves vector completeness to
    /// admin or scheduled embedding backfill. Without an embedder,
    /// vector search is skipped and `ReaderPool::hybrid_search` uses
    /// FTS5 + graph expansion.
    #[must_use]
    pub fn with_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Borrow the optional embedder (used by the `ai-memory embed`
    /// backfill command).
    #[must_use]
    pub fn embedder(&self) -> Option<&Arc<dyn Embedder>> {
        self.embedder.as_ref()
    }

    /// Borrow the git adapter (for callers wiring auto-commit).
    #[must_use]
    pub fn git(&self) -> &GitAdapter {
        &self.git
    }

    /// Stage + commit the entire wiki tree. Returns `Ok(None)` if there
    /// was nothing to commit.
    ///
    /// # Errors
    /// Propagates [`WikiError`] from the git adapter.
    pub fn commit_all(&self, message: &str) -> WikiResult<Option<git2::Oid>> {
        self.git.commit_all(message)
    }

    /// Path of the wiki root on disk.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolve the on-disk root for a project: `<wiki_root>/<ws>/<proj>`.
    /// All page files for this project live under this directory.
    #[must_use]
    pub fn project_root(&self, workspace_id: WorkspaceId, project_id: ProjectId) -> PathBuf {
        self.root
            .join(workspace_id.to_string())
            .join(project_id.to_string())
    }

    /// Absolute on-disk path for a page within a specific project.
    #[must_use]
    pub fn abs_path(
        &self,
        workspace_id: WorkspaceId,
        project_id: ProjectId,
        path: &PagePath,
    ) -> PathBuf {
        self.project_root(workspace_id, project_id)
            .join(path.as_str())
    }

    /// Read the page at `path` from disk for the given project.
    ///
    /// # Errors
    /// Returns [`WikiError::Io`] if the file is missing or unreadable, or
    /// [`WikiError::Yaml`] if the frontmatter block is malformed.
    pub fn read_page(
        &self,
        workspace_id: WorkspaceId,
        project_id: ProjectId,
        path: &PagePath,
    ) -> WikiResult<Markdown> {
        let abs = self.abs_path(workspace_id, project_id, path);
        let raw = std::fs::read_to_string(&abs)?;
        parse(&raw)
    }

    /// Delete the on-disk file for `path` within the given project.
    ///
    /// Returns `Ok(())` when the file was removed or did not exist (idempotent).
    /// The file watcher will observe the deletion; the sha256 short-circuit in
    /// the watcher's reindex path means a missing file produces a graceful
    /// no-op rather than an error.
    ///
    /// # Errors
    /// Returns [`WikiError::Io`] for any OS error other than "not found".
    pub fn delete_page(
        &self,
        workspace_id: WorkspaceId,
        project_id: ProjectId,
        path: &PagePath,
    ) -> WikiResult<()> {
        let abs = self.abs_path(workspace_id, project_id, path);
        match std::fs::remove_file(&abs) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(crate::WikiError::Io(e)),
        }
    }

    /// Cloneable handle to the underlying store writer.
    #[must_use]
    pub fn writer(&self) -> &WriterHandle {
        &self.writer
    }

    /// Re-index the page on disk at `path` into the store *without*
    /// rewriting the file.
    ///
    /// Called by the watcher when an external editor (Obsidian, vim) has
    /// changed a file we did not write. The store-side sha256 short-circuit
    /// makes this idempotent: if the on-disk content already matches the
    /// latest version, no supersession happens.
    ///
    /// # Errors
    /// Returns [`WikiError`] for any filesystem, parsing, or store error.
    pub async fn reindex_page(
        &self,
        workspace_id: WorkspaceId,
        project_id: ProjectId,
        path: PagePath,
    ) -> WikiResult<PageId> {
        let md = self.read_page(workspace_id, project_id, &path)?;
        let title = derive_title(&md.frontmatter, &md.body, &path);
        let links = extract_links(&md.body, &path);
        let pinned = is_slot_path(&path);
        let id = self
            .writer
            .upsert_page(NewPage {
                workspace_id,
                project_id,
                path,
                title,
                body: md.body,
                tier: Tier::Semantic,
                frontmatter_json: md.frontmatter,
                pinned,
                links,
            })
            .await?;
        Ok(id)
    }

    /// Atomically apply a batch of page writes. Either all pages land
    /// (one SQL transaction) and their files are renamed into place,
    /// or no DB row changes and tempfiles are dropped.
    ///
    /// # Errors
    /// Returns [`WikiError`] for any filesystem, parsing, or store
    /// error.
    pub async fn apply_batch(&self, requests: Vec<WritePageRequest>) -> WikiResult<Vec<PageId>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        // Pre-compute markdown + tempfile for each request.
        let mut staged: Vec<(
            WritePageRequest,
            tempfile::NamedTempFile,
            std::path::PathBuf,
        )> = Vec::with_capacity(requests.len());
        for mut req in requests {
            // Defence-in-depth scrub at the batch boundary too.
            req.body = self.sanitizer.scrub(&req.body);
            if let Some(t) = req.title.take() {
                req.title = Some(self.sanitizer.scrub(&t));
            }
            let title = derive_title(&req.frontmatter, &req.body, &req.path);
            let markdown = Markdown {
                frontmatter: req.frontmatter.clone(),
                body: req.body.clone(),
            };
            let emitted = emit(&markdown)?;
            let abs = self.abs_path(req.workspace_id, req.project_id, &req.path);
            let parent = abs.parent().ok_or_else(|| {
                ai_memory_wiki_error("page path has no parent (cannot stage tempfile)")
            })?;
            std::fs::create_dir_all(parent)?;
            let mut tmp = tempfile::Builder::new()
                .prefix(".ai-memory-tmp.")
                .tempfile_in(parent)?;
            use std::io::Write as _;
            tmp.write_all(emitted.as_bytes())?;
            tmp.as_file().sync_data()?;
            let req_with_title = WritePageRequest {
                title: Some(title),
                ..req
            };
            staged.push((req_with_title, tmp, abs));
        }

        // Build NewPage batch with the precomputed titles.
        let pages: Vec<ai_memory_core::NewPage> = staged
            .iter()
            .map(|(req, _, _)| ai_memory_core::NewPage {
                workspace_id: req.workspace_id,
                project_id: req.project_id,
                path: req.path.clone(),
                title: req.title.clone().unwrap_or_default(),
                body: req.body.clone(),
                tier: req.tier,
                frontmatter_json: req.frontmatter.clone(),
                pinned: req.pinned || is_slot_path(&req.path),
                links: extract_links(&req.body, &req.path),
            })
            .collect();

        let ids = self.writer.upsert_pages_batch(pages).await?;

        // SQL succeeded; rename tempfiles into place.
        for (_, tmp, abs) in staged {
            let persisted = tmp.persist(&abs)?;
            persisted.sync_data()?;
        }

        Ok(ids)
    }

    /// Write `body` (with optional `frontmatter`) atomically to
    /// `<wiki_root>/<workspace_id>/<project_id>/<path>` and upsert the
    /// matching page row in the store.
    ///
    /// The store side does the sha256 short-circuit + supersession dance.
    /// Returns the id of the page version that is now `is_latest = 1`.
    ///
    /// # Errors
    /// Returns [`WikiError`] for any filesystem, parsing, or store error.
    pub async fn write_page(&self, req: WritePageRequest) -> WikiResult<PageId> {
        let WritePageRequest {
            workspace_id,
            project_id,
            path,
            frontmatter,
            body,
            tier,
            pinned,
            title: explicit_title,
        } = req;

        // Defence-in-depth: scrub the body before we touch disk or the
        // store, regardless of caller. The hook ingress already scrubs
        // observation text; this catches LLM-rewritten consolidation
        // bodies, manual `write-page` CLI inputs, and anything an MCP
        // tool slips through.
        let body = self.sanitizer.scrub(&body);

        let title = explicit_title
            .clone()
            .map(|t| self.sanitizer.scrub(&t))
            .unwrap_or_else(|| derive_title(&frontmatter, &body, &path));
        let links = extract_links(&body, &path);
        let pinned = pinned || is_slot_path(&path);
        let markdown = Markdown {
            frontmatter: frontmatter.clone(),
            body: body.clone(),
        };
        let emitted = emit(&markdown)?;
        let abs = self.abs_path(workspace_id, project_id, &path);
        if let Some(parent) = abs.parent() {
            std::fs::create_dir_all(parent)?;
        }
        atomic::write_atomic(&abs, emitted.as_bytes())?;

        let page_id = self
            .writer
            .upsert_page(NewPage {
                workspace_id,
                project_id,
                path,
                title,
                body: body.clone(),
                tier,
                frontmatter_json: frontmatter,
                pinned,
                links,
            })
            .await?;
        // Embed if configured. We do this on the caller's task so the
        // tool reply still happens "indexes commit in the same
        // transaction" (basic-memory #763 lesson): no fire-and-forget
        // background embedding.
        if let Some(embedder) = &self.embedder {
            match embedder.embed(&body).await {
                Ok(vec) => {
                    let bytes = f32_vec_to_bytes(&vec);
                    self.writer
                        .store_embedding(
                            page_id,
                            bytes,
                            embedder.provider().to_string(),
                            embedder.model().to_string(),
                            embedder.dim(),
                        )
                        .await?;
                }
                Err(e) => {
                    tracing::warn!(error = %e, path = %page_id, "embedding failed; page indexed without it");
                }
            }
        }
        Ok(page_id)
    }
}

/// Input bundle for [`Wiki::write_page`]. Carries the full 3-tuple
/// identity (`workspace_id`, `project_id`, `path`) plus body & metadata.
#[derive(Debug, Clone)]
pub struct WritePageRequest {
    /// Owning workspace.
    pub workspace_id: WorkspaceId,
    /// Owning project.
    pub project_id: ProjectId,
    /// Relative wiki path.
    pub path: PagePath,
    /// Optional frontmatter (JSON object). May be `Null` for no frontmatter.
    pub frontmatter: serde_json::Value,
    /// Markdown body (excluding any frontmatter block).
    pub body: String,
    /// Tier classification.
    pub tier: Tier,
    /// `true` if the user has pinned this page.
    pub pinned: bool,
    /// Optional pre-derived title (used by `apply_batch` to share the
    /// title between the staged markdown file + the store row).
    #[doc(hidden)]
    pub title: Option<String>,
}

fn ai_memory_wiki_error(msg: &str) -> crate::WikiError {
    crate::WikiError::Io(std::io::Error::other(msg.to_string()))
}

fn is_slot_path(path: &PagePath) -> bool {
    path.as_str().starts_with("_slots/")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_memory_store::Store;
    use tempfile::TempDir;

    #[tokio::test]
    async fn project_root_is_wiki_root_joined_with_ws_and_proj() {
        let tmp = TempDir::new().unwrap();
        let store = Store::open(tmp.path()).unwrap();
        let wiki = Wiki::new(tmp.path(), store.writer.clone()).unwrap();
        let ws = WorkspaceId::new();
        let proj = ProjectId::new();
        assert_eq!(
            wiki.project_root(ws, proj),
            tmp.path()
                .join("wiki")
                .join(ws.to_string())
                .join(proj.to_string()),
        );
    }

    fn req(
        ws: WorkspaceId,
        proj: ProjectId,
        path: &str,
        body: &str,
        fm: serde_json::Value,
    ) -> WritePageRequest {
        WritePageRequest {
            workspace_id: ws,
            project_id: proj,
            path: PagePath::new(path).unwrap(),
            frontmatter: fm,
            body: body.into(),
            tier: Tier::Semantic,
            pinned: false,
            title: None,
        }
    }

    #[tokio::test]
    async fn write_page_writes_file_and_indexes_in_store() {
        let tmp = TempDir::new().unwrap();
        let store = Store::open(tmp.path()).unwrap();
        let ws = store
            .writer
            .get_or_create_workspace("default")
            .await
            .unwrap();
        let proj = store
            .writer
            .get_or_create_project(ws, "scratch", None)
            .await
            .unwrap();
        let wiki = Wiki::new(tmp.path(), store.writer.clone()).unwrap();

        let id = wiki
            .write_page(req(
                ws,
                proj,
                "notes/karpathy.md",
                "Karpathy says: compile, do not retrieve.\n",
                serde_json::json!({ "title": "Karpathy LLM Wiki" }),
            ))
            .await
            .unwrap();
        let _ = id; // any non-zero PageId is sufficient

        // File is on disk at the per-project location.
        let on_disk = std::fs::read_to_string(wiki.abs_path(
            ws,
            proj,
            &PagePath::new("notes/karpathy.md").unwrap(),
        ))
        .unwrap();
        assert!(on_disk.starts_with("---\n"));
        assert!(on_disk.contains("title: Karpathy LLM Wiki"));
        assert!(on_disk.contains("Karpathy says"));

        // FTS5 finds it via the store reader.
        let hits = store
            .reader
            .search_pages("karpathy".into(), 5)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].title, "Karpathy LLM Wiki");
        assert!(hits[0].snippet.contains("compile"));
    }

    /// Defence-in-depth: anything that reaches `write_page` gets
    /// scrubbed at the wiki boundary, even if upstream callers (LLM
    /// consolidation output, manual `write-page` CLI input, MCP tool
    /// args) skipped the hook-ingress sanitizer.
    #[tokio::test]
    async fn write_page_scrubs_secrets_at_the_wiki_boundary() {
        let tmp = TempDir::new().unwrap();
        let store = Store::open(tmp.path()).unwrap();
        let ws = store
            .writer
            .get_or_create_workspace("default")
            .await
            .unwrap();
        let proj = store
            .writer
            .get_or_create_project(ws, "scratch", None)
            .await
            .unwrap();
        let wiki = Wiki::new(tmp.path(), store.writer.clone()).unwrap();

        let body = "we agreed to use ANTHROPIC_API_KEY=sk-ant-leak-1234567890abcdef \
                    and the canary id sk-canary-LEAK_ME_PLEASE_xxxxxxxxxxxx — see \
                    postgres://admin:hunter2@db.internal/prod for details";
        wiki.write_page(req(
            ws,
            proj,
            "notes/leaky.md",
            body,
            serde_json::json!({ "title": "leaky" }),
        ))
        .await
        .unwrap();

        let on_disk = std::fs::read_to_string(wiki.abs_path(
            ws,
            proj,
            &PagePath::new("notes/leaky.md").unwrap(),
        ))
        .unwrap();
        // The on-disk page must not contain any of the planted
        // secrets; each should have been replaced with [REDACTED].
        assert!(
            on_disk.contains("[REDACTED]"),
            "expected redaction in: {on_disk}"
        );
        assert!(
            !on_disk.contains("sk-ant-leak"),
            "anthropic key leaked: {on_disk}"
        );
        assert!(
            !on_disk.contains("LEAK_ME_PLEASE"),
            "canary leaked: {on_disk}"
        );
        assert!(
            !on_disk.contains("hunter2"),
            "DB password leaked: {on_disk}"
        );

        // The store-indexed body must also be scrubbed (so FTS5 + the
        // MCP query path never surface the raw secret either).
        let hits = store
            .reader
            .search_pages("REDACTED".into(), 5)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert!(!hits[0].snippet.contains("sk-ant-leak"));
        assert!(!hits[0].snippet.contains("hunter2"));
    }

    #[tokio::test]
    async fn slot_pages_are_pinned_automatically() {
        let tmp = TempDir::new().unwrap();
        let store = Store::open(tmp.path()).unwrap();
        let ws = store
            .writer
            .get_or_create_workspace("default")
            .await
            .unwrap();
        let proj = store
            .writer
            .get_or_create_project(ws, "scratch", None)
            .await
            .unwrap();
        let wiki = Wiki::new(tmp.path(), store.writer.clone()).unwrap();

        wiki.write_page(req(
            ws,
            proj,
            "_slots/current_focus.md",
            "Keep this tiny and durable.",
            serde_json::json!({ "title": "Current focus", "kind": "slot" }),
        ))
        .await
        .unwrap();

        let candidates = store.reader.decay_candidates(ws, proj).await.unwrap();
        assert_eq!(candidates.len(), 1);
        assert!(candidates[0].pinned, "slot pages should be decay-immune");
    }

    #[tokio::test]
    async fn apply_batch_persists_all_pages_in_one_transaction() {
        let tmp = TempDir::new().unwrap();
        let store = Store::open(tmp.path()).unwrap();
        let ws = store
            .writer
            .get_or_create_workspace("default")
            .await
            .unwrap();
        let proj = store
            .writer
            .get_or_create_project(ws, "scratch", None)
            .await
            .unwrap();
        let wiki = Wiki::new(tmp.path(), store.writer.clone()).unwrap();
        let batch: Vec<_> = (0..5)
            .map(|i| WritePageRequest {
                workspace_id: ws,
                project_id: proj,
                path: PagePath::new(format!("batch/{i}.md")).unwrap(),
                frontmatter: serde_json::json!({"title": format!("Page {i}")}),
                body: format!("batch page {i} body line"),
                tier: Tier::Semantic,
                pinned: false,
                title: None,
            })
            .collect();
        let ids = wiki.apply_batch(batch).await.unwrap();
        assert_eq!(ids.len(), 5);
        for i in 0..5 {
            let path = wiki.abs_path(ws, proj, &PagePath::new(format!("batch/{i}.md")).unwrap());
            assert!(path.is_file(), "missing file {i}");
            let body = std::fs::read_to_string(&path).unwrap();
            assert!(body.contains(&format!("Page {i}")));
        }
        let counts = store.reader.status_counts().await.unwrap();
        assert_eq!(counts.pages_latest, 5);
        let hits = store.reader.search_pages("batch".into(), 10).await.unwrap();
        assert_eq!(hits.len(), 5);
    }

    /// Two projects writing the same relative path must produce two distinct
    /// files under their respective UUID-namespaced directories.
    #[tokio::test]
    async fn two_projects_same_path_no_collision() {
        let tmp = TempDir::new().unwrap();
        let store = Store::open(tmp.path()).unwrap();
        let ws = store
            .writer
            .get_or_create_workspace("default")
            .await
            .unwrap();
        let proj_a = store
            .writer
            .get_or_create_project(ws, "alpha", None)
            .await
            .unwrap();
        let proj_b = store
            .writer
            .get_or_create_project(ws, "beta", None)
            .await
            .unwrap();
        let wiki = Wiki::new(tmp.path(), store.writer.clone()).unwrap();

        wiki.write_page(WritePageRequest {
            workspace_id: ws,
            project_id: proj_a,
            path: PagePath::new("decisions/foo.md").unwrap(),
            frontmatter: serde_json::json!({"title": "Alpha decision"}),
            body: "Alpha body".into(),
            tier: Tier::Semantic,
            pinned: false,
            title: None,
        })
        .await
        .unwrap();

        wiki.write_page(WritePageRequest {
            workspace_id: ws,
            project_id: proj_b,
            path: PagePath::new("decisions/foo.md").unwrap(),
            frontmatter: serde_json::json!({"title": "Beta decision"}),
            body: "Beta body".into(),
            tier: Tier::Semantic,
            pinned: false,
            title: None,
        })
        .await
        .unwrap();

        let page = PagePath::new("decisions/foo.md").unwrap();
        let path_a = wiki.abs_path(ws, proj_a, &page);
        let path_b = wiki.abs_path(ws, proj_b, &page);

        assert!(path_a.is_file(), "alpha file must exist");
        assert!(path_b.is_file(), "beta file must exist");
        assert_ne!(path_a, path_b, "distinct paths on disk");

        let content_a = std::fs::read_to_string(&path_a).unwrap();
        let content_b = std::fs::read_to_string(&path_b).unwrap();
        assert!(content_a.contains("Alpha body"), "alpha content intact");
        assert!(content_b.contains("Beta body"), "beta content intact");
    }

    #[tokio::test]
    async fn rewriting_same_body_is_idempotent() {
        let tmp = TempDir::new().unwrap();
        let store = Store::open(tmp.path()).unwrap();
        let ws = store
            .writer
            .get_or_create_workspace("default")
            .await
            .unwrap();
        let proj = store
            .writer
            .get_or_create_project(ws, "scratch", None)
            .await
            .unwrap();
        let wiki = Wiki::new(tmp.path(), store.writer.clone()).unwrap();

        let r = |body: &str| req(ws, proj, "a.md", body, serde_json::json!({ "title": "A" }));

        let a = wiki.write_page(r("body one")).await.unwrap();
        let b = wiki.write_page(r("body one")).await.unwrap();
        assert_eq!(a, b);
        let c = wiki.write_page(r("body two")).await.unwrap();
        assert_ne!(b, c);
    }
}
