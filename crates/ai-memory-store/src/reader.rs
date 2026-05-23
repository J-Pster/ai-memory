//! Read-only connection pool and query helpers.
//!
//! WAL mode lets us have unlimited concurrent readers alongside the single
//! writer, so the pool is mostly about bounding file-descriptor usage and
//! avoiding `Connection::open` overhead on hot paths. Pool eviction is a
//! soft cap: a connection that comes back when the pool is already full
//! is simply dropped.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ai_memory_core::{
    AgentKind, Handoff, HandoffId, HandoffState, Observation, ObservationId, ObservationKind,
    PageId, PagePath, ProjectId, SessionId, WorkspaceId,
};
use parking_lot::Mutex;
use rusqlite::{Connection, OpenFlags, OptionalExtension, params};
use serde::Serialize;
// `ai_memory_core::Tier` is referenced via fully-qualified path inside the
// DecayCandidate struct definition above to avoid a top-level import
// for a single use-site.

use crate::error::{StoreError, StoreResult};

/// One hit returned by [`ReaderPool::search_pages`].
#[derive(Debug, Clone, Serialize)]
pub struct PageHit {
    /// Stable identifier for this page version.
    pub id: PageId,
    /// Relative path within the wiki tree.
    pub path: PagePath,
    /// Page title.
    pub title: String,
    /// FTS5 snippet of the body around the matched terms (HTML-marked).
    pub snippet: String,
    /// FTS5 rank score (lower is better — closer to query terms).
    pub rank: f64,
}

/// Aggregate counts surfaced by [`ReaderPool::status_counts`] and consumed
/// by `ai-memory status`.
#[derive(Debug, Clone, Default, Serialize)]
pub struct StatusCounts {
    /// Pages with `is_latest = 1`.
    pub pages_latest: u64,
    /// All page versions including superseded ones.
    pub pages_all: u64,
    /// Total sessions ever recorded.
    pub sessions: u64,
    /// Total observations across all sessions.
    pub observations: u64,
}

/// Rolling activity counters over a fixed time window. Surfaced by
/// [`ReaderPool::briefing`] so the caller (or an LLM-driven `memory_explore`)
/// can calibrate verbosity against how busy the project's been.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ActivityWindow {
    /// Window size in days (e.g. 7 or 30).
    pub days: u32,
    /// Sessions whose `created_at` falls in the window.
    pub sessions: u64,
    /// Observations whose `created_at` falls in the window.
    pub observations: u64,
    /// Pages whose `updated_at` falls in the window — counts only
    /// `is_latest = 1`. Supersession of an old version into a new one
    /// counts as one update (the new row).
    pub pages_updated: u64,
}

/// Snapshot used by `memory_briefing` and the LLM-driven
/// `memory_explore`. Pure SQL aggregation; no LLM, no schema reads
/// outside the existing `pages` / `sessions` / `observations` /
/// `handoffs` tables.
#[derive(Debug, Clone, Default, Serialize)]
pub struct BriefingSnapshot {
    /// Lifetime totals — same shape `memory_status` returns today.
    pub counts: StatusCounts,
    /// Activity over the last 7 days.
    pub activity_7d: ActivityWindow,
    /// Activity over the last 30 days.
    pub activity_30d: ActivityWindow,
    /// Timestamp of the most recent observation (ISO-8601), or `null`
    /// if no observations exist. The `now - last_observation_at` gap
    /// is the signal `memory_explore` uses to scale its verbosity.
    pub last_observation_at: Option<String>,
    /// Number of open (un-accepted) handoffs.
    pub pending_handoff_count: u64,
    /// All pages currently under `_rules/` — small, surfaced verbatim
    /// because they're the highest-signal type of memory.
    pub rules: Vec<BriefingPage>,
    /// Top-N most-recently-updated `is_latest = 1` pages.
    pub recent_pages: Vec<BriefingPage>,
}

/// Trimmed page view for the briefing — path, title, kind, updated_at
/// timestamp. Body and snippets are intentionally omitted (the caller
/// can follow up with `memory_query` if they need detail).
#[derive(Debug, Clone, Serialize)]
pub struct BriefingPage {
    /// Relative wiki path.
    pub path: String,
    /// Page title (first H1 / frontmatter title).
    pub title: String,
    /// Semantic classification — `decision` / `gotcha` / `rule` / `fact`.
    pub kind: String,
    /// ISO-8601 timestamp of the last update.
    pub updated_at: String,
}

/// One row per (workspace, project) with aggregate stats.
/// Returned by [`ReaderPool::list_projects_with_stats`].
#[derive(Debug, Clone, Serialize)]
pub struct ProjectSummary {
    /// Name of the workspace.
    pub workspace_name: String,
    /// Name of the project within the workspace.
    pub project_name: String,
    /// Number of `is_latest = 1` pages.
    pub page_count: u64,
    /// ISO-8601 timestamp of the newest `updated_at`, or `None` when
    /// the project has no pages yet.
    pub last_updated: Option<String>,
}

/// Page summary for tree-view rendering (no body).
/// Returned by [`ReaderPool::list_pages`].
#[derive(Debug, Clone, Serialize)]
pub struct PageSummary {
    /// Relative path within the wiki tree.
    pub path: String,
    /// Page title.
    pub title: String,
    /// Semantic kind: `fact` | `rule` | `decision` | `gotcha` | …
    pub kind: String,
    /// Memory tier: `working` | `episodic` | `semantic` | `procedural`.
    pub tier: String,
    /// ISO-8601 timestamp of last update.
    pub updated_at: String,
}

/// Full page metadata for the page-view template.
/// Returned by [`ReaderPool::page_meta`].
#[derive(Debug, Clone, Serialize)]
pub struct PageMeta {
    /// Name of the workspace.
    pub workspace_name: String,
    /// Name of the project.
    pub project_name: String,
    /// Relative wiki path.
    pub path: String,
    /// Page title.
    pub title: String,
    /// Semantic kind.
    pub kind: String,
    /// Memory tier.
    pub tier: String,
    /// Whether the page is pinned (decay-immune).
    pub pinned: bool,
    /// ISO-8601 creation timestamp.
    pub created_at: String,
    /// ISO-8601 last-update timestamp.
    pub updated_at: String,
    /// Path of the page this one supersedes, if any.
    pub supersedes: Option<String>,
}

/// Cheap, cloneable read-only connection pool handle.
#[derive(Clone)]
pub struct ReaderPool {
    inner: Arc<Inner>,
}

struct Inner {
    db_path: PathBuf,
    pool: Mutex<Vec<Connection>>,
    soft_cap: usize,
}

impl ReaderPool {
    /// Initialise the pool. Connections are opened lazily on first use.
    ///
    /// # Errors
    /// Currently infallible, but reserved so we can pre-open connections
    /// in a later milestone.
    pub fn new(db_path: &Path, soft_cap: usize) -> StoreResult<Self> {
        Ok(Self {
            inner: Arc::new(Inner {
                db_path: db_path.to_path_buf(),
                pool: Mutex::new(Vec::with_capacity(soft_cap.max(1))),
                soft_cap: soft_cap.max(1),
            }),
        })
    }

    /// Run a synchronous closure against a pooled read-only connection.
    ///
    /// The closure runs on the tokio blocking pool so it never starves the
    /// async runtime. If the pool is empty we open a fresh connection;
    /// on return we keep it only when the pool is below its soft cap.
    ///
    /// # Errors
    /// Returns [`StoreError::PoolPanic`] if the blocking task panics; any
    /// error returned by the closure is propagated unchanged.
    pub async fn with_conn<F, T>(&self, f: F) -> StoreResult<T>
    where
        F: FnOnce(&Connection) -> StoreResult<T> + Send + 'static,
        T: Send + 'static,
    {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let conn = checkout(&inner)?;
            let result = f(&conn);
            checkin(&inner, conn);
            result
        })
        .await
        .map_err(|e| StoreError::PoolPanic(e.to_string()))?
    }

    /// Run a full-text search against the FTS5 index and return the top
    /// matches, limited to `is_latest = 1` rows.
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    pub async fn search_pages(&self, query: String, limit: usize) -> StoreResult<Vec<PageHit>> {
        self.with_conn(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT pages.id, pages.path, pages.title, \
                        snippet(pages_fts, 1, '<mark>', '</mark>', '…', 24) AS snip, \
                        pages_fts.rank \
                 FROM pages_fts \
                 JOIN pages ON pages.rowid = pages_fts.rowid \
                 WHERE pages_fts MATCH ?1 AND pages.is_latest = 1 \
                 ORDER BY pages_fts.rank \
                 LIMIT ?2",
            )?;
            #[allow(clippy::cast_possible_wrap)]
            let rows = stmt.query_map(params![query, limit as i64], |row| {
                let id_bytes: Vec<u8> = row.get(0)?;
                let path: String = row.get(1)?;
                let title: String = row.get(2)?;
                let snippet: String = row.get(3)?;
                let rank: f64 = row.get(4)?;
                Ok((id_bytes, path, title, snippet, rank))
            })?;

            let mut hits = Vec::new();
            for row in rows {
                let (id_bytes, path, title, snippet, rank) = row?;
                hits.push(PageHit {
                    id: PageId::from_slice(&id_bytes)?,
                    path: PagePath::new(path)?,
                    title,
                    snippet,
                    rank,
                });
            }
            Ok(hits)
        })
        .await
    }

    /// Return the N most-recently-updated `is_latest = 1` pages.
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    pub async fn recent_pages(&self, limit: usize) -> StoreResult<Vec<PageHit>> {
        self.with_conn(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT id, path, title, \
                        substr(body, 1, 240) AS snip, \
                        CAST(updated_at AS REAL) AS rank \
                 FROM pages \
                 WHERE is_latest = 1 \
                 ORDER BY updated_at DESC \
                 LIMIT ?1",
            )?;
            #[allow(clippy::cast_possible_wrap)]
            let rows = stmt.query_map(params![limit as i64], |row| {
                let id_bytes: Vec<u8> = row.get(0)?;
                let path: String = row.get(1)?;
                let title: String = row.get(2)?;
                let snippet: String = row.get(3)?;
                let rank: f64 = row.get(4)?;
                Ok((id_bytes, path, title, snippet, rank))
            })?;
            let mut hits = Vec::new();
            for row in rows {
                let (id_bytes, path, title, snippet, rank) = row?;
                hits.push(PageHit {
                    id: PageId::from_slice(&id_bytes)?,
                    path: PagePath::new(path)?,
                    title,
                    snippet,
                    rank,
                });
            }
            Ok(hits)
        })
        .await
    }

    /// Return all observations for the given session, ordered by
    /// `created_at` ascending.
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    pub async fn observations_for_session(
        &self,
        session_id: SessionId,
    ) -> StoreResult<Vec<Observation>> {
        self.with_conn(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT id, session_id, workspace_id, project_id, kind, title, body, \
                        importance, created_at \
                 FROM observations \
                 WHERE session_id = ?1 \
                 ORDER BY created_at ASC",
            )?;
            let rows = stmt.query_map(params![session_id.as_bytes()], row_to_observation)?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r??);
            }
            Ok(out)
        })
        .await
    }

    /// True when any `is_latest = 1` page row anywhere in the store
    /// still references the given wiki path. Used by purge-project to
    /// avoid deleting a wiki file on disk when another project still
    /// has a page row pointing at the same path (the wiki is flat on
    /// disk, so paths can collide across projects).
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    pub async fn path_still_referenced(&self, path: String) -> StoreResult<bool> {
        self.with_conn(move |conn| {
            let n: i64 = conn.query_row(
                "SELECT EXISTS (SELECT 1 FROM pages WHERE path = ?1 AND is_latest = 1)",
                params![path],
                |row| row.get(0),
            )?;
            Ok(n != 0)
        })
        .await
    }

    /// Look up the `(workspace_id, project_id)` a session belongs to.
    /// Returns `None` when no such session row exists.
    ///
    /// Used by the consolidator + lint pass to write pages into the
    /// SESSION'S project, not the server's startup defaults — every
    /// session row carries the project_id the hook router resolved
    /// from its per-cwd basename heuristic, which is the correct
    /// target for any wiki page derived from that session.
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    pub async fn session_project_ids(
        &self,
        session_id: SessionId,
    ) -> StoreResult<Option<(WorkspaceId, ProjectId)>> {
        self.with_conn(move |conn| {
            let mut stmt =
                conn.prepare("SELECT workspace_id, project_id FROM sessions WHERE id = ?1")?;
            let mut rows = stmt.query(params![session_id.as_bytes()])?;
            let Some(row) = rows.next()? else {
                return Ok(None);
            };
            let ws_bytes: Vec<u8> = row.get(0)?;
            let proj_bytes: Vec<u8> = row.get(1)?;
            let ws = WorkspaceId::from_slice(&ws_bytes)
                .map_err(|_| rusqlite::Error::IntegralValueOutOfRange(0, 0))?;
            let proj = ProjectId::from_slice(&proj_bytes)
                .map_err(|_| rusqlite::Error::IntegralValueOutOfRange(1, 0))?;
            Ok(Some((ws, proj)))
        })
        .await
    }

    /// Load every `is_latest=1` page's embedding for the project, but
    /// only when the stored `(provider, model, dim)` matches the
    /// caller's expectation. Mismatched rows are skipped (the
    /// refuse-on-mismatch check is `embedding_meta_for_mismatch`).
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    pub async fn load_embeddings(
        &self,
        workspace_id: WorkspaceId,
        project_id: ProjectId,
        provider: String,
        model: String,
        dim: u32,
    ) -> StoreResult<Vec<StoredEmbedding>> {
        self.with_conn(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT page_embeddings.page_id, page_embeddings.vector, pages.path \
                 FROM page_embeddings \
                 JOIN pages ON pages.id = page_embeddings.page_id \
                 WHERE pages.workspace_id = ?1 \
                   AND pages.project_id = ?2 \
                   AND pages.is_latest = 1 \
                   AND page_embeddings.provider = ?3 \
                   AND page_embeddings.model = ?4 \
                   AND page_embeddings.dim = ?5",
            )?;
            let rows = stmt.query_map(
                params![
                    workspace_id.as_bytes(),
                    project_id.as_bytes(),
                    provider,
                    model,
                    dim,
                ],
                |row| {
                    let id_bytes: Vec<u8> = row.get(0)?;
                    let vec_bytes: Vec<u8> = row.get(1)?;
                    let path: String = row.get(2)?;
                    Ok((id_bytes, vec_bytes, path))
                },
            )?;
            let mut out = Vec::new();
            for r in rows {
                let (id_bytes, vec_bytes, path) = r?;
                let id = PageId::from_slice(&id_bytes)?;
                let path = PagePath::new(path)?;
                let vector = bytes_to_f32_vec(&vec_bytes, dim)?;
                out.push(StoredEmbedding { id, path, vector });
            }
            Ok(out)
        })
        .await
    }

    /// Return any `(provider, model, dim)` triples currently stored
    /// that *don't* match the caller's expectation. An empty vec
    /// means "all clean". Used at startup for the refuse-on-mismatch
    /// (agentmemory #469 lesson).
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    pub async fn embedding_meta_for_mismatch(
        &self,
        provider: String,
        model: String,
        dim: u32,
    ) -> StoreResult<Vec<(String, String, u32, u64)>> {
        self.with_conn(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT provider, model, dim, COUNT(*) \
                 FROM page_embeddings \
                 WHERE NOT (provider = ?1 AND model = ?2 AND dim = ?3) \
                 GROUP BY provider, model, dim",
            )?;
            let rows = stmt.query_map(params![provider, model, dim], |row| {
                let provider: String = row.get(0)?;
                let model: String = row.get(1)?;
                let dim: i64 = row.get(2)?;
                let count: i64 = row.get(3)?;
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                Ok((
                    provider,
                    model,
                    dim as u32,
                    u64::try_from(count).unwrap_or(0),
                ))
            })?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r?);
            }
            Ok(out)
        })
        .await
    }

    /// Return decay-evaluation candidates for the M8 forget sweep.
    ///
    /// Walks `pages` rows with `is_latest = 1` and returns the columns
    /// the forget sweep needs to compute the retention formula. The
    /// sweep itself filters by tier (only `episodic`) + pinned flag,
    /// so this method does not pre-filter -- it just hands the data
    /// over.
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    pub async fn decay_candidates(
        &self,
        workspace_id: WorkspaceId,
        project_id: ProjectId,
    ) -> StoreResult<Vec<DecayCandidate>> {
        self.with_conn(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT id, path, tier, pinned, updated_at, access_count, last_accessed_at, \
                        frontmatter_json \
                 FROM pages \
                 WHERE workspace_id = ?1 AND project_id = ?2 AND is_latest = 1",
            )?;
            let rows = stmt.query_map(
                params![workspace_id.as_bytes(), project_id.as_bytes()],
                row_to_decay_candidate,
            )?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r??);
            }
            Ok(out)
        })
        .await
    }

    /// Hybrid search: RRF-fuse FTS5 results with cosine-similarity
    /// over the stored embeddings of the matching `(provider, model,
    /// dim)`. Returns the top-`limit` pages by fused score.
    ///
    /// When `query_vec` is `None`, falls back to pure FTS5
    /// (equivalent to [`Self::search_pages`]). When no embeddings
    /// exist for the configured triple, also falls back to pure FTS5.
    ///
    /// k=60 is the canonical RRF constant.
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    #[allow(clippy::too_many_arguments)]
    pub async fn hybrid_search(
        &self,
        workspace_id: WorkspaceId,
        project_id: ProjectId,
        query: String,
        query_vec: Option<Vec<f32>>,
        provider: String,
        model: String,
        dim: u32,
        limit: usize,
    ) -> StoreResult<Vec<PageHit>> {
        // Fetch FTS5 hits first.
        let fts_hits = self.search_pages(query, limit * 2).await?;
        let Some(qv) = query_vec else {
            // No query vector → pure FTS5.
            let mut out = fts_hits;
            out.truncate(limit);
            return Ok(out);
        };
        // Compute cosines against all stored embeddings.
        let stored = self
            .load_embeddings(workspace_id, project_id, provider, model, dim)
            .await?;
        if stored.is_empty() {
            let mut out = fts_hits;
            out.truncate(limit);
            return Ok(out);
        }
        let mut vec_hits: Vec<(PageId, PagePath, f32)> = stored
            .iter()
            .map(|s| {
                let score = dot(&qv, &s.vector);
                (s.id, s.path.clone(), score)
            })
            .collect();
        vec_hits.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        vec_hits.truncate(limit * 2);

        // RRF fuse: score(d) = Σ 1/(k + rank_i(d)) over rankers.
        let k = 60.0_f64;
        let mut fused: std::collections::HashMap<PageId, (PagePath, String, String, f64, f64)> =
            std::collections::HashMap::new();

        for (rank, h) in fts_hits.iter().enumerate() {
            let contrib = 1.0 / (k + (rank + 1) as f64);
            fused
                .entry(h.id)
                .and_modify(|entry| entry.3 += contrib)
                .or_insert((
                    h.path.clone(),
                    h.title.clone(),
                    h.snippet.clone(),
                    contrib,
                    h.rank,
                ));
        }
        for (rank, (id, path, _score)) in vec_hits.iter().enumerate() {
            let contrib = 1.0 / (k + (rank + 1) as f64);
            fused
                .entry(*id)
                .and_modify(|entry| entry.3 += contrib)
                .or_insert((path.clone(), String::new(), String::new(), contrib, 0.0));
        }

        let mut out: Vec<PageHit> = fused
            .into_iter()
            .map(|(id, (path, title, snippet, fused_rank, _orig))| PageHit {
                id,
                path,
                title,
                snippet,
                rank: -fused_rank, // lower = better (matches FTS5 convention)
            })
            .collect();
        out.sort_by(|a, b| {
            a.rank
                .partial_cmp(&b.rank)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out.truncate(limit);
        Ok(out)
    }

    /// Return the latest open handoff for the project, optionally
    /// filtered to a specific `cwd`.
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    pub async fn latest_open_handoff(
        &self,
        workspace_id: WorkspaceId,
        project_id: ProjectId,
        cwd_filter: Option<String>,
    ) -> StoreResult<Option<Handoff>> {
        self.with_conn(move |conn| {
            let mut stmt: rusqlite::Statement<'_> = if let Some(_cwd) = cwd_filter.as_deref() {
                conn.prepare(
                    "SELECT id, workspace_id, project_id, from_session_id, from_agent, to_agent, \
                            cwd, summary, open_questions, next_steps, files_touched, state, \
                            created_at, accepted_by, accepted_at, accepted_by_session \
                     FROM handoffs \
                     WHERE workspace_id = ?1 AND project_id = ?2 AND cwd = ?3 \
                       AND state = 'open' \
                     ORDER BY created_at DESC LIMIT 1",
                )?
            } else {
                conn.prepare(
                    "SELECT id, workspace_id, project_id, from_session_id, from_agent, to_agent, \
                            cwd, summary, open_questions, next_steps, files_touched, state, \
                            created_at, accepted_by, accepted_at, accepted_by_session \
                     FROM handoffs \
                     WHERE workspace_id = ?1 AND project_id = ?2 AND state = 'open' \
                     ORDER BY created_at DESC LIMIT 1",
                )?
            };
            let row_opt = if let Some(c) = cwd_filter.as_deref() {
                stmt.query_row(
                    params![workspace_id.as_bytes(), project_id.as_bytes(), c],
                    row_to_handoff,
                )
                .optional()?
            } else {
                stmt.query_row(
                    params![workspace_id.as_bytes(), project_id.as_bytes()],
                    row_to_handoff,
                )
                .optional()?
            };
            row_opt.transpose()
        })
        .await
    }

    /// Look up a handoff by id.
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    pub async fn handoff_by_id(&self, handoff_id: HandoffId) -> StoreResult<Option<Handoff>> {
        self.with_conn(move |conn| {
            let row = conn
                .query_row(
                    "SELECT id, workspace_id, project_id, from_session_id, from_agent, to_agent, \
                            cwd, summary, open_questions, next_steps, files_touched, state, \
                            created_at, accepted_by, accepted_at, accepted_by_session \
                     FROM handoffs WHERE id = ?1",
                    params![handoff_id.as_bytes()],
                    row_to_handoff,
                )
                .optional()?;
            row.transpose()
        })
        .await
    }

    /// Snapshot the database to `dest_path` using SQLite's online backup
    /// API. The source DB stays writable for the duration of the copy.
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    pub async fn snapshot_to(&self, dest_path: PathBuf) -> StoreResult<()> {
        self.with_conn(move |conn| {
            conn.backup(rusqlite::DatabaseName::Main, &dest_path, None)
                .map_err(StoreError::from)
        })
        .await
    }

    /// Assemble a [`BriefingSnapshot`] — pure SQL aggregation across
    /// the `pages` / `sessions` / `observations` / `handoffs` tables.
    /// No LLM, no schema reads outside what's already there.
    ///
    /// `recent_pages_limit` caps the `recent_pages` array; pass a
    /// small number (5-20) — this is meant to be skimmed, not paged
    /// through.
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    #[allow(clippy::too_many_lines)]
    pub async fn briefing(&self, recent_pages_limit: usize) -> StoreResult<BriefingSnapshot> {
        let recent_limit = recent_pages_limit.clamp(1, 100) as i64;
        self.with_conn(move |conn| {
            let now_us = jiff::Timestamp::now().as_microsecond();
            let day_us: i64 = 86_400 * 1_000_000;
            let cutoff_7d = now_us - 7 * day_us;
            let cutoff_30d = now_us - 30 * day_us;

            let counts = StatusCounts {
                pages_latest: count(conn, "SELECT COUNT(*) FROM pages WHERE is_latest = 1")?,
                pages_all: count(conn, "SELECT COUNT(*) FROM pages")?,
                sessions: count(conn, "SELECT COUNT(*) FROM sessions")?,
                observations: count(conn, "SELECT COUNT(*) FROM observations")?,
            };

            let activity_7d = window_activity(conn, 7, cutoff_7d)?;
            let activity_30d = window_activity(conn, 30, cutoff_30d)?;

            let last_observation_at: Option<i64> = conn
                .query_row("SELECT MAX(created_at) FROM observations", [], |row| {
                    row.get::<_, Option<i64>>(0)
                })
                .optional()?
                .flatten();
            let last_observation_at = last_observation_at
                .and_then(|us| jiff::Timestamp::from_microsecond(us).ok())
                .map(|ts| ts.to_string());

            let pending_handoff_count: u64 =
                count(conn, "SELECT COUNT(*) FROM handoffs WHERE state = 'open'")?;

            // Rules: any `is_latest = 1` page under `_rules/`.
            // Routed there automatically by the consolidator when
            // `kind = "rule"` — see consolidator.rs::slugify_for_rule.
            let mut rules_stmt = conn.prepare(
                "SELECT path, title, \
                        COALESCE(json_extract(frontmatter_json, '$.kind'), 'fact') AS kind, \
                        updated_at \
                 FROM pages \
                 WHERE is_latest = 1 AND path LIKE '_rules/%' \
                 ORDER BY updated_at DESC",
            )?;
            let rules: Vec<BriefingPage> = rules_stmt
                .query_map([], briefing_page_from_row)?
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?;

            let mut recent_stmt = conn.prepare(
                "SELECT path, title, \
                        COALESCE(json_extract(frontmatter_json, '$.kind'), 'fact') AS kind, \
                        updated_at \
                 FROM pages \
                 WHERE is_latest = 1 \
                 ORDER BY updated_at DESC \
                 LIMIT ?1",
            )?;
            let recent_pages: Vec<BriefingPage> = recent_stmt
                .query_map(params![recent_limit], briefing_page_from_row)?
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?;

            Ok(BriefingSnapshot {
                counts,
                activity_7d,
                activity_30d,
                last_observation_at,
                pending_handoff_count,
                rules,
                recent_pages,
            })
        })
        .await
    }

    /// Look up a page's workspace and project names by its path (across all
    /// workspaces and projects). Returns the first `is_latest = 1` match.
    ///
    /// Used by the web search handler to resolve workspace/project for a hit
    /// without a per-hit SQL join.
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    pub async fn page_meta_by_path(&self, path: &str) -> StoreResult<Option<PageMeta>> {
        let path = path.to_owned();
        self.with_conn(move |conn| {
            let row_opt = conn
                .query_row(
                    "SELECT w.name, p.name, pg.path, pg.title, \
                            COALESCE(json_extract(pg.frontmatter_json, '$.kind'), 'fact'), \
                            pg.tier, pg.pinned, pg.created_at, pg.updated_at, \
                            sp.path AS supersedes_path \
                     FROM pages pg \
                     JOIN projects p ON p.id = pg.project_id \
                     JOIN workspaces w ON w.id = pg.workspace_id \
                     LEFT JOIN pages sp ON sp.id = pg.supersedes \
                     WHERE pg.path = ?1 AND pg.is_latest = 1 \
                     LIMIT 1",
                    params![path],
                    |row| {
                        let workspace_name: String = row.get(0)?;
                        let project_name: String = row.get(1)?;
                        let path: String = row.get(2)?;
                        let title: String = row.get(3)?;
                        let kind: String = row.get(4)?;
                        let tier: String = row.get(5)?;
                        let pinned: i64 = row.get(6)?;
                        let created_us: i64 = row.get(7)?;
                        let updated_us: i64 = row.get(8)?;
                        let supersedes_path: Option<String> = row.get(9)?;
                        Ok((
                            workspace_name,
                            project_name,
                            path,
                            title,
                            kind,
                            tier,
                            pinned,
                            created_us,
                            updated_us,
                            supersedes_path,
                        ))
                    },
                )
                .optional()?;

            let Some((
                workspace_name,
                project_name,
                path,
                title,
                kind,
                tier,
                pinned,
                created_us,
                updated_us,
                supersedes,
            )) = row_opt
            else {
                return Ok(None);
            };

            let created_at = jiff::Timestamp::from_microsecond(created_us)
                .map(|ts| ts.to_string())
                .unwrap_or_default();
            let updated_at = jiff::Timestamp::from_microsecond(updated_us)
                .map(|ts| ts.to_string())
                .unwrap_or_default();

            Ok(Some(PageMeta {
                workspace_name,
                project_name,
                path,
                title,
                kind,
                tier,
                pinned: pinned != 0,
                created_at,
                updated_at,
                supersedes,
            }))
        })
        .await
    }

    /// Return one row per (workspace, project) with page-count and
    /// last-updated aggregates. Used by the web UI project-list view.
    ///
    /// Only `is_latest = 1` pages are counted.
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    pub async fn list_projects_with_stats(&self) -> StoreResult<Vec<ProjectSummary>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT w.name AS workspace_name, \
                        p.name AS project_name, \
                        COUNT(pg.id) AS page_count, \
                        MAX(pg.updated_at) AS last_updated_us \
                 FROM workspaces w \
                 JOIN projects p ON p.workspace_id = w.id \
                 LEFT JOIN pages pg ON pg.project_id = p.id AND pg.is_latest = 1 \
                 GROUP BY w.id, p.id \
                 ORDER BY last_updated_us DESC NULLS LAST",
            )?;
            let rows = stmt.query_map([], |row| {
                let workspace_name: String = row.get(0)?;
                let project_name: String = row.get(1)?;
                let page_count: i64 = row.get(2)?;
                let last_updated_us: Option<i64> = row.get(3)?;
                Ok((workspace_name, project_name, page_count, last_updated_us))
            })?;
            let mut out = Vec::new();
            for r in rows {
                let (workspace_name, project_name, page_count, last_updated_us) = r?;
                let last_updated = last_updated_us
                    .and_then(|us| jiff::Timestamp::from_microsecond(us).ok())
                    .map(|ts| ts.to_string());
                #[allow(clippy::cast_sign_loss)]
                out.push(ProjectSummary {
                    workspace_name,
                    project_name,
                    page_count: page_count.max(0) as u64,
                    last_updated,
                });
            }
            Ok(out)
        })
        .await
    }

    /// All `is_latest = 1` pages under a given (workspace, project),
    /// ordered by path ascending. Used by the web UI tree view.
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    pub async fn list_pages(
        &self,
        workspace: &str,
        project: &str,
    ) -> StoreResult<Vec<PageSummary>> {
        let workspace = workspace.to_owned();
        let project = project.to_owned();
        self.with_conn(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT pg.path, pg.title, \
                        COALESCE(json_extract(pg.frontmatter_json, '$.kind'), 'fact') AS kind, \
                        pg.tier, pg.updated_at \
                 FROM pages pg \
                 JOIN projects p ON p.id = pg.project_id \
                 JOIN workspaces w ON w.id = pg.workspace_id \
                 WHERE w.name = ?1 AND p.name = ?2 AND pg.is_latest = 1 \
                 ORDER BY pg.path ASC",
            )?;
            let rows = stmt.query_map(params![workspace, project], |row| {
                let path: String = row.get(0)?;
                let title: String = row.get(1)?;
                let kind: String = row.get(2)?;
                let tier: String = row.get(3)?;
                let updated_us: i64 = row.get(4)?;
                Ok((path, title, kind, tier, updated_us))
            })?;
            let mut out = Vec::new();
            for r in rows {
                let (path, title, kind, tier, updated_us) = r?;
                let updated_at = jiff::Timestamp::from_microsecond(updated_us)
                    .map(|ts| ts.to_string())
                    .unwrap_or_default();
                out.push(PageSummary {
                    path,
                    title,
                    kind,
                    tier,
                    updated_at,
                });
            }
            Ok(out)
        })
        .await
    }

    /// Full page metadata for the page-view template (body comes from
    /// `Wiki::read_page`). Returns `None` when no `is_latest = 1` row
    /// matches the given path.
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    pub async fn page_meta(
        &self,
        workspace: &str,
        project: &str,
        page_path: &str,
    ) -> StoreResult<Option<PageMeta>> {
        let workspace = workspace.to_owned();
        let project = project.to_owned();
        let page_path = page_path.to_owned();
        self.with_conn(move |conn| {
            let row_opt = conn
                .query_row(
                    "SELECT w.name, p.name, pg.path, pg.title, \
                            COALESCE(json_extract(pg.frontmatter_json, '$.kind'), 'fact'), \
                            pg.tier, pg.pinned, pg.created_at, pg.updated_at, \
                            sp.path AS supersedes_path \
                     FROM pages pg \
                     JOIN projects p ON p.id = pg.project_id \
                     JOIN workspaces w ON w.id = pg.workspace_id \
                     LEFT JOIN pages sp ON sp.id = pg.supersedes \
                     WHERE w.name = ?1 AND p.name = ?2 AND pg.path = ?3 AND pg.is_latest = 1",
                    params![workspace, project, page_path],
                    |row| {
                        let workspace_name: String = row.get(0)?;
                        let project_name: String = row.get(1)?;
                        let path: String = row.get(2)?;
                        let title: String = row.get(3)?;
                        let kind: String = row.get(4)?;
                        let tier: String = row.get(5)?;
                        let pinned: i64 = row.get(6)?;
                        let created_us: i64 = row.get(7)?;
                        let updated_us: i64 = row.get(8)?;
                        let supersedes_path: Option<String> = row.get(9)?;
                        Ok((
                            workspace_name,
                            project_name,
                            path,
                            title,
                            kind,
                            tier,
                            pinned,
                            created_us,
                            updated_us,
                            supersedes_path,
                        ))
                    },
                )
                .optional()?;

            let Some((
                workspace_name,
                project_name,
                path,
                title,
                kind,
                tier,
                pinned,
                created_us,
                updated_us,
                supersedes,
            )) = row_opt
            else {
                return Ok(None);
            };

            let created_at = jiff::Timestamp::from_microsecond(created_us)
                .map(|ts| ts.to_string())
                .unwrap_or_default();
            let updated_at = jiff::Timestamp::from_microsecond(updated_us)
                .map(|ts| ts.to_string())
                .unwrap_or_default();

            Ok(Some(PageMeta {
                workspace_name,
                project_name,
                path,
                title,
                kind,
                tier,
                pinned: pinned != 0,
                created_at,
                updated_at,
                supersedes,
            }))
        })
        .await
    }

    /// Look up a workspace id by name without creating it.
    ///
    /// Returns `None` when no workspace with the given name exists.
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    pub async fn find_workspace(&self, name: String) -> StoreResult<Option<WorkspaceId>> {
        self.with_conn(move |conn| {
            let row_opt = conn
                .query_row(
                    "SELECT id FROM workspaces WHERE name = ?1",
                    params![name],
                    |row| {
                        let bytes: Vec<u8> = row.get(0)?;
                        Ok(bytes)
                    },
                )
                .optional()?;
            row_opt
                .map(|bytes| WorkspaceId::from_slice(&bytes).map_err(StoreError::from))
                .transpose()
        })
        .await
    }

    /// Look up a project id by `(workspace_id, name)` without creating it.
    ///
    /// Returns `None` when no project with the given name exists in the workspace.
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    pub async fn find_project(
        &self,
        workspace_id: WorkspaceId,
        name: String,
    ) -> StoreResult<Option<ProjectId>> {
        self.with_conn(move |conn| {
            let row_opt = conn
                .query_row(
                    "SELECT id FROM projects WHERE workspace_id = ?1 AND name = ?2",
                    params![workspace_id.as_bytes(), name],
                    |row| {
                        let bytes: Vec<u8> = row.get(0)?;
                        Ok(bytes)
                    },
                )
                .optional()?;
            row_opt
                .map(|bytes| ProjectId::from_slice(&bytes).map_err(StoreError::from))
                .transpose()
        })
        .await
    }

    /// Return aggregate counts for the `status` view.
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    pub async fn status_counts(&self) -> StoreResult<StatusCounts> {
        self.with_conn(|conn| {
            let pages_latest: u64 = count(conn, "SELECT COUNT(*) FROM pages WHERE is_latest = 1")?;
            let pages_all: u64 = count(conn, "SELECT COUNT(*) FROM pages")?;
            let sessions: u64 = count(conn, "SELECT COUNT(*) FROM sessions")?;
            let observations: u64 = count(conn, "SELECT COUNT(*) FROM observations")?;
            Ok(StatusCounts {
                pages_latest,
                pages_all,
                sessions,
                observations,
            })
        })
        .await
    }
}

fn row_to_observation(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoreResult<Observation>> {
    let id_bytes: Vec<u8> = row.get(0)?;
    let session_bytes: Vec<u8> = row.get(1)?;
    let workspace_bytes: Vec<u8> = row.get(2)?;
    let project_bytes: Vec<u8> = row.get(3)?;
    let kind_str: String = row.get(4)?;
    let title: String = row.get(5)?;
    let body: String = row.get(6)?;
    let importance: i64 = row.get(7)?;
    let created_us: i64 = row.get(8)?;
    Ok(materialise_observation(
        id_bytes,
        session_bytes,
        workspace_bytes,
        project_bytes,
        kind_str,
        title,
        body,
        importance,
        created_us,
    ))
}

#[allow(clippy::too_many_arguments)]
fn materialise_observation(
    id_bytes: Vec<u8>,
    session_bytes: Vec<u8>,
    workspace_bytes: Vec<u8>,
    project_bytes: Vec<u8>,
    kind_str: String,
    title: String,
    body: String,
    importance: i64,
    created_us: i64,
) -> StoreResult<Observation> {
    Ok(Observation {
        id: ObservationId::from_slice(&id_bytes)?,
        session_id: SessionId::from_slice(&session_bytes)?,
        workspace_id: WorkspaceId::from_slice(&workspace_bytes)?,
        project_id: ProjectId::from_slice(&project_bytes)?,
        kind: kind_str
            .parse::<ObservationKind>()
            .map_err(StoreError::from)?,
        title,
        body,
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        importance: importance.clamp(1, 10) as u8,
        created_at: jiff::Timestamp::from_microsecond(created_us).map_err(|e| {
            StoreError::Memory(ai_memory_core::MemoryError::MalformedRecord(format!(
                "bad timestamp: {e}"
            )))
        })?,
    })
}

/// One stored embedding row, materialised for the vector path.
#[derive(Debug, Clone)]
pub struct StoredEmbedding {
    /// Page identifier (always the `is_latest=1` row's id).
    pub id: PageId,
    /// Relative wiki path.
    pub path: PagePath,
    /// Unit-normalised vector.
    pub vector: Vec<f32>,
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn bytes_to_f32_vec(bytes: &[u8], dim: u32) -> StoreResult<Vec<f32>> {
    let expected = (dim as usize) * 4;
    if bytes.len() != expected {
        return Err(StoreError::Memory(
            ai_memory_core::MemoryError::MalformedRecord(format!(
                "embedding bytes {} != expected {}",
                bytes.len(),
                expected
            )),
        ));
    }
    let mut out = Vec::with_capacity(dim as usize);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

/// Pack a `&[f32]` into little-endian bytes for storage. Inverse of
/// [`bytes_to_f32_vec`].
#[must_use]
pub fn f32_vec_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// One row's worth of input for the M8 retention formula.
#[derive(Debug, Clone, Serialize)]
pub struct DecayCandidate {
    /// Stable identifier.
    pub id: PageId,
    /// Relative wiki path.
    pub path: PagePath,
    /// Tier (the sweep only considers `episodic`).
    pub tier: ai_memory_core::Tier,
    /// Pinned flag — true means "never decay".
    pub pinned: bool,
    /// `updated_at` in microseconds since epoch.
    pub updated_at_us: i64,
    /// Total query/access hits.
    pub access_count: u32,
    /// `last_accessed_at` in microseconds since epoch, or `None` if never accessed.
    pub last_accessed_at_us: Option<i64>,
    /// Frontmatter JSON; the sweep peeks at it for an explicit
    /// `pinned: true` (which overrides the schema flag).
    pub frontmatter_json: String,
}

fn row_to_decay_candidate(
    row: &rusqlite::Row<'_>,
) -> rusqlite::Result<StoreResult<DecayCandidate>> {
    let id_bytes: Vec<u8> = row.get(0)?;
    let path: String = row.get(1)?;
    let tier_str: String = row.get(2)?;
    let pinned: i64 = row.get(3)?;
    let updated_at_us: i64 = row.get(4)?;
    let access_count: i64 = row.get(5)?;
    let last_accessed_at_us: Option<i64> = row.get(6)?;
    let frontmatter_json: String = row.get(7)?;
    Ok(materialise_decay_candidate(
        id_bytes,
        path,
        tier_str,
        pinned,
        updated_at_us,
        access_count,
        last_accessed_at_us,
        frontmatter_json,
    ))
}

#[allow(clippy::too_many_arguments)]
fn materialise_decay_candidate(
    id_bytes: Vec<u8>,
    path: String,
    tier_str: String,
    pinned: i64,
    updated_at_us: i64,
    access_count: i64,
    last_accessed_at_us: Option<i64>,
    frontmatter_json: String,
) -> StoreResult<DecayCandidate> {
    Ok(DecayCandidate {
        id: PageId::from_slice(&id_bytes)?,
        path: PagePath::new(path)?,
        tier: tier_str
            .parse::<ai_memory_core::Tier>()
            .map_err(StoreError::from)?,
        pinned: pinned != 0,
        updated_at_us,
        access_count: u32::try_from(access_count.max(0)).unwrap_or(u32::MAX),
        last_accessed_at_us,
        frontmatter_json,
    })
}

fn row_to_handoff(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoreResult<Handoff>> {
    let id_bytes: Vec<u8> = row.get(0)?;
    let ws_bytes: Vec<u8> = row.get(1)?;
    let pj_bytes: Vec<u8> = row.get(2)?;
    let from_session_bytes: Option<Vec<u8>> = row.get(3)?;
    let from_agent: String = row.get(4)?;
    let to_agent: Option<String> = row.get(5)?;
    let cwd: Option<String> = row.get(6)?;
    let summary: String = row.get(7)?;
    let open_q_json: String = row.get(8)?;
    let next_s_json: String = row.get(9)?;
    let files_json: String = row.get(10)?;
    let state: String = row.get(11)?;
    let created_us: i64 = row.get(12)?;
    let accepted_by: Option<String> = row.get(13)?;
    let accepted_at_us: Option<i64> = row.get(14)?;
    let accepted_by_session_bytes: Option<Vec<u8>> = row.get(15)?;
    Ok(materialise_handoff(
        id_bytes,
        ws_bytes,
        pj_bytes,
        from_session_bytes,
        from_agent,
        to_agent,
        cwd,
        summary,
        open_q_json,
        next_s_json,
        files_json,
        state,
        created_us,
        accepted_by,
        accepted_at_us,
        accepted_by_session_bytes,
    ))
}

#[allow(clippy::too_many_arguments)]
fn materialise_handoff(
    id_bytes: Vec<u8>,
    ws_bytes: Vec<u8>,
    pj_bytes: Vec<u8>,
    from_session_bytes: Option<Vec<u8>>,
    from_agent: String,
    to_agent: Option<String>,
    cwd: Option<String>,
    summary: String,
    open_q_json: String,
    next_s_json: String,
    files_json: String,
    state: String,
    created_us: i64,
    accepted_by: Option<String>,
    accepted_at_us: Option<i64>,
    accepted_by_session_bytes: Option<Vec<u8>>,
) -> StoreResult<Handoff> {
    let open_questions: Vec<String> = serde_json::from_str(&open_q_json)?;
    let next_steps: Vec<String> = serde_json::from_str(&next_s_json)?;
    let files_touched: Vec<String> = serde_json::from_str(&files_json)?;
    let from_session = from_session_bytes
        .as_deref()
        .map(SessionId::from_slice)
        .transpose()?;
    let accepted_session = accepted_by_session_bytes
        .as_deref()
        .map(SessionId::from_slice)
        .transpose()?;
    Ok(Handoff {
        id: HandoffId::from_slice(&id_bytes)?,
        workspace_id: WorkspaceId::from_slice(&ws_bytes)?,
        project_id: ProjectId::from_slice(&pj_bytes)?,
        from_session_id: from_session,
        from_agent: parse_agent(&from_agent),
        to_agent: to_agent.as_deref().map(parse_agent),
        cwd,
        summary,
        open_questions,
        next_steps,
        files_touched,
        state: state.parse::<HandoffState>().map_err(StoreError::from)?,
        created_at: jiff::Timestamp::from_microsecond(created_us).map_err(|e| {
            StoreError::Memory(ai_memory_core::MemoryError::MalformedRecord(format!(
                "bad created_at: {e}"
            )))
        })?,
        accepted_by: accepted_by.as_deref().map(parse_agent),
        accepted_at: accepted_at_us
            .map(jiff::Timestamp::from_microsecond)
            .transpose()
            .map_err(|e| {
                StoreError::Memory(ai_memory_core::MemoryError::MalformedRecord(format!(
                    "bad accepted_at: {e}"
                )))
            })?,
        accepted_by_session: accepted_session,
    })
}

fn parse_agent(s: &str) -> AgentKind {
    match s {
        "claude-code" => AgentKind::ClaudeCode,
        "codex" => AgentKind::Codex,
        "open-code" => AgentKind::OpenCode,
        _ => AgentKind::Other,
    }
}

fn count(conn: &Connection, sql: &str) -> StoreResult<u64> {
    let n: Option<i64> = conn.query_row(sql, [], |row| row.get(0)).optional()?;
    Ok(u64::try_from(n.unwrap_or(0)).unwrap_or(0))
}

/// Count rows in a time-bounded window. Used by [`ReaderPool::briefing`]
/// to compute "last 7 days" / "last 30 days" activity slices.
fn window_activity(conn: &Connection, days: u32, cutoff_us: i64) -> StoreResult<ActivityWindow> {
    let count_since = |sql: &str| -> StoreResult<u64> {
        let n: Option<i64> = conn
            .query_row(sql, params![cutoff_us], |row| row.get(0))
            .optional()?;
        Ok(u64::try_from(n.unwrap_or(0)).unwrap_or(0))
    };
    Ok(ActivityWindow {
        days,
        // `sessions` schema uses `started_at`, not `created_at` — easy
        // to forget because the other tables all use `created_at`.
        sessions: count_since("SELECT COUNT(*) FROM sessions WHERE started_at > ?1")?,
        observations: count_since("SELECT COUNT(*) FROM observations WHERE created_at > ?1")?,
        pages_updated: count_since(
            "SELECT COUNT(*) FROM pages WHERE is_latest = 1 AND updated_at > ?1",
        )?,
    })
}

/// Materialise one row from the briefing's recent-pages / rules queries
/// into a [`BriefingPage`]. The row shape is `(path, title, kind,
/// updated_at_us)` — all queries above MUST select those columns in
/// that order.
fn briefing_page_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoreResult<BriefingPage>> {
    let path: String = row.get(0)?;
    let title: String = row.get(1)?;
    let kind: String = row.get(2)?;
    let updated_us: i64 = row.get(3)?;
    Ok(jiff::Timestamp::from_microsecond(updated_us)
        .map(|ts| BriefingPage {
            path,
            title,
            kind,
            updated_at: ts.to_string(),
        })
        .map_err(|e| {
            StoreError::Memory(ai_memory_core::MemoryError::MalformedRecord(format!(
                "bad updated_at: {e}"
            )))
        }))
}

fn checkout(inner: &Inner) -> StoreResult<Connection> {
    if let Some(conn) = inner.pool.lock().pop() {
        return Ok(conn);
    }
    open_read_only(&inner.db_path)
}

fn checkin(inner: &Inner, conn: Connection) {
    let mut pool = inner.pool.lock();
    if pool.len() < inner.soft_cap {
        pool.push(conn);
    }
}

fn open_read_only(path: &Path) -> StoreResult<Connection> {
    let flags = OpenFlags::SQLITE_OPEN_READ_ONLY
        | OpenFlags::SQLITE_OPEN_URI
        | OpenFlags::SQLITE_OPEN_NO_MUTEX;
    let conn = Connection::open_with_flags(path, flags)?;
    conn.pragma_update(None, "busy_timeout", 5_000)?;
    Ok(conn)
}
