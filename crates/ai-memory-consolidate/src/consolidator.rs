//! Single-page session consolidator.
//!
//! Reads the observation log for a session, asks the configured LLM
//! for an updated [`ConsolidatedPage`], then writes it via
//! [`Wiki::write_page`] so the supersession chain + git auto-commit
//! kicks in automatically.

use std::sync::Arc;

use ai_memory_core::{
    Observation, ObservationKind, PagePath, ProjectId, SessionId, Tier, WorkspaceId,
};
use ai_memory_llm::{ChatMessage, ChatRequest, LlmError, LlmProvider, Role, complete_structured};
use ai_memory_store::{ReaderPool, WriterHandle};
use ai_memory_wiki::{Wiki, WritePageRequest};
use thiserror::Error;
use tracing::{debug, info};

use crate::types::{ConsolidatedBatch, ConsolidatedPage, ConsolidationOutcome};

/// Errors raised by the consolidator.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ConsolidatorError {
    /// Domain-level error (e.g. invalid `PagePath`).
    #[error(transparent)]
    Memory(#[from] ai_memory_core::MemoryError),

    /// Underlying store error.
    #[error(transparent)]
    Store(#[from] ai_memory_store::StoreError),

    /// Underlying wiki error.
    #[error(transparent)]
    Wiki(#[from] ai_memory_wiki::WikiError),

    /// Underlying LLM error.
    #[error(transparent)]
    Llm(#[from] LlmError),

    /// JSON error.
    #[error("serde: {0}")]
    Serde(String),

    /// Session was not found.
    #[error("session not found: {0}")]
    SessionNotFound(SessionId),

    /// Session had no observations to consolidate.
    #[error("session {0} has no observations")]
    EmptySession(SessionId),
}

impl From<serde_json::Error> for ConsolidatorError {
    fn from(value: serde_json::Error) -> Self {
        Self::Serde(value.to_string())
    }
}

/// Result alias used by the consolidator.
pub type ConsolidatorResult<T> = Result<T, ConsolidatorError>;

/// Karpathy-style single-page consolidator. Holds handles to the
/// store, wiki, and LLM provider so it can be reused across many
/// `consolidate_session` calls.
pub struct Consolidator {
    reader: ReaderPool,
    writer: WriterHandle,
    wiki: Wiki,
    llm: Arc<dyn LlmProvider>,
    workspace_id: WorkspaceId,
    project_id: ProjectId,
}

impl Consolidator {
    /// Construct a consolidator. Caller is responsible for selecting
    /// the LLM provider via the `ai-memory-llm` factory.
    #[must_use]
    pub fn new(
        reader: ReaderPool,
        writer: WriterHandle,
        wiki: Wiki,
        llm: Arc<dyn LlmProvider>,
        workspace_id: WorkspaceId,
        project_id: ProjectId,
    ) -> Self {
        Self {
            reader,
            writer,
            wiki,
            llm,
            workspace_id,
            project_id,
        }
    }

    /// Consolidate a single session into a refreshed
    /// `sessions/<id>.md` page.
    ///
    /// # Errors
    /// Returns [`ConsolidatorError`] for any store, wiki, or LLM
    /// failure.
    pub async fn consolidate_session(
        &self,
        session_id: SessionId,
        dry_run: bool,
    ) -> ConsolidatorResult<ConsolidationOutcome> {
        let observations = self.reader.observations_for_session(session_id).await?;
        if observations.is_empty() {
            return Err(ConsolidatorError::EmptySession(session_id));
        }

        let path = PagePath::new(format!("sessions/{session_id}.md"))?;
        let current_body = self
            .wiki
            .read_page(&path)
            .map(|md| md.body)
            .unwrap_or_default();
        let request = build_request(session_id, &observations, &current_body);
        debug!(
            session = %session_id,
            provider = self.llm.name(),
            model = self.llm.model(),
            "consolidating session"
        );
        let page: ConsolidatedPage = complete_structured(&*self.llm, request).await?;

        if dry_run {
            return Ok(ConsolidationOutcome {
                path,
                dry_run: true,
                new_title: page.title,
                new_body_markdown: page.body_markdown,
                page_id: None,
                tags: page.tags,
            });
        }

        let frontmatter = build_frontmatter(&page);
        let id = self
            .wiki
            .write_page(WritePageRequest {
                workspace_id: self.workspace_id,
                project_id: self.project_id,
                path: path.clone(),
                frontmatter,
                body: page.body_markdown.clone(),
                tier: Tier::Episodic,
                pinned: false,
                title: None,
            })
            .await?;
        // Auto-commit the result so the supersession lands in git.
        let _ = self
            .wiki
            .commit_all(&format!(
                "consolidate(session {}): {}",
                short_id(&session_id.to_string()),
                page.title.chars().take(60).collect::<String>(),
            ))
            .map_err(|e| {
                tracing::warn!(error = %e, "consolidate auto-commit failed");
                e
            });
        info!(
            session = %session_id,
            page = %id,
            "session consolidated via LLM",
        );
        Ok(ConsolidationOutcome {
            path,
            dry_run: false,
            new_title: page.title,
            new_body_markdown: page.body_markdown,
            page_id: Some(id),
            tags: page.tags,
        })
    }

    /// Borrow the underlying writer (used by the MCP tool to ack the
    /// consolidate operation in the audit log).
    #[must_use]
    pub fn writer(&self) -> &WriterHandle {
        &self.writer
    }

    /// M7b multi-page consolidation: ask the LLM for a batch of page
    /// updates spanning sessions/, concepts/, decisions/, then write
    /// them all atomically (one SQL transaction).
    ///
    /// # Errors
    /// Returns [`ConsolidatorError`] for any store, wiki, or LLM
    /// failure. On error, no pages are written and no files moved.
    pub async fn consolidate_session_multi(
        &self,
        session_id: SessionId,
        dry_run: bool,
    ) -> ConsolidatorResult<Vec<ConsolidationOutcome>> {
        let observations = self.reader.observations_for_session(session_id).await?;
        if observations.is_empty() {
            return Err(ConsolidatorError::EmptySession(session_id));
        }
        let request = build_batch_request(session_id, &observations);
        debug!(
            session = %session_id,
            provider = self.llm.name(),
            "consolidating session (multi-page)",
        );
        let batch: ConsolidatedBatch =
            ai_memory_llm::complete_structured(&*self.llm, request).await?;

        let mut requests = Vec::with_capacity(batch.updates.len());
        let mut outcomes_preview = Vec::with_capacity(batch.updates.len());
        for upd in &batch.updates {
            let path = PagePath::new(upd.path.clone())?;
            let tier = upd.tier.parse::<Tier>().unwrap_or(Tier::Semantic);
            let mut fm = serde_json::Map::new();
            fm.insert("title".into(), serde_json::Value::String(upd.title.clone()));
            fm.insert(
                "tier".into(),
                serde_json::Value::String(tier_as_str(tier).into()),
            );
            if !upd.tags.is_empty() {
                fm.insert(
                    "tags".into(),
                    serde_json::Value::Array(
                        upd.tags
                            .iter()
                            .map(|t| serde_json::Value::String(t.clone()))
                            .collect(),
                    ),
                );
            }
            fm.insert("consolidated".into(), serde_json::Value::Bool(true));
            requests.push(WritePageRequest {
                workspace_id: self.workspace_id,
                project_id: self.project_id,
                path: path.clone(),
                frontmatter: serde_json::Value::Object(fm),
                body: upd.body_markdown.clone(),
                tier,
                pinned: false,
                title: Some(upd.title.clone()),
            });
            outcomes_preview.push(ConsolidationOutcome {
                path,
                dry_run,
                new_title: upd.title.clone(),
                new_body_markdown: upd.body_markdown.clone(),
                page_id: None,
                tags: upd.tags.clone(),
            });
        }

        if dry_run {
            return Ok(outcomes_preview);
        }

        let ids = self.wiki.apply_batch(requests).await?;
        let rationale_short = batch.rationale.chars().take(60).collect::<String>();
        let _ = self.wiki.commit_all(&format!(
            "consolidate-batch(session {}): {} page(s) — {}",
            short_id(&session_id.to_string()),
            ids.len(),
            rationale_short,
        ));

        let outcomes = outcomes_preview
            .into_iter()
            .zip(ids)
            .map(|(mut o, id)| {
                o.dry_run = false;
                o.page_id = Some(id);
                o
            })
            .collect();
        Ok(outcomes)
    }
}

const fn tier_as_str(t: Tier) -> &'static str {
    match t {
        Tier::Working => "working",
        Tier::Episodic => "episodic",
        Tier::Semantic => "semantic",
        Tier::Procedural => "procedural",
    }
}

fn build_batch_request(session_id: SessionId, observations: &[Observation]) -> ChatRequest {
    let mut buf = String::new();
    buf.push_str(
        "You are compiling a Karpathy-style multi-page wiki update. Given the \
         session's observation log, produce a ConsolidatedBatch:\n\n",
    );
    buf.push_str("Session id: ");
    buf.push_str(&session_id.to_string());
    buf.push_str("\n\nObservations:\n");
    for o in observations {
        buf.push_str(&format!("- {} | {}\n", o.kind.as_str(), one_line(&o.title)));
        if !o.body.trim().is_empty() {
            buf.push_str(&format!("    body: {}\n", one_line(&o.body)));
        }
    }
    buf.push_str(
        "\nProduce up to 5 page updates. Use these path conventions:\n\
         - sessions/<session_id>.md  (episodic, this run's narrative)\n\
         - concepts/<slug>.md         (semantic, evergreen concept pages)\n\
         - decisions/<short>.md       (semantic, ADR-style records)\n\
         - gotchas/<slug>.md          (semantic, failure modes / surprises)\n\
         \nEach update must include a title, markdown body, tier, and tags.\n",
    );
    ChatRequest {
        system: Some(BATCH_SYSTEM_PROMPT.into()),
        messages: vec![ChatMessage {
            role: Role::User,
            content: buf,
        }],
        max_tokens: 4000,
        temperature: Some(0.2),
    }
}

const BATCH_SYSTEM_PROMPT: &str = "You are the maintainer of a Karpathy-style LLM wiki \
for a software engineer. Your job is to compile *durable* knowledge -- not just \
restate the session log. Produce a ConsolidatedBatch with 1-5 page updates. Favour \
extracting concept / decision / gotcha pages alongside the session summary when the \
session yields reusable insight; otherwise return only the session page.";

fn build_request(
    session_id: SessionId,
    observations: &[Observation],
    current_body: &str,
) -> ChatRequest {
    let mut buf = String::new();
    buf.push_str("Session id: ");
    buf.push_str(&session_id.to_string());
    buf.push_str("\nObservations (in order):\n\n");
    for o in observations {
        buf.push_str(&format!("- {} | {}\n", o.kind.as_str(), one_line(&o.title)));
        if !o.body.trim().is_empty() {
            buf.push_str(&format!("    body: {}\n", one_line(&o.body)));
        }
    }
    if !current_body.trim().is_empty() {
        buf.push_str("\nCurrent (heuristic) page body:\n\n```\n");
        buf.push_str(current_body);
        buf.push_str("\n```\n");
    }

    ChatRequest {
        system: Some(SYSTEM_PROMPT.into()),
        messages: vec![ChatMessage {
            role: Role::User,
            content: buf,
        }],
        max_tokens: 1500,
        temperature: Some(0.2),
    }
}

fn build_frontmatter(page: &ConsolidatedPage) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    map.insert(
        "title".into(),
        serde_json::Value::String(page.title.clone()),
    );
    map.insert("tier".into(), serde_json::Value::String("episodic".into()));
    if !page.tags.is_empty() {
        let tags = page
            .tags
            .iter()
            .map(|t| serde_json::Value::String(t.clone()))
            .collect();
        map.insert("tags".into(), serde_json::Value::Array(tags));
    }
    map.insert("consolidated".into(), serde_json::Value::Bool(true));
    serde_json::Value::Object(map)
}

fn one_line(s: &str) -> String {
    s.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .take(3)
        .collect::<Vec<_>>()
        .join(" / ")
        .chars()
        .take(240)
        .collect()
}

fn short_id(s: &str) -> String {
    s.chars().take(8).collect()
}

/// Suppress the unused-variant lint for now; consumers will use
/// [`ObservationKind`] via the observations parameter.
const _OBSERVATION_KIND: Option<ObservationKind> = None;

const SYSTEM_PROMPT: &str = "You are the maintainer of a Karpathy-style LLM wiki for a \
software engineer. You receive the chronological observation log of one coding-agent \
session plus the current heuristic page body. Compile a clean, durable markdown page \
that future agents and the user can read to recover context.\n\nRules:\n\
1. Title: short, descriptive (<= 80 chars). No filler.\n\
2. Body: well-formed markdown. Use sections (## Heading) when useful.\n\
3. Focus on decisions made, problems encountered, code/file references, and open \
   questions.\n\
4. Do NOT include redundant per-tool-call detail. Aggregate.\n\
5. Do NOT echo timestamps or session ids (frontmatter already has them).\n\
6. Tags: 0-5 short kebab-case tags surfaced to frontmatter.\n";
