//! Public-facing consolidation types.

use ai_memory_core::{PageId, PagePath};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// JSON-schema-validated structured output from the LLM. The Karpathy
/// wiki pattern is "compile then keep current"; this is what one
/// compile step produces for a single page.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ConsolidatedPage {
    /// Page title; rendered as the first H1 by the wiki layer.
    pub title: String,
    /// Markdown body (no frontmatter; the wiki layer adds that).
    pub body_markdown: String,
    /// Up to ~5 short tags surfaced into the page's frontmatter.
    #[serde(default)]
    pub tags: Vec<String>,
}

/// One update inside a multi-page consolidation batch (M7b).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ConsolidatedPageUpdate {
    /// Relative wiki path (`concepts/foo.md`, `decisions/0001.md`, …).
    pub path: String,
    /// Tier (`semantic`, `episodic`, `procedural`, `working`).
    pub tier: String,
    /// New page title.
    pub title: String,
    /// New markdown body.
    pub body_markdown: String,
    /// Optional tags surfaced into frontmatter.
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Batch produced by [`ConsolidatorMulti`].
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ConsolidatedBatch {
    /// Pages to create / update.
    pub updates: Vec<ConsolidatedPageUpdate>,
    /// Brief LLM-authored note about *why* this batch was produced.
    /// Surfaced in the auto-commit message.
    #[serde(default)]
    pub rationale: String,
}

/// Outcome of a single consolidation call.
#[derive(Debug, Clone, Serialize)]
pub struct ConsolidationOutcome {
    /// Path of the page that was (or would be) written.
    pub path: PagePath,
    /// Whether the call ran in dry-run mode.
    pub dry_run: bool,
    /// New title.
    pub new_title: String,
    /// New body. Hidden when content has not changed.
    pub new_body_markdown: String,
    /// Identifier of the page that is now `is_latest = 1`. `None` on
    /// dry-run.
    pub page_id: Option<PageId>,
    /// Tags applied to the page.
    pub tags: Vec<String>,
}
