//! AI "first-pass normalization" of imported wiki pages (phase 1).
//!
//! After the deterministic `import` writes raw 1:1 pages into the wiki,
//! this pass asks the LLM to normalize each one *faithfully*:
//!
//! 1. Classify a [`PageKind`](crate::types::PageKind) and confirm/adjust
//!    the [`Tier`] from the page content (imported pages arrive with no
//!    kind and `tier = semantic`).
//! 2. Clean the body conservatively: repair high-confidence
//!    double-encoded UTF-8 mojibake (Portuguese text stored as
//!    Latin-1-misread UTF-8) and tidy H1 / whitespace structure.
//! 3. Preserve all factual content, every `[[..]]` / markdown link, and
//!    the page `path` verbatim. The LLM must not invent content, this
//!    is a memory system and fabrication corrupts it.
//!
//! The per-page output is a [`ConsolidatedPageUpdate`] reusing the
//! consolidator's structured-output type; each is written via the
//! existing write/supersession path (non-destructive, the raw version
//! stays in the supersession chain + git), making the pass idempotent
//! (re-running re-normalizes the current latest).
//!
//! ## Scope, phase 1 only
//!
//! This is strictly per-page. There is NO cross-page deduplication or
//! merge here; that is a documented future v2 (see
//! `docs/import-claude-memory.md`). Each input page maps to exactly one
//! output update with the same `path`.
//!
//! ## Crate placement
//!
//! Only the system prompt + the [`build_normalize_request`] builder live
//! here (pure, no IO). The server-side load → batch → LLM → write loop
//! lives in the MCP admin handler (`POST /admin/import-normalize`); the
//! CLI flags live in `ai-memory-cli`.

use ai_memory_core::Tier;
use ai_memory_llm::{ChatMessage, ChatRequest, Role};

use crate::consolidator::tier_as_str;
use crate::types::PageKind;

/// System prompt for the import-normalization pass. Loaded at compile
/// time from `prompts/import_normalize_system.md` so the prompt is
/// plain-text-editable + version-controlled as Markdown alongside the
/// code. Public so off-tree harnesses (`evals/`) can inspect the exact
/// prompt without duplicating it.
pub const NORMALIZE_SYSTEM_PROMPT: &str = include_str!("../prompts/import_normalize_system.md");

/// One imported page handed to the normalize pass: its wiki `path`, the
/// current `body`, and the page's current `kind` / `tier` (imported
/// pages have no kind and `tier = semantic`, but the builder is generic
/// so a re-run sees the latest values).
#[derive(Debug, Clone)]
pub struct NormalizeInputPage {
    /// Wiki path, echoed back unchanged; it is the supersession key.
    pub path: String,
    /// Current page body (markdown without frontmatter).
    pub body: String,
    /// Current semantic classification, or `None` when unclassified
    /// (the imported default).
    pub current_kind: Option<PageKind>,
    /// Current memory tier.
    pub current_tier: Tier,
}

/// Character budget for a batch's page bodies rendered into the prompt.
/// Mirrors the consolidator's conservative chars/4 sizing: ~400k chars
/// ≈ ~100k tokens of input, leaving the other ~100k of a 200k-context
/// model for the system prompt, schema, and the output reservation.
const PAGE_BODY_BUDGET_CHARS: usize = 400_000;

/// Per-page body clip so a single oversized imported page cannot starve
/// the rest of the batch. A page longer than this is truncated in the
/// prompt with an explicit marker (the full page is still superseded
/// faithfully only when the LLM echoes it; an over-long page should be
/// rare for imported memories).
const MAX_PAGE_BODY_CHARS: usize = 12_000;

/// Build the exact [`ChatRequest`] the import-normalize pass sends for
/// one batch of pages. Each page contributes its `path`, current
/// `kind`/`tier`, and (clipped) `body`; the LLM returns one
/// [`ConsolidatedPageUpdate`](crate::types::ConsolidatedPageUpdate) per
/// page inside a [`ConsolidatedBatch`](crate::types::ConsolidatedBatch).
///
/// Pure: no network, no store. The server-side handler is responsible
/// for batching under a token budget and writing the results.
#[must_use]
pub fn build_normalize_request(pages: &[NormalizeInputPage]) -> ChatRequest {
    let mut buf = String::with_capacity(8_192);
    buf.push_str(
        "Normalize the following imported wiki pages. Return ONE update per page, \
         in the same order, echoing each page's `path` verbatim.\n\n",
    );

    let mut spent = 0usize;
    for (idx, page) in pages.iter().enumerate() {
        buf.push_str(&format!("=== page {} ===\n", idx + 1));
        buf.push_str("path: ");
        buf.push_str(&page.path);
        buf.push('\n');
        buf.push_str("current_kind: ");
        buf.push_str(page.current_kind.map_or("(none)", PageKind::as_str));
        buf.push('\n');
        buf.push_str("current_tier: ");
        buf.push_str(tier_as_str(page.current_tier));
        buf.push_str("\nbody:\n");
        let clipped = clip_body(&page.body, MAX_PAGE_BODY_CHARS);
        // Once the whole-batch budget is exhausted, still include the
        // page header (path/kind/tier) so the LLM keeps emitting one
        // update per page, but elide the body to stay under context.
        if spent >= PAGE_BODY_BUDGET_CHARS {
            buf.push_str("[body omitted to fit the batch token budget; preserve it unchanged]\n\n");
            continue;
        }
        spent = spent.saturating_add(clipped.len());
        buf.push_str(&clipped);
        buf.push_str("\n\n");
    }

    buf.push_str(
        "Produce a ConsolidatedBatch with exactly one update per page above. \
         Required keys on every update: `path` (echo verbatim), `title`, \
         `body_markdown` (cleaned per the rules), `tier` (one of: working | \
         episodic | semantic | procedural), `kind` (one of: decision | gotcha \
         | rule | fact), `tags` (array, may be []). Do NOT add a `slot_kind`. \
         Set `rationale` to one short sentence.\n",
    );

    ChatRequest {
        system: Some(NORMALIZE_SYSTEM_PROMPT.into()),
        messages: vec![ChatMessage {
            role: Role::User,
            content: buf,
        }],
        // Generous: 32K covers a multi-page normalization comfortably.
        // Cheaper to over-allocate than to truncate JSON mid-response.
        max_tokens: 32_000,
        // Low temperature: normalization is a faithful transform, not a
        // creative one.
        temperature: Some(0.1),
    }
}

/// Clip a page body to `max_chars`, appending a marker when truncated.
/// Char-boundary safe.
fn clip_body(body: &str, max_chars: usize) -> String {
    let mut chars = body.chars();
    let mut out: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        out.push_str("\n[imported page body truncated for the prompt; preserve it unchanged]");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page(path: &str, body: &str) -> NormalizeInputPage {
        NormalizeInputPage {
            path: path.into(),
            body: body.into(),
            current_kind: None,
            current_tier: Tier::Semantic,
        }
    }

    #[test]
    fn request_includes_each_page_path_and_body() {
        let pages = vec![
            page("imported/users.md", "Sobre os usuÃ¡rios do sistema."),
            page("omc/payments.md", "ACH batch billing flow."),
        ];
        let request = build_normalize_request(&pages);
        let prompt = &request.messages[0].content;
        assert!(prompt.contains("imported/users.md"));
        assert!(prompt.contains("omc/payments.md"));
        // Bodies are passed through verbatim (the LLM does the repair).
        assert!(prompt.contains("Sobre os usuÃ¡rios do sistema."));
        assert!(prompt.contains("ACH batch billing flow."));
        // One header per page so the model emits one update per page.
        assert!(prompt.contains("=== page 1 ==="));
        assert!(prompt.contains("=== page 2 ==="));
    }

    #[test]
    fn request_carries_current_kind_and_tier() {
        let pages = vec![NormalizeInputPage {
            path: "imported/a.md".into(),
            body: "x".into(),
            current_kind: Some(PageKind::Rule),
            current_tier: Tier::Procedural,
        }];
        let request = build_normalize_request(&pages);
        let prompt = &request.messages[0].content;
        assert!(prompt.contains("current_kind: rule"));
        assert!(prompt.contains("current_tier: procedural"));
    }

    #[test]
    fn unclassified_kind_renders_as_none() {
        let request = build_normalize_request(&[page("imported/a.md", "body")]);
        assert!(request.messages[0].content.contains("current_kind: (none)"));
    }

    #[test]
    fn system_prompt_is_attached_and_forbids_invention() {
        let request = build_normalize_request(&[page("imported/a.md", "body")]);
        let system = request.system.expect("system prompt set");
        assert!(system.contains("FAITHFULNESS"));
        assert!(system.contains("mojibake"));
        assert!(system.to_lowercase().contains("must not"));
    }

    #[test]
    fn oversized_page_body_is_clipped_with_marker() {
        let big = "y".repeat(MAX_PAGE_BODY_CHARS + 5_000);
        let request = build_normalize_request(&[page("imported/big.md", &big)]);
        let prompt = &request.messages[0].content;
        assert!(prompt.contains("truncated for the prompt"));
        assert!(prompt.len() < big.len() + 2_000);
    }
}
