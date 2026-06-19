//! Deterministic, LLM-free import of a Claude Code "dual-store" memory
//! setup into ai-memory wiki pages.
//!
//! The dual store is:
//! - a `@modelcontextprotocol/server-memory` knowledge graph
//!   (`memory.jsonl`: one `entity`/`relation` object per line), and
//! - an `mcp-server-qdrant` collection (one point per memory, payload
//!   `{ document, metadata }`).
//!
//! The two stores are bridged by convention: a Qdrant point's
//! `metadata.entityName` equals the `name` of a Memory Graph entity, so
//! the same concept lives in both (graph = structure, Qdrant = semantic
//! summary).
//!
//! This module is the **pure transform**: it takes the parsed source
//! shapes and produces a [`Vec<ImportedPage>`] with no IO. The CLI's
//! `import` command parses the sources (file + HTTP) and POSTs each
//! resulting page to the server. Mapping rules live in
//! `docs/import-claude-memory.md`.

use std::collections::HashMap;
use std::fmt::Write as _;

/// A Memory Graph entity (one `{"type":"entity", ...}` line).
#[derive(Debug, Clone)]
pub struct GraphEntity {
    /// Unique entity name. Bridges to Qdrant via `metadata.entityName`,
    /// and is the target of relations.
    pub name: String,
    /// Free-form entity type (`Decision`, `Bug`, `User`, …). Passed
    /// through verbatim; no closed vocabulary is enforced.
    pub entity_type: String,
    /// Observations attached to the entity, rendered as a bullet list.
    pub observations: Vec<String>,
}

/// A Memory Graph relation (one `{"type":"relation", ...}` line).
#[derive(Debug, Clone)]
pub struct GraphRelation {
    /// Source entity name (the relation's owner page).
    pub from: String,
    /// Target entity name (becomes a wikilink; may not exist as an
    /// entity — unresolved forward links are valid in ai-memory).
    pub to: String,
    /// Relation kind (`decided_by`, `depends_on`, …), passed through.
    pub relation_type: String,
}

/// A Qdrant point's distilled payload.
#[derive(Debug, Clone)]
pub struct QdrantPoint {
    /// `metadata.entityName` — the bridge to a Memory Graph entity.
    pub entity_name: String,
    /// `payload.document` — the semantic summary text.
    pub document: String,
    /// `metadata.project`, when present.
    pub project: Option<String>,
    /// `metadata.topic`, when present.
    pub topic: Option<String>,
    /// `metadata.date`, when present.
    pub date: Option<String>,
    /// `metadata.repos`, when present.
    pub repos: Vec<String>,
}

/// One wiki page produced by the importer, ready to POST to
/// `/admin/write-page`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportedPage {
    /// Relative wiki path, always `imported/<slug>.md`.
    pub path: String,
    /// Page title (the entity name, or the orphan point's entityName).
    pub title: String,
    /// Rendered markdown body.
    pub body: String,
    /// Tier the page is written at — always `semantic` for imports.
    pub tier: String,
    /// Frontmatter tags — the source `entityType` (one element), or
    /// empty for an orphan Qdrant point with no entity.
    pub tags: Vec<String>,
    /// Whether the page is pinned (exempt from the decay sweep).
    pub pinned: bool,
}

/// Slugify a name into a stable, collision-free path component.
///
/// Lowercase; every run of non-`[a-z0-9]` collapses to a single `-`;
/// leading/trailing `-` trimmed. The `used` map records every slug
/// already handed out so a second distinct name colliding onto the same
/// base slug gets a `-2`, `-3`, … suffix. Stable: the same name always
/// resolves to the same slug within one run, so a relation's target slug
/// matches the target entity's page slug.
fn slugify(name: &str, used: &mut HashMap<String, usize>) -> String {
    let mut base = String::with_capacity(name.len());
    let mut prev_dash = false;
    for ch in name.chars() {
        let lower = ch.to_ascii_lowercase();
        if lower.is_ascii_lowercase() || lower.is_ascii_digit() {
            base.push(lower);
            prev_dash = false;
        } else if !prev_dash {
            base.push('-');
            prev_dash = true;
        }
    }
    let base = base.trim_matches('-').to_string();
    // An all-symbol name (or empty) slugs to empty; fall back to a
    // stable placeholder so the path is never `imported/.md`.
    let base = if base.is_empty() {
        "untitled".to_string()
    } else {
        base
    };

    let count = used.entry(base.clone()).or_insert(0);
    *count += 1;
    if *count == 1 {
        base
    } else {
        format!("{base}-{count}")
    }
}

/// Build the importer's wiki pages from the parsed dual-store sources.
///
/// Each Memory Graph entity becomes one `imported/<slug>.md` page with an
/// `## Observations` list, an optional `## Summary` (from a matching
/// Qdrant point), and an `## Related` list of wikilinks (its outgoing
/// relations). A Qdrant point whose `entity_name` matches no entity
/// becomes its own standalone page. Slugs are assigned once per name so
/// relations resolve to the same page slug as their target entity.
///
/// `pinned` marks every produced page pinned. Text is copied
/// byte-for-byte (no transcoding).
#[must_use]
pub fn build_import_pages(
    entities: &[GraphEntity],
    relations: &[GraphRelation],
    points: &[QdrantPoint],
    pinned: bool,
) -> Vec<ImportedPage> {
    // First, assign a stable slug to every entity, by position. Entities
    // are slugged before orphan points so a relation's target (an entity)
    // always wins the un-suffixed base slug. We keep two indexes:
    // `entity_slugs[i]` is the unique page slug for the i-th entity (so
    // even two entities sharing a name get distinct pages), while
    // `entity_slug[name]` records the FIRST occurrence's slug — the
    // canonical target a relation to `name` resolves to.
    let mut used_slugs: HashMap<String, usize> = HashMap::new();
    let mut entity_slugs: Vec<String> = Vec::with_capacity(entities.len());
    let mut entity_slug: HashMap<&str, String> = HashMap::new();
    for entity in entities {
        let slug = slugify(&entity.name, &mut used_slugs);
        entity_slug
            .entry(entity.name.as_str())
            .or_insert_with(|| slug.clone());
        entity_slugs.push(slug);
    }

    // Index Qdrant points by entityName so each entity can pick up its
    // matching summary in O(1). A later point for the same entityName
    // overwrites an earlier one (last write wins).
    let mut point_by_entity: HashMap<&str, &QdrantPoint> = HashMap::new();
    for point in points {
        point_by_entity.insert(point.entity_name.as_str(), point);
    }

    // Group outgoing relations by their `from` entity.
    let mut relations_by_from: HashMap<&str, Vec<&GraphRelation>> = HashMap::new();
    for relation in relations {
        relations_by_from
            .entry(relation.from.as_str())
            .or_default()
            .push(relation);
    }

    let mut pages = Vec::with_capacity(entities.len());

    for (i, entity) in entities.iter().enumerate() {
        let slug = &entity_slugs[i];
        let matched = point_by_entity.get(entity.name.as_str()).copied();
        let outgoing = relations_by_from
            .get(entity.name.as_str())
            .map_or(&[][..], |v| v.as_slice());

        let body = render_entity_body(entity, matched, outgoing, &entity_slug);
        pages.push(ImportedPage {
            path: format!("imported/{slug}.md"),
            title: entity.name.clone(),
            body,
            tier: "semantic".to_string(),
            tags: vec![entity.entity_type.clone()],
            pinned,
        });
    }

    // Orphan Qdrant points: a point whose entityName matches no entity
    // becomes its own page, preserving Qdrant-only memories.
    for point in points {
        if entity_slug.contains_key(point.entity_name.as_str()) {
            continue;
        }
        let slug = slugify(&point.entity_name, &mut used_slugs);
        let body = render_orphan_point_body(point);
        pages.push(ImportedPage {
            path: format!("imported/{slug}.md"),
            title: point.entity_name.clone(),
            body,
            tier: "semantic".to_string(),
            tags: Vec::new(),
            pinned,
        });
    }

    pages
}

/// One oh-my-claudecode (OMC) Karpathy-wiki page parsed from a flat dir
/// of markdown files with YAML frontmatter. The IO + frontmatter parsing
/// lives in the CLI; this is the already-split, pure shape the transform
/// consumes.
#[derive(Debug, Clone)]
pub struct OmcWikiPage {
    /// Original on-disk filename, e.g. `ach-requests-....md` (already
    /// ends in `.md`). Preserved verbatim as the page's path component.
    pub filename: String,
    /// Frontmatter `title`, when present. Falls back to the first body
    /// `# H1`, then to the filename stem.
    pub title: Option<String>,
    /// Frontmatter `tags`. Merged with `category` (de-duplicated,
    /// order-stable) to form the page tags.
    pub tags: Vec<String>,
    /// Frontmatter `category`, when present. Appended to `tags` unless
    /// already present.
    pub category: Option<String>,
    /// Frontmatter `links` — bare filenames of sibling OMC wiki pages.
    /// Rendered as `[[omc/<link>]]` wikilinks under a trailing
    /// `## Related (OMC wiki)` section.
    pub links: Vec<String>,
    /// The markdown body (everything after the frontmatter), preserved
    /// byte-for-byte.
    pub body: String,
}

/// Build wiki pages from a parsed OMC Karpathy wiki (a flat dir of
/// markdown pages with YAML frontmatter).
///
/// Each page maps to one `omc/<original-filename>.md` page:
/// - title from frontmatter `title`, falling back to the first body
///   `# H1`, then the filename stem;
/// - tags = frontmatter `tags` plus `category` (de-duplicated,
///   order-stable);
/// - tier `semantic`;
/// - body preserved byte-for-byte, then (when the page has frontmatter
///   `links`) a trailing `## Related (OMC wiki)` section of
///   `[[omc/<link>]]` wikilinks so ai-memory resolves them into its link
///   graph. The section is omitted entirely when there are no links.
///
/// `pinned` marks every produced page pinned. Text is copied
/// byte-for-byte (no transcoding).
#[must_use]
pub fn build_omc_wiki_pages(pages: &[OmcWikiPage], pinned: bool) -> Vec<ImportedPage> {
    let mut out = Vec::with_capacity(pages.len());
    for page in pages {
        let title = resolve_omc_title(page);
        let tags = merge_tags(&page.tags, page.category.as_deref());
        let body = render_omc_body(&page.body, &page.links);
        out.push(ImportedPage {
            path: format!("omc/{}", page.filename),
            title,
            body,
            tier: "semantic".to_string(),
            tags,
            pinned,
        });
    }
    out
}

/// Resolve an OMC page title: frontmatter `title`, else the first body
/// `# H1`, else the filename stem (the filename with a trailing `.md`
/// stripped). Empty candidates are skipped so the title is never blank.
fn resolve_omc_title(page: &OmcWikiPage) -> String {
    if let Some(title) = page
        .title
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        return title.to_string();
    }
    if let Some(h1) = first_h1(&page.body) {
        return h1;
    }
    page.filename
        .strip_suffix(".md")
        .unwrap_or(&page.filename)
        .to_string()
}

/// Extract the first markdown `# H1` heading text from a body, if any.
/// Matches a line whose first non-whitespace run is a single `#`
/// followed by space(s); returns the trimmed heading text. Pure, no IO.
fn first_h1(body: &str) -> Option<String> {
    for line in body.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("# ") {
            let text = rest.trim();
            if !text.is_empty() {
                return Some(text.to_string());
            }
        }
    }
    None
}

/// Merge frontmatter `tags` with an optional `category`, de-duplicated
/// and order-stable (tags first in their original order, then the
/// category appended only if not already present).
fn merge_tags(tags: &[String], category: Option<&str>) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(tags.len() + 1);
    for tag in tags {
        if !out.iter().any(|t| t == tag) {
            out.push(tag.clone());
        }
    }
    if let Some(cat) = category.filter(|s| !s.is_empty())
        && !out.iter().any(|t| t == cat)
    {
        out.push(cat.to_string());
    }
    out
}

/// Render an OMC page body: the original body byte-for-byte, then (when
/// `links` is non-empty) a trailing `## Related (OMC wiki)` section with
/// one `[[omc/<link>]]` wikilink per link. The section is omitted
/// entirely when there are no links.
fn render_omc_body(body: &str, links: &[String]) -> String {
    if links.is_empty() {
        return body.to_string();
    }
    let mut out = String::with_capacity(body.len() + 32 + links.len() * 16);
    out.push_str(body);
    out.push_str("\n\n## Related (OMC wiki)\n");
    for link in links {
        let _ = writeln!(out, "- [[omc/{link}]]");
    }
    out
}

/// Render the markdown body for an entity page: title, type, optional
/// metadata header + summary (from a matched Qdrant point), observations,
/// and related wikilinks. Sections with no content are omitted.
fn render_entity_body(
    entity: &GraphEntity,
    matched: Option<&QdrantPoint>,
    outgoing: &[&GraphRelation],
    entity_slug: &HashMap<&str, String>,
) -> String {
    let mut body = String::new();
    let _ = writeln!(body, "# {}", entity.name);
    let _ = writeln!(body);
    let _ = writeln!(body, "**Type:** {}", entity.entity_type);

    if let Some(point) = matched {
        write_metadata_header(&mut body, point);
    }

    if !entity.observations.is_empty() {
        let _ = writeln!(body);
        let _ = writeln!(body, "## Observations");
        for obs in &entity.observations {
            let _ = writeln!(body, "- {obs}");
        }
    }

    if let Some(point) = matched
        && !point.document.is_empty()
    {
        let _ = writeln!(body);
        let _ = writeln!(body, "## Summary");
        let _ = writeln!(body, "{}", point.document);
    }

    if !outgoing.is_empty() {
        let _ = writeln!(body);
        let _ = writeln!(body, "## Related");
        for relation in outgoing {
            let target_slug = resolve_target_slug(&relation.to, entity_slug);
            let _ = writeln!(
                body,
                "- {} -> [[imported/{}.md|{}]]",
                relation.relation_type, target_slug, relation.to
            );
        }
    }

    body
}

/// Resolve the slug a relation target should link to.
///
/// If the target is a known entity, reuse its already-assigned slug so
/// the link resolves to the real page. Otherwise (an orphan forward
/// reference to an entity that does not exist) compute a slug for the
/// name without reserving it as a collision — unresolved forward links
/// are valid, and we must not perturb the disambiguation counters that
/// real pages depend on. We therefore slug it in an isolated map.
fn resolve_target_slug(target: &str, entity_slug: &HashMap<&str, String>) -> String {
    if let Some(slug) = entity_slug.get(target) {
        return slug.clone();
    }
    // Forward link to a non-existent entity: derive the base slug in an
    // isolated counter so it matches what the target page *would* get as
    // a first occurrence, without disturbing real slug assignments.
    let mut isolated = HashMap::new();
    slugify(target, &mut isolated)
}

/// Render the body for an orphan Qdrant point (no matching entity):
/// title, metadata header, and the summary.
fn render_orphan_point_body(point: &QdrantPoint) -> String {
    let mut body = String::new();
    let _ = writeln!(body, "# {}", point.entity_name);
    write_metadata_header(&mut body, point);
    if !point.document.is_empty() {
        let _ = writeln!(body);
        let _ = writeln!(body, "## Summary");
        let _ = writeln!(body, "{}", point.document);
    }
    body
}

/// Append the `**Project:** … · **Date:** …` / `**Repos:** …` metadata
/// header for a Qdrant point. Each field is emitted only when present;
/// the whole header is skipped when the point carries no metadata.
fn write_metadata_header(body: &mut String, point: &QdrantPoint) {
    let mut header = String::new();
    match (&point.project, &point.date) {
        (Some(project), Some(date)) => {
            let _ = write!(header, "**Project:** {project}  ·  **Date:** {date}");
        }
        (Some(project), None) => {
            let _ = write!(header, "**Project:** {project}");
        }
        (None, Some(date)) => {
            let _ = write!(header, "**Date:** {date}");
        }
        (None, None) => {}
    }
    if !header.is_empty() {
        let _ = writeln!(body, "{header}");
    }
    if !point.repos.is_empty() {
        let _ = writeln!(body, "**Repos:** {}", point.repos.join(", "));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entity(name: &str, etype: &str, obs: &[&str]) -> GraphEntity {
        GraphEntity {
            name: name.to_string(),
            entity_type: etype.to_string(),
            observations: obs.iter().map(|s| (*s).to_string()).collect(),
        }
    }

    fn relation(from: &str, to: &str, rtype: &str) -> GraphRelation {
        GraphRelation {
            from: from.to_string(),
            to: to.to_string(),
            relation_type: rtype.to_string(),
        }
    }

    fn point(entity_name: &str, document: &str) -> QdrantPoint {
        QdrantPoint {
            entity_name: entity_name.to_string(),
            document: document.to_string(),
            project: None,
            topic: None,
            date: None,
            repos: Vec::new(),
        }
    }

    /// Case 1: an entity plus a Qdrant point that matches it by
    /// `entityName` yields a single page carrying both an `## Observations`
    /// list and a `## Summary` section (plus the metadata header).
    #[test]
    fn entity_with_matching_point_renders_observations_and_summary() {
        let entities = vec![entity(
            "Auth Decision",
            "Decision",
            &["chose cookies over JWT", "rejected vector RAG"],
        )];
        let mut p = point("Auth Decision", "We standardised on session cookies.");
        p.project = Some("bidchex".to_string());
        p.date = Some("2026-06-16".to_string());
        p.repos = vec!["backend".to_string(), "frontend".to_string()];
        let points = vec![p];

        let pages = build_import_pages(&entities, &[], &points, false);
        assert_eq!(pages.len(), 1, "one entity, matched point folds in");
        let page = &pages[0];
        assert_eq!(page.path, "imported/auth-decision.md");
        assert_eq!(page.title, "Auth Decision");
        assert_eq!(page.tags, vec!["Decision".to_string()]);
        assert_eq!(page.tier, "semantic");
        assert!(!page.pinned);

        assert!(page.body.contains("# Auth Decision"));
        assert!(page.body.contains("**Type:** Decision"));
        assert!(
            page.body
                .contains("**Project:** bidchex  ·  **Date:** 2026-06-16")
        );
        assert!(page.body.contains("**Repos:** backend, frontend"));
        assert!(page.body.contains("## Observations"));
        assert!(page.body.contains("- chose cookies over JWT"));
        assert!(page.body.contains("- rejected vector RAG"));
        assert!(page.body.contains("## Summary"));
        assert!(page.body.contains("We standardised on session cookies."));
    }

    /// Case 2: outgoing relations render as wikilinks under `## Related`,
    /// and each link's slug matches the target entity's own page slug.
    #[test]
    fn relations_render_as_wikilinks_with_matching_slugs() {
        let entities = vec![
            entity("Order Service", "Component", &[]),
            entity("Payment Gateway", "Component", &[]),
        ];
        let relations = vec![relation("Order Service", "Payment Gateway", "depends_on")];

        let pages = build_import_pages(&entities, &relations, &[], false);
        let order = pages
            .iter()
            .find(|p| p.title == "Order Service")
            .expect("order page present");
        let payment = pages
            .iter()
            .find(|p| p.title == "Payment Gateway")
            .expect("payment page present");

        assert!(order.body.contains("## Related"));
        assert!(
            order
                .body
                .contains("- depends_on -> [[imported/payment-gateway.md|Payment Gateway]]"),
            "link should use the target entity's slug; body was:\n{}",
            order.body
        );
        // The wikilink target slug must equal the target page's real path.
        assert_eq!(payment.path, "imported/payment-gateway.md");
    }

    /// Case 3: a Qdrant point whose entityName matches no entity becomes
    /// its own standalone page.
    #[test]
    fn orphan_point_becomes_standalone_page() {
        let mut p = point("Lonely Memory", "A fact with no graph entity.");
        p.date = Some("2026-01-02".to_string());
        let pages = build_import_pages(&[], &[], &[p], false);
        assert_eq!(pages.len(), 1);
        let page = &pages[0];
        assert_eq!(page.path, "imported/lonely-memory.md");
        assert_eq!(page.title, "Lonely Memory");
        assert!(page.tags.is_empty(), "orphan point has no entityType tag");
        assert!(page.body.contains("# Lonely Memory"));
        assert!(page.body.contains("**Date:** 2026-01-02"));
        assert!(page.body.contains("## Summary"));
        assert!(page.body.contains("A fact with no graph entity."));
    }

    /// Case 4: a relation pointing at a non-existent entity still emits a
    /// wikilink (an unresolved forward link is valid in ai-memory).
    #[test]
    fn orphan_relation_target_still_emits_wikilink() {
        let entities = vec![entity("Source Entity", "Feature", &[])];
        let relations = vec![relation("Source Entity", "Ghost Target", "relates_to")];

        let pages = build_import_pages(&entities, &relations, &[], false);
        assert_eq!(pages.len(), 1, "only the source entity yields a page");
        let page = &pages[0];
        assert!(
            page.body
                .contains("- relates_to -> [[imported/ghost-target.md|Ghost Target]]"),
            "forward link to a missing entity must still render; body:\n{}",
            page.body
        );
    }

    /// Case 5: slug derivation is stable for a name, and two distinct
    /// names that collapse to the same base slug are disambiguated with a
    /// numeric suffix.
    #[test]
    fn slug_is_stable_and_collisions_get_suffix() {
        // "Foo Bar" and "foo/bar" both slugify to base "foo-bar".
        let entities = vec![
            entity("Foo Bar", "Fact", &[]),
            entity("foo/bar", "Fact", &[]),
            entity("Foo Bar", "Fact", &[]), // third distinct-by-position collision
        ];
        let pages = build_import_pages(&entities, &[], &[], false);
        let paths: Vec<&str> = pages.iter().map(|p| p.path.as_str()).collect();
        assert_eq!(
            paths,
            vec![
                "imported/foo-bar.md",
                "imported/foo-bar-2.md",
                "imported/foo-bar-3.md",
            ],
            "collisions must disambiguate deterministically"
        );

        // Stability: the same input set always produces the same slugs.
        let again = build_import_pages(&entities, &[], &[], false);
        let paths_again: Vec<&str> = again.iter().map(|p| p.path.as_str()).collect();
        assert_eq!(paths, paths_again, "slug assignment must be deterministic");
    }

    /// Case 6: observation and summary text is copied byte-for-byte (no
    /// transcoding, no escaping, no normalisation).
    #[test]
    fn text_is_passed_through_byte_for_byte() {
        let tricky = "naïve café — façade \u{1F600} <html> & \"quotes\" \\back";
        let entities = vec![entity("Encoding Test", "Reference", &[tricky])];
        let points = vec![point("Encoding Test", tricky)];

        let pages = build_import_pages(&entities, &[], &points, false);
        let body = &pages[0].body;
        // The exact bytes appear in both the observation bullet and the
        // summary, unaltered.
        assert!(
            body.contains(&format!("- {tricky}")),
            "observation must be verbatim; body:\n{body}"
        );
        assert!(
            body.contains(tricky),
            "summary must contain the verbatim document text"
        );
    }

    /// An entity with neither observations nor a matched summary still
    /// produces a page (title + type + relations) so relation targets
    /// always resolve.
    #[test]
    fn bare_entity_still_produces_a_page() {
        let entities = vec![entity("Empty Node", "Fact", &[])];
        let pages = build_import_pages(&entities, &[], &[], false);
        assert_eq!(pages.len(), 1);
        let body = &pages[0].body;
        assert!(body.contains("# Empty Node"));
        assert!(body.contains("**Type:** Fact"));
        assert!(!body.contains("## Observations"));
        assert!(!body.contains("## Summary"));
        assert!(!body.contains("## Related"));
    }

    /// `pinned = true` propagates onto every produced page (entity pages
    /// and orphan point pages alike).
    #[test]
    fn pinned_flag_propagates_to_all_pages() {
        let entities = vec![entity("E", "Fact", &[])];
        let points = vec![point("Orphan", "doc")];
        let pages = build_import_pages(&entities, &[], &points, true);
        assert_eq!(pages.len(), 2);
        assert!(pages.iter().all(|p| p.pinned));
    }

    fn omc_page(
        filename: &str,
        title: Option<&str>,
        tags: &[&str],
        category: Option<&str>,
        links: &[&str],
        body: &str,
    ) -> OmcWikiPage {
        OmcWikiPage {
            filename: filename.to_string(),
            title: title.map(str::to_string),
            tags: tags.iter().map(|s| (*s).to_string()).collect(),
            category: category.map(str::to_string),
            links: links.iter().map(|s| (*s).to_string()).collect(),
            body: body.to_string(),
        }
    }

    /// OMC case 1: a full-frontmatter page maps to `omc/<file>` with the
    /// frontmatter title, tags = tags+category de-duplicated, the body
    /// preserved, and a `## Related (OMC wiki)` section of `[[omc/<link>]]`
    /// wikilinks.
    #[test]
    fn omc_full_frontmatter_page_maps_completely() {
        let pages = build_omc_wiki_pages(
            &[omc_page(
                "ach-requests.md",
                Some("ACH Requests & Batches"),
                &["payments", "ach"],
                Some("architecture"),
                &["payments-batch.md", "payments-external.md"],
                "# ACH Requests\n\nBody text here.",
            )],
            false,
        );
        assert_eq!(pages.len(), 1);
        let page = &pages[0];
        assert_eq!(page.path, "omc/ach-requests.md");
        assert_eq!(page.title, "ACH Requests & Batches");
        assert_eq!(
            page.tags,
            vec![
                "payments".to_string(),
                "ach".to_string(),
                "architecture".to_string(),
            ]
        );
        assert_eq!(page.tier, "semantic");
        assert!(!page.pinned);
        assert!(
            page.body.starts_with("# ACH Requests\n\nBody text here."),
            "original body must be preserved first; body:\n{}",
            page.body
        );
        assert!(page.body.contains("\n\n## Related (OMC wiki)\n"));
        assert!(page.body.contains("- [[omc/payments-batch.md]]\n"));
        assert!(page.body.contains("- [[omc/payments-external.md]]\n"));
    }

    /// OMC case 2: a page with no frontmatter title falls back to the
    /// first `# H1`, then the filename stem; with no `# H1` it uses the
    /// stem. Tags are empty and there is no Related section.
    #[test]
    fn omc_no_frontmatter_title_falls_back_to_h1_then_stem() {
        // H1 present -> title from H1.
        let pages = build_omc_wiki_pages(
            &[omc_page(
                "billing-flow.md",
                None,
                &[],
                None,
                &[],
                "# Billing Flow\n\nSome body.",
            )],
            false,
        );
        let page = &pages[0];
        assert_eq!(page.title, "Billing Flow");
        assert!(page.tags.is_empty());
        assert!(!page.body.contains("## Related (OMC wiki)"));

        // No H1 -> title from filename stem.
        let pages = build_omc_wiki_pages(
            &[omc_page(
                "just-body.md",
                None,
                &[],
                None,
                &[],
                "plain body, no heading",
            )],
            false,
        );
        assert_eq!(pages[0].title, "just-body");
    }

    /// OMC case 3: a page with no links produces no `## Related` section.
    #[test]
    fn omc_empty_links_omits_related_section() {
        let pages = build_omc_wiki_pages(
            &[omc_page(
                "lonely.md",
                Some("Lonely"),
                &["misc"],
                None,
                &[],
                "Just a body.",
            )],
            false,
        );
        let page = &pages[0];
        assert_eq!(page.body, "Just a body.");
        assert!(!page.body.contains("## Related (OMC wiki)"));
    }

    /// OMC case 4: `category` merges into tags without duplicating a tag
    /// that already carries that value.
    #[test]
    fn omc_category_merges_without_duplicating() {
        let pages = build_omc_wiki_pages(
            &[omc_page(
                "p.md",
                Some("T"),
                &["payments", "architecture"],
                Some("architecture"),
                &[],
                "body",
            )],
            false,
        );
        assert_eq!(
            pages[0].tags,
            vec!["payments".to_string(), "architecture".to_string()],
            "category already present as a tag must not duplicate"
        );
    }

    /// OMC case 5: the body is preserved byte-for-byte (accents/emoji),
    /// and the appended Related section follows it verbatim.
    #[test]
    fn omc_body_is_preserved_byte_for_byte() {
        let tricky = "naïve café \u{1F600} integração — fluxo\nlinha dois";
        let pages = build_omc_wiki_pages(
            &[omc_page(
                "enc.md",
                Some("Enc"),
                &[],
                None,
                &["other.md"],
                tricky,
            )],
            false,
        );
        let body = &pages[0].body;
        assert_eq!(
            body,
            &format!("{tricky}\n\n## Related (OMC wiki)\n- [[omc/other.md]]\n"),
            "body must be verbatim then the Related section; body:\n{body}"
        );
    }
}
