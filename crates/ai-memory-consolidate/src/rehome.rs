//! Deterministic "re-home by kind" transform.
//!
//! After pages are classified with a `kind`, they should live under
//! ai-memory's native kind folder rather than under their import-provenance
//! folder (`imported/`, `omc/`). This module is the PURE core of that move:
//! it computes the old→new path map and rewrites every wikilink / markdown
//! link that points at a moved page, so the link graph never dangles. All
//! IO (reading pages, writing at new paths, deleting old paths) lives in the
//! caller (the server's `/admin/rehome-by-kind` handler).
//!
//! Hard rules, mirrored from `docs/ai-ingestion-playbook.md` (Step 2.5):
//! - The slug (filename) NEVER changes; only the folder does.
//! - A move is write-at-new + delete-old; here we only compute WHAT to move.
//! - Every folder-qualified link to a moved page is rewritten through the
//!   global map. Bare links (`[[slug]]`, no folder) resolve by filename and
//!   are left untouched. Cross-scope links (`[[proj:path]]`) and URLs are
//!   left untouched.
//! - Collisions never clobber: if two pages would land on the same path, or
//!   a target is already occupied by a non-moving page, BOTH are skipped and
//!   reported rather than silently overwriting.

use std::collections::{BTreeMap, BTreeSet};

/// One page's identity for planning: its current path and classified kind.
#[derive(Debug, Clone)]
pub struct RehomePage {
    /// Current wiki path, e.g. `imported/foo.md`.
    pub path: String,
    /// Classified `kind` (frontmatter), e.g. `decision`. Empty / unknown
    /// kinds have no native folder and are left in place.
    pub kind: String,
}

/// A planned move: a page travels from `old_path` to `new_path` (same slug,
/// different folder).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RehomeMove {
    /// Current wiki path the page moves FROM.
    pub old_path: String,
    /// Native kind-folder path the page moves TO (same slug).
    pub new_path: String,
}

/// A page that COULD have moved by kind but was held back, with why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RehomeSkip {
    /// The page that was held back (kept at its current path).
    pub path: String,
    /// Why it was held back (e.g. target collision / occupied).
    pub reason: String,
}

/// The computed re-home plan: the moves to perform and the conflicts that
/// were skipped. `map` is `old_path -> new_path` for the moves (the link
/// rewriter's input).
#[derive(Debug, Clone, Default)]
pub struct RehomePlan {
    /// The moves to perform (write-at-new + delete-old).
    pub moves: Vec<RehomeMove>,
    /// Move candidates held back to avoid clobbering, with reasons.
    pub skipped: Vec<RehomeSkip>,
    /// `old_path -> new_path` for the moves, the link rewriter's input.
    pub map: BTreeMap<String, String>,
}

/// The native wiki folder for a classified `kind`, or `None` when the kind
/// has no home (so the page stays where it is). `fact` and `concept` share
/// `concepts/`; `rule` uses the reserved `_rules/` (never `rules/`).
#[must_use]
pub fn kind_folder(kind: &str) -> Option<&'static str> {
    match kind.trim() {
        "decision" => Some("decisions"),
        "gotcha" => Some("gotchas"),
        "rule" => Some("_rules"),
        "fact" | "concept" => Some("concepts"),
        "procedure" => Some("procedures"),
        "note" => Some("notes"),
        _ => None,
    }
}

/// The last `/`-separated segment of a path, the slug (filename), which a
/// re-home preserves verbatim.
fn slug_of(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// The top-level folder of a path (everything before the first `/`), or the
/// empty string when the path has no folder.
fn top_folder(path: &str) -> &str {
    match path.find('/') {
        Some(i) => &path[..i],
        None => "",
    }
}

/// Build the re-home plan from every page's `(path, kind)`.
///
/// A page is a move candidate when its kind has a native folder
/// ([`kind_folder`]) AND it is not already under that folder. Candidates
/// are then filtered so a move never clobbers another page:
/// - two candidates targeting the same new path → both skipped;
/// - a target already occupied by a page that is NOT itself moving away →
///   that candidate skipped.
///
/// The result is deterministic (inputs are sorted) regardless of the order
/// pages are supplied.
#[must_use]
pub fn build_rehome_plan(pages: &[RehomePage]) -> RehomePlan {
    let existing: BTreeSet<&str> = pages.iter().map(|p| p.path.as_str()).collect();

    // First pass: raw candidates (old -> new), sorted by old for determinism.
    let mut sorted: Vec<&RehomePage> = pages.iter().collect();
    sorted.sort_by(|a, b| a.path.cmp(&b.path));

    let mut candidates: Vec<RehomeMove> = Vec::new();
    let mut skipped: Vec<RehomeSkip> = Vec::new();
    for page in &sorted {
        let Some(folder) = kind_folder(&page.kind) else {
            continue; // no native home for this kind; leave in place
        };
        if top_folder(&page.path) == folder {
            continue; // already home
        }
        let new_path = format!("{folder}/{}", slug_of(&page.path));
        candidates.push(RehomeMove {
            old_path: page.path.clone(),
            new_path,
        });
    }

    // Detect target collisions: multiple candidates onto one new path.
    let mut target_count: BTreeMap<&str, usize> = BTreeMap::new();
    for c in &candidates {
        *target_count.entry(c.new_path.as_str()).or_insert(0) += 1;
    }
    let vacating: BTreeSet<&str> = candidates.iter().map(|c| c.old_path.as_str()).collect();

    let mut moves: Vec<RehomeMove> = Vec::new();
    for c in &candidates {
        if target_count.get(c.new_path.as_str()).copied().unwrap_or(0) > 1 {
            skipped.push(RehomeSkip {
                path: c.old_path.clone(),
                reason: format!("target {} claimed by multiple pages", c.new_path),
            });
            continue;
        }
        // Target already exists as a different page that is NOT moving away.
        if existing.contains(c.new_path.as_str()) && !vacating.contains(c.new_path.as_str()) {
            skipped.push(RehomeSkip {
                path: c.old_path.clone(),
                reason: format!("target {} already occupied", c.new_path),
            });
            continue;
        }
        moves.push(c.clone());
    }

    let map: BTreeMap<String, String> = moves
        .iter()
        .map(|m| (m.old_path.clone(), m.new_path.clone()))
        .collect();

    RehomePlan {
        moves,
        skipped,
        map,
    }
}

/// Rewrite every link in `body` whose target is a moved page, using the
/// old→new `map`. Handles `[[target]]`, `[[target|label]]`, and markdown
/// `[label](target)`. Bare wikilinks (no `/`), cross-scope wikilinks
/// (`proj:path`), and URLs are left untouched. The target's `.md`-ness is
/// preserved (a link written without `.md` stays without it).
///
/// Returns the rewritten body and the number of links changed.
#[must_use]
pub fn rewrite_links(body: &str, map: &BTreeMap<String, String>) -> (String, usize) {
    let mut out = String::with_capacity(body.len());
    let mut changed = 0usize;
    let bytes = body.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        // Wikilink: [[ ... ]]
        if bytes[i] == b'['
            && i + 1 < bytes.len()
            && bytes[i + 1] == b'['
            && let Some(close) = find_sub(body, i + 2, "]]")
            && let Some(rewritten) = rewrite_wikilink_inner(&body[i + 2..close], map)
        {
            out.push_str("[[");
            out.push_str(&rewritten);
            out.push_str("]]");
            changed += 1;
            i = close + 2;
            continue;
        }
        // Markdown link: ](target)
        if bytes[i] == b']'
            && i + 1 < bytes.len()
            && bytes[i + 1] == b'('
            && let Some(close) = find_sub(body, i + 2, ")")
            && let Some(new_target) = remap_target(&body[i + 2..close], map)
        {
            out.push_str("](");
            out.push_str(&new_target);
            out.push(')');
            changed += 1;
            i = close + 1;
            continue;
        }
        // Default: copy one char (respect UTF-8 boundaries).
        let ch_len = utf8_len(bytes[i]);
        out.push_str(&body[i..i + ch_len]);
        i += ch_len;
    }
    (out, changed)
}

/// Rewrite the inside of a `[[ ... ]]` wikilink (without the brackets).
/// Returns `Some(new_inner)` when the target maps, else `None`.
fn rewrite_wikilink_inner(inner: &str, map: &BTreeMap<String, String>) -> Option<String> {
    let (target, label) = match inner.split_once('|') {
        Some((t, l)) => (t, Some(l)),
        None => (inner, None),
    };
    let new_target = remap_target(target.trim(), map)?;
    Some(match label {
        Some(l) => format!("{new_target}|{l}"),
        None => new_target,
    })
}

/// Map a link target path to its new home, preserving `.md`-ness. Returns
/// `None` when the target is bare (no `/`), cross-scope (`:`), a URL, or
/// simply not a moved page.
fn remap_target(target: &str, map: &BTreeMap<String, String>) -> Option<String> {
    if target.contains(':') || !target.contains('/') {
        return None; // cross-scope, URL, or bare slug, leave untouched
    }
    let had_md = target.ends_with(".md");
    let key = if had_md {
        target.to_string()
    } else {
        format!("{target}.md")
    };
    let new = map.get(&key)?;
    Some(if had_md {
        new.clone()
    } else {
        new.strip_suffix(".md").unwrap_or(new).to_string()
    })
}

/// Find the byte index of `needle` in `haystack` at or after `from`.
fn find_sub(haystack: &str, from: usize, needle: &str) -> Option<usize> {
    haystack[from..].find(needle).map(|rel| from + rel)
}

/// Byte length of a UTF-8 sequence from its leading byte.
const fn utf8_len(lead: u8) -> usize {
    if lead < 0x80 {
        1
    } else if lead < 0xE0 {
        2
    } else if lead < 0xF0 {
        3
    } else {
        4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page(path: &str, kind: &str) -> RehomePage {
        RehomePage {
            path: path.to_string(),
            kind: kind.to_string(),
        }
    }

    #[test]
    fn kind_folder_maps_known_kinds() {
        assert_eq!(kind_folder("decision"), Some("decisions"));
        assert_eq!(kind_folder("gotcha"), Some("gotchas"));
        assert_eq!(kind_folder("rule"), Some("_rules"));
        assert_eq!(kind_folder("fact"), Some("concepts"));
        assert_eq!(kind_folder("concept"), Some("concepts"));
        assert_eq!(kind_folder("note"), Some("notes"));
        assert_eq!(kind_folder("unknown"), None);
        assert_eq!(kind_folder(""), None);
    }

    #[test]
    fn plan_moves_by_kind_and_skips_already_home() {
        let pages = vec![
            page("imported/a.md", "decision"),
            page("omc/b.md", "gotcha"),
            page("imported/c.md", "fact"),
            page("decisions/d.md", "decision"), // already home → no move
            page("imported/e.md", "rule"),
        ];
        let plan = build_rehome_plan(&pages);
        let got: BTreeMap<_, _> = plan
            .moves
            .iter()
            .map(|m| (m.old_path.as_str(), m.new_path.as_str()))
            .collect();
        assert_eq!(got.get("imported/a.md"), Some(&"decisions/a.md"));
        assert_eq!(got.get("omc/b.md"), Some(&"gotchas/b.md"));
        assert_eq!(got.get("imported/c.md"), Some(&"concepts/c.md"));
        assert_eq!(got.get("imported/e.md"), Some(&"_rules/e.md"));
        assert!(!got.contains_key("decisions/d.md"));
        assert_eq!(plan.moves.len(), 4);
    }

    #[test]
    fn plan_skips_colliding_targets() {
        // Two facts with the same slug from different sources → both skipped.
        let pages = vec![page("imported/x.md", "fact"), page("omc/x.md", "fact")];
        let plan = build_rehome_plan(&pages);
        assert!(plan.moves.is_empty());
        assert_eq!(plan.skipped.len(), 2);
    }

    #[test]
    fn plan_skips_occupied_target() {
        // A real concepts/x.md already exists and is NOT moving.
        let pages = vec![page("imported/x.md", "fact"), page("concepts/x.md", "fact")];
        let plan = build_rehome_plan(&pages);
        assert!(plan.moves.is_empty(), "{:?}", plan.moves);
        assert_eq!(plan.skipped.len(), 1);
        assert_eq!(plan.skipped[0].path, "imported/x.md");
    }

    #[test]
    fn rewrite_wikilink_with_and_without_label() {
        let mut map = BTreeMap::new();
        map.insert("imported/a.md".to_string(), "decisions/a.md".to_string());
        let (out, n) = rewrite_links(
            "see [[imported/a.md|The A]] and [[imported/a.md]] here",
            &map,
        );
        assert_eq!(n, 2);
        assert_eq!(
            out,
            "see [[decisions/a.md|The A]] and [[decisions/a.md]] here"
        );
    }

    #[test]
    fn rewrite_preserves_md_suffix_absence() {
        let mut map = BTreeMap::new();
        map.insert("omc/b.md".to_string(), "gotchas/b.md".to_string());
        let (out, n) = rewrite_links("[[omc/b]] and [[omc/b.md]]", &map);
        assert_eq!(n, 2);
        assert_eq!(out, "[[gotchas/b]] and [[gotchas/b.md]]");
    }

    #[test]
    fn rewrite_markdown_link() {
        let mut map = BTreeMap::new();
        map.insert("imported/a.md".to_string(), "concepts/a.md".to_string());
        let (out, n) = rewrite_links("text [label](imported/a.md) end", &map);
        assert_eq!(n, 1);
        assert_eq!(out, "text [label](concepts/a.md) end");
    }

    #[test]
    fn rewrite_leaves_bare_and_unmapped_and_crossscope() {
        let mut map = BTreeMap::new();
        map.insert("imported/a.md".to_string(), "decisions/a.md".to_string());
        let body = "bare [[a]] unmapped [[imported/z.md]] cross [[proj:imported/a.md]] url [x](http://e/a.md)";
        let (out, n) = rewrite_links(body, &map);
        assert_eq!(n, 0);
        assert_eq!(out, body);
    }

    #[test]
    fn rewrite_handles_unicode_body() {
        let mut map = BTreeMap::new();
        map.insert("imported/a.md".to_string(), "decisions/a.md".to_string());
        let (out, n) = rewrite_links("ção é ótimo [[imported/a.md]] não", &map);
        assert_eq!(n, 1);
        assert_eq!(out, "ção é ótimo [[decisions/a.md]] não");
    }
}
