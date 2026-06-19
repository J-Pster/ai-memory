# AI Ingestion Playbook (agent-driven normalization)

> The deterministic `ai-memory import` writes raw 1:1 pages. This playbook
> is the **second pass**: an AI agent (Claude Code, or any model the user
> picks) reads the imported pages and turns them into a clean, classified,
> de-noised corpus, using only ai-memory's own tools. It is the
> agent-driven alternative to the headless `import --normalize` Rust pass:
> faster, monitorable, model-agnostic, and free of an extra API key (it
> uses whatever agent the user already runs).
>
> An agent given this file + a list of page paths can execute it directly.

## Tools the agent uses
- `ai-memory read-page --project <P> --path <path>` (or MCP `memory_read_page`) — read a page.
- `ai-memory write-page --project <P> --path <path> --body - --kind <k> --tier <t> [--tag <t> ...]` (or MCP `memory_write_page`) — write a normalized page (supersedes the previous version; non-destructive).
- `ai-memory delete-pages --project <P> --prefix <pfx>` / `--paths-file <f>` — bulk-prune noise in one git commit.
- `ai-memory search --project <P> "<q>"` — find duplicates / related pages.

## Step 1 — Prune noise (delete unnecessary pages)
Auto-captured imports often carry low-value pages that pollute the wiki.
Identify and bulk-delete them BEFORE normalizing (don't spend tokens on junk):
- **Session logs**: pages titled `Session Log <date>` or path `*/session-log-*` — per-turn capture noise, almost never durable. Delete all.
- Empty/near-empty pages, pure changelog dumps, or exact-duplicate scaffolding.
Use `delete-pages --prefix` for a whole family (e.g. `omc/session-log-`), or `--paths-file` for a hand-picked list. One git commit, not one-per-page.

## Step 2 — Normalize each remaining page (FAITHFULLY)
For each page, read it, then write it back with:
1. **`kind`** — exactly one of `decision` | `gotcha` | `rule` | `fact`:
   - `decision` — an architectural/process choice with rationale ("we chose X over Y because…").
   - `gotcha` — a bug, footgun, or surprising behavior + how it was handled.
   - `rule` — a durable project convention/policy ("always/never …").
   - `fact` — everything else: feature overviews, data-flow, references, status.
2. **`tier`** — `semantic` for durable knowledge (the default for imports); `procedural` for repeated how-to patterns. Keep `semantic` unless clearly procedural.
3. **Clean** — repair obvious double-encoded UTF-8 mojibake (e.g. `usuÃ¡rios`→`usuários`, `nÃ£o`→`não`), fix broken headings/whitespace.
4. **Preserve** — every fact, every `[[wikilink]]` / markdown link, the page slug (filename), and existing tags. Keep bodies tight (100–400 words of dense fact). The folder may change in Step 2.5; the slug never does.

### Faithfulness (the one hard rule)
This is a MEMORY system; fabrication corrupts it. Do **NOT** invent dates,
versions, file paths, error codes, "alternatives considered", "best
practices", or any detail not already in the page. You only RECLASSIFY,
CLEAN, and TIGHTEN — you never add knowledge the page didn't contain.

## Step 2.5 — Re-home pages by kind (organize the wiki)
The deterministic importer lands every page under its source-provenance
folder (`imported/` for claude-memory, `omc/` for omc-wiki). Once each page
has a `kind`, MOVE it into ai-memory's native kind folder so the wiki is
organized by TYPE, not by where it came from. Keep the same slug:
- `decision` → `decisions/<slug>.md`
- `gotcha`   → `gotchas/<slug>.md`
- `rule`     → `_rules/<slug>.md`  (note the leading underscore — never `rules/`)
- `fact`     → `concepts/<slug>.md`

**Just run the built-in command** — it does the whole thing deterministically:

```
ai-memory rehome --project <P>            # live
ai-memory rehome --project <P> --dry-run  # preview the moves + link count
```

`rehome` moves every classified page into its kind folder, rewrites every
link to a moved page, and is **idempotent** (pages already home are left
alone, so a second run is a no-op). It also runs as the final step of
`import` when you pass `--rehome` (pair it with `--normalize` so kinds exist
first): `ai-memory import --source omc-wiki --omc-wiki-dir <d> --normalize --rehome`.

The single hard rule it enforces (and you must too, if you ever do this by
hand):

> **Rewrite every wikilink to the moved pages, or the link graph dangles.**
> Folder-qualified links (`[[imported/<slug>.md|Label]]`, `[[omc/<slug>.md]]`)
> point at the OLD path and break the moment the page moves. Bare wikilinks
> (`[[<slug>]]`, no folder) resolve by filename and keep working since the
> slug is unchanged — leave them alone. Targets not in the map are
> already-dangling links to deleted pages; leave them as-is.

Collisions never clobber: if two pages would land on the same path, or a
target is already occupied by a non-moving page, `rehome` skips both and
reports them rather than overwriting.

## Step 3 — De-duplicate (optional, careful)
When two pages clearly cover the same concept (often one from each import
source), `search` to confirm, then merge: write the richer page, and
either delete the thinner one (`delete-pages`) or leave a one-line stub
linking to the canonical page. Never merge pages that only look similar by
title — read both first.

## Why agent-driven (vs the Rust `--normalize` pass)
The Rust pass batches pages to a configured LLM (e.g. gpt-5-mini) and is
the right choice for headless/no-agent automation. But when a capable
coding agent is already in the loop, having THAT agent do the pass is
faster to iterate, fully observable (you watch each decision), needs no
separate provider/API key, and lets the user pick the model. Both paths
write through the same supersession-safe `write-page`, so they are
interchangeable and idempotent.
