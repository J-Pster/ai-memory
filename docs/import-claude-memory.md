# `ai-memory import`, Claude Code memory stack importer

> Status: proposed feature. Imports an existing Claude Code "dual-store"
> memory setup (the `@modelcontextprotocol/server-memory` knowledge
> graph + an `mcp-server-qdrant` collection) into the ai-memory wiki as
> native, git-versioned markdown pages with resolved cross-links.

## Motivation

A common Claude Code memory setup before adopting ai-memory is a
**dual store**:

- **Memory Graph**, the `@modelcontextprotocol/server-memory` MCP
  server. Persists a knowledge graph to a JSON-Lines file
  (`memory.jsonl`): one object per line, either an `entity`
  (`name`, `entityType`, `observations[]`) or a `relation`
  (`from`, `to`, `relationType`).
- **Qdrant**, the `mcp-server-qdrant` MCP server. Stores one point per
  memory with payload `{ document: <text>, metadata: { ... } }`. By
  convention the two stores are bridged: `metadata.entityName` on a
  Qdrant point equals the `name` of a Memory Graph entity, so the same
  concept lives in both stores (graph = structure, Qdrant = semantic
  summary).

ai-memory replaces that whole arrangement with a single
source-of-truth wiki. This importer migrates the accrued knowledge in
one shot, **without an LLM** (deterministic, lossless), mapping the
dual store directly onto ai-memory's wiki + link graph:

| Source concept                  | ai-memory target                                  |
|---------------------------------|---------------------------------------------------|
| Memory Graph entity             | one wiki page (`imported/<slug>.md`)              |
| entity `observations[]`         | `## Observations` bullet list                     |
| Memory Graph relation           | wikilink `[[imported/<slug(to)>.md\|<to>]]` under `## Related` |
| Qdrant point (matched by bridge)| `## Summary` section on the matched entity page    |
| Qdrant point (orphan, no entity)| its own `imported/<slug(entityName)>.md` page      |

Relations become real ai-memory links: the server resolves them into a
backlink graph, so the imported memory keeps its structure as
first-class wiki edges (not flattened prose).

## Usage

```bash
ai-memory import \
  --source claude-memory \
  --memory-graph-file ~/.../server-memory/dist/memory.jsonl \
  --qdrant-url http://localhost:6333 \
  --qdrant-collection memory \
  --workspace default \
  --project my-project \
  [--pinned] [--dry-run]
```

- `--source` selects the importer. Only `claude-memory` ships today; the
  flag exists so future sources (other MCP memory servers) slot in
  without a breaking CLI change.
- At least one of `--memory-graph-file` / `--qdrant-url` must be given.
  Either store alone imports fine (graph-only, or Qdrant-only).
- `--dry-run` prints the planned pages (path + title + section summary)
  and writes nothing, mirroring `bootstrap --dry-run`.
- `--pinned` marks every imported page pinned (exempt from the decay
  sweep). Off by default; imported pages are written at the `semantic`
  tier regardless, so they do not decay unless re-tiered.

The command is a **thin HTTP client** like every other state-touching
subcommand: it reads the sources locally, builds the pages, and POSTs
each to `POST /admin/write-page` on the running server. It never opens
the SQLite store or the wiki files directly.

## Generic, not BidChex-specific

The importer assumes only the two public payload shapes above
(`memory.jsonl` entity/relation lines, and the `document` +
`metadata.entityName` Qdrant convention). It hardcodes no project
names, entity types, or topics. Unknown `entityType` values pass
through verbatim into the page frontmatter/body. Text is copied
byte-for-byte (no transcoding), so a source store with pre-existing
encoding quirks round-trips unchanged rather than being "fixed"
lossily.

## Mapping details

### Slug

`slug(name)` = lowercase, every run of non-`[a-z0-9]` collapsed to a
single `-`, leading/trailing `-` trimmed. Deterministic and stable so a
relation's target slug always matches the target entity's page slug.
Collisions (two distinct names → same slug) are disambiguated with a
`-2`, `-3`, … suffix, recorded so relations still resolve.

### Page body (entity)

```markdown
# <entity name>

**Type:** <entityType>
<!-- when a Qdrant point matched, also: -->
**Project:** <metadata.project>  ·  **Date:** <metadata.date>
**Repos:** <metadata.repos joined>

## Observations
- <observation 1>
- <observation 2>

## Summary
<Qdrant document text, when a point matched this entityName>

## Related
- <relationType> → [[imported/<slug(to)>.md|<to name>]]
```

Sections with no content are omitted. An entity with neither
observations nor a matched summary still produces a page (title + type +
relations) so relation targets always resolve.

### Orphan Qdrant point

A point whose `metadata.entityName` matches no entity becomes its own
page: title = `entityName`, body = the `## Summary` (document) plus the
metadata header. This preserves Qdrant-only memories.

### Write request

Each page is sent as the existing `WritePageBody`:
`{ workspace, project, path: "imported/<slug>.md", body, title,
kind: None, tier: "semantic", tags: [<entityType>], pinned }`.

## Crate placement

- **Pure transform** (sources → `Vec<ImportedPage>`), no IO, with unit
  tests: `ai-memory-consolidate` (it already owns the bootstrap
  source→page pipeline). Input types: `GraphEntity`, `GraphRelation`,
  `QdrantPoint`. Output: `ImportedPage { path, title, body, tier, tags,
  pinned }`.
- **IO + orchestration**: `crates/ai-memory-cli/src/commands/import.rs`
 , parse the jsonl file, scroll the Qdrant collection over HTTP
  (`POST /collections/<c>/points/scroll`, paging via
  `next_page_offset`), call the pure mapper, then POST each page (or
  print under `--dry-run`).
- **CLI surface**: `ImportArgs` in `cli.rs`, dispatch in
  `commands/mod.rs` + `main.rs`.

## Tests

Pure-mapper unit tests (no network, no server):
1. entity + matching Qdrant point → one page with Observations +
   Summary sections.
2. relations → wikilinks under Related, slugs match target pages.
3. orphan Qdrant point → standalone page.
4. orphan relation target (no entity) → wikilink still emitted
   (unresolved forward link is valid in ai-memory).
5. slug stability + collision suffixing.
6. byte-for-byte text passthrough (no transcoding).

## Source: `omc-wiki` (oh-my-claudecode Karpathy wiki)

A second importer, selected with `--source omc-wiki`, ingests an
oh-my-claudecode (OMC) Karpathy wiki: a **flat directory of markdown
pages**, each carrying YAML frontmatter, plus one `index.md` manifest
that is skipped. Unlike the dual-store importer this source needs no
LLM and no network, it reads the directory and maps each page directly.

### Source format (one OMC wiki page)

```
---
title: "ACH Requests, Batches & Resident Fee Collection (NACHA debit + credit)"
tags: ["payments","ach","nacha"]
created: 2026-06-03T19:55:31.180Z
updated: 2026-06-03T19:55:31.180Z
sources: []
links: ["payments-fluxo-de-dados-batch-billing.md","payments-integra-es-externas.md"]
category: architecture
confidence: medium
schemaVersion: 1
---

# Title

...markdown body...
```

Frontmatter is YAML between leading `---` fences. `links[]` are bare
filenames of sibling wiki pages. Pages may have no frontmatter or missing
fields; both are tolerated (only a present-but-invalid YAML block is an
error, and even then that single page is skipped with a warning).

### Mapping

For each `*.md` file (except `index.md`) → one `ImportedPage`:

| Source                              | ai-memory target                                   |
|-------------------------------------|----------------------------------------------------|
| filename (verbatim, already `.md`)  | path `omc/<original-filename>`                      |
| frontmatter `title`                 | page title (fallbacks: first `# H1`, then stem)    |
| frontmatter `tags` + `category`     | page tags (de-duplicated, order-stable)            |
| markdown body                       | page body, preserved byte-for-byte                 |
| frontmatter `links[]`               | `## Related (OMC wiki)` of `[[omc/<link>]]` links  |

- **path**: `omc/<original-filename>`, the filename is preserved
  verbatim (it already ends in `.md`); no slugging.
- **title**: frontmatter `title`; else the first markdown `# H1`; else
  the filename stem.
- **tags**: frontmatter `tags` PLUS `category` (when present),
  de-duplicated and order-stable (tags first, then category if new).
- **tier**: always `semantic`. **pinned**: from `--pinned`.
- **body**: the markdown body byte-for-byte, then, only when the page
  has `links`, a trailing section:

  ```markdown

  ## Related (OMC wiki)
  - [[omc/<link-1>]]
  - [[omc/<link-2>]]
  ```

  so ai-memory resolves the links into its backlink graph. The section is
  omitted entirely when there are no links.

### Usage

```bash
ai-memory import \
  --source omc-wiki \
  --omc-wiki-dir ~/.omc/wiki \
  --workspace default \
  --project my-project \
  [--pinned] [--dry-run]
```

`--omc-wiki-dir` is required for this source. `--dry-run` prints the
planned pages and writes nothing (shared with the `claude-memory` source
via the same `write_pages` tail). The dual-store flags
(`--memory-graph-file`, `--qdrant-url`, `--qdrant-collection`) are ignored
by this source.

### Crate placement & tests

Same split as `claude-memory`: the pure transform (`OmcWikiPage` →
`Vec<ImportedPage>`) and its unit tests live in
`ai-memory-consolidate::import` (`build_omc_wiki_pages`); the IO +
frontmatter parsing (`serde_yaml`) lives in
`crates/ai-memory-cli/src/commands/import.rs`. Pure-mapper unit tests:
full-frontmatter mapping; H1/stem title fallback; empty-links omits the
Related section; `category` merges into tags without duplicating;
byte-for-byte body passthrough.

## AI normalization (`--normalize`)

The deterministic importers above are intentionally lossless and 1:1:
every source memory becomes one wiki page, copied byte-for-byte, with no
`kind` classification and `tier = semantic` by default. That faithfulness
is the right default for migration, but it leaves the imported corner of
the wiki rougher than a session-grown one, unclassified, and (for
sources that round-tripped through a Latin-1 mojibake) sometimes garbled.

`--normalize` runs an **LLM first-pass normalization** over the pages
that were just imported. It is **opt-in**: nothing runs unless you pass
the flag, and the server must have an LLM provider configured
(`AI_MEMORY_LLM_PROVIDER`) or the pass fails loud with a clear error.

### Phase-1 contract (per page)

The pass reuses the consolidator's faithfulness guardrails. For EACH
imported page the LLM:

1. **Classifies `kind`** as one of `decision` | `gotcha` | `rule` |
   `fact`, and **confirms or adjusts `tier`** (`working` | `episodic` |
   `semantic` | `procedural`) from the page content.
2. **Cleans the body conservatively**: repairs high-confidence
   double-encoded UTF-8 mojibake (Portuguese text stored as a
   Latin-1-misread UTF-8, e.g. `usuÃ¡rios` → `usuários`, `nÃ£o` → `não`)
   and tidies the H1 / whitespace / broken-markdown structure.
3. **Preserves everything else**: all factual content, every `[[..]]` /
   markdown link, and the page `path` are kept verbatim. The LLM must
   NOT invent dates, sections, alternatives, or any content not present
   in the page. This is a memory system; fabrication corrupts it.

The per-page output is a `ConsolidatedPageUpdate` (same `path`, new
`kind`, `tier`, cleaned `body`), written by **supersession** through the
existing write path. The write is **non-destructive**: the raw imported
version stays in the supersession chain and in git. The pass is
**idempotent**, re-running re-normalizes whatever is currently latest.

### Scope, phase 1 only (no cross-page dedup yet)

This is strictly per-page. There is **no** cross-page deduplication or
merge: two imported pages describing the same concept stay two pages.
Cross-page consolidation (detect duplicates, merge into one canonical
page with redirects/backlinks) is a **deferred phase-2 (v2)** that is
deliberately out of scope here, because a safe merge needs a separate
similarity + supersession design and a human-in-the-loop confirmation
step that phase-1's faithful per-page transform does not.

### Usage

```bash
# Import, then normalize what was imported (one scope, one pass):
ai-memory import \
  --source claude-memory \
  --memory-graph-file ~/.../memory.jsonl \
  --qdrant-url http://localhost:6333 \
  --project my-project \
  --normalize

# Normalize an ALREADY-imported project without re-importing.
# --memory-graph-file / --qdrant-url / --omc-wiki-dir are NOT required:
ai-memory import \
  --source claude-memory \
  --project my-project \
  --normalize-only

# Cheap test: normalize at most 5 pages, write nothing, print the plan:
ai-memory import --source omc-wiki --project my-project \
  --normalize-only --normalize-limit 5 --dry-run
```

- `--normalize` runs the normalize pass **after** a normal import.
- `--normalize-only` **skips** the import/source step and runs only the
  normalize pass over the scope. The source flags are not required.
- `--normalize-limit <N>` caps how many pages are normalized (cheap
  testing).
- `--normalize-max-tokens <N>` overrides the per-batch input-token budget
  (chars/4). Omitted → the server's small reasoning-model-safe default
  (~8k tokens, roughly 5-6 pages/batch). Raise it for fast non-reasoning
  models; lower it if large batches still time out.
- `--dry-run` (shared with the import step) makes the normalize pass
  **compute and print the plan WITHOUT calling the LLM**, pages
  considered, batch count, estimated input tokens, and the list of page
  paths that would be normalized, and write nothing. Because the new
  `kind`/`tier` only exist after the (skipped) LLM call, the plan lists
  paths only; it does not show reclassifications. Dry-run is the
  cost-estimate path, so it must be free.

The scope is **this project + the source's path prefix**:
`claude-memory` normalizes `imported/`, `omc-wiki` normalizes `omc/`.

### Cost note

Unlike the deterministic import (free, no LLM), the normalize pass calls
the LLM once per batch. Pages are batched under a chars/4 input-token
budget, so cost scales with the total size of the imported pages, not the
page count. The default budget is deliberately **small** (~8k input
tokens, roughly 5-6 pages/batch): reasoning-capable models (e.g.
`gpt-5-mini`) spend a long time on large structured-output batches, and a
~40k-token batch reliably overflows the provider's HTTP timeout while an
8k one returns in seconds. Override it with `--normalize-max-tokens` when
you know the model is fast. `--dry-run` is **free**, it never calls the
LLM, so run it first to see the batch count and estimated input tokens
before paying for a live run, and use `--normalize-limit` to bound a test.

### Resilience (retry + skip-and-continue)

Each batch's LLM call is retried up to 3 attempts on **transient**
failures (transport/timeout/5xx) with a short backoff (1s, then 2s). If a
batch still fails after retries, the pass does **not** abort: it skips
that batch, records its page paths, and continues the remaining batches.
The CLI then prints a `⚠ N pages failed (re-run to retry)` line and the
response carries them in `pages_failed`. The pass only fails fast when the
**first** batch fails on a **non-transient** error (auth/400/schema) , 
i.e. the provider is misconfigured and every batch would hit the same
wall. Re-running re-processes the still-raw + failed pages naturally (it
always re-normalizes whatever is currently latest).

### Speed

Two things keep the normalize pass fast:

- **`reasoning_effort=low` for GPT-5 models.** When the configured provider
  is OpenAI (`Official` dialect) and the model id starts with `gpt-5`, every
  LLM request is sent with `reasoning_effort: "low"`. GPT-5 accepts
  `minimal|low|medium|high`; `low` cuts the time the model spends on internal
  reasoning, which dominates latency for summarization-style calls. This
  applies to all GPT-5 LLM calls (consolidate / lint / normalize), not only
  normalize. It is not sent for non-gpt-5 models or the `Compat` dialect.
- **Concurrent batches.** The per-batch LLM calls run with bounded
  concurrency (up to 6 in flight) instead of one-after-another. Each batch
  keeps its own retry + skip-and-continue behaviour; only the page WRITES are
  serialized (they go through the single writer actor). Write order doesn't
  matter, each page is an independent supersession.

### Pruning imported pages (`delete-pages`)

To remove a large set of pages (e.g. discard a bad import before
re-running), use the bulk `delete-pages` command instead of calling
`delete-page` once per page. `delete-page` commits the wiki git repo on
every call (~860ms each), so deleting thousands of pages is dominated by
git; `delete-pages` does all removals then commits **once**.

```bash
# Delete every page under a prefix (one git commit at the end):
ai-memory delete-pages --project myproj --prefix imported/

# Delete an explicit list (newline-separated paths), plus a prefix:
ai-memory delete-pages --project myproj \
  --paths-file ./to-delete.txt --prefix omc/
```

At least one of `--prefix` / `--paths-file` is required. The server expands
`--prefix` to every LATEST page under `(workspace, project)` whose path
starts with it, unions that with the explicit paths, deletes them all, and
returns `{ deleted, not_found }` (explicit paths that matched no page show
up in `not_found`; prefix-derived paths never do).

### Surface

- Server endpoint: `POST /admin/delete-pages` with body
  `{ workspace, project, paths?, path_prefix? }`. Deletes the union of
  `paths` and every latest page under `path_prefix`, committing the wiki
  git repo ONCE. Returns `{ deleted, not_found, checkpoint? }`. A typo'd
  scope returns 404 (no auto-create).
- Server endpoint: `POST /admin/import-normalize` with body
  `{ workspace, project, path_prefixes, limit?, dry_run, max_input_tokens? }`.
  `max_input_tokens` is optional: absent (or `0`) → the small
  reasoning-model-safe default (~8k tokens). It loads the latest pages
  whose path starts with any prefix, batches them under the token budget,
  runs the LLM per batch (with retry + skip-and-continue), and (unless
  `dry_run`) writes each updated page via the write/supersession path.
  Returns `{ pages_considered, batches, estimated_input_tokens,
  pages_updated: [{path, kind, tier}], pages_failed: [path], dry_run }`.
  With no LLM provider configured it returns 4xx with a "configure
  `AI_MEMORY_LLM_PROVIDER`" error.
- Pure logic (system prompt + `build_normalize_request`) lives in
  `crates/ai-memory-consolidate/src/import_normalize.rs`, reusing
  `ConsolidatedPageUpdate` / `ConsolidatedBatch` as the per-page output.

## Re-home by kind (`--rehome` / `ai-memory rehome`)

The deterministic importer lands pages under their source-provenance folder
(`imported/`, `omc/`). Once each page has a `kind` (from `--normalize` or an
agent-driven classify pass), **re-home** moves it into ai-memory's native
kind folder so the wiki is organized by TYPE, not by where it came from:

| kind | folder |
|------|--------|
| `decision` | `decisions/` |
| `gotcha` | `gotchas/` |
| `rule` | `_rules/` (reserved; never `rules/`) |
| `fact` / `concept` | `concepts/` |
| `procedure` | `procedures/` |
| `note` | `notes/` |

The slug (filename) is preserved; only the folder changes. The critical
part: **every link to a moved page is rewritten** (`[[imported/x.md|L]]` →
`[[concepts/x.md|L]]`, and `[label](imported/x.md)` likewise) so the link
graph never dangles. Bare wikilinks (`[[slug]]`, no folder) resolve by
filename and are left untouched. It is **deterministic, LLM-free, and
idempotent**, pages already home are skipped, so a second run is a no-op.
Collisions (two pages onto one path, or an occupied target) are skipped and
reported, never clobbered.

```bash
# Standalone, over an already-classified project:
ai-memory rehome --project myproj
ai-memory rehome --project myproj --dry-run   # preview moves + link count

# As the final step of import (pair with --normalize so kinds exist first):
ai-memory import --source omc-wiki --omc-wiki-dir ~/.omc/wiki \
  --project myproj --normalize --rehome
```

### Surface

- Server endpoint: `POST /admin/rehome-by-kind` with body
  `{ workspace, project, dry_run? }`. Lists the scope's pages, builds the
  `old→new` map by kind, rewrites links across every page, writes movers at
  their new path, deletes the old paths, and commits **once**. Returns
  `{ dry_run, pages_considered, pages_moved, links_rewritten, moves, skipped, checkpoint? }`.
  A typo'd scope returns 404 (no auto-create).
- Pure logic (`build_rehome_plan`, `rewrite_links`, `kind_folder`) lives in
  `crates/ai-memory-consolidate/src/rehome.rs`, no IO, fully unit-tested
  (collisions, occupied targets, `.md`-suffix preservation, unicode bodies).

## Importing the playbook (`import-instructions`)

`ai-memory import-instructions` prints `docs/ai-ingestion-playbook.md` (the
agent-driven second-pass contract) to stdout. The doc is embedded in the
binary via `include_str!`, so an agent can fetch the prune → classify →
re-home → de-dup workflow with one command, no repo-layout knowledge.

## Skipping session-log noise (`omc-wiki`)

oh-my-claudecode writes one auto-capture "session log" page per session.
The `omc-wiki` importer **skips these by default** (filenames `session-log-*`
or titles `Session Log …`) so they don't flood the wiki; pass
`--include-session-logs` to import them anyway.
