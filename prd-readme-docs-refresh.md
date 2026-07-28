# PRD: README & docs refresh (hero, Storybook, screenshots)

- **Date:** 2026-07-28
- **Status:** Approved
- **Related:** Follows v3.0 platform hardening (React dashboard, virtual context, metrics, supervise/Docker). No Jira ticket.

## Problem Statement

The README is accurate but hard to scan: it buries the product goal under a long feature list and duplicates agent workflows that already live in `AGENTS.md` / skills. The dashboard Storybook still documents the removed **templ/HTML** UI, so README screenshots and `make dashboard-screenshot` no longer represent what operators see on port 7830. Humans evaluating the project, operators installing it, and agents configuring MCP all need an up-to-date, layered front door without three conflicting sources of truth.

## Goals

1. Give **humans** a half-screen hero that states the goal and the highest-value capabilities (token savings, context survival, local-first, live dashboard).
2. Give **operators** a clear quick start and pointers to deeper config (embed backends moved out of the README body).
3. Give **agents** a thin MCP install pointer plus link to `AGENTS.md` (no full workflow duplication in README).
4. Replace Storybook with **real React `ui/` stories** for every surface needed for screenshots.
5. Produce **dark-theme** screenshots from Storybook, **visually verified** against production with Webwright as fallback/confirmation; commit images under `docs/images/`.
6. Keep status **badges** (release/version, license) and in-repo `docs/storybook-static/` unless a better hosting path is adopted later.

## Non-Goals (Out of Scope)

- Rewriting `AGENTS.md` / `skills/*` content (README links to them; no merge of full agent manuals into README).
- Light-theme screenshots or marketing site redesign beyond README + Storybook.
- Pixel-perfect CI diff against live dashboard (visual QA is enough).
- Changing MCP protocol, tool tiers, or product behavior (docs/presentation only).
- Full Storybook coverage of every micro-component (only what’s needed for agreed screenshots).
- Automating Webwright in CI unless already trivial; capture path remains Makefile/Storybook-first.

## Functional Requirements

### README structure & audiences

1. The README MUST open with a **half-screen hero** aimed at all three audiences (evaluators, operators, agents): product name, one short goal statement, primary-feature bullets, and the Overview hero screenshot(s).
2. The README MUST use a **layered** layout after the hero: Quick Start (operators) → Configure editor / MCP JSON (agents + operators) → Features (prioritized) → Architecture / links → License.
3. Agent workflow detail MUST NOT be duplicated at length in the README; the README MUST provide a **thin pointer** to [`AGENTS.md`](AGENTS.md) (and MAY link `skills/usage`) plus **MCP JSON** snippets for Cursor/OpenCode/etc.
4. Detailed **embedding backend** tables and env dumps MUST be **moved out** of the README body into operator-facing docs (e.g. `docs/` and/or `skills/operator`) with a short pointer remaining in README.
5. The README MUST call out **all** product capabilities somewhere, but MUST feature primary-goal capabilities higher (see Feature priority below).
6. The README MUST retain **License** and **badges**; it MUST add/keep **release/version** and **License** badges (status-relevant). Other badges MAY be added if useful and low-maintenance.
7. Sacred/keep content that MUST remain findable: license, badges, mcp-local pointer (may be shortened).

### Feature priority (ordering in README)

**Hero / primary (human-facing, top):** in this order:

1. Token-efficient code search (modes, session dedup, measured Tokens saved) — **A1**
2. Virtual context (survive host compaction via `ctx_*`) — **A2**
3. Local-first / no cloud / no account — **A8**
4. Operator dashboard (live status, embed queue, savings) — **A9**
5. Then remaining A-list in existing inventory order: KV repair (**A3**), structured memory (**A4**), precise AST hybrid search (**A5**), impact graph (**A6**), RAG `retrieve` (**A7**), doc caching (**A10**)

**Supporting (later / secondary):**

1. Configurable embed backends — **B3**
2. Aux embedder catch-up pool — **B4**
3. Multi-language indexing — **B1**
4. File watcher + incremental re-index — **B2**
5. Remaining B-list in any reasonable order: tool tiers, code-mode, pin/queue, analysis, bundles, supervise/Docker, Prometheus metrics, stuck-worker recover, 3-DB/WAL, pipeline stats, mcp-local pointer

### Storybook

8. Storybook MUST be updated to render **real React components from `ui/`** (not legacy templ HTML fixtures / `styles.css`-only snapshots).
9. Storybook MUST include stories sufficient for all screenshot subjects defined below.
10. Storybook docs/README MUST describe the React dashboard (port 7830), not the removed HTMX/templ path.
11. `docs/storybook-static/` MUST remain buildable into the repo (`npm run build-storybook` / `make storybook` as applicable) unless replaced by an explicitly better in-repo approach.
12. `make dashboard-screenshot` (or a clearly documented successor) SHOULD remain the official capture entrypoint; if replaced, README and Makefile MUST point to one command.

### Screenshots

13. Screenshots MUST be **dark theme only**.
14. Screenshots MUST be **committed** under `docs/images/` (or an agreed subpath) and referenced from the README.
15. **Overview:** MUST include up to **2** images (hero is most important; second MAY show Index & runtime / confidence strip).
16. **Memory:** MUST include up to **2** images.
17. Additional pages: implementer SHOULD add up to **2** images per chosen page; recommended minimum set: **Embeddings / Index & runtime** (1–2) and **Settings** (1), unless Storybook time is constrained—then Overview + Memory alone still satisfy the hero/examples bar.
18. Capture MUST use **Storybook** as the primary renderer.
19. **Webwright** MUST be used as fallback and/or to **visually confirm** Storybook output matches what production React looks like; **visual judgment is enough** (no pixel-diff gate required).
20. README MUST embed at least the Overview hero image near the top; other images SHOULD appear near the sections they illustrate or in a short Gallery subsection.

### Process / tooling

21. Updating Storybook fixtures/stories MUST be part of the same effort as the README rewrite (acceptance requires all three: README, Storybook, screenshots).
22. Capture scripts MUST work from a built Storybook (or documented Storybook URL) without requiring a live indexed codebase, using fixtures/mocks for API data.

## Non-Functional Requirements

1. **Scannability:** A new reader MUST understand goal + primary value in under ~30 seconds from the hero alone.
2. **Maintainability:** Screenshots SHOULD be regenerable via one documented Makefile/`npm` path after UI changes.
3. **Size:** Committed PNGs SHOULD stay reasonably sized for GitHub (compress if needed); no full-page `full_page` capture requirement beyond what Storybook stories need.
4. **Consistency:** Storybook stories SHOULD use the same MUI theme/tokens as production `ui/` so visual QA is meaningful.
5. **No secrets:** Fixtures MUST NOT include real API keys, tokens, or private paths beyond anonymized placeholders.

## User Workflows

### Human evaluator

1. Open README on GitHub.
2. Read hero goal + primary bullets + Overview screenshot.
3. Decide whether to clone / install from Quick Start.

### Operator

1. Follow Quick Start (`make setup` / `make run` or `ast-mcp`).
2. Open dashboard; optionally regenerate screenshots via Storybook Makefile target after UI changes.
3. Follow pointer for embed backend details in docs/operator skill.

### Agent / integrator

1. Copy MCP JSON from README.
2. Follow link to `AGENTS.md` for workflow, tiers, virtual context, and tool usage.
3. Does not need to parse full token/pipeline essays from README.

### Doc maintainer (screenshot refresh)

1. Update React Storybook stories if UI changed.
2. Run Storybook build + capture command.
3. Optionally run Webwright visual check against Storybook (and/or live `:7830` if available).
4. Commit images + README references.

## Integration Points

| System | Role |
|--------|------|
| `ui/` React dashboard | Source of truth for Storybook stories and visual look |
| `dashboard-storybook/` | Storybook app; must stop depending on dead templ HTML as production truth |
| `docs/images/` | Committed screenshots for README |
| `docs/storybook-static/` | Built Storybook static site in-repo |
| `Makefile` (`storybook`, `dashboard-screenshot`, etc.) | Official build/capture entrypoints |
| Webwright / Playwright | Visual confirmation / fallback capture |
| `AGENTS.md`, `skills/*` | Canonical agent + operator deep docs |
| New/updated operator doc for embed backends | Destination for tables moved out of README |
| GitHub README badges | Release/version + license |

## Acceptance Criteria

1. Given a cold README view, when a human reads only the hero, then they can state the product goal and at least token savings + virtual context + local-first + dashboard as primary themes.
2. Given the README, when searching for long agent workflows, then they are directed to `AGENTS.md` rather than a duplicated full manual.
3. Given the README, when looking up embed backend env tables, then those tables live outside the main README body with a working link.
4. Given Storybook, when opening Overview/Memory/(agreed) stories, then they render React `ui/` components (not legacy templ HTML fixtures as the documented production UI).
5. Given `make dashboard-screenshot` (or successor), when run successfully, then Overview images under `docs/images/` update and are referenced by README.
6. Given Webwright (or equivalent Playwright visual check), when comparing Storybook Overview to production React look, then a maintainer judges them acceptably the same (dark theme).
7. Given Memory stories, when screenshots are captured, then up to two Memory images exist and are linked or gallery-listed in README.
8. Given badges on README, when viewed on GitHub, then release/version and license badges render.
9. Given `docs/storybook-static/`, when built, then it reflects the updated React Storybook.
10. Given the PR for this work, when reviewed, then README + Storybook + committed screenshots ship together.

## Open Questions

1. Exact badge URLs/services for **release/version** (GitHub release, Shields.io `v` from `VERSION`, GitHub Package, etc.) — implementer SHOULD pick the simplest accurate badge unless the user specifies.
2. Final non-Overview/Memory screenshot set beyond the recommended Embeddings + Settings — implementer MAY trim if schedule-constrained; Overview:2 + Memory:2 remain MUST.
3. Whether embed-backend content moves to a new `docs/embedding-backends.md` vs extending `skills/operator/SKILL.md` only — either is fine if README links clearly.
4. Whether Webwright confirmation is a checked-in script under the repo vs a one-off maintainer run documented in Storybook README — SHOULD document at least one reproducible command.

## Decisions captured from refinement

| Topic | Decision |
|-------|----------|
| Audiences | Humans, operators, and agents (layered README) |
| Hero | Half-screen |
| Agent docs | Link `AGENTS.md`; thin pointer + MCP JSON |
| Embed tables | Move out of README |
| Feature order | A1>A2>A8>A9 then A3…A10; B3>B4>B1>B2 then rest |
| Screenshots | Overview:2, Memory:2; dark only; commit PNGs; Storybook primary; Webwright visual confirm |
| Storybook | Real `ui/` React stories; keep static build in-repo |
| Capture command | Keep `make dashboard-screenshot` unless a better single entrypoint replaces it |
| Done definition | README + Storybook + screenshots |

## Added during refinement (review)

- Explicit ban on treating templ HTML Storybook as production truth (current Storybook README is wrong post-v3).
- Recommend Embeddings + Settings shots as SHOULD, not only Overview/Memory.
- Require fixtures without secrets.
- Require layered README so agents are not the primary voice in the hero.
