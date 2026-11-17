# PRD: Unified Knowledge Base & Human-Readable Dashboard UI

- **Date:** 2026-11-15
- **Status:** Draft
- **Target Repository:** `ast-context-cache`

---

## Problem Statement

Today, `ast-context-cache` splits agent knowledge across narrow primitives: `structured_memory` only supports compact two-line `fact` and `procedure` entries (auto-superseding facts on matching `subject+predicate` and truncating embeddings at 400 characters), `context_notes` are ephemeral `session_id`-bound dumps subject to quotas and LRU eviction, `scratchpad` entries expire with 7-day handoff trees, and `doc_sources` only cache external URLs. Agents lack first-class ways to record timestamped events, configuration changes with before/after rationale, durable user preferences, and general discoveries—and humans have no UI anywhere in the Dashboard to browse, search, verify, diff, edit, or curate stored memories, `ctx_*` notes, or cached documentation content.

---

## Goals

1. **Unified Knowledge Base Model:** Extend `structured_memory` so agents can store and retrieve all six knowledge kinds (`fact`, `procedure`, `preference`, `event`, `config_change`, `discovery`) with two-tier payloads (compact summary + optional rich Markdown body and structured metadata) without adding MCP tool bloat.
2. **Kind-Accurate Lifecycles:** Preserve append-only chronological integrity for `event`, linked version chains for `config_change`, non-decaying supersession for `preference`, and persistent conflict-checked storage for `discovery`.
3. **Human-Readable Knowledge Hub:** Transform the Dashboard's `Memory` tab into a 5-sub-tab Knowledge Base hub (`Explorer`, `Timeline`, `Preferences`, `Virtual Context`, `Doc Sources`) supporting full-text + semantic search, Markdown rendering, config diffs, human curation (create, edit, verify, forget, restore, quarantine review), and Markdown/JSON export.
4. **Seamless Agent & RAG Ergonomics:** Fix existing MCP friction (`session_id` required for `project`/`global` writes, `retrieve(include_memory=true)` skipping memory when `session_id` is absent, and `FACT:`/`RULE:`-only extraction) while protecting stored knowledge against accidental secret leakage.

---

## Non-Goals (Out of Scope for v1)

1. **No LLM-Based Auto-Extraction or Summarization:** The server will not call external or local LLMs on write to auto-summarize entries, infer facts from arbitrary prose, or resolve conflicts automatically.
2. **No Automated Repo File-Watcher Config Diffing:** Watching arbitrary project files or git commits in indexed repositories to automatically synthesize `config_change` entries is out of scope.
3. **No Internal Operational Noise Logging:** Routine `ast-mcp` telemetry (indexing job ticks, embedder retries, KV repair events, file watcher heartbeats) will not be written into the Knowledge Base timeline; only deliberate `ast-mcp` configuration/flag changes are auto-logged.
4. **No Multi-User Accounts or RBAC:** Access control continues to use the existing single-operator `httpguard` Dashboard and MCP authentication model.
5. **No Cloud Sync or External Database Backend:** All storage remains local-first in SQLite (`context.db`, `index.db`, `usage.db`); manual UI export covers portability for v1.

---

## Functional Requirements

### 1. Knowledge Base Data Model & Entry Kinds

- **FR-1.1 (MUST):** `structured_memory` MUST support six entry kinds: existing `fact` and `procedure`, plus four new kinds: `preference`, `event`, `config_change`, and `discovery`.
- **FR-1.2 (MUST):** Every Knowledge Base entry MUST support a two-tier payload:
  - **Tier 1 (Compact):** Short headline/summary (`title` or `subject` + `predicate` + `object`, or `rule`), `kind`, `scope` (`session`, `project`, `global`), `tags`, `actor`, `occurred_at` / `created_at`, and key metadata fields.
  - **Tier 2 (Detail):** Optional rich Markdown `body` and kind-specific detail fields (`before_value`, `after_value`, `reason`, `target`, `category`, `severity`), returned on demand rather than inflating default compact recall.
- **FR-1.3 (MUST):** Single-entry Tier-2 `body` size MUST be capped at 16 KB (approx. 4,000 tokens). Writes exceeding this limit MUST be rejected with a clear validation error directing the caller to store oversized content via `store_context` and reference its `ctx_*` ref in `source_ref`.
- **FR-1.4 (MUST):** Every entry MUST support optional `tags` (normalized lowercase string list), `actor` (identifying the creating agent, user, or system component), `source_ref` (file path, symbol, commit SHA, URL, or `ctx_*` ref), and a human-curation `verified` flag.
- **FR-1.5 (MUST):** Kind-specific structured fields MUST be supported:
  - `event`: `title` (or `subject`/`object` fallback), `occurred_at` (ISO 8601 / SQL timestamp, defaulting to write time), `category` (e.g., `deployment`, `migration`, `incident`, `decision`, `discovery`, `workaround`), `severity` (`info`, `warning`, `error`, `critical`; default `info`), `target`, and optional Markdown `body`.
  - `config_change`: `target` (config key, file path, or service identifier), `before_value`, `after_value`, `reason`, `actor`, `occurred_at` (defaulting to write time), and optional Markdown `body`.
  - `preference`: `subject` (whose preference, e.g., `user`, `project`, `agent`), `predicate` (topic/key), `object` (preferred behavior/value), and optional `body` for rationale/examples.
  - `discovery`: `title` (or `subject`/`object`), optional Markdown `body`, `category`, `tags`, and `source_ref`.
- **FR-1.6 (MUST — added during refinement):** Existing `fact` and `procedure` callers omitting the new fields MUST remain 100% backward-compatible with current `store_memory`, `recall_memory`, and `forget_memory` inputs and outputs.
- **FR-1.7 (SHOULD):** Vector embeddings for entries with a `title`/`body` SHOULD embed a structured composite of kind, headline, tags, and leading body summary (rather than truncating blindly mid-word at 400 characters) while remaining compatible with the existing 768-dimension vector index.

### 2. Lifecycle, Supersession, Decay & Retention

- **FR-2.1 (MUST):** `preference` entries MUST default to `global` (or `project` when `project_path` is provided and `scope` is omitted), MUST NOT decay with time, and MUST auto-supersede prior active `preference` entries sharing the same `(subject, predicate, scope[, project_path])` unless `invalidate_previous=false`.
- **FR-2.2 (MUST):** `event` entries MUST be append-only (MUST NEVER auto-supersede prior events), MUST NOT decay out of timeline or search ranking, and MUST default to `project` scope when `project_path` is set (otherwise `global` if `session_id` is absent, or `session` if only `session_id` is set).
- **FR-2.3 (MUST):** `config_change` entries MUST form a linked version chain per `(target, scope[, project_path])`: recording a new change for the same `target` MUST link the prior latest entry via `superseded_by` while preserving all prior changes in the Timeline and history queries; `config_change` entries MUST NOT decay with time.
- **FR-2.4 (MUST):** `discovery` entries MUST default to `project` scope when `project_path` is set (otherwise `global`), MUST persist without time-decay (supporting optional explicit `ttl` / `expires_at` only), and MUST run duplicate/conflict candidate detection (`possible_conflicts`) on store just like `fact`.
- **FR-2.5 (MUST):** Active entries of all kinds MUST be retained indefinitely unless explicitly forgotten (`forget_memory` / UI Forget), rejected from quarantine, superseded (for superseding kinds), or expired via an explicit caller-supplied TTL. Soft-deleted/superseded archived entries MUST continue to respect the existing 365-day archive purge window.
- **FR-2.6 (SHOULD):** Entries flagged as `verified` by a human in the UI SHOULD receive a ranking boost in `recall_memory` and MUST NOT be invalidated by automatic TTL/stale-anchor cleanup without explicit human action.

### 3. Capture & Secret Safety

- **FR-3.1 (MUST):** AI agents MUST be able to store all six kinds explicitly via `store_memory`.
- **FR-3.2 (MUST):** `ast-mcp` MUST automatically append a `config_change` entry (with `scope="global"`, `actor="dashboard"` or `"api"`, `target` set to the setting/flag key, `before_value`, `after_value`, and `occurred_at`) whenever `ast-mcp` settings or feature flags are changed through Dashboard settings/flags endpoints. Routine operational events (indexing runs, embedder retries, KV repairs, file watcher ticks) MUST NOT be logged to the Knowledge Base.
- **FR-3.3 (MUST):** All Knowledge Base writes (via MCP, Dashboard UI, or auto-captured `ast-mcp` settings changes) MUST scan string fields (`title`, `subject`, `predicate`, `object`, `rule`, `body`, `before_value`, `after_value`, `reason`) for common secret patterns (API keys, bearer/JWT tokens, private key blocks, URL/connection-string credentials, and high-entropy secret assignments).
- **FR-3.4 (MUST):** When a secret pattern is matched, the server MUST redact the matched secret value in-place with `[REDACTED:<pattern_type>]` before persisting to SQLite, FTS5, or vector storage, and MUST return a non-fatal warning (`warnings` / `redacted_fields`) to the caller identifying what was redacted.
- **FR-3.5 (MUST — added during refinement):** `store_memory` MUST enforce secret redaction before conflict checking and embedding so raw secrets never enter `vectors` or `structured_memory_fts`.

### 4. MCP Tool Enhancements & Auto-Extraction

- **FR-4.1 (MUST):** `store_memory` and `forget_memory` MUST remain in `TierExtended`, and `recall_memory` MUST remain in `TierCore`, without introducing separate redundant top-level MCP tools.
- **FR-4.2 (MUST):** `store_memory` MUST make `session_id` optional in its JSON schema and validation whenever `scope` is `project` or `global`. `session_id` MUST remain required when `scope` is `session` (the default when neither `scope` nor `project_path` is provided, or when explicitly `session`).
- **FR-4.3 (MUST):** `recall_memory` MUST support new filter parameters:
  - `kinds` (array or comma-separated list of kinds; backward-compatible with `kind`)
  - `scope` (`session`, `project`, `global`, or `all`)
  - `tags` (match entries containing all or any specified tags; default any-match with clear parameter docs)
  - `actor` (filter by creator/actor)
  - `since` and `until` (time-range filtering against `occurred_at` for `event`/`config_change` and `created_at`/`valid_from` for other kinds)
  - `verified_only` (boolean)
  - `expand` (boolean; default `false` returns Tier-1 compact lines + summary metadata; `true` includes Tier-2 Markdown `body`, `before_value`, `after_value`, `reason`, `tags`, and provenance)
- **FR-4.4 (MUST):** `recall_memory` MUST support fetching specific entries directly by `refs` (array of `mem_*` IDs) to retrieve Tier-2 details on demand without searching again.
- **FR-4.5 (MUST):** `retrieve(include_memory=true)` MUST retrieve relevant `global` and `project` Knowledge Base entries (including `preference`, `discovery`, recent `config_change`, and relevant `event`/`fact`/`procedure`) even when `session_id` is omitted.
- **FR-4.6 (MUST):** `ExtractFromText` (used by `store_context(..., extract_memory=true)` and `handoff(action="complete")`) MUST be extended to recognize `PREFERENCE:`, `EVENT:`, `CONFIG_CHANGE:`, and `DISCOVERY:` non-fenced line prefixes in addition to existing `FACT:` and `RULE:` prefixes, applying secret redaction before persisting.
- **FR-4.7 (SHOULD):** MCP tool descriptions (`store_memory`, `recall_memory`, `retrieve`) and repository agent/skill guidance (`AGENTS.md`, `CLAUDE.md`, `skills/usage/SKILL.md`) SHOULD document when and how agents should store `preference`, `event`, `config_change`, and `discovery` entries.

### 5. Dashboard UI: Knowledge Base Hub (`Memory` Tab Expansion)

- **FR-5.1 (MUST):** The existing `Memory` tab in the Dashboard UI MUST be expanded into a Knowledge Base hub with five sub-tabs:
  1. `Explorer`
  2. `Timeline`
  3. `Preferences`
  4. `Virtual Context`
  5. `Doc Sources`
- **FR-5.2 (MUST):** When a project is selected in the Dashboard top-bar project selector, `Explorer`, `Timeline`, and `Preferences` MUST display entries belonging to that project (including WTG sibling worktrees via existing project linking) **plus** `global` entries by default, while providing explicit scope filter pills (`All`, `Global`, `Project`, `Session`) to narrow down.

#### 5A. `Explorer` Sub-Tab

- **FR-5.3 (MUST):** `Explorer` MUST provide combined keyword + semantic search across all Knowledge Base entries, with filters for:
  - Kind (`fact`, `procedure`, `preference`, `event`, `config_change`, `discovery`)
  - Scope (`All`, `Global`, `Project`, `Session`)
  - Lifecycle Status (`Active`, `Superseded & Forgotten`, `Quarantined / Review`)
  - Tags, Actor, Date Range (`since` / `until`), and `Verified only`
- **FR-5.4 (MUST):** Clicking an entry MUST open a human-readable detail view (drawer or split pane) rendering:
  - Formatted Markdown for `body` (safely sanitized against XSS — added during refinement)
  - Provenance badges (`ref`, `kind`, `scope`, `actor`, `session_id`, `project_path`, `source_ref`, `created_at` / `occurred_at`, `access_count`, `verified`)
  - Structured `before_value` → `after_value` and `reason` block for `config_change`
  - Linked supersession / version chain (`superseded_by` and predecessor entries) with clickable navigation
- **FR-5.5 (MUST):** Humans MUST be able to create new Knowledge Base entries directly from `Explorer` (choosing kind, scope, project, tags, and Tier-1/Tier-2 fields, with client-side validation for the 16 KB body limit and server-side secret redaction).
- **FR-5.6 (MUST):** Humans MUST be able to edit existing entries in `Explorer` with kind-appropriate versioning:
  - Editing `discovery`, `event`, `fact`, or `procedure` descriptive details MUST update the entry in place (immediately updating FTS5 and re-queuing/updating vector embeddings), with an optional `"Save as new version"` toggle that supersedes the prior entry instead.
  - Editing substantive values of a `preference` (`subject`/`predicate`/`object`) or `config_change` (`target`/`after_value`) MUST create a linked new version (`superseded_by`) preserving the prior version in history. Metadata-only edits (`tags`, `verified` toggle) MAY update in place.
- **FR-5.7 (MUST):** Humans MUST be able to Forget (soft-delete), Restore (`action="restore"`), and toggle `Verified` on entries from `Explorer`.
- **FR-5.8 (MUST):** The `Quarantined / Review` filter MUST surface entries where `quarantined = 1` (and optionally entries flagged with unresolved conflict candidates), allowing humans to **Approve** (clear `quarantined = 0` so the entry enters normal recall) or **Reject** (invoke `RejectQuarantined` / invalidate with reason `"quarantine"`).
- **FR-5.9 (MUST):** `Explorer` MUST provide an Export action allowing the user to download the currently filtered (or selected) Knowledge Base entries as human-readable **Markdown** or **JSON**.

#### 5B. `Timeline` Sub-Tab

- **FR-5.10 (MUST):** `Timeline` MUST display a reverse-chronological feed of `event` and `config_change` entries (ordered by `occurred_at` descending, falling back to `created_at`), grouped by calendar day.
- **FR-5.11 (MUST):** `Timeline` MUST support filtering by kind (`Events & Config Changes`, `Events only`, `Config Changes only`, or `All KB Activity`), severity (`info`, `warning`, `error`, `critical`), category, actor, project/scope, and date range (`since` / `until`).
- **FR-5.12 (MUST):** Each `config_change` row in `Timeline` MUST render an inline readable diff of `before_value` → `after_value`, the `target` key, `reason`, `actor`, and a link to view the full version chain for that `target`.
- **FR-5.13 (SHOULD):** `Timeline` SHOULD provide a quick `"Log Event / Config Change"` button for human entries.

#### 5C. `Preferences` Sub-Tab

- **FR-5.14 (MUST):** `Preferences` MUST present active `preference` and `procedure` entries in a clean, human-scannable layout grouped by `Global` vs `Project` scope.
- **FR-5.15 (MUST):** Humans MUST be able to quick-add a preference/rule, edit an existing preference (creating a linked version on value change), toggle `Verified`, view past superseded preferences for a key, and retire/restore preferences directly in this sub-tab.

#### 5D. `Virtual Context` Sub-Tab

- **FR-5.16 (MUST):** `Virtual Context` MUST preserve the existing aggregate stat cards (`Inventory tokens`, `30d utilization`, `Orphans`) and orphan flush controls, AND add a human-readable browser for individual `ctx_*` notes.
- **FR-5.17 (MUST):** The `ctx_*` note browser MUST support searching/filtering notes (by query, `session_id`, `project_path`, `kind`, `tags`), reading rendered Markdown note bodies with metadata (`ref`, `kind`, `token_est`, `revision`, `access_count`, `last_accessed_at`), inspecting `edit_context` revision history (`context_note_revisions`) with diffs between revisions, and deleting individual notes (with confirmation).

#### 5E. `Doc Sources` Sub-Tab

- **FR-5.18 (MUST):** `Doc Sources` MUST preserve the existing URL source management controls (add source, starter doc pack, sort, refresh, delete, pagination), AND allow clicking a doc source (or searching within cached docs) to inspect its cached `doc_content` pages/chunks in human-readable rendered Markdown/text form.

### 6. Dashboard HTTP API & Security

- **FR-6.1 (MUST):** The Dashboard backend MUST expose HTTP endpoints for listing/searching Knowledge Base entries, fetching single entry details + version history, creating/updating/forgetting/restoring/verifying/approving/rejecting KB entries, exporting KB entries (Markdown/JSON), browsing/reading/deleting `ctx_*` notes and their revisions, and browsing/reading `doc_content` entries for a doc source.
- **FR-6.2 (MUST — added during refinement):** All new Dashboard mutation endpoints MUST pass through the existing `httpguard` authentication/CSRF/origin protections and MUST NOT bypass local/remote access policies.
- **FR-6.3 (MUST — added during refinement):** Destructive UI actions (forget entry, reject quarantined entry, delete `ctx_*` note, bulk operations) MUST require explicit user confirmation before executing.

---

## Non-Functional Requirements

- **NFR-1 (Performance — Search & UI Responsiveness):** Dashboard KB listing, filtering, and FTS search MUST respond in < 100 ms p95 for up to 10,000 stored `structured_memory` rows on local SQLite; `Explorer` and `Timeline` MUST paginate (e.g., 50 items per page) rather than loading unbounded rows into browser memory.
- **NFR-2 (Performance — Recall Latency):** `recall_memory` and `retrieve(include_memory=true)` MUST maintain < 150 ms p95 (excluding cold embedder startup) by keeping Tier-1 recall queries off the Tier-2 `body` column unless `expand=true` or `refs` lookup is requested.
- **NFR-3 (Safety — XSS & Injection):** All rendered Markdown in the Dashboard UI (`body`, `ctx_*` notes, `doc_content`) MUST sanitize HTML / disallow raw script execution to prevent stored XSS from untrusted web docs or agent-written notes.
- **NFR-4 (Reliability — Index Consistency):** Every UI edit, restore, quarantine approval/rejection, or auto-captured `config_change` MUST update `structured_memory_fts` atomically in the same write path and update/delete the corresponding vector in `index.db` (`vectors`), logging non-fatal embed errors without failing the primary SQLite write.
- **NFR-5 (Backward Compatibility & Migration):** Existing `context.db` databases MUST migrate cleanly via the existing `PRAGMA user_version` additive migration pipeline without losing or corrupting existing `fact`, `procedure`, `context_notes`, or `doc_content` rows.
- **NFR-6 (Observability):** KB reads/writes via MCP (`store_memory`, `recall_memory`, `forget_memory`) MUST continue to be logged in `usage.db` (`queries` and `memory_access`), and WebSocket `/ws` updates SHOULD notify open Dashboard tabs when KB entries change.

---

## User Workflows

### Workflow 1: Agent Logs a Configuration Change and Recalls It Later

1. An agent updates a service configuration (e.g., changes PostgreSQL `max_connections` or Docker port mapping) during a session.
2. Agent calls `store_memory(kind="config_change", scope="project", project_path="/abs/repo", target="docker-compose.yml:postgres.max_connections", before_value="100", after_value="200", reason="Avoid connection exhaustion under integration test load", actor="claude-code", body="...details...")`.
3. Server scans all string fields for secrets, redacts any matches with a warning, writes the new entry, links any previous `config_change` for the same `target` in scope via `superseded_by`, indexes FTS5, and queues the vector embedding.
4. Later, another agent (even without a `session_id`) calls `recall_memory(query="postgres connections", kinds=["config_change"], project_path="/abs/repo")` or `retrieve(..., include_memory=true)`.
5. Server returns compact Tier-1 summary lines; if the agent needs the full diff and rationale, it calls `recall_memory(refs=["mem_..."], expand=true)` to load Tier-2 fields.

### Workflow 2: Human Reviews Timeline, Verifies a Discovery, and Updates a Preference

1. Human opens the Dashboard (`http://localhost:7830`), optionally selects a project in the top bar, and opens the `Memory` (Knowledge Base) tab.
2. Human switches to `Timeline` to review recent `event` and `config_change` entries, seeing inline `before_value` → `after_value` diffs, `ast-mcp` flag/setting changes, and agent-logged incidents.
3. Human switches to `Preferences`, notices an outdated coding preference, clicks **Edit**, changes the preferred rule, and saves.
4. Server creates a linked new `preference` version superseding the old one, updates FTS5 and vectors, and displays the updated preference as active while keeping the old version accessible in history.
5. Human switches to `Explorer` → `Quarantined / Review` filter, inspects a flagged/quarantined `discovery`, and clicks **Approve** (making it active and searchable) or **Reject** (confirming dialog, soft-deleting with reason `"quarantine"`).

### Workflow 3: Agent Stores Oversized Output or Secret by Mistake (Error / Edge Paths)

1. **Oversized body:** Agent calls `store_memory(kind="discovery", ..., body=<25 KB log dump>)`. Server rejects the write with an explicit validation error stating the 16 KB limit and advising `store_context` + `source_ref`.
2. **Accidental secret:** Agent calls `store_memory(kind="config_change", after_value="API_KEY=sk-live-12345...", ...)`. Server replaces the secret with `[REDACTED:api_key]`, persists the safe entry, and includes a `warnings` field in the MCP response telling the agent redaction occurred.

### Workflow 4: Human Inspects Virtual Context Notes & Cached Docs

1. Human opens `Memory` → `Virtual Context`, filters `ctx_*` notes by session or keyword, clicks a note to read its rendered Markdown body, and opens the **Revisions** tab to inspect diffs across `edit_context` revisions.
2. Human opens `Memory` → `Doc Sources`, clicks a tracked source (e.g., `React`), and browses or searches its cached `doc_content` pages/chunks in human-readable view.

---

## Integration Points

- **Existing Storage (`context.db`, `index.db`, `usage.db`):** Additive `PRAGMA user_version` schema migrations on `structured_memory` and `structured_memory_fts`; vector upserts/deletes in `index.db` (`vectors`, `doc_type="memory"`); access tracking in `usage.db` (`memory_access`, `queries`).
- **Project Linking (`projectlinks`):** Scope resolution continues to use `ResolveScopeWithRepoSiblings` so `project`-scoped KB entries are shared across WTG sibling worktrees in both MCP tools and Dashboard UI queries.
- **MCP Server (`internal/mcp`):** Extended schemas and handlers for `store_memory`, `recall_memory`, `forget_memory`, `retrieve`, and `ExtractFromText` (`store_context` / `handoff`); respects `AST_MCP_TIER` (`TierCore` vs `TierExtended`) and existing `feature_memory_v2` / `feature_*` flags.
- **Dashboard Settings & Flags (`internal/dashboard`, `internal/flags`):** Settings/flag update paths emit auto-captured `config_change` entries into the KB store.
- **Dashboard UI (`ui/`):** Built into `internal/dashboard/ui/dist` using existing React 19 + MUI 9 + Vite toolchain, reusing `/ws` WebSocket notifications, `ToastContext`, and `httpguard` middleware.

---

## Acceptance Criteria

1. **Given** an agent calls `store_memory` with `kind="event"`, `scope="project"`, and no `session_id`, **when** the request is processed, **then** the entry is persisted with `occurred_at`, is NOT auto-superseded by subsequent events on the same subject, and is returned by `recall_memory` with `since`/`until` filters.
2. **Given** two sequential `store_memory(kind="config_change")` calls for the same `target` in `project` scope, **when** both complete, **then** the first entry has `superseded_by` pointing to the second entry, both entries appear in the Dashboard `Timeline` with `before_value`/`after_value` diffs, and neither decays over time.
3. **Given** an agent calls `store_memory(kind="preference", scope="global", subject="user", predicate="test_runner", object="vitest")` without `session_id`, **when** a second preference for `("user", "test_runner")` is stored with `object="playwright"`, **then** the first preference is superseded, the second is active, and `retrieve(include_memory=true)` returns the active preference even when `session_id` is omitted.
4. **Given** a `store_memory` or UI create/edit payload containing a recognizable secret (e.g., `sk-...`, Bearer token, or PEM private key), **when** persisted, **then** the stored text in SQLite, FTS5, and vector index contains `[REDACTED:<type>]` instead of the raw secret and the response includes a redaction warning.
5. **Given** a `store_memory` payload with a `body` larger than 16 KB, **when** submitted, **then** the write fails with a validation error referencing the 16 KB limit and `store_context`.
6. **Given** a human toggles a feature flag or updates a setting in the Dashboard `Settings` tab, **when** saved, **then** a corresponding `config_change` entry (`before_value`, `after_value`, `target`, `actor`) appears in the Knowledge Base `Timeline`.
7. **Given** a `store_context(..., extract_memory=true)` or `handoff(action="complete")` body containing `PREFERENCE:`, `EVENT:`, `CONFIG_CHANGE:`, or `DISCOVERY:` lines, **when** processed, **then** corresponding typed entries are created in `structured_memory`.
8. **Given** a human opens the Dashboard `Memory` tab, **when** navigating across `Explorer`, `Timeline`, `Preferences`, `Virtual Context`, and `Doc Sources`, **then** they can search/filter KB entries, create/edit/verify/forget/restore entries, approve/reject quarantined entries, export Markdown/JSON, read `ctx_*` notes and their revisions, and inspect cached `doc_content` pages.

---

## Open Questions

1. **Secret Pattern False-Positive Handling:** Should v1 rely solely on tuned regex patterns (with unit tests for common API keys/tokens/PEM blocks), or also expose a per-entry opt-out/allowlist setting if benign identifiers (like git hashes or fixture strings) occasionally trigger false-positive redaction?
2. **MCP `recall_memory` Token Budget Cap:** Currently `feature_memory_v2` hard-caps `recall_memory` at `token_budget <= 2000`. When an agent explicitly passes `expand=true` or requests specific `refs` to read Tier-2 Markdown bodies (up to 4k tokens each), should `expand=true` / `refs` lookups allow a higher budget cap (e.g., 8,000 tokens) or truncate individual bodies with a hint?
3. **Duplicate `ctx_*` Deletion vs. `flush_context`:** In the `Virtual Context` sub-tab, when a human deletes an individual `ctx_*` note (including `offload` or `handoff_result` notes), should `handoff_result` notes be protected/warned against deletion while an active handoff tree exists?
