# Plan: Unified Knowledge Base & Human-Readable Dashboard UI

- **Summary:** Extend `structured_memory` into a 6-kind, two-tier Knowledge Base with secret redaction, automatic `ast-mcp` config-change logging, improved MCP/RAG retrieval, and a 5-sub-tab human-readable Dashboard Knowledge Hub.
- **Date:** 2026-11-15
- **Related Ticket/Issue:** N/A
- **Source PRD:** [`prd-knowledge-base-and-ui.md`](prd-knowledge-base-and-ui.md)

---

## Context

### Codebase Exploration Summary
- **Storage & Migrations (`internal/db/`):** Three WAL-mode SQLite databases (`index.db`, `context.db`, `usage.db`) managed by `internal/db/pools.go`. Additive migrations use `PRAGMA user_version` in `internal/db/migrate.go` (`contextSteps` currently at `v5` in `internal/db/migrate_steps.go:52-58`), strictly enforced to be additive (no `DROP`, `RENAME`, `ALTER COLUMN`) by `TestStepSQLIsAdditive` in `internal/db/migrate_test.go:105-128`. `structured_memory_fts` (`internal/db/schema_context.go:144`) is a standalone FTS5 table without triggers, maintained manually by `internal/memory/fts.go`.
- **Structured Memory (`internal/memory/`):** Supports `fact` and `procedure` (`types.go:6-10`). `Store` (`store.go:132-237`) uses `factSupersessionMu` to serialize `invalidateConflicting` (`invalidate.go:29-54`). `Recall` (`recall.go:119-189`) fuses FTS5 (`searchFTS`), `LIKE` (`searchLike`), and 768-dim vectors (`search.Cache.SearchMemory` via `internal/search/vector.go:367-400`, truncated to 400 bytes in `embed.go:23`) using RRF (`search.RRF`), applies decay & TTL (`recall_rank.go:124-156`), and defaults to compact lines under `feature_memory_v2` (`flags.KeyMemoryV2`). `quarantined` column (`v5` migration) and `RejectQuarantined` (`prune.go:216-238`) exist, though `validityClause`/`filterClauses` in `recall.go` do not yet exclude `quarantined = 1` rows.
- **MCP Surface (`internal/mcp/`):** `store_memory` (`TierExtended`, `tools.go:380-408`) currently marks `session_id` as required; `recall_memory` (`TierCore`) and `forget_memory` (`TierExtended`) route through `handleMemoryTool` (`memory_tools.go:60-69`). `HandleRetrieve` (`retrieve.go:178-194`) skips `include_memory` whenever `sessionID == ""` and omits `IncludeRepoSiblings: true`. `ExtractFromText` (`internal/memory/extract.go:37-65`) only parses `FACT:` and `RULE:` lines when called from `handleStoreContext` (`context_tools.go:53-68`) and `handoff.Complete` (`internal/handoff/complete.go:93, 164-180`).
- **Context Notes & Docs (`internal/contextnotes/`, `internal/docs/`):** `contextnotes.List` (`store.go:517-578`), `Peek` (`handoff.go:17-30`), and `ListRevisions` (`revision.go:66-81`) already exist; `docs.ListEntriesBySource` (`fetcher.go:220-233`) reads `doc_content` for a source. Neither is exposed for individual browsing in the UI.
- **Dashboard Backend & Security (`internal/dashboard/`, `internal/realtime/`, `internal/httpguard/`):** `NewHandler` (`api.go:70-156`) mounts `registerReactAPI` (`react_api.go:24-39`) and wraps the `ServeMux` in `httpguard.Middleware` (`guard.go:99-109`). `components.MemoryData` (`components/memory.go:5-38`) already includes unused structured memory counts. Settings writes go through `applySettings` (`settings_save.go:224-315`) and `postFlag` (`flags_api.go:32-47`). `realtime.Notify` (`internal/realtime/realtime.go:48-59`) debounces panel refreshes via `flushRefreshBroadcast` (`realtime_bridge.go:15-35`), with `memory` currently refreshing on every 10s `Stats` tick.
- **Dashboard Frontend (`ui/`):** React 19 + MUI 9 + Vite 8 (`ui/package.json`), built into `internal/dashboard/ui/dist` (`ui/vite.config.ts`). `App.tsx` uses local `useState<TabId>` for 6 sidebar tabs; `MemoryTab.tsx` renders only 3 Virtual Context stat cards and the `DocSources` table. No Markdown rendering library is currently installed. `ui/scripts/verify-stories.mjs` runs Playwright smoke checks against Storybook including `dashboard-memory--healthy`.

### Architectural Constraints
1. **Additive SQL Migrations Only:** `TestStepSQLIsAdditive` forbids dropping or altering `structured_memory_fts` directly; expanded FTS columns must be added via a new additive FTS5 table (e.g., `structured_memory_kb_fts`) backfilled during migration.
2. **Tier & Flag Compatibility:** Keep `store_memory`/`forget_memory` in `TierExtended` and `recall_memory` in `TierCore` without adding redundant MCP tools. Features ship always-on respecting `feature_memory_v2` and `AST_MCP_TIER`.
3. **Two-Tier Recall Performance:** Default `recall_memory` and `retrieve(include_memory=true)` must not read or return Tier-2 `body`/diff payloads unless `expand=true` or `refs` lookup is requested.
4. **Vector Delete Ordering:** Always delete vectors (`search.Cache.DeleteRefs`) before deleting/invalidating rows or ensure index-quiesce errors prevent orphan vectors (`prune.go:58-63`).
5. **Dist Assets in Repo:** Updating `ui/` requires rebuilding `internal/dashboard/ui/dist` via `make ui-build` so `go build` embeds the updated SPA.

---

## Requirements

### Functional Requirements (from `prd-knowledge-base-and-ui.md`)
- **FR-1.1 (MUST):** Support 6 kinds in `structured_memory`: `fact`, `procedure`, `preference`, `event`, `config_change`, `discovery`.
- **FR-1.2 (MUST):** Support two-tier payloads: Tier-1 compact summary/metadata (`title`, `subject`/`predicate`/`object`/`rule`, `kind`, `scope`, `tags`, `actor`, timestamps) and Tier-2 rich Markdown `body` + kind-specific detail fields (`before_value`, `after_value`, `reason`, `target`, `category`, `severity`).
- **FR-1.3 (MUST):** Cap Tier-2 `body` at 16 KB (~4,000 tokens); reject oversized writes with validation guidance pointing to `store_context` + `source_ref`.
- **FR-1.4 (MUST):** Support `tags` (normalized lowercase list), `actor`, `source_ref`, and human-curation `verified` flag on every entry.
- **FR-1.5 (MUST):** Support kind-specific fields for `event` (`title`, `occurred_at`, `category`, `severity`, `target`, `body`), `config_change` (`target`, `before_value`, `after_value`, `reason`, `actor`, `occurred_at`, `body`), `preference` (`subject`, `predicate`, `object`, `body`), and `discovery` (`title`, `body`, `category`, `tags`, `source_ref`).
- **FR-1.6 (MUST):** Maintain 100% backward compatibility for existing `fact` and `procedure` callers.
- **FR-1.7 (SHOULD):** Embed a structured composite of kind, headline, tags, and leading body summary (up to ~1,000–1,500 chars / model context budget rather than mid-word 400-byte cut) compatible with 768-dim `vectors`.
- **FR-2.1 (MUST):** `preference` defaults to `global` (or `project` if `project_path` provided without `scope`), never decays, and auto-supersedes active `preference` with same `(subject, predicate, scope[, project_path])` unless `invalidate_previous=false`.
- **FR-2.2 (MUST):** `event` is append-only (never auto-superseded), never decays, defaults to `project` (if `project_path` set), else `global` (if `session_id` empty), else `session`.
- **FR-2.3 (MUST):** `config_change` forms a linked version chain per `(target, scope[, project_path])` via `superseded_by`, preserves full history in Timeline/history queries, and never decays.
- **FR-2.4 (MUST):** `discovery` defaults to `project` (if `project_path` set, else `global`), persists without time-decay (optional explicit TTL only), and runs `ConflictCandidates` on store.
- **FR-2.5 (MUST):** Active entries persist indefinitely unless forgotten, quarantined/rejected, superseded, or TTL-expired; archived rows retain the 365-day `PruneArchived` window.
- **FR-2.6 (SHOULD):** Human-`verified` entries receive a ranking boost in `recall_memory` and are exempt from automated stale-anchor/TTL invalidation.
- **FR-3.1–3.5 (MUST):** Explicit MCP capture + auto-logging of `ast-mcp` Dashboard settings/flag changes (`config_change` per changed non-secret key; `"(updated)"` / `"(cleared)"` for secret keys `embed_http_bearer`, `embed_openai_api_key`, `remote_access_token`, `instance_favicon_upload`; skip operational noise like indexing/embedder/KV repair ticks). Scan all string fields on every write (MCP, UI, auto-capture) using tuned regex patterns (v1, no allowlist/bypass) and redact matches in-place to `[REDACTED:<type>]` before SQLite, FTS5, conflict checks, and vectors, returning non-fatal `warnings` / `redacted_fields`.
- **FR-4.1–4.6 (MUST) & FR-4.7 (SHOULD):** Keep `store_memory`/`forget_memory` (`TierExtended`) and `recall_memory` (`TierCore`); make `session_id` optional for `project`/`global` in `store_memory`; add `kinds`, `scope`, `tags`, `actor`, `since`, `until`, `verified_only`, `expand`, and `refs` to `recall_memory` (2,000-token cap for Tier-1, 8,000-token cap when `expand=true` or `refs` lookup is used, truncating oversized single entries with a hint); fix `retrieve(include_memory=true)` to work without `session_id` and include repo siblings; extend `ExtractFromText` for `PREFERENCE:`, `EVENT:`, `CONFIG_CHANGE:`, `DISCOVERY:`; update tool descriptions and docs (`AGENTS.md`, `CLAUDE.md`, `skills/usage/SKILL.md`).
- **FR-5.1–5.18 (MUST, FR-5.13 SHOULD):** Expand `Memory` tab into a 5-sub-tab Knowledge Hub (`Explorer`, `Timeline`, `Preferences`, `Virtual Context`, `Doc Sources`) with URL query/hash state (`?tab=memory&sub=...&ref=...`), project+global default scoping with scope pills, FTS+semantic search, sanitized Markdown rendering (`react-markdown` + `remark-gfm`), provenance badges, `before_value`→`after_value` diffs, version chains, create/edit (in-place with optional version toggle for `discovery`/`event`/`fact`/`procedure`; linked version on value change for `preference`/`config_change`), forget/restore/verify, Quarantine review queue (Approve/Reject), Markdown/JSON export, `ctx_*` note + `context_note_revisions` browser (warning when deleting active `handoff_result`), and `doc_content` reader inside `Doc Sources`.
- **FR-6.1–6.3 (MUST):** Dashboard HTTP API endpoints under `httpguard.Middleware`, destructive action confirmation dialogs, and dedicated `realtime.KnowledgeBase` WebSocket refresh isolated from open edit modals.

### Non-Functional Requirements
- **NFR-1 (MUST):** Dashboard KB search/list < 100 ms p95 up to 10,000 rows; 50-item pagination.
- **NFR-2 (MUST):** `recall_memory` / `retrieve(include_memory=true)` < 150 ms p95 (excluding cold embedder), avoiding Tier-2 `body` reads unless `expand=true` or `refs` set.
- **NFR-3 (MUST):** Sanitized Markdown rendering (`react-markdown` without raw HTML injection) preventing stored XSS.
- **NFR-4 (MUST):** Atomic FTS5 + SQLite updates and immediate vector re-sync on UI edits/restores/rejections/auto-logs.
- **NFR-5 (MUST):** Clean additive `PRAGMA user_version` migration preserving all existing data.
- **NFR-6 (MUST/SHOULD):** MCP KB queries logged in `usage.db`; `/ws` `realtime.KnowledgeBase` notifies open tabs on KB mutations.

### Out of Scope (Non-Goals)
- LLM-based auto-extraction or summarization on write.
- Automated repo file-watcher/git-hook `config_change` synthesis.
- Logging internal operational telemetry (indexing jobs, embedder retries, KV repair ticks) into the KB.
- Multi-user accounts/RBAC or cloud sync.
- Per-entry secret allowlist/bypass flags in v1.

---

## Approach

### High-Level Strategy
1. **Additive Schema & Unified `internal/memory` Core:** Add new columns to `structured_memory` and create a new `structured_memory_kb_fts` FTS5 virtual table (plus backfill) in `internal/db/migrate_steps.go` (`contextSteps` v6). Extend `memory.Entry`, `StoreInput`, and `RecallInput` cleanly, keeping Tier-1 recall queries off the `body` column unless `expand` or `refs` is requested. Introduce `internal/memory/redact.go` for deterministic regex secret redaction before persistence/conflict-check/embedding.
2. **Backward-Compatible MCP & Auto-Capture:** Update `internal/mcp/tools.go`, `memory_tools.go`, `retrieve.go`, and `internal/memory/extract.go` so existing `fact`/`procedure` callers work identically while unlocking optional `session_id`, `since`/`until`, `tags`, `refs`, `expand` (8k cap), and `include_memory` without `session_id`. Hook `applySettings` and `postFlag` in `internal/dashboard/` to emit `config_change` entries through a shared helper.
3. **Dedicated Dashboard APIs & `realtime.KnowledgeBase`:** Add `/api/dashboard/kb/*` (and `/api/dashboard/context-notes*`, `/api/dashboard/doc-sources/{id}/entries`) in `internal/dashboard/kb_api.go` under `httpguard.Middleware`. Add a dedicated `realtime.KnowledgeBase` bit in `internal/realtime/realtime.go` wired to a `"kb"` panel in `realtime_bridge.go` and `useWebSocket.ts` so KB views refresh live on actual KB/settings/doc/context mutations without interrupting typing on 10s `Stats` ticks.
4. **Modular React 5-Sub-Tab Knowledge Hub:** Refactor `ui/src/tabs/MemoryTab.tsx` into a container managing the 5 sub-tabs (`Explorer`, `Timeline`, `Preferences`, `Virtual Context`, `Doc Sources`) synced to URL search params (`?tab=memory&sub=...&ref=...`) alongside `App.tsx`, with `react-markdown` + `remark-gfm` for safe rendering and draft-isolated modals/drawers for curation.

### Rejected Alternatives
- **Altering `structured_memory_fts` via `DROP` + `CREATE`:** Rejected because `TestStepSQLIsAdditive` explicitly prohibits destructive DDL in `migrate_steps.go`. Adding `structured_memory_kb_fts` and dual-writing both FTS tables (or querying `structured_memory_kb_fts` for all KB queries while keeping `structured_memory_fts` populated for safety) is strictly additive and zero-downtime.
- **Separate `kb_events` / `kb_configs` tables or new top-level MCP tools:** Rejected per PRD to avoid tool bloat and keep unified hybrid search, conflict checks, `source_ref` provenance, and `recall_memory` ergonomics in one model.
- **Reusing `realtime.Stats` / `"memory"` panel for KB lists:** Rejected because `startLiveRefresh()` fires `realtime.Stats` every 10s (`realtime_bridge.go:79`), which would cause disruptive re-fetches while browsing or editing entries.

---

## Style Guide Notes

*(Note: `~/repos/STYLEGUIDE.md` does not exist in this environment; conventions below come from `CLAUDE.md` and the codebase.)*
- **Errors (`internal/errs`):** Use `errs.NewCode(errs.CodeInvalidInput, ...)`, `errs.CodeNotFound`, `errs.CodeConflict`, etc., rather than raw `fmt.Errorf` for domain errors so MCP and HTTP handlers can inspect codes via `errs.HasCode`.
- **Structured Logging (`internal/logging`):** Declare `var logger = logging.Tagged("<pkg>")` and always pass errors as keyed attributes (`logger.Warn("msg", "error", err)`), required by `go vet`'s slog analyzer.
- **SQLite Concurrency & Additive Migrations:** Write migrations as idempotent `schemaStep` entries in `internal/db/migrate_steps.go` using `CREATE ... IF NOT EXISTS` / `ALTER TABLE ... ADD COLUMN` only. Keep `factSupersessionMu` locking around supersession/version-chain read-then-write sequences.
- **Vector Safety:** Delete vectors via `search.Cache.DeleteRefs("memory", ...)` before invalidating/pruning DB rows (`prune.go:58-63`).
- **Build & Lint Hygiene:** Always build/test Go with `GOWORK=off` and `-tags sqlite_fts5`, format with `gofumpt`, and lint UI with `oxlint` (`make lint`, `make ui-test`).

---

## Detailed Implementation Steps

### Phase 1: Schema Migration, Secret Redaction & `internal/memory` Core

1. **Additive DB Migration (`internal/db/migrate_steps.go`, `internal/db/schema_context.go`):**
   - Add `contextSteps` version `6` (`"structured_memory_kb"`) in `internal/db/migrate_steps.go:52-58` (and include columns in `createStructuredMemoryTable` at `schema_context.go:121-143` for fresh DBs) adding columns to `structured_memory`:
     - `title TEXT`, `body TEXT`, `tags TEXT` (JSON array or comma-delimited normalized tags; JSON array string recommended for exact tag matching), `actor TEXT`, `category TEXT`, `severity TEXT`, `target TEXT`, `before_value TEXT`, `after_value TEXT`, `reason TEXT`, `occurred_at TEXT`, `verified INTEGER DEFAULT 0`, `verified_at TEXT`, `redacted_fields TEXT`.
   - Add indexes: `idx_struct_mem_kind_occurred(kind, occurred_at)`, `idx_struct_mem_target(kind, target, scope, valid_until)`.
   - Add `CREATE VIRTUAL TABLE IF NOT EXISTS structured_memory_kb_fts USING fts5(ref, title, subject, predicate, object, rule, body, tags, target, before_value, after_value, reason, category, actor)` and backfill existing `structured_memory` rows inside step `v6` so pre-existing facts/procedures are immediately searchable via the new FTS table.

2. **Types & Validation (`internal/memory/types.go`, `internal/memory/store.go`):**
   - Add `KindPreference = "preference"`, `KindEvent = "event"`, `KindConfigChange = "config_change"`, `KindDiscovery = "discovery"` to `types.go:6-10`, plus `IsValidKind(Kind) bool`.
   - Extend `Entry`, `CompactLine` (include `Title`, `OccurredAt`, `Severity`, `Actor`, `Target`, `Tags`, `Verified`, `Scope`, `ProjectPath`, `CreatedAt` as lightweight Tier-1 metadata), `StoreInput`, and `StoreResult` (add `Warnings []string`, `RedactedFields []string`, `BodyTruncated bool`).
   - Enforce `maxBodyBytes = 16 * 1024` (16 KB) on `StoreInput.Body` in `Store` (`store.go:132`): return `errs.NewCode(errs.CodeInvalidInput, "body exceeds 16 KB limit; use store_context for large content and pass ctx_* ref in source_ref")`.
   - Update scope defaulting (`normalizeScope` in `store.go`):
     - `preference`: default `ScopeProject` if `ProjectPath != ""`, else `ScopeGlobal`.
     - `event`, `discovery`, `config_change`: default `ScopeProject` if `ProjectPath != ""`, else `ScopeGlobal` if `SessionID == ""`, else `ScopeSession`.
     - `fact`, `procedure`: preserve existing behavior (`ScopeSession` default; require `session_id` only when `scope == ScopeSession`).
   - Add per-kind validation in `Store`: normalize `Tags` (lowercase, trimmed, deduplicated), normalize `OccurredAt` (default to `time.Now().UTC().Format(db.SQLTime)` for `event` and `config_change`), validate `Severity` (`"" | info | warning | error | critical`, default `"info"` for `event`), require `Title` or `Subject`+`Object` for `event`/`discovery`, require `Target` and (`AfterValue` or `Body`) for `config_change`, require `Subject`+`Object` (or `Rule`) for `preference`.

3. **Secret Scanning & Auto-Redaction (`internal/memory/redact.go` — new file):**
   - Implement `RedactSecrets(in *StoreInput) (redactedFields []string, warnings []string)` scanning `Title`, `Subject`, `Predicate`, `Object`, `Rule`, `Body`, `BeforeValue`, `AfterValue`, `Reason`.
   - Include tuned regex patterns for common secret formats: OpenAI/Anthropic/Stripe/GitHub/AWS keys (`sk-...`, `sk-ant-...`, `ghp_`/`github_pat_`, `AKIA...`), `Bearer <token>` / JWT (`eyJ...`), PEM private key blocks (`-----BEGIN ... PRIVATE KEY-----`), URL embedded credentials (`://user:pass@host`), and explicit key=value secret assignments (`password=`, `api_key=`, `secret=`, `token=`).
   - Replace matched secret spans with `[REDACTED:<type>]` in-place prior to token estimation, SQLite insert, FTS indexing, `ConflictCandidates`, and `EmbedEntryWithVector`. Return `RedactedFields` and `Warnings` on `StoreResult`.

4. **Lifecycle, Supersession, Decay & Quarantine (`internal/memory/invalidate.go`, `store.go`, `recall.go`, `recall_rank.go`, `history.go`, `prune.go`, `restore.go`):**
   - **Supersession in `Store` (`store.go:215-224`, `invalidate.go:29-54`):**
     - `fact` & `preference`: under `factSupersessionMu`, supersede active entries of same `kind` + `(subject, predicate, scope[, session_id|project_path])` when `InvalidatePrevious` is true.
     - `config_change`: under `factSupersessionMu`, locate latest active `config_change` matching `(target, scope[, session_id|project_path])` and link it (`superseded_by = newRef`, `valid_until = datetime('now')`, `invalidated_reason = ReasonContradiction` or a new `ReasonSuperseded = "superseded"`). Ensure `History` and Dashboard `Timeline` display both active and superseded `config_change` rows while marking the active row as `"current"`.
     - `event` & `discovery` & `procedure`: never auto-supersede.
   - **Conflict Detection (`conflicts.go`):** Run `ConflictCandidates` for both `KindFact` and `KindDiscovery` (matching same kind in scope).
   - **Quarantine Filtering (`recall.go`, `prune.go`):**
     - Update `validityClause` / `filterClauses` in `recall.go` so `recall_memory` and `retrieve` exclude `quarantined = 1` rows by default (unless `IncludeQuarantined: true` for Dashboard review queries).
     - Add `ApproveQuarantined(ref string) error` alongside `RejectQuarantined(ref string) error` (`prune.go:216-238`) to clear `quarantined = 0`, re-index FTS, and queue vector embedding.
   - **Decay & Expiry (`recall_rank.go:124-156`, `prune.go:95-199`):**
     - Exempt `preference`, `event`, `config_change`, `discovery`, and `procedure` (unless explicitly toggled) from `applyDecay` time-decay.
     - Apply ranking boost (e.g. `1.25x`, clamped within existing max ceiling or `[0.3, 1.75]`) for `verified = 1` entries in `applyDecay`.
     - Exempt `verified = 1` entries from automatic `RunStaleAnchorJob` invalidation (`prune.go:151-199`).
   - **Two-Tier `Recall` & Filters (`recall.go`, `recall_rank.go`, `history.go`):**
     - Add `Refs []string`, `Tags []string`, `Actor string`, `Since string`, `Until string`, `VerifiedOnly bool`, `IncludeQuarantined bool`, `IncludeSuperseded bool` to `RecallInput`.
     - Default `Recall` SQL queries select Tier-1 columns only (omitting `body`, `before_value`, `after_value`, `reason` unless `in.Expand || len(in.Refs) > 0`).
     - When `in.Expand || len(in.Refs) > 0`, raise `feature_memory_v2` token budget hard cap from `2000` to `8000`; if an individual expanded entry would still exceed remaining budget on its own, include a cleanly truncated preview + `"... [truncated; fetch ref via Dashboard or smaller budget]"` hint rather than returning empty.
     - Add `Since`/`Until` SQL predicates against `COALESCE(NULLIF(occurred_at,''), created_at)`.
     - Fix `historyByQuery` (`history.go:181-211`) to respect `scope`, `project_path`, and `session_id` filters and search across `title`/`target`/`reason` too.

5. **FTS & Embeddings (`internal/memory/fts.go`, `embed.go`, `format.go`):**
   - Update `fts.go` (`indexFTS`, `deleteFTS`) to maintain both `structured_memory_fts` and `structured_memory_kb_fts`, and query `structured_memory_kb_fts` in `searchFTS` (`recall_rank.go:180-187`).
   - Update `FormatLine` (`format.go`) to format `preference`, `event`, `config_change`, and `discovery` into clean, high-signal single-line Tier-1 summaries (e.g. `[event:warning] <title> (target, occurred_at)` or `[config_change] <target>: <before> -> <after> (reason)`).
   - Add `EmbedText(e Entry) string` in `embed.go` building a composite text (kind, title/triple/rule, tags, target, leading body summary up to 1,200 bytes) instead of truncating blindly at 400 bytes.

6. **UI/Service Helper Operations (`internal/memory/curate.go` — new file):**
   - Implement `UpdateEntry(in UpdateInput, emb embedder.Interface) (*Entry, []string, error)` supporting:
     - In-place edits for `discovery`, `event`, `fact`, `procedure` (or metadata-only edits like `tags` / `verified` on any kind), running `RedactSecrets`, enforcing 16 KB `body` limit, updating `structured_memory`, atomically re-indexing FTS, and re-embedding.
     - Linked-version edits (`NewVersion: true` or substantive value edits on `preference` / `config_change`), creating a new entry under `factSupersessionMu` that supersedes the edited ref.
   - Implement `SetVerified(ref string, verified bool) error`, `QueryKB(in KBQueryInput) (*KBQueryResult, error)` (paginated listing/search for Dashboard Explorer/Timeline/Preferences with counts, status filters `active|archived|quarantined|all`, project+global scope resolution via `projectlinks.ResolveScopeWithRepoSiblings`), and `ExportEntries(in KBQueryInput, format string) ([]byte, error)` (Markdown or JSON).

---

### Phase 2: MCP Tools, `retrieve`, Auto-Extraction & Settings Auto-Capture

1. **MCP Schemas & Handlers (`internal/mcp/tools.go`, `internal/mcp/memory_tools.go`):**
   - Update `store_memory` schema (`tools.go:380-408`, keep `TierExtended`):
     - Change `required` from `["kind", "session_id"]` to `["kind"]` (enforce `session_id` dynamically in `memory.Store` only when `scope == "session"`).
     - Update `kind` description to list `fact`, `procedure`, `preference`, `event`, `config_change`, `discovery`.
     - Add properties: `title`, `body`, `tags` (array of string), `actor`, `category`, `severity`, `target`, `before_value`, `after_value`, `reason`, `occurred_at`.
   - Update `handleStoreMemory` (`memory_tools.go:71-176`) to map new args, run `ConflictCandidates` for `KindFact` and `KindDiscovery`, and include `warnings` / `redacted_fields` in the response JSON when non-empty. Also emit `realtime.Notify(realtime.KnowledgeBase)` on success.
   - Update `recall_memory` schema (`tools.go:409-435`, keep `TierCore`) and `handleRecallMemory` (`memory_tools.go:178-265`):
     - Add properties: `refs` (array of string), `tags`, `actor`, `since`, `until`, `verified_only`.
     - Update `expand` description to explain Tier-1 compact vs Tier-2 full Markdown body/diffs (and 8k budget cap).
     - Pass `scope`, `project_path`, `session_id` into `HistoryInput` when `history=true`.
   - Update `handleForgetMemory` (`memory_tools.go:299-388`) to emit `realtime.Notify(realtime.KnowledgeBase)` on forget/restore.

2. **`retrieve` Integration (`internal/mcp/retrieve.go`):**
   - Update `HandleRetrieve` (`retrieve.go:178-194`): remove the `&& sessionID != ""` gate on `includeMemory`; pass `IncludeRepoSiblings: true` and `SessionID: sessionID` (when `sessionID == ""`, `memory.Recall` naturally queries `global` + `project` entries).

3. **Auto-Extraction (`internal/memory/extract.go`, `store.go`):**
   - Extend `splitMarker` and `Extracted` in `extract.go` to recognize `PREFERENCE:`, `EVENT:`, `CONFIG_CHANGE:`, and `DISCOVERY:` prefixes (case-insensitive, skipping fenced code blocks).
   - Define clean single-line parsing conventions (e.g., `PREFERENCE: user | topic | value`, `EVENT: [severity] title | detail`, `CONFIG_CHANGE: target | before -> after | reason`, `DISCOVERY: title | detail`, with graceful fallback to `Skipped` if unparseable).
   - Update `StoreExtracted` (`store.go:247-279`) to persist extracted `preference`, `event`, `config_change`, and `discovery` items (respecting secret redaction; defaulting `preference`/`discovery` to `project` or `global` if `scope` allows, or keeping caller-provided `scope` with `handoff`/`store_context` guidance).

4. **Automatic `ast-mcp` Settings & Flag Change Logging (`internal/dashboard/settings_save.go`, `internal/dashboard/flags_api.go`, `internal/dashboard/kb_autolog.go` — new file):**
   - Create `recordSettingsConfigChange(target, beforeVal, afterVal, actor string)` in `internal/dashboard/kb_autolog.go`:
     - Calls `memory.Store` with `Kind: KindConfigChange`, `Scope: ScopeGlobal`, `Target: target`, `BeforeValue`, `AfterValue`, `Actor: actor` (`"dashboard"`), `Reason: "Dashboard configuration updated"`.
     - For `secretSettingKeys[key] || key == instance.FaviconUploadKey`: never read or store raw secret values; record `beforeVal`/`afterVal` as `"(set)"`/`"(unset)"` -> `"(updated)"`/`"(cleared)"` based on whether a value existed before and whether the submitted value is non-empty.
   - Hook `applySettings` (`settings_save.go:224-315`) to snapshot prior values of non-secret keys (and presence booleans for secret keys) before writing, and emit one `config_change` entry per actually-changed key after commit, followed by `realtime.Notify(realtime.KnowledgeBase)`.
   - Hook `postFlag` (`flags_api.go:32-47`) to snapshot prior flag state via `flags.State()` and emit a `config_change` entry on change (`target = req.Key`, `before_value = "true"/"false"`, `after_value = ...`), plus `realtime.Notify(realtime.KnowledgeBase)`.

5. **Agent & Skill Documentation (`AGENTS.md`, `CLAUDE.md`, `skills/usage/SKILL.md`):**
   - Update structured memory sections to document the 6 kinds, optional `session_id` for `project`/`global`, Tier-2 `expand`/`refs`, `since`/`until`, secret redaction warnings, and `PREFERENCE:`/`EVENT:`/`CONFIG_CHANGE:`/`DISCOVERY:` extraction prefixes.

---

### Phase 3: Go Dashboard HTTP APIs & Realtime

1. **Realtime Bitmask & WebSocket Panel (`internal/realtime/realtime.go`, `internal/dashboard/realtime_bridge.go`, `ws_react.go`):**
   - Add `KnowledgeBase Reason = 1 << iota` in `internal/realtime/realtime.go:11-23`.
   - Add `{name: "kb"}` to `dashboardPanels` (`ws_react.go:136`) and map `case "kb": return mask&realtime.KnowledgeBase != 0` in `panelMatchesMask` (`realtime_bridge.go:46-70`) and `panelUsesIndexDB("kb") = true` (or `false` if Explorer lists come from `context.db` only—note `vectors` are in `index.db`, so gating semantic search gracefully when index is quiesced is safer, while plain SQLite KB list remains available!).
   - Trigger `realtime.Notify(realtime.KnowledgeBase)` on MCP `store_memory`/`forget_memory`, `store_context`/`edit_context`/`flush_context`, `add_doc_source`/`update_doc_source`/`remove_doc_source`, settings/flag changes, and Dashboard KB/doc/context mutations.

2. **Dashboard Memory Stats Fix (`internal/dashboard/components/memory.go`, `internal/dashboard/react_api.go`):**
   - Ensure `MemoryData` populates counts for all 6 kinds (`ActiveFacts`, `ActiveProcedures`, `ActivePreferences`, `ActiveEvents`, `ActiveConfigChanges`, `ActiveDiscoveries`, `QuarantinedCount`, `StructuredMemoryTokens`).

3. **Knowledge Base HTTP Endpoints (`internal/dashboard/kb_api.go` — new file, registered in `react_api.go:24-39`):**
   - `GET /api/dashboard/kb`: paginated query (`q`, `kinds`, `scope`, `project_id`, `session_id`, `status` [`active|archived|quarantined|all`], `tags`, `actor`, `since`, `until`, `verified_only`, `page`, `per_page`, `sort`). Uses `projectlinks.ResolveScopeWithRepoSiblings(projectID, true)` + `global` entries by default when `project_id` is set.
   - `GET /api/dashboard/kb/entry`: `?ref=mem_...` returning full Tier-2 `Entry` + linked version/supersession chain (`memory.History`) + `ConflictCandidates` (if applicable).
   - `POST /api/dashboard/kb/entry`: create new entry from UI (`actor="human"` default), returns created entry + any redaction `warnings`.
   - `PUT /api/dashboard/kb/entry`: update entry (`new_version` bool supported; substantive `preference`/`config_change` edits force linked new version), returns updated entry + redaction `warnings`.
   - `POST /api/dashboard/kb/entry/action`: dispatch `forget`, `restore`, `verify` (`verified: bool`), `approve_quarantine`, `reject_quarantine`.
   - `GET /api/dashboard/kb/export`: `?format=markdown|json` with same filter params as `/api/dashboard/kb`, returning `Content-Disposition: attachment; filename="knowledge-base.<md|json>"`.

4. **Virtual Context & Doc Content Browsing Endpoints (`internal/dashboard/kb_api.go` or `context_stats.go`):**
   - `GET /api/dashboard/context-notes`: paginated search/list across `context_notes` (`q`, `session_id`, `project_id`, `kind`, `tags`, `page`, `per_page`) returning metadata + snippet; plus `GET /api/dashboard/context-notes/detail?ref=ctx_...` returning full `content` + `context_note_revisions` (`ListRevisions` + revision content for diffing) + whether `kind == "handoff_result"` belongs to an active `handoff_trees` row.
   - `DELETE /api/dashboard/context-notes?ref=ctx_...`: deletes a single `ctx_*` note (reusing `contextnotes.Flush` ref deletion), notifying `realtime.KnowledgeBase` + `realtime.Stats`.
   - `GET /api/dashboard/doc-sources/{id}/entries` (or `/api/dashboard/doc-entries?source_id=&q=&page=`): wraps `docs.ListEntriesBySource` / `docs.SearchDocsLexical` so humans can browse and search cached `doc_content` pages/chunks in the UI.

---

### Phase 4: React 5-Sub-Tab Knowledge Base Hub & Verification

1. **Dependencies & Security (`ui/package.json`):**
   - Install `react-markdown` and `remark-gfm` in `ui/`. Render all Markdown (`body`, `ctx_*` content, `doc_content`) through a shared `<MarkdownView content={...} />` component (`ui/src/components/MarkdownView.tsx`) using `react-markdown` with `remark-gfm` and **without** raw HTML plugins (`rehype-raw` explicitly omitted), preventing stored XSS.

2. **Types, API Client & Live Refresh (`ui/src/api/types.ts`, `ui/src/api/client.ts`, `ui/src/hooks/useWebSocket.ts`, `ui/src/App.tsx`):**
   - Add TypeScript interfaces for `KBEntry`, `KBListResponse`, `KBDetailResponse`, `KBCreateInput`, `KBUpdateInput`, `ContextNoteItem`, `ContextNoteDetail`, `DocContentEntry`, and update `MemoryData`.
   - Add typed methods to `api` in `ui/src/api/client.ts`.
   - Map `kb: ['kb']` in `panelToQuery` (`ui/src/hooks/useWebSocket.ts:3-15`) and pass a `kbRefreshNonce` (or reload callback) into `MemoryTab` in `App.tsx` so active views update without resetting open create/edit forms or search inputs.
   - Sync `tab`, `sub` (sub-tab), and `ref` with `window.location.search` / `popstate` in `App.tsx` + `MemoryTab.tsx` so `?tab=memory&sub=timeline&ref=mem_...` URLs are bookmarkable and shareable while preserving sidebar click behavior.

3. **Knowledge Base Hub Components (`ui/src/tabs/MemoryTab.tsx` + `ui/src/components/kb/`):**
   - Refactor `MemoryTab.tsx` into a 5-sub-tab MUI `Tabs` shell (`Explorer`, `Timeline`, `Preferences`, `Virtual Context`, `Doc Sources`), keeping `"Virtual context"` and `"Memory"` visible labels intact for `verify-stories.mjs`.
   - **`KBExplorerSubtab.tsx`:** Search input, filter bar (Kind chips, Scope pills `All`/`Global`/`Project`/`Session`, Status `Active`/`Superseded & Forgotten`/`Quarantined`, Tag/Actor/Date inputs, `Verified only`), paginated table/cards (50/page), `+ New Entry` dialog, Export (`Markdown` / `JSON`) menu, and slide-over **`KBDetailDrawer.tsx`** (Markdown body, provenance chips, `before_value`→`after_value` diff block, clickable version/supersession chain, Edit form with `Save as new version` toggle for non-versioned kinds, Verify/Forget/Restore/Approve/Reject buttons, confirmation dialogs for destructive actions, and secret-redaction toast warnings).
   - **`KBTimelineSubtab.tsx`:** Day-grouped reverse-chronological timeline of `event` and `config_change` entries (with filter for `All Activity`, severity chips, category/actor/date filters, inline readable `before_value` → `after_value` diff, `reason`, link to open target version chain in `KBDetailDrawer`, and `+ Log Event / Config Change` button).
   - **`KBPreferencesSubtab.tsx`:** Human-scannable view grouping active `preference` and `procedure` entries under `Global` and `Project` sections, with quick-add bar, inline/modal edit (auto-creating linked version on value change), Verify toggle, history drawer, and Retire/Restore actions.
   - **`VirtualContextSubtab.tsx`:** Preserves the 3 existing `StatCard`s + `Flush orphans` button, plus searchable `ctx_*` notes table (session/project/kind/tags filters) and slide-over reader displaying rendered Markdown, token/access metadata, `context_note_revisions` selector/diff viewer, and Delete confirmation (displaying an extra warning banner if `kind === 'handoff_result'` belongs to an active handoff tree).
   - **`DocSourcesSubtab.tsx`:** Preserves existing `DocSources` CRUD table + starter pack installer, plus clicking a source (or searching) opens a `doc_content` drawer/panel rendering cached pages/chunks via `<MarkdownView />`.

4. **Storybook, Build & Assets (`ui/src/stories/`, `internal/dashboard/ui/dist`):**
   - Update/add Storybook stories for `MemoryTab` sub-tabs (`Explorer`, `Timeline`, `Preferences`, `Virtual Context`, `Doc Sources`) using mocked API/props so `dashboard-memory--healthy` continues to pass `ui/scripts/verify-stories.mjs`.
   - Run `make ui-build` to regenerate `internal/dashboard/ui/dist`.

---

## Testing Strategy

### Test Frameworks & Conventions
- **Go (`internal/db`, `internal/memory`, `internal/mcp`, `internal/dashboard`):** `dbtest.Init(t)` (`store_test.go:14-17`), `stubEmbedder` / `unitVector` (`vector_recall_test.go:21-38`), `httptest.NewRecorder()` (`memory_tools_test.go`), `make test`, `make race`, `make lint`.
- **Frontend (`ui/`):** Vitest + jsdom (`ui/vitest.config.ts`, `SettingsTab.test.tsx`), `oxlint`, and Playwright Storybook smoke (`make verify-stories`).

### Acceptance Criteria Mapping
| PRD AC | Target Test File(s) | Test Cases to Implement |
|---|---|---|
| **AC-1** (`event` without `session_id`, `occurred_at`, append-only, `since`/`until`) | `internal/memory/kb_test.go`, `internal/mcp/memory_tools_test.go` | `TestStoreEventOptionalSessionAppendOnly`; `TestRecallMemorySinceUntilFilters` |
| **AC-2** (`config_change` version chain, timeline history, no decay) | `internal/memory/kb_test.go`, `internal/memory/recall_rank_test.go` | `TestConfigChangeVersionChainAndNoDecay`; `TestHistoryConfigChangeTargetChain` |
| **AC-3** (`preference` supersession, no decay, `retrieve` without `session_id`) | `internal/memory/kb_test.go`, `internal/mcp/retrieve_test.go` | `TestPreferenceSupersessionAndNoDecay`; `TestRetrieveIncludeMemoryWithoutSessionID` |
| **AC-4** (Secret auto-redaction before DB/FTS/vectors + warning) | `internal/memory/redact_test.go`, `internal/mcp/memory_tools_test.go` | `TestRedactSecretsBeforePersistenceAndEmbedding`; `TestStoreMemoryReturnsRedactionWarning` |
| **AC-5** (16 KB body limit rejection) | `internal/memory/kb_test.go`, `internal/mcp/memory_tools_test.go` | `TestStoreMemoryRejectsOversizedBodyWithHint` |
| **AC-6** (Dashboard settings/flags auto-log `config_change` + secret masking) | `internal/dashboard/settings_save_test.go`, `internal/dashboard/flags_api_test.go` | `TestSettingsBatchAutoLogsConfigChangeAndMasksSecrets`; `TestFlagToggleAutoLogsConfigChange` |
| **AC-7** (`ExtractFromText` for `PREFERENCE:`, `EVENT:`, `CONFIG_CHANGE:`, `DISCOVERY:`) | `internal/memory/extract_test.go`, `internal/mcp/memory_tools_test.go` | `TestExtractFromTextAllKBPrefixes`; `TestStoreContextExtractsAllKBKinds` |
| **AC-8** (5-sub-tab Dashboard UI CRUD, Quarantine, Export, `ctx_*` revisions, `doc_content`) | `internal/dashboard/kb_api_test.go`, `ui/src/tabs/MemoryTab.test.tsx`, `ui/scripts/verify-stories.mjs` | Go API tests for `/api/dashboard/kb*`, `/api/dashboard/context-notes*`, `/api/dashboard/doc-entries`, `httpguard` checks; Vitest tests for sub-tab switching, URL state, secret warning toast, Markdown XSS sanitization, `handoff_result` delete warning, confirmation dialogs; `make verify-stories` |
| **Migration & Additive Rule** | `internal/db/migrate_test.go` | Existing `TestStepSQLIsAdditive` + new `TestContextStepsV6BackfillsKBFTS` |
| **Two-Tier Budget & Expand** | `internal/memory/kb_test.go`, `internal/mcp/memory_tools_test.go` | `TestRecallTwoTierOmitBodyDefaultExpand8kCap` |

### Verification Commands
- `make lint` (`go vet` + `gofumpt` check)
- `make test` & `make race`
- `make ui-test` & `cd ui && npm run lint`
- `make ui-build` & `make verify-stories`
- `make build`

---

## Risks & Open Questions

1. **Supply-Chain Audit on New npm Packages (`react-markdown`, `remark-gfm`):** Installing new JS dependencies updates `ui/package-lock.json`; verify clean install via `npm ci` and check against repo npm/pnpm hygiene rules (minimum release age / no lifecycle scripts) before committing.
2. **Migration Backfill Time on Large `context.db`:** Step `v6` backfills `structured_memory_kb_fts` from existing `structured_memory` rows inside an `IMMEDIATE` transaction. Existing `structured_memory` tables are typically small (hundreds to low thousands of rows), so backfill should take milliseconds, but should still be tested with a 10,000-row benchmark fixture to validate **NFR-1**.
3. **Secret Regex False Positives:** Because v1 ships tuned regexes without an allowlist/override (`PRD Open Question 1`), generic high-entropy hex strings (e.g., git SHAs, `ctx_*`/`mem_*` refs) must NOT trigger redaction; pattern tests must explicitly assert zero false positives on git SHAs, UUIDs, and `ast-context-cache` refs.
4. **Storybook Smoke Signal Compatibility:** `ui/scripts/verify-stories.mjs` asserts `dashboard-memory--healthy` renders `"Virtual context"` and `"Memory"`; ensure the expanded `MemoryTab` header and sub-tab labels keep those exact strings visible in the default story state.
