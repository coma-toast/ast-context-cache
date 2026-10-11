# Plan: ast-context-cache 5.0, token savings, memory, and repair

This plan makes ast-context-cache return less and stay cache-stable, and makes its token measurement honest. It also gives memory history, conflict candidates and recency-aware recall, and lets the data heal itself. The work ships in five phase PRs (P0–P4).

- **Date:** 2026-10-09
- **Source PRD:** [prd-context-techniques-v5.md](prd-context-techniques-v5.md)
- **Related:**
  - Research report `reports/Agent memory and token techniques.md` in `~/git/ast-context-cache` (git-excluded).
  - Paperclip JD-208.
  - No Jira ticket.
- **Delivery:**
  - One PR per phase. Each phase branch is cut from `main` after the previous phase merges; branches are not stacked.
  - One wtg space for the implementation (`ai-v5-context`), with a new branch per phase: `NO-TICKET-v5-p0-fixes`, `-p1-tokens`, `-p2-memory`, `-p3-repair`, `-p4-agents`.
  - Agents work in parallel per independent stream within a phase, committing and pushing after each verified step. PRs need the user's approval; the user merges.
  - P0 ships as 4.0.x. P1 sets `VERSION` to 5.0.0 by hand and adds a `## Migrating to 5.0` section to `docs/MIGRATING.md`. P2–P4 ship as 5.x, auto-bumped on merge.
- **Code references** are as of `440defe` (v4.0.11). Each phase re-checks line numbers against the merged previous phase.
- **Exploration digest:** `../exploration-notes.md` in the space root, which is not committed.

---

## Context

### Explored

**MCP envelope** (`internal/mcp/server.go`)
- `handleToolCall` (229–670) parses arguments and sets `sid` at 280. It then runs the per-tool switch (285–649) and builds one envelope at 655–669:
  - `content[0].text` is the `json.Marshal` of the result (657).
  - An optional second content block carries the claims notice (658–660); it is the precedent for extra lines.
  - `isError` is set via `resultIsError` (663).
- Nothing is put in `_meta`. `modernize` (`protocol.go:236–258`) preserves an existing `result._meta`, so the place to add one is between 664 and 665.
- The deny envelope is at 250–262.
- Each code tool re-marshals its own body:
  - capsule: 330
  - search_semantic: 431
  - get_file_context: 466 (via `annotateFileContext`)
  - retrieve: 600
- **retrieve already takes a `format` argument** (markdown|xml|json|plain; `retrieve.go:117–120`) for its assembled context.

**Packing**
- Capsule:
  - `internal/context/handler.go` `handleGetContext` (34–138) hard-codes `Limit: 30` (63).
  - Its loop (83–113) dedups, then calls `EffectiveMode` and `ApplyMode`, and `break`s at the first result over budget (104–106). It records the *requested* mode as delivered (112).
- search_semantic: `PackScoredResults` (143–198) mirrors that loop, with its `break` at 182–184.
- get_file_context: `internal/mcp/handlers.go` `handleFileContextWithMeta` (160–277) passes `maxScore=1.0`, so `auto` there means `full`.
- `EffectiveMode` (`pack.go:21–39`) picks full, skeleton or summary by score ratio. That contradicts the "top 3 full" text in the tool descriptions.
- retrieve: `internal/mcp/retrieve.go`
  - It serializes the assembled `Context` plus `Chunks[].Content`, so the content is sent twice.
  - The include_memory split has an 800-token floor that can exceed the caller's budget (126–149).
  - `rankAndDedup` (397–416) mixes RRF-scale and doc-scale scores.
  - The stats JSON is pinned by `TestRetrieveStatsJSONGolden` (`retrieve_test.go:13–38`).

**Ordering bugs**
- `search/vector.go` `topMatches` (233–278) leaves results unsorted when there are ≤ `limit` candidates. `SearchDoc` (364–375) and `SearchNote` (415–426) share the bug.
- `hybrid.go` fuses results from map iteration and sorts by score only, with an unstable sort (77–82).
- The BM25 SQL has no secondary order key (`bm25.go` 20–22, 29–31). The fallback has no ORDER BY (32–34).
- `docs/search.go` fuse (214–218) has the same problem.
- No sort anywhere breaks ties.

**Session dedup**
- The key is `SymbolDedupKey` = `file|name|line` (`context/session.go:18–20`). `handoff/snapshot.go` `parseDedupKey` (343–357) parses that format, so it must not change.
- `sessionSet` (`session_store.go:32–39`) holds keys only. usage `sessions.mode` exists but is never read back.

**Tool list** (`internal/mcp/tools.go`)
- `GetTools` (123–644) is a static list, filtered by `toolAccess` (77–91) and `FilterTools` (647).
- `dispatch` (`server.go:169–170`) answers `tools/list` for both protocol eras and has no session id.
- Legacy sessions:
  - They are minted at `initialize` for protocol versions ≥2025-03-26 (`server.go:122–128`, `stream.go` `newSession` 100–112).
  - They are identified by the `Mcp-Session-Id` header.
  - `mcpSession` (34–38) has room for a frozen tool list.
- `onFlagChange` (`stream.go:93–98`) broadcasts once per changed flag; `flags.notify` (296–302) is synchronous per key. `TestLegacyStreamReceivesListChanged` (stream_test.go:112–129) pins today's behavior.
- `FlagState` (`flags/flags.go:40–48`) has no `affects_tools` field.
- `export_bundle` and `import_bundle` are stubs (`tools.go:509–533`).

**Tokens**
- `db.EstimateTokens` is `len/4` (`db.go:132–134`), with 68 non-test callers.
- No BPE library is in `go.mod`. `daulet/tokenizers` is used only by the embedder.
- `SavingsMeta` / `ComputeSavings` / `ApplyTo` are in `context/savings.go` (43–89).
- Query logging:
  - `QueryLogMetrics` is at `writebatch.go:42–57`, and the insert at 12–18 has 22 columns.
  - Quirk: `full_baseline_tokens` always receives `SymbolBaseline` (204–210).
- The dashboard "Tokens saved" total comes from `tokensSavedSum` (`queries_filter.go:10`). It excludes only `file_watcher`, so writes to virtual context are counted.

**Memory** (`internal/memory`)
- `Store` (store.go:86–164) ignores `InvalidatePrevious` (46). Supersession uses exact subject and predicate only (`invalidateConflicting` 173–202).
- `Recall` (recall.go:94–131):
  - FTS ranks by `access_count` (311). LIKE is a fallback, and vector search runs only if both FTS and LIKE are empty (271–294).
  - It fetches `Limit*2` rows and never trims back to the limit. The kind filter runs after the SQL LIMIT.
  - `as_of` is compared as a raw string (171–181).
- `EmbedEntry` (embed.go:13–34) stores the *session id* in `vectors.project_path`. `SearchMemory` (`search/vector.go:463–466`) then hides project and global memory from other sessions.
- `PruneSuperseded` (prune.go:23–65) has a local-time `T`-format cutoff bug, runs only manually, and hard-deletes.
- The handoff `promoteResultMemory` (complete.go:168–182) never embeds.
- `CompactLine.Score` (types.go:42–47) is unused.
- The FTS table indexes the `ref` column too, so a query like "mem" matches every row.

**Context notes** (`internal/contextnotes`)
- `deleteRefs` (store.go:259–297) leaves revisions behind.
- "LRU" eviction orders by `created_at` (23–24).
- `Edit` (edit.go:76–155) computes `after` at 132 and calls `checkEditGrowth` at 133, which returns nil when the note shrinks. `applyEdit` (176–198) is the op switch; `commitEdit` is at 298–313.
- `RetireFn` (fn.go:211) is the tombstone precedent.
- `last_accessed_at` is stored as RFC3339 while `created_at` uses SQLite datetime format.

**Hooks**
- `internal/hooks/hooks.go` dispatches 4 events (switch at 148; unknown events at 158). The `Input` struct (81–94) has no compaction fields. The client decodes only `Content[0].Text` (client.go:109).
- The installer (`component_hooks.go:44–49`) is Claude-only. `installed()` (244) requires every spec to be present.
- Fixtures for PreCompact, PostCompact, PostToolUse and Stop exist under `docs/spikes/fixtures/`.
- The unknown-event tests use `"post-compact"` (hooks_test.go:497) and `"stop"` (cli_hook_test.go:65).

**Handoff**
- The scratchpad accepts `finding` and `dead_end` (scratchpad.go:70–72; types.go:72–80, `Valid` 652–658).
- The tool enum is at `handoff_tools.go:119`. The schema budget test (handoff_tools_test.go:117–123) caps it at 1,200 tokens.
- `returnStub` (complete.go:223–225) always includes the result ref.
- Fork seeding happens at `open.go:257`.

**Database**
- Driver: `sql.Open("sqlite3")` (pools.go:76). `applyPragmas` (86–92) reaches only one pooled connection, and foreign keys are off.
- There is no `user_version`. The `init*Schema` functions run idempotent CREATE/ALTER statements and ignore errors.
- There are no integrity checks and no snapshots. `VACUUM INTO` exists only for the data-dir move (datadir_move.go:16, 113–200).
- `quiesceIndexPool` / `restoreIndexPool` (quiesce.go:42–76) work for index.db only and do not re-run the schema.
- There is no write pause for context or usage, and calling `Init` again is unsafe.
- `drive_monitor.go:119–130` is the precedent for pausing and announcing.
- mattn's amalgamation (SQLite 3.53.0) has no `sqlite3_recover` and no DBPAGE vtab.
- Maintenance runs from `db.StartWALCheckpoint` (db.go:233–271), with 24h tickers, and `startBackgroundServices` / `runEvery` (main.go:578–632).
- `watcher.PostIndexHook` is set late (main.go:443) and is skipped if the embedder fails.
- Import cycles: `memory` → `context` would pull in embedqueue and watcher, and `watcher` must never import `memory`. A leaf package is needed for symbol fingerprints.

**Dashboard and UI**
- The UI `dist` is committed and embedded.
- Stats:
  - `react_api.go:19–21` and 89–119.
  - Digest: `confidence.go:16–39` and 226–241.
  - Settings validation: an if-chain at `api.go:822–1011`; insert new checks after 886.
- `OverviewTab.tsx` has the Tokens saved card (73–82) and `WeekCard` (156–241).
- `SettingsTab.tsx`: `StorageSection` is at 617–820.
- `FeaturesSection.tsx`: the warning slots are at 77/90 and 129–133.
- `MemoryTab.tsx` has no structured-memory UI.
- `api-stub.ts` (50–115) is missing the prune and move stubs.
- `verify-stories.mjs` is at 12–73.
- UI logic is tested only through `lib/*.test.ts`.
- CI (`test.yml`) runs `test`, `lint`, `race` and `ui-test`; it does not run `verify-stories`.

### Key patterns to follow

| Need | Follow |
|---|---|
| New feature package layout | `internal/contextnotes/`, `internal/memory/` (`store.go`, `limits.go`, `types.go`, `stats.go`) |
| Setting with env override | `db.SettingInt(settingKey, envKey, def)` (`internal/db/settings_helpers.go:13`), `contextnotes.envOrSetting` (`limits.go:44`) |
| Feature flag | `internal/flags/registry.go` entry: Key, Env `AST_FEATURE_*`, Default, Tools/Actions; check with `flags.Enabled` |
| MCP tool arguments | `strArg` / `boolArg` (`memory_tools.go:198,205`), `intArg` (`handoff_tools.go:464`), schema helpers `prop` / `enumProp` (`handoff_tools.go:497–513`) |
| Structured tool error | `errs` codes (`internal/errs/codes.go`), `contextnotes.LimitErrorMap`, `handoff.ErrorMap` |
| Extra notice line | second content block, `claimsGrantedNotice` (`server.go:658–660`) |
| Background job | `runEvery` (main.go:621) + STYLEGUIDE §11 loop shape; status struct pattern from `datadir_move.go` |
| Hook variable to avoid import cycles | `db.ParserVersion`, `db.RestartProcess`, `indexer.OnFilePurged` (`prune.go:49`) |
| Realtime refresh | `realtime.Reason` (`realtime.go:11–25`) → `realtime_bridge.go:48–75` → `useWebSocket.ts:3–15` → `App.tsx` `load` |
| Prometheus | `registerPrometheusMetrics` (`metrics_prom.go:36–93`); `handoff.Collectors()` for package-owned collectors |
| Settings UI | `SettingsTab.tsx` `save(key, value)` (92–100); numeric `TextField` `onBlur` (167–173) |
| Storybook | fixture in `storybook/fixtures.ts`, stub in `api-stub.ts`, story in `stories/`, entry in `verify-stories.mjs`, id in `STORY_IDS.md` |
| DB tests | `dbtest.Init(t)` (`dbtest.go:25`); `-tags sqlite_fts5` is required |
| MCP tool tests | `callTool` (`memory_tools_test.go:26`), `callToolTexts` (`handoff_tools_test.go:44`), `indexedPython` (`summary_tool_test.go:19`), `newMCPServer` (`protocol_test.go:19`) |
| Hook tests | `fakeMCP` + `fixture()` (`hooks_test.go:46,97`) |
| Driver connect hook | `sqlite3.SQLiteDriver{ConnectHook}` precedent `fts_rebuild_test.go:56–67` |

### Architectural constraints

- **No cross-DB transactions.** index.db holds every vector, including note and memory vectors. context.db holds notes, memory and handoff data. usage.db holds settings, query logs and sessions. Orphan checks that span databases therefore run in Go over ref sets.
- **`db` imports only errs, logging, realtime and startup.** Anything that needs to reindex, embed or classify is wired from `main` through hook variables.
- **The UI `dist` is committed.** Every UI change needs `make ui-build`, and the rebuilt `internal/dashboard/ui/dist` goes in the same commit.
- **Skills are embedded** (`skills/embed.go`). Doc changes under `skills/` ship in the binary. The `.cursor/skills/*` copies are separate files and must be updated too.
- **Flags that change `tools/list`** emit `list_changed`. None of the new flags hide tools, so none of them trigger it. The schemas get new arguments unconditionally, so `tools/list` stays stable across flag toggles.
- **Additive schema only (DB-4).** New tables and columns only: no DROP, RENAME or type changes, so a 4.x binary still runs against a 5.x data directory.
- **Build flags.** The vendored SQLite recover extension needs `-DSQLITE_ENABLE_DBPAGE_VTAB` in `CGO_CFLAGS` for the mattn amalgamation. It must be set in the Makefile `CGO_FLAGS`, in `.goreleaser.yaml` `env`, in CI and in the Docker build.

---

## Requirements (from PRD)

Priorities are preserved from the PRD; † marks requirements added during PRD refinement.

| IDs | Priority | Summary | Phase |
|---|---|---|---|
| BF-1–BF-12 | MUST | Fixes: vector sort, tie-breaks, recall limit, invalidate_previous, timestamps, revision orphans, LRU, cross-session memory vectors, promoted-memory embedding, Tokens saved scope, FTS relevance, fused recall | P0 |
| BF-13 †, BF-14 † | SHOULD | Scoped `forget all` with confirm; capsule `limit` honored | P0 |
| DB-1–DB-5 (DB-5 †) | MUST | Per-connection pragmas, FK enforcement + report, `user_version` steps, additive schema, pre-5.0 snapshot | P0 |
| ME-1, ME-2 | MUST | Token benchmark harness and committed baseline | P0 |
| TL-1 | MUST | Embedded o200k tokenizer (moved from P1 to P0, decided in planning) | P0 |
| TS-1–TS-4, TS-6 | MUST | Freeze legacy sessions; coalesce stateless notifications; byte-stable list; dashboard warning; remove bundle stubs | P1 |
| TS-5, TS-7 | SHOULD | Static descriptions; "when to call" impact descriptions | P1 |
| DR-1–DR-4 | MUST | Byte-identical responses; timings in `_meta`; stable-first ordering; frozen stub formats | P1 |
| OF-1–OF-3 | MUST | Text output by default, `output=json`, retrieve without duplicated content | P1 |
| PR-1–PR-5 | MUST | Relative cutoff with `withheld`, no_match, collapse, collapse skip, `output=locations` | P1 |
| PR-6 | SHOULD | Packing skips oversized results | P1 |
| MO-1–MO-3 | MUST | `auto` = 3 full + skeletons; `mode=edit`; summary is navigation-only | P1 |
| MO-4 | SHOULD | Mode-aware dedup | P1 |
| OL-1–OL-3 | MUST | Offload over 2k tokens; separate bucket and 24h TTL; expired-ref response | P1 |
| OL-4 | SHOULD | Host context-clearing guidance | P1 |
| TL-2–TL-4, TL-6 | MUST | Three ledgers, baseline definitions plus a conservative baseline, recount and flag history, digest/Prometheus | P1 |
| TL-5 | SHOULD | Opt-in transcript usage ingest | P1 |
| MEM-1–MEM-8, MEM-10, MEM-12–MEM-14 | MUST | Invalidate-not-delete with reasons; 365-day archive; history; restore; world time; decay; rules exempt; TTL; index-first output; conflict candidates with sync embed + FTS fallback; exact supersession kept | P2 |
| MEM-9 | SHOULD | Recall `debug` demotions | P2 |
| PIN-1, PIN-2, PIN-4 | MUST | One pinned note per project, 500 tokens; returned by open_handoff and SessionStart; dashboard edit | P2 |
| PIN-3 | SHOULD | "Change rarely" guidance | P2 |
| HC-1–HC-3 | MUST | Scheduled checks, orphan counts, surfaced in dashboard, `/health` and metrics | P3 |
| SN-1–SN-3 | MUST | Daily snapshots, keep 7, skipped at low disk; also before migrations and restores; verified | P3 |
| SN-4 | SHOULD | Dashboard snapshot list and take-now | P3 |
| AH-1–AH-4 | MUST | Auto-rebuild index.db; automatic heal ladder; heal report and session notice; keep corrupt copies | P3 |
| TR-1–TR-4 | MUST | Origin and sha256; untrusted docs; quarantine; integrity flag | P3 |
| TR-5 | SHOULD | Near-duplicate burst flag | P3 |
| AN-1–AN-4 | MUST | Explicit and inferred anchors; reclassify on reindex; advisory staleness | P3 |
| AN-5 | SHOULD | Staleness rate | P3 |
| UI-1 | MUST | Review queue | P3 |
| NT-1–NT-5 | MUST | Shrink guard; playbook bullets and ops; dedup on add; ranked budgeted fetch | P4 |
| NT-6 | SHOULD | Votes hook from Bash outcomes | P4 |
| CH-1–CH-4, CH-6 | MUST | PreCompact archive, PostCompact checkpoint, SessionStart briefing, installer flag, goal recitation | P4 |
| CH-5 | SHOULD | Codex installer support; Cursor docs only | P4 |
| HO-1–HO-4 | MUST | Decision posts, conflict flags, `inherit_seen`, result ref always present | P4 |
| HO-5 | SHOULD | Tree token report | P4 |
| EV-1 | MUST | Deterministic coding-memory eval | P4 |
| NF-1–NF-8 | NFR | Latency targets (Go benchmark, not a CI gate), storage, safety, security, compatibility, observability, determinism, tests | all |

**Decisions made in planning (they refine the PRD):**
1. **Tokenizer:**
   - `pkoukk/tiktoken-go` with an embedded o200k_base vocabulary loaded through a custom offline BPE loader.
   - It **replaces** the implementation behind `db.EstimateTokens`, so budgets, quotas and ledgers all agree, and it is memoized.
   - It moves to P0.
2. **Recovery:** vendor SQLite's `ext/recover` (`sqlite3recover.c` and `dbdata.c`) through cgo, with the DBPAGE vtab enabled. A P3 spike gates this. If linking fails, ask the user before falling back to the `sqlite3` CLI.
3. **Output argument:** `output=text|json|locations` on the four code tools. retrieve keeps `format` for its context assembly.
4. **Conservative baseline:** each returned symbol's full source plus 20 lines above and below, with ranges merged per file.
5. **Thresholds:** tuned on labeled harness scenarios to zero lost expected hits. NF-1 is checked by a Go benchmark, not a CI gate. There is no CI gate on token counts.
6. **Compaction hooks:**
   - Gated by a separate `feature_compaction_hooks` flag (default off).
   - The transcript archive lives 7 days after last access, in its own bucket with a 2 MB cap per session.
   - Transcript usage ingest (TL-5) is opt-in and reads all `~/.claude/projects` folders; it was not restricted.
7. **Codex hooks** are installable through the installer. **Cursor** gets documentation only.
8. **Heal pause:** writes are blocked for at most 60 s. If healing hasn't finished by then, the server goes read-only and shows a dashboard alert until the user acts.
9. **Text layout:**
   - A summary line.
   - Per symbol, a `### <file>:<start>-<end> <kind> <name> (<mode>)` heading followed by a fenced block.
   - Stats go in `_meta`.
   - The stub format is `[ctx_… <tool> <target>, <N> tok — head shown, fetch_context for all]`.
10. **Memory archive:** invalidated rows stay in `structured_memory` (`valid_until` set, FTS row and vector removed) instead of moving to a separate table. A daily job deletes rows invalidated more than 365 days ago.

**Out of scope** (PRD non-goals): the tier 3 research items, an LLM inside the server, an LSP backend, real bundle export/import, off-machine backup, the project rename, and changing how hosts compact.

---

## Approach

**Phase order: P0 → P1 → P2 → P3 → P4**, each a mergeable, green PR.

- **P0 fixes the base everything else measures against.** It covers ranking determinism, the recall bugs, timestamp normalization and database hygiene. It also adds the real tokenizer and the benchmark harness, so P1–P4 numbers are comparable. Nothing changes defaults except bug fixes.
- **P1 is the token work, and it is the 5.0.0 break.** It changes default output, `auto` semantics and tool-list freezing, each behind its own flag. The four code tools build their responses as before, and a single renderer in `handleToolCall` turns them into text, locations or JSON. Building on the existing maps avoids a risky refactor of four packers into typed structs.
- **P2 is memory.** Ranking, conflicts, history and pinned notes all live in `internal/memory`. Recall fusion is built once, then shared by conflict lookup and the eval.
- **P3 is repair.** Integrity, snapshots and healing live in `internal/db`, since they need pools and paths. Re-indexing and re-embedding are wired from `main` through hooks. A new leaf package, `internal/symref`, holds symbol fingerprints, lookup and change classification; handoff, anchors and `internal/context` all use it without import cycles.
- **P4 covers notes, hooks and handoff.** Its pieces are independent.

**Alternatives considered:**
- Typed response structs for all code tools. Rejected for now: the renderer reads the existing maps, at much lower risk. Typed structs can follow later.
- A separate `memory_archive` table. Rejected: keeping rows in place with validity, FTS and vector removal meets MEM-2 with less code and no migration of existing rows.
- Freezing the tool list per stateless request. Impossible, since stateless requests have no session. TS-2 coalescing plus the existing `ttlMs` is the stateless answer.
- Using the HuggingFace CGO tokenizer for o200k. Rejected because it needs a downloaded vocabulary and is a CGO path. tiktoken-go is pure Go, offline and memoizable.
- A real call graph for `mode=edit`. Rejected: the index stores only import edges (`schema_index.go:24–35`). Callees are approximated by call-like identifiers in the target's body that resolve to project symbols, the same name-based approach the impact tools use (`impact/deletion.go:94–131`).

---

## Style Guide Notes (`~/git/STYLEGUIDE.md`)

- **Errors:** use `errs.New` / `WrapMessage` / `WrapCodeMessage` with lowercase messages and key/value context. New codes go in `internal/errs/codes.go`: `CodeRepairing`, `CodeShrinkGuard`, `CodeExpired`, `CodeConflict` (already exists; reuse it). No `fmt.Errorf`.
- **Logging:** use `logging.Tagged("<pkg>")`. Messages start uppercase, and errors are keyed (`"error", err`) because go vet rejects bare arguments. New tags: `tokens`, `render`, `health`, `heal`, `snapshot`, `symref`, `anchors`, `transcripts`, `tokenbench`.
- **SQL:** queries are package-level `const` blocks at the top of each file, never inline (§10, Anti-pattern 11). That includes the new migration-step SQL.
- **Migrations:** append steps to the **end** of each step list (§10, Anti-pattern 14), and never reorder them.
- **Compact code:** no blank lines inside functions except between logical chunks. Don't combine a multi-line call with `if err` (§9). Use early returns.
- **Background loops:** follow §11. Accept a ctx, log start and stop with defer, keep the body in a separate method, and log errors as warnings without returning them.
- **Services:** new stateful components (heal runner, snapshot manager, transcript ingester) follow the `Service` interface + `realService` pattern only when they're mocked in tests. Otherwise use package functions, matching `memory` and `contextnotes`.
- **Receivers:** single letters. **Mutexes:** comment what they protect, and put them last in the struct.
- **TypeScript:** arrow functions, interfaces for props and data, logic in `ui/src/lib/*.ts` with Vitest tests, and compact code.
- **Tests:** testify (`require`, `assert`), table-driven, `t.Parallel()` where package state allows. Most DB tests can't run in parallel because they share globals; match the existing files.

---

## Detailed Implementation Steps

### Phase 0: fixes and foundations (4.0.x)

#### 0.1 Space and branch
- Create the space with `wtg new ai-v5-context ast-context-cache --branch NO-TICKET-v5-p0-fixes --base origin/main`. Symlink `libtokenizers.a` and `model/` from `~/git/ast-context-cache`.
- Write SPACE.md and a Paperclip task (org jd) that links JD-208.

#### 0.2 Tokenizer (TL-1, NF-1)
- **Dependency:** add `github.com/pkoukk/tiktoken-go` to `go.mod`, plus `github.com/cespare/xxhash/v2` (already an indirect dependency) for memoization.
- **New package `internal/tokens`:**
  - **`vocab.go`:** `//go:embed o200k_base.tiktoken.gz`. Fetch the vocabulary once from the official tiktoken URL during development, check it against the published sha256 (constant in the file), and commit the gzipped copy (about 2 MB). A test verifies the sha256.
  - **`loader.go`:** a type implementing `tiktoken.BpeLoader` that serves the embedded file, registered with `tiktoken.SetBpeLoader` in a `sync.Once`. **Check first** whether the upstream offline loader (`pkoukk/tiktoken-go-loader`) already ships o200k_base. If it does, use it instead and drop the custom loader.
  - **`count.go`:**
    - `Count(text string) int` loads lazily. On failure it logs one warning and falls back to `len(text)/4`.
    - Results for texts of 512 bytes or more are memoized in a bounded LRU (4,096 entries, keyed by xxhash, mutex-guarded).
    - `Method() string` returns `o200k` or `bytes4`.
  - **`count_test.go`:** known counts for fixed strings, the fallback path, concurrency, and memoization. Also `BenchmarkCount4k` for NF-1 (≤2 ms p95; checked in the PR, not gated).
- **`internal/db/db.go:132–134`:** `EstimateTokens` delegates to `tokens.Count`. `db` may import `tokens`, since `tokens` imports only logging, tiktoken and xxhash. All 68 callers switch automatically.
- **Indexer cost check:** `indexer.IndexFile` calls `EstimateTokens` per symbol. Measure the time to index the fixture (and this repo) before and after.
  - If it regresses by more than 10%, add `tokens.Approx(text)`, a calibrated `bytes/3.6` for code, and use it only in `internal/indexer/*`.
  - Record the decision in the PR.

#### 0.3 Deterministic ordering (BF-1, BF-2)
- **New `internal/search/order.go`:** `LessScored(a, b ScoredResult) bool`. It compares score (descending), then file, then start_line, then name; `LessVector` does the same for `VectorEntry`. Both are used everywhere below.
- **`search/vector.go`:**
  - `topMatches` (233–278) always sorts with `sort.SliceStable(... LessVector)`, whatever the length; then it truncates.
  - Apply the same change to `SearchDoc` (364–375), `SearchNote` (415–426) and `SearchMemory` (471–483).
  - `selectAllVectorsQuery` (21) gains `ORDER BY id` so the cache order is deterministic.
- **`search/hybrid.go:77–82`:** fuse into a slice. Iterate keys in sorted order, then sort with `sort.SliceStable` on score, using `fusedEntry.key` for ties.
- **`search/bm25.go`:**
  - The FTS and trigram queries use `ORDER BY f.rank, s.id` (20–22, 29–31).
  - The fallback adds `ORDER BY s.id` before LIMIT (32–34).
  - The sort at 218–220 becomes `SliceStable` with `LessScored`.
- **`internal/mcp/retrieve.go` `rankAndDedup` (397–416):** a stable sort on score, then Type, File, StartLine, Name.
- **`internal/docs/search.go:214–218`:** fuse in sorted-key order, with `Entry.ID` as the tie-break.
- **Tests:**
  - `search/order_test.go`: a table of ties.
  - `vector_test.go`: three candidates with limit 10 come back sorted (AC1).
  - `hybrid` test: two runs over a tied fixture return identical order.

#### 0.4 Memory and context fixes (BF-3–BF-9, BF-11–BF-13)
- **BF-5, timestamps:**
  - New `internal/db/timefmt.go` with `SQLTime(t time.Time) string`, which returns `t.UTC().Format(time.DateTime)`.
  - Also `NormalizeSQLTime(s string) (string, error)`, which accepts RFC3339, `DateTime` and `DateOnly` and returns `SQLTime`. Errors use `errs.CodeInvalidInput`.
  - Apply them:
    - `memory_tools.go:90`: `as_of` is normalized, and a bad value is an error.
    - `memory/prune.go:27`: the cutoff becomes `SQLTime(time.Now().UTC().AddDate(0,0,-n))`.
    - `contextnotes/stats.go:184–187, 160–161, 175–176, 203–205` and `kv_repair.go:160–162`: writes and cutoffs use `SQLTime`.
  - Migration step context#1 / usage#1 (0.6) normalizes existing RFC3339 values with `UPDATE … SET col = datetime(col) WHERE col LIKE '%T%'`. Columns:
    - `context_notes.last_accessed_at`
    - `context_note_access.accessed_at`
    - `context_session_stats.last_store_at` and `last_access_at`
- **BF-6:** `contextnotes/store.go` gains `deleteRevisionsByRefQuery`, executed for each ref in `deleteRefs` (287–290) and in `deleteBySession` / `deleteAll` (299, 322). Test: flush a note that has revisions, and no revision rows remain.
- **BF-7:** `selectOldestSessionNoteQuery` (23–24) orders by `COALESCE(last_accessed_at, created_at) ASC, access_count ASC`. Update `TestLRUEviction` (store_test.go:103) to assert that a recently fetched older note survives.
- **BF-3:**
  - `memory/recall.go`: `Recall` trims to `in.Limit` after ranking and before `applyTokenBudget` (between 112 and 113).
  - The kind filter moves into SQL, as an `andKindInClause` const appended beside the validity and scope clauses at 258–265, 279–286, 303–310 and 340–347. `filterKinds` (133–148) is removed.
- **BF-4:** `store.go:145` supersedes only when `in.InvalidatePrevious`. `memory_tools.go:47–49` defaults the flag to true when the argument is absent. `StoreExtracted` (217) keeps passing true.
- **BF-11, BF-12** go in a new `memory/recall_rank.go`:
  - **FTS:** `searchFTS` matches only the content columns (`{subject predicate object rule} : <query>`) and orders by `bm25(structured_memory_fts)`, then `sm.ref`. Change `searchEntriesFTSQuery` (23–27).
  - **`searchEntries` (271–294)** collects the FTS, LIKE and vector lists. Vector runs when `emb != nil` and is no longer a fallback.
  - **`fuseEntries`:** RRF with k=60, using `contextnotes.fuseNoteResults` (store.go:728) as the pattern. Extract a shared `search.RRF(lists ...[]string) map[string]float64` and use it from both places.
  - `CompactLine.Score` is filled with the fused score. The recency factor (MEM-6) plugs in here in P2.
- **BF-8:**
  - `search/vector.go` `SearchMemory(vec, sessionID string, includeSessionless bool, limit int)`: when `includeSessionless` is true (the scope isn't session-only), skip the session filter at 466, and the SQL re-select at `recall.go:335–347` scopes the results.
  - Raise the candidate pool to `max(limit*5, 50)`.
  - Update `vector_recall_test.go:87`, which encodes the old behavior, and add a cross-session project-memory case (AC for BF-8).
- **BF-9:** `handoff/complete.go:178–180`. In the `promoteResultMemory` loop, `go memory.EmbedEntry(r.Ref, string(parent), r.Line, s.emb)` when `s.emb != nil`. Test with the stub embedder: the promoted memory has a vector row.
- **BF-13:**
  - The `forget_memory` schema (`tools.go:394–409`) adds `project_path` and `confirm`.
  - `recall.go` `Forget` all mode (421–441) applies `scopeClauseFor` when a scope, session or project is given. With none of them and no `confirm=true`, it returns `CodeInvalidInput` with the message "all=true without scope requires confirm=true".
- **BF-14:** the capsule `limit` argument is added to the schema (`tools.go:128–142`; default 30, max 100). `handler.go:63` uses `Limit: limit`.

#### 0.5 Tokens saved scope (BF-10)
- **`internal/dashboard/queries_filter.go`:** `savingsToolsClause` = `tool_name IN ('get_context_capsule','search_semantic','get_file_context','retrieve','execute_code')`, used by `tokensSavedSum`, `dedupTokensSum` and `savingsVsFilesSum` (10–12).
- **Apply the same clause** to:
  - `confidence.go` `selectTopToolsBaseQuery` (28–29)
  - `stats_today.go:9`
  - `metrics_prom.go:17`
  - the legacy `api.go:40` and 45–47
  - `tool_stats.go:14–15`
  - `recent_build.go:16`
- **Docs:** fix the wording in `skills/usage/SKILL.md:176`, `skills/operator/SKILL.md:78`, the `.cursor/skills` copies, `AGENTS.md:312–321` and `CLAUDE.md:141–148`.
- **Tests:** `confidence_test.go` with a store_context row that doesn't change the total (AC12). Add a new `TestDashboardStatsExcludesVirtual` in `react_api_test.go`.

#### 0.6 Database hygiene (DB-1–DB-5)
- **DB-1 (`pools.go`):**
  - Register `astcacheDriver = "sqlite3_astcache"` once in `init()` with `sql.Register(astcacheDriver, &sqlite3.SQLiteDriver{ConnectHook: applyConnPragmas})`.
  - `applyConnPragmas(c *sqlite3.SQLiteConn) error` runs `busy_timeout=15000`, `synchronous=NORMAL`, `cache_size=-32000`, `wal_autocheckpoint=200` and `foreign_keys=ON` through `c.Exec`, keeping the queries in a const block.
  - `openPoolWith` (72–84) uses `astcacheDriver`. `applyPragmas` is deleted, and only `journal_mode` stays in the DSN.
  - Ad-hoc opens (`checkpoint.go:134`, `datadir_move.go:213`, `migrate_split.go`) keep using `"sqlite3"`.
  - Test: `PRAGMA cache_size` is checked on 4 separate connections obtained from the pool via `conn.Conn(ctx)`.
- **DB-2:** on first start (when `user_version` < 1), run `PRAGMA foreign_key_check` on context.db. Log violations as a warning with counts, and store the count in `health` state (P3 surfaces it). Delete nothing.
  - **Risk check:** grep every `DELETE FROM doc_sources` path (`docs` package) and make sure `doc_content` is deleted first. Add a test that removes a doc source with FKs on.
- **DB-3, new `internal/db/migrate.go`:**
  - `type schemaStep struct { version int; name string; run func(tx *sql.Tx) error }`.
  - `runSteps(conn *sql.DB, dbName string, steps []schemaStep) error`:
    1. Read `PRAGMA user_version`.
    2. For each step with a higher version, in order: `BEGIN IMMEDIATE`, run it, then `PRAGMA user_version = N`, then commit.
    3. On error, roll back and return `errs.WrapMessage("schema step failed", err, "db", dbName, "step", s.name)`.
  - Step lists `indexSteps`, `contextSteps` and `usageSteps` live in `migrate_steps.go`, appended in order.
  - `Init` (db.go:79–81) runs the idempotent `init*Schema` (the baseline), then `runSteps` per database. `InitUsage` (usage_only.go:32) does the same.
  - **P0 steps:**
    - context#1: normalize timestamps (BF-5).
    - usage#1: normalize timestamps.
    - usage#2: `ALTER TABLE queries ADD COLUMN estimate_method TEXT DEFAULT 'bytes4'`. New rows write `tokens.Method()` (TL-4 foundation; `writebatch.go:12–18`, 204–210).
  - The existing `full_baseline_tokens` quirk (it always receives `SymbolBaseline`) stays as is and gets a code comment. Nothing reads that column for the ledgers, and the conservative baseline gets its own column in P1.
- **DB-4:**
  - `migrate_test.go` adds a lint-style test. It scans every step's SQL constants for `DROP`, `RENAME` or `ALTER COLUMN` and fails if any are found.
  - The manual test plan covers starting the v4.0.11 release binary against a P0 data directory.
- **DB-5:**
  - In `Init`, before `runSteps` on context and usage, and only when `user_version == 0`, the file exists and is non-empty, and `snapshots/pre-5.0/` is missing: open a temporary `"sqlite3"` connection and run `VACUUM INTO <dataDir>/snapshots/pre-5.0/<name>.db`.
  - `vacuumIntoQuery` from `datadir_move.go:16` is reused.
  - On failure, `Init` returns an error and no step runs. Test with a read-only snapshot dir.

#### 0.7 Benchmark harness (ME-1, ME-2) and hash embedder
- **New `internal/embedder/hash.go`:** `NewHashEmbedder(dims int) Interface`. It hashes tokens (xxhash) into a 768-dimension bag-of-words vector, L2-normalized and deterministic. It is used by the harness and by tests. It implements `EmbedSingle` and `EmbedBatch`, matching the interface in `embedder/factory.go`.
- **New `internal/tokenbench/`:**
  - **`testdata/fixture/`** is a small multi-language repo (~40 files) containing:
    - Go: a config loader with `_test.go`, a `mocks/` dir, a `vendor/` copy, and overloads (methods sharing a name on different receivers).
    - Python: a module plus `test_*.py`.
    - TypeScript: `*.ts` with `*.spec.ts` and `__mocks__/`.
    - YAML.
  - **`scenarios.yaml`:** a list of `{name, tool, args, expect: [file:symbol...], negative: bool}`. Coverage:
    - ~25 scenarios: capsule (5), semantic (4), file_context (4), retrieve (4), recall (4, with seeded memories), handoff open (2).
    - Plus 4 negative queries that should come back as no-match.
  - **`bench_test.go` `TestTokenBench`:**
    - Index the fixture with `dbtest.Init` and the hash embedder, and seed memory.
    - Run each scenario twice through `httptest.NewServer(mcp.NewHandler())` (the `protocol_test.go:19` pattern), with `session_id` per scenario.
    - Record tokens (`tokens.Count` of `content[0].text`), result count, expected-hit recall, whether a negative query correctly came back with no match, and byte equality between the two runs.
    - **Assertions (CI):** recall of `expect` must be at least the baseline's.
    - **Report only:** tokens against `baseline.json`, and determinism (asserted from P1).
    - `-update` (via `AST_TOKENBENCH_UPDATE=1`) rewrites `baseline.json`.
  - **Makefile:** `bench-tokens` runs `TEST_PKGS=./internal/tokenbench/... AST_TOKENBENCH_VERBOSE=1 make test` and prints the table. `bench-tokens-update` rewrites the baseline.
  - `TestTokenBench` also runs in the normal `make test` (small, about 2 s).
- **ME-2:** commit `baseline.json` in the P0 PR. Every later phase PR includes the diff table in its description.

#### 0.8 P0 docs and PR
- `CHANGELOG`-style notes go in the PR, along with the README "How it saves tokens" formula note.
- The release goes out as 4.0.x via the normal auto-bump.

### Phase 1: token savings (5.0.0)

#### 1.1 Flags
- Add these to `internal/flags/registry.go`, all defaulting on, and none hiding tools:
  - `feature_stable_responses`
  - `feature_text_format`
  - `feature_relevance_floor`
  - `feature_mode_v2`
  - `feature_result_offload`
- `FlagState` (`flags.go:40–48`) gains `AffectsTools bool \`json:"affects_tools"\``, filled from `AffectsTools(key)` in `State()`.

#### 1.2 Tool surface (TS-1–TS-7)
- **TS-1:**
  - `stream.go` `mcpSession` gains `tools []Tool`, guarded by `streamHub.mu`.
  - `dispatch(w, rpcReq)` becomes `dispatch(w, rpcReq, sessionID string)`. `handleLegacy` (server.go:132) passes the `Mcp-Session-Id` header value when `hub.touch(id)` reports the session live. `handleModern` passes "".
  - `tools/list` (server.go:169–170) goes through `toolsFor(sessionID)`. That function returns the session's frozen list, capturing it from `FilterTools(GetConfig())` on first use, or the live list when there is no session.
  - `onFlagChange` no longer sends `list_changed` to legacy session streams, because their lists are frozen.
  - `toolAccessByName` at call time still uses the live config, so a tool disabled after the freeze returns the existing deny message.
- **TS-2:**
  - New `listChangedCoalescer` in `stream.go`. `onFlagChange` marks the list dirty and arms one timer: send after `listChangedDebounce` (default 400 ms, a package var), and no sooner than `listChangedMinInterval` (default 5 s, a var) after the last frame.
  - When the timer fires, it sends one `notifications/tools/list_changed` to the modern subscribers and calls `realtime.Notify(SettingsChanged)` once.
  - Test with the vars shortened: toggling `feature_handoff` gives exactly one frame (AC3).
- **TS-1/TS-2 tests:**
  - Rewrite `TestLegacyStreamReceivesListChanged` (stream_test.go:112–129) as `TestLegacySessionToolListFrozen`: a legacy session's `tools/list` is unchanged after a flag flip, and a new `initialize` sees the change.
  - Keep `TestBroadcastOncePerSession` for modern subscribers.
- **TS-3:** `tools_test.go` `TestToolsListBytesStable` marshals `FilterTools` twice, and across `SetConfig` round-trips, and asserts the bytes are equal. Add a golden sha for the default configuration, updated deliberately.
- **TS-4:**
  - `ui/src/components/FeaturesSection.tsx`: before toggling a flag with `affects_tools`, show a confirm Dialog: "Changing this updates the tool list. Connected agents lose their prompt cache; sessions that started before the change keep the old list until they reconnect."
  - The toggle logic goes in `ui/src/lib/flags.ts` (`needsToolListWarning`), with a test.
  - Add the `affects_tools` field to `types.ts` `FlagState`.
- **TS-5:**
  - Audit the descriptions in `tools.go`.
  - In the capsule description, replace "~90%/~94%" with mode semantics, and update `auto` to "full source for the top 3, skeletons for the rest" (MO-3).
  - search_docs "max ~0.033" (536) becomes a documented limit, so keep it.
- **TS-6:**
  - Delete `export_bundle` and `import_bundle` from `GetTools` (509–533), their handler cases (`handlers.go:526–555`, `server.go:479–482`), and the tier/docs references (`skills/*`, `AGENTS.md`, README).
  - Update the goldens in `tools_example_test.go:17,29,41` and `context_tools_test.go`.
  - **Cross-repo:** check `~/git/mcp-local/internal/asttools/default_tiers.go` for these names, and open a small follow-up PR there if they are listed (needs user approval).
- **TS-7:** add one "Call this before …" sentence to the descriptions of `get_impact_graph`, `diff_impact` and `check_deletion_safety`.

#### 1.3 Deterministic responses (DR-1–DR-5)
- **New `internal/mcp/respmeta.go`:**
  - `type toolMeta map[string]any`, plus `metaKey = "ast-context-cache/stats"`.
  - `handleToolCall` keeps a `meta toolMeta` per call. When `feature_stable_responses` is on and `debug` is not true, each tool branch moves these fields into `meta`:
    - capsule: `pipeline`, `cache_hit`.
    - semantic: `total_vectors`, `cache_hit`.
    - retrieve: `search_time_ms`, `code_retrieve_ms`, `docs_retrieve_ms`, `dedup_budget_ms`, plus the pipeline counts.
  - Do this by deleting the keys from the parsed map before re-marshalling (capsule 325–330, semantic 414–431, retrieve 600). `_meta` is attached at 664–665.
- **`RetrieveStats` (retrieve.go:56–79):** remove the four timing fields from the struct and fill a separate `retrieveTimings` struct. It goes into `_meta`, or back into the body as `timings` when `debug=true` or the flag is off. Update `TestRetrieveStatsJSONGolden` (13–38).
  - The server's stats re-parse (`server.go:609–630`) reads timings from the timings struct, not from the body.
- **DR-3:**
  - New `internal/mcp/orderedjson.go`: `type orderedMap struct { keys []string; vals map[string]any }` with a `MarshalJSON` that emits keys in insertion order.
  - The four tools' `output=json` responses are built as `orderedMap`: results first, then counts and budget, then savings, then hints. The text output (1.4) is ordered by construction.
- **`debug` argument:** add `debug` (boolean) to the schemas of the four tools.
- **DR-4:** new `docs/stubs.md` documents the `[ctx_…]`, `[handoff hof_…]`, `[result ctx_… for hof_…]` and offload stub grammars. Add `stubs_test.go` in `internal/mcp` with a regex for each producer: `contextnotes`, `handoff/create.go`, `complete.go:223–225`, and offload.
- **DR-1/NF-7:** the `TestTokenBench` determinism assertion becomes enforced (AC2).

#### 1.4 Output format (OF-1–OF-4) and locations (PR-5)
- **New package `internal/render`:**
  - `Text(r Response) string`, where `Response` is `{Tool, Query string; Results []map[string]any; Total, Withheld, WithheldTokens int; Collapsed []Collapse; NoMatch *NoMatch; Notes []string}`.
  - **Summary line:** `<tool> "<query>" · <n> results · <tok> tok[ · withheld N (X tok)][ · also N similar in tests][ · no match (best 0.12)]`.
  - **Per result:** `### <relfile>:<start>-<end> <kind> <name> (<mode>)`, then a fenced block with the language from the extension. Move `langFromExt` from `retrieve.go:474` into render and re-export it.
  - The body comes from the `source`, `skeleton` or `summary` key.
  - `Locations(r Response) string` prints one line per hit: `<relfile>:<start>-<end> <kind> <name> <score>`.
  - Tests are table-driven with golden strings in `render/testdata`.
- **`handleToolCall`:**
  - Read `output := strArg(toolArgs,"output")` near 280. The default is `text` when `feature_text_format` is on, `json` otherwise.
  - For the four tools, build `render.Response` from the parsed map. The capsule and semantic `results` arrays need the score field kept: add `score` to the result maps in `hitFromScored` (`savings.go:171–186`) if it isn't there.
  - Set `content[0].text` to `render.Text` or `render.Locations`. `resultIsError` still reads the JSON.
  - **`output=locations`:** the packers still run, which applies dedup and the floor. Only the rendering changes, and the token count is computed on the rendered text.
- **retrieve (OF-3):**
  - New `include_chunks` argument (default false). When false, `Chunks[].Content` is cleared before marshalling (retrieve.go:229).
  - In text output, retrieve renders the summary line, then `result.Context` (already assembled per `format`), then a compact chunk list (`file:lines score`).
- **`code_script_hints`** appear only in `output=json`. In text they go in `_meta.hints`.
- **Tests:**
  - `callTool` (memory_tools_test.go:26–46) injects `"output":"json"` for the four code tools when absent, so existing assertions keep working.
  - New `output_test.go` covers text, json and locations for all four tools (AC4).
- **Docs:** `AGENTS.md`, `skills/usage/SKILL.md` (and the `.cursor` copy), README examples, and `scripts/code-mode/README.md`, which needs `output=json`.

#### 1.5 Precision (PR-1–PR-4, PR-6, PR-7)
- **New `internal/context/precision.go`:**
  - **`ApplyRelevanceFloor(scored []search.ScoredResult, cfg FloorConfig) (kept []search.ScoredResult, withheld []search.ScoredResult)`:** a relative cutoff `score/top < cfg.MinRelative`, with a default to be tuned. Always keep at least one hit unless no-match applies.
  - **`WeakMatch(scored, path string, cfg) (bool, float64)`:**
    - Hybrid path: weak unless the top hit appears in both the BM25 and vector lists, **or** its vector similarity is at least `cfg.VectorMin`, **or** its BM25 term coverage is at least `cfg.CoverageMin`. Reuse `docs/relevance.go` coverage (113–141), extracted to `search.TermCoverage`.
    - Semantic path: similarity ≥ `VectorMin`.
    - Retrieve: per code part.
  - **`FloorConfig`** loads from settings (`relevance_min_relative`, `relevance_vector_min`, `relevance_coverage_min`; env `AST_RELEVANCE_*`) via `db.SettingInt` / float helpers. Add `db.SettingFloat` beside `settings_helpers.go:13`.
  - **New `internal/context/distractors.go`:**
    - `IsDistractorPath(rel string, globs []string) bool` uses built-in patterns: `*_test.go`, `test_*.py`, `*_test.py`, `*.spec.ts`, `*.test.ts`, `*.spec.js`, `*.test.js`, `__mocks__/`, `mocks/`, `mock_*.go`, `vendor/`, `node_modules/`, `testdata/`, `*.pb.go`, `*_gen.go`. The setting `collapse_globs` adds more.
    - `CollapseDistractors(results []map[string]any, query string, enabled bool) (kept []map[string]any, collapsed []render.Collapse)` groups distractor hits, and same-signature duplicates (same name, kind and skeleton hash), under the first non-distractor result with the same name.
    - Collapse is skipped when the query matches `(?i)\b(test|spec|mock)` or the caller passes `collapse=false` (PR-4).
- **Wiring:**
  - Capsule (handler.go between 64 and 83): floor → weak check → collapse.
  - `PackScoredResults` (143–198) and `retrieveCode` (261–337) follow the same order.
  - File context: no floor (the file is explicit).
  - Responses carry `withheld` `{count, tokens}`, with the token count estimated from the skeleton of each withheld hit, capped at 20 hits. They also carry `no_match` `{best_score, hint}` and `collapsed`.
  - New arguments: `min_relative_score` (number) and `collapse` (boolean) on the capsule, semantic and retrieve schemas.
- **PR-6:** the budget loops `continue` instead of `break`, giving up after 5 consecutive misses (handler.go:104–106, 182–184; retrieve `budgetChunks` 424–426; file_context `overBudget` 218–221). Withheld results that come from the budget are counted in `withheld` too.
- **Tuning:**
  - Run `make bench-tokens` with a threshold sweep: `AST_RELEVANCE_SWEEP=1` runs a grid and prints, for each setting, recall of expected hits, negatives correctly flagged, and total tokens.
  - Choose the defaults with zero lost expected hits and maximum tokens withheld. Commit the defaults and record the sweep table in the PR.

#### 1.6 Modes (MO-1–MO-5)
- **MO-1:**
  - `EffectiveMode` (pack.go:21–39) becomes `EffectiveMode(mode string, rank int) string`. For `auto` it returns `full` when `rank < 3` and `skeleton` otherwise. It never returns summary.
  - Update the callers: handler.go:96, 175; handlers.go:211; retrieve.go `retrieveCode`.
  - `get_file_context` with auto behaves as skeleton (handlers.go:181 `maxScore` logic is removed).
- **MO-2, new `internal/context/editmode.go`:**
  - `EditView(projectPath string, hit PackHit, fileCache) (target map[string]any, callees []map[string]any)`.
    - The target is the exact source span, from `ReadSourceRange`.
    - Callees come from identifiers in the target body that match `\b([A-Za-z_][A-Za-z0-9_]*)\s*\(`. Drop the target's own name and language keywords (a list per language in a `const`). Resolve them with `SELECT … FROM symbols WHERE project_path=? AND name IN (…) AND kind IN ('function','method')` as a const query.
    - Take at most 10 callees, prefer the same file, and render them as skeletons.
  - **Capsule `mode=edit`:** the top non-collapsed hit becomes the target. The other hits are dropped and reported in `withheld`.
  - **`get_file_context mode=edit`:** requires a new `symbol` argument (name or fqn) and errors if it is missing.
  - **retrieve:** `mode=edit` isn't supported and returns a clear error.
  - Add `edit` to the mode enums. `mode=edit` is gated by `feature_mode_v2`; with the flag off it falls back to `full` for the top hit.
- **MO-3:** update the descriptions and skills as in 1.2.
- **MO-4:**
  - `sessionSet` (session_store.go:32–39) gains `modes map[string]string`. `hydrate` (150–161) loads `mode` (extend `selectReturnedSymbolsQuery` at session.go:10–13). `MarkReturned` records the **effective** mode.
  - Fix the delivered `Mode` at handler.go:112 and 190, handlers.go:226 and retrieve.go:347.
  - The dedup check in each loop skips only when `modeRank(prev) >= modeRank(requested)`, with ranks locations=0 < summary=1 < skeleton=2 < full=edit=3.
  - `ReturnedKeys` keeps its signature. Add `ReturnedModes(sessionID) map[string]string`.
  - Handoff `parseDedupKey` is untouched.

#### 1.7 Self-offloading results (OL-1–OL-5)
- **Settings:**
  - `result_offload_threshold` (default 2000; env `AST_RESULT_OFFLOAD_THRESHOLD`).
  - `context_offload_max_tokens_global` (default 500000).
  - `offload_ttl_hours` (24).
- **`handleToolCall`:** after the text, locations or JSON is produced for the four tools, and when the flag is on, `offload` isn't false, and `tokens.Count(text) > threshold`:
  1. Store the full text with `contextnotes.Store(sid, text, label, projectPath, nil, contextnotes.KindOffload, meta{tool, args}, nil)`. Skip embedding for offload notes: they're re-fetched by ref, so pass a nil embedder.
  2. Re-render a **head**: keep results in rank order while the rendered size is at or below the threshold minus 80 tokens.
  3. Prepend the stub line `[ctx_… <tool> <target>, <N> tok — head shown, fetch_context for all]`, where target is the query or file.
  - **No `session_id`:** use the synthetic session `offload-<UTC date>`, so TTL cleanup still works and Fetch of the ref succeeds. Fetch only guards by session when the caller passes one.
- **`contextnotes`:**
  - `KindOffload = "offload"`.
  - `checkLimits` and `evictSessionLRU` exclude `kind IN ('offload','transcript_archive')` from session and global counts. The `notOffloadClause` const is used by `checkLimits` queries 189–207.
  - A separate `checkOffloadLimit` evicts the oldest offloads when over `context_offload_max_tokens_global`.
- **TTL:** `PurgeExpiredOffloads(ttl)` deletes offload notes where `COALESCE(last_accessed_at, created_at) < now - ttl`. It first writes a tombstone to the new table `context_note_tombstones(ref TEXT PRIMARY KEY, kind TEXT, tool TEXT, args_json TEXT, expired_at TEXT)` (context step #2). It runs hourly via `runEvery` in `startBackgroundServices`.
  - `FlushSession` also deletes offloads for the session; the existing `deleteBySession` covers this.
- **OL-3:** `Fetch` (store.go:433–477) returns `{ref, status: "expired", tool, args}` for refs found in the tombstones table. It also prunes tombstones older than 30 days in the same hourly job.
- **OL-4:** add a "Host context clearing" section to `docs/USAGE.md` and to the skills.
- **Tests (AC9, AC10):** a 5k-token result yields a head of at most ~2k tokens with the stub, and `fetch_context` returns the full result. Use a backdated note for the expiry case.

#### 1.8 Ledgers (TL-2–TL-6)
- **Schema, usage step #3:** `queries` adds `conservative_baseline_tokens INTEGER DEFAULT 0` and `ledger TEXT DEFAULT ''`. Extend `insertQueryLogQuery` (writebatch.go:12–18) and `QueryLogMetrics` (42–57).
  - `ledger` is `compression` for the five savings tools from BF-10, `virtual` for `store_context`, `fetch_context`, `edit_context`, `store_memory`, `recall_memory`, `apply_context_fn` and `handoff`, and `none` otherwise. It is set in `logToolQuery` (server.go:672–689) from a const map.
- **TL-3:**
  - `context/savings.go` gains `ConservativeBaselineTokens(results []PackHit, fileCache) int`. Per file, it merges `[start-20, end+20]` ranges, reads the lines, and counts them with `tokens.Count`.
  - `ComputeSavings` takes the value and stores it in `SavingsMeta.ConservativeBaseline`, wired through `ApplyTo` and `logToolQuery`.
  - `ApplyTo` writes it only into `_meta` (stable responses) and the query log, not the body.
- **Dashboard API:**
  - `components.Stats` (`components/stats.go`) gains `CompressionSaved`, `DedupSaved`, `ConservativeSaved`, `VirtualStoredTokens`, `VirtualFetchedTokens`, `VirtualRecalledTokens`, `EstimatedRows`, `BaselineDefinitions map[string]string`.
  - The SQL goes in `react_api.go:19–21` and in `queries_filter.go` sums keyed by `ledger`. "Tokens saved" = compression + dedup. `ConservativeSaved` = `SUM(max(0, conservative_baseline_tokens - tokens_used))` over the compression ledger.
  - `EstimatedRows` = the count where `estimate_method='bytes4'`.
- **TL-4, context step #3:** recount the stored text with `tokens.Count` and update `token_est` on `context_notes`, `context_note_revisions` and `structured_memory`. Batched in 500s, inside the step's transaction. `tokens` has no DB dependency, so steps can call it. Existing `queries` rows keep `bytes4`.
- **TL-6:** `confidence.go` `WeeklyDigest` (74–83) gains the ledger fields. `metrics_prom.go` adds `astcache_tokens_saved_today{ledger="compression|dedup"}` and keeps the old name for compatibility.
- **UI:**
  - `OverviewTab.tsx` Tokens saved card (73–82) shows compression + dedup, with a detail line giving the definitions (tooltip text from `BaselineDefinitions`), and a "conservative: X" line.
  - A footnote reads "N earlier rows estimated" when `EstimatedRows > 0`.
  - `VirtualContextCard` (243–338) shows the virtual ledger.
  - Logic goes in `ui/src/lib/ledger.ts` with a test; types in `types.ts`; fixtures and story updates.
- **TL-5, new package `internal/transcripts`:**
  - Setting `transcript_usage_ingest` (default off; Settings → Storage toggle).
  - `IngestOnce(root string)`:
    - Walks `~/.claude/projects/*/*.jsonl`.
    - For each file, resumes from `host_usage_offsets(path, offset, mtime)` (usage step #4).
    - Reads lines, JSON-decoding only `{timestamp, message:{usage:{input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens}}}` into a minimal struct, and never keeps the text.
    - Aggregates into `host_usage_daily(day, project_dir, input, output, cache_read, cache_write, PRIMARY KEY(day, project_dir))`.
  - Runs hourly via `runEvery`.
  - Dashboard: `/api/dashboard/host-usage` (30-day series) and a small chart card on the Overview when enabled.
  - Tests use fixture JSONL files in `internal/transcripts/testdata`.
- **5.0.0:**
  - `VERSION` → `5.0.0`.
  - `docs/MIGRATING.md` gets a 5.0 section covering: text output (use `output=json`), `auto` semantics, frozen legacy tool lists, timings moved to `_meta`, removed bundle stubs, the new ledger meaning, every new flag, and how to turn each off.
  - The README "How it saves tokens" section is updated, including the formula and ledgers.

### Phase 2: memory (5.x)

#### 2.1 Schema (context step #4)
- `structured_memory` adds:
  - `invalidated_reason TEXT`
  - `world_valid_at TEXT`
  - `world_invalid_at TEXT`
  - `expires_at TEXT`
- Index `idx_struct_mem_expires(expires_at)`.
- Every const goes in `schema`-style declarations inside `migrate_steps.go`.

#### 2.2 Invalidate, archive, history, restore (MEM-1–MEM-4)
- **`memory/invalidate.go`:**
  - `invalidate(tx, ref, reason, supersededBy string)` sets `valid_until`, `superseded_by` and `invalidated_reason`, then deletes the FTS row and the `mem:<ref>` vector (via the vector cache, outside the tx).
  - Reason constants: `ReasonContradiction`, `ReasonUserForget`, `ReasonTTL`, `ReasonStaleAnchor`, `ReasonQuarantine`, `ReasonRestored`.
  - Route `invalidateConflicting` (store.go:173–202), `Forget` modes (recall.go:412–501) and `forgetRefs` through it.
- **MEM-2:**
  - `PruneSuperseded` is renamed `PruneArchived(maxAgeDays int)`. It defaults to the `memory_archive_max_age_days` setting (365) and deletes only rows whose `valid_until` is older than the cutoff (SQLTime).
  - It runs daily via `runEvery(ctx, 24h, …)`.
  - The dashboard prune (`dashboard/prune.go:88–94`) calls it with the setting.
- **MEM-3:**
  - `memory/history.go`: `History(in HistoryInput) ([]Entry, error)` takes a ref, or subject + predicate, plus scope.
  - With a ref, it walks the `superseded_by` chain forward and back (const queries). With subject and predicate, it lists every row ordered by `valid_from`.
  - `recall_memory` gains `history` (boolean) and `ref` arguments (`tools.go:370–390`).
  - The output lines include `valid_from`, `valid_until` and `reason`.
- **MEM-4:**
  - `memory/restore.go`: `Restore(refs []string) (*RestoreResult, error)`. Under `factSupersessionMu`, for each ref:
    - It must be invalid; otherwise `already_active`.
    - If `superseded_by` = B, and B is active: invalidate B with `ReasonRestored`, then reactivate A (clear `valid_until`, `superseded_by` and `invalidated_reason`), reindex FTS, and re-embed asynchronously.
    - If B is itself invalid with a successor C: return `conflict` and change nothing.
  - `forget_memory` gains `action` (`forget` default | `restore`).
  - Tests: AC13, AC14.

#### 2.3 World time (MEM-5)
- `store_memory` adds `valid_at` and `invalid_at`, both normalized.
- `validityClause` (recall.go:171–181) uses `COALESCE(world_valid_at, valid_from)` and `COALESCE(world_invalid_at, valid_until)` unless `as_of_system=true` (new argument).
- Add a const variant for each clause pair.
- Test: AC15.

#### 2.4 Ranking, TTL, debug, index-first (MEM-6–MEM-11)
- **`memory/recency.go`:** `recencyFactor(e Entry, now time.Time, cfg DecayConfig) float64`.
  - Δ = now − COALESCE(last_accessed_at, created_at).
  - h = the half-life by scope: session `memory_halflife_session_hours` (24), project `memory_halflife_project_days` (30), global 0 (no decay).
  - r = 2^(−Δ/h).
  - f = min(1, ln(1+access_count)/ln(21)).
  - factor = clamp(0.3, 1.5, 0.3 + 1.2·(0.7r + 0.3f)).
  - It returns 1.0 for global scope, and for procedures when `memory_decay_rules` is false (MEM-7, the default).
- **`recall_rank.go`:** score = fused × factor, when `feature_memory_v2` is on.
- **MEM-8:**
  - `store_memory` adds `ttl` (Go duration, or `Nd` days) and `expires_at` (normalized).
  - The validity clause adds `AND (expires_at IS NULL OR expires_at > ?)`.
  - An hourly job, `memory.ExpireDue()`, invalidates expired rows with `ReasonTTL`.
- **MEM-9:** with `debug=true`, recall returns `demoted: [{ref, line, score, reason: "budget"|"limit"|"decay"}]`. `decay` means it ranked out only because of the factor.
- **MEM-10:**
  - **Recall** output by default is `lines` (CompactLine: ref, kind, line, score, plus flags in P3). `formatted` is dropped unless `expand=true`. `expand=true` adds full `entries` (Entry JSON).
  - Hard caps: 50 lines and 2,000 tokens.
  - `retrieve` include_memory (retrieve.go:138–147) uses the lines.
- **list_context** (context_tools.go:100–117) returns `index: ["ctx_… · label · N tok · kind · age"]` by default. `expand=true` returns today's `notes` objects. Same caps.
- **Flag:** `feature_memory_v2` (default on) gates MEM-6 through MEM-10. History, restore and world time are always on.

#### 2.5 Conflict candidates (MEM-12–MEM-14)
- **`memory/conflicts.go`:** `ConflictCandidates(e Entry, emb embedder.Interface, timeout time.Duration) ([]Conflict, string)`.
  - It embeds `FormatLine(e)` in a goroutine with a select on `time.After(timeout)`, with timeout `memory_conflict_timeout_ms` (default 300).
  - On success, it calls `search.Cache.SearchMemory` (with the BF-8 scope) for the top 10, then re-selects the active facts in the same scope and siblings. It excludes the new ref and any refs the exact supersession just invalidated.
  - On timeout or a nil embedder, it falls back to FTS over subject and object terms, and the method is `fts`.
  - It returns `possible_conflicts: [{ref, line, similarity}]` and `conflict_method`.
- **`handleStoreMemory` (memory_tools.go:28–70):**
  - For facts, call it after `Store`.
  - Reuse the computed vector for the stored entry (new `EmbedEntryWithVector`) instead of embedding asynchronously a second time.
- Tests: AC17, AC18.

#### 2.6 Pinned note (PIN-1–PIN-4)
- **`contextnotes/pinned.go`:**
  - `KindPinned`.
  - `StorePinned(projectPath, content, origin string) (*StoreResult, error)`. It requires a project, rejects content over `pinned_max_tokens` (default 500; AC19), and keeps one note per project.
  - If a pinned note exists, it applies the change as an `EditRewrite` with `force` (the shrink guard is exempt for pinned notes, since they are small by design). The ref and revisions are kept.
  - `Pinned(projectPath) (*Note, error)`.
- **`store_context`:** `kind=pinned` routes to `StorePinned`.
- **Handoff:** `open.go` `openDigest` (287–337) adds a `pinned` section first, with its tokens counted in the open budget.
- **Dashboard:**
  - `GET /api/dashboard/pinned?project=` and `PUT` (origin `user`).
  - A `PinnedNoteCard` in `MemoryTab.tsx` with a textarea and a token counter. The counter comes from a new `tokens` endpoint, `POST /api/dashboard/count-tokens`, rather than a client-side estimate.
- **PIN-3:** description and skill text.

#### 2.7 Memory UI
- **API:**
  - `/api/dashboard/memory/entries?project=&q=&include_invalid=` (paginated).
  - `/api/dashboard/memory/history?ref=`.
  - `POST /api/dashboard/memory/restore`.
  - `POST /api/dashboard/memory/forget`.
- **`MemoryTab.tsx`:** a new "Structured memory" section with a searchable table, a history drawer and restore/forget buttons (with confirmation, NF-3).
- `buildMemory` (partials_data.go:262–305) gains per-project counts.
- Logic in `lib/memory.ts` with a test; fixtures, stub and story `Memory/Structured`.

### Phase 3: repair and trust (5.x)

#### 3.0 Spike: vendored recover (gate)
- **Vendor the extension.** Copy `ext/recover/sqlite3recover.c`, `sqlite3recover.h` and `dbdata.c` for SQLite **3.53.0**, which must match mattn v1.14.44's amalgamation, into `internal/db/recover/`.
  - Add `sqlite3.h` 3.53.0, which is public domain.
  - Add `recover.go` (cgo) exposing `Recover(srcPath, dstPath string) error`. It uses `sqlite3_recover_init`, `_run` and `_finish` on a C-level connection opened with `sqlite3_open_v2`.
- **Build flags.** Add `-DSQLITE_ENABLE_DBPAGE_VTAB` to `CGO_CFLAGS`:
  - Makefile `CGO_FLAGS` (Makefile:17–28)
  - `.goreleaser.yaml` `env`
  - CI `test.yml`
  - `docker/ast-mcp/Dockerfile`
- **Verify:**
  - It links against mattn's symbols with no duplicate-symbol errors on darwin_arm64 and linux_arm64. Run the release workflow dry run on the PR.
  - It recovers a deliberately corrupted fixture database.
- **Gate:** if linking or symbol resolution fails, stop and ask the user. The fallback is the `sqlite3` CLI `.recover`, behind `exec.LookPath`.

#### 3.1 Leaf package `internal/symref`
- Move `SymbolRow`, `SymbolFingerprint` and `LookupSymbol` from `internal/context/fingerprint.go` (27–82) into `internal/symref`. That package imports only db, errs, indexer and projectlinks. `internal/context` keeps thin forwarding functions for its API.
- Extract the pure classifier from `handoff/expand.go:156–204`. `Classify(projectPath string, a Anchor) (Change, *SymbolRow)` returns `ChangeFresh`, `ChangeMoved`, `ChangeModified`, `ChangeDeleted` or `ChangeFileMissing`; the constants move from `handoff/types.go:83–91` and are re-exported.
- `expandPointer` calls `symref.Classify`, then renders. The existing handoff tests must stay green.

#### 3.2 Health checks (HC-1–HC-3)
- **`internal/db/health.go`:**
  - `type CheckResult struct { DB, Kind string; OK bool; Detail string; CheckedAt time.Time }`.
  - `QuickCheck(db string)`, `IntegrityCheck(db)` and `ForeignKeyCheck(db)` run on a dedicated connection.
  - `Orphans() map[string]int` counts in Go across databases:
    - edges whose `source_file` has no symbols
    - code vectors whose `symbol_id` is missing
    - note and memory vectors whose ref is missing in context.db
    - `superseded_by` refs that don't exist
    - revisions without notes
    - anchors to unknown projects (P3.6)
  - Results are stored in memory (mutex) and in usage table `db_health_checks` (usage step #5).
- **Scheduling, `db.StartHealthLoop(ctx)` started from `startBackgroundServices`:**
  - Quick check at startup, +2 minutes.
  - Daily quick check plus orphans, run when `EmbedQueueIdleHook()` reports idle, deferred up to 6 h otherwise.
  - Weekly integrity and FK checks.
  - A loop body that follows STYLEGUIDE §11.
- **Surfacing:**
  - `/health` (`main.go:499–527`) adds `db_ok: {index, context, usage}`.
  - `buildHealthData` (react_api.go:56–87) adds `DBHealth`.
  - `metrics_prom.go` adds GaugeVec `astcache_db_integrity_ok{db}`, `astcache_db_orphans{kind}`, `astcache_snapshot_age_seconds` and `astcache_disk_free_bytes`.
  - API: `/api/dashboard/db-health` and `POST /api/dashboard/db-health/check`.

#### 3.3 Snapshots (SN-1–SN-4)
- **`internal/db/snapshot.go`:**
  - `TakeSnapshot(reason string) (*SnapshotInfo, error)` writes `VACUUM INTO <snapdir>/<UTC 20061009T150405Z>-<reason>/{context,usage}.db` and then runs `quick_check` on each copy through a fresh connection. If a copy fails, it deletes that copy and returns an error (SN-3).
  - It skips with `CodeUnavailable` when `SampleDiskSpace().Level != DiskOK`.
  - `ListSnapshots()` and `pruneSnapshots(keep)` (default 7, setting `snapshot_keep`; directory setting `snapshot_dir`, default `<dataDir>/snapshots`) never touch `pre-5.0/` or `corrupt-*/`.
- **Scheduling:** daily via the health loop.
- **`runSteps`** takes a snapshot when there are pending steps and `user_version` ≥ 1. That generalizes DB-5.
- **Restore** always takes a snapshot first (SN-2).
- **Dashboard:** a Storage section sub-box (`SettingsTab.tsx` between 816 and 817) with a list, "Take snapshot now" (`POST /api/dashboard/snapshots`) and settings.

#### 3.4 Auto-heal (AH-1–AH-4)
- **`internal/db/heal.go`:**
  - `var repairing atomic.Bool`, `Repairing() bool`, and a `HealState` struct (status struct pattern from `datadir_move.go:18–55`).
  - Add `CodeRepairing` to `internal/errs/codes.go`.
- **The write gate in `handleToolCall`** goes after 250–264 and before 278:
  - When `db.Repairing()` is true and the tool is in `writeTools`, return the deny envelope with `{error:"repairing", message}`.
  - `writeTools` is a const set in `internal/mcp/writetools.go`. It is explicit, because `ReadOnly` is incomplete:
    - `index_files`, `cache_summary`, `store_context`, `edit_context`, `flush_context`
    - `store_memory`, `forget_memory`, `report_kv_repair_event`
    - `define_context_fn`, `apply_context_fn`
    - `handoff`, `open_handoff`, `scratchpad`
    - `fetch_doc`, `add_doc_source`, `remove_doc_source`, `update_doc_source`
  - While repairing, usage logging is buffered: `stopWriteBatchers()` keeps the in-memory queue. Verify in `writebatch.go`, and add a bounded hold if needed.
- **AH-2, `HealContextOrUsage(name string)`**, triggered by a failed check:
  1. Set `repairing`.
  2. Call `BeforeForceCheckpoint`, which pauses embedding.
  3. Flush where possible, and close the affected pool(s): `ContextDB` + `HandoffWriteDB`, or `DB` plus a stop of the batchers.
  4. Move the file and its `-wal`/`-shm` to `snapshots/corrupt-<ts>/`.
  5. Run `recover.Recover(corrupt, new)` (3.0), then `integrity_check` on the result. On failure, copy the newest passing snapshot into place.
  6. Reopen the pool(s), run `init*Schema` and `runSteps`, and resume the batchers.
  7. Clear `repairing` and record a heal event.
  - A 60 s timer: if healing hasn't finished, it stays in **read-only** mode (`repairing` stays true) and raises a dashboard alert with "Retry heal" and "Restore snapshot X" actions (API `POST /api/dashboard/heal`).
  - A test helper corrupts a database: close the pools, `WriteAt` random bytes over pages 2–4, then reopen (`internal/db/heal_test.go`).
- **AH-1, `HealIndex()`:**
  1. Snapshot the project list (`SELECT DISTINCT project_path FROM symbols` if readable; otherwise `projectmeta.DiscoverPaths()` plus pinned projects).
  2. `quiesceIndexPool`, move the files aside, then `restoreIndexPool`.
  3. `initIndexSchema`, `createFTSTriggers`, `startFTSRebuild`, `search.Cache.Unload()`, `cache.Candidates.ClearAll()`.
  4. Call the hook `db.ReindexHook(projects []string)`, set in main. It calls `startIndexJob` for each project, then `docs.EmbedAllSources()`, and new bulk backfills `contextnotes.ReembedAll(emb)` and `memory.ReembedAll(emb)`.
- **AH-3:**
  - Heal events go in the usage table `heal_events(id, db, started_at, finished_at, method recover|snapshot|rebuild, snapshot_age_seconds, before_json, after_json, corrupt_path)` (usage step #6). Counts come from the last successful health check and are compared after the heal.
  - **Session notice:** `mcp/healnotice.go` keeps a `map[sessionID]healID` of sessions already told. `handleToolCall` appends one content line: "ast-context-cache repaired context.db at …: N notes/M memories since the last snapshot may be missing (see dashboard)".
- **AH-4:** the dashboard lists `corrupt-*` folders with sizes and a delete button behind a confirm dialog (`DELETE /api/dashboard/corrupt?name=`).

#### 3.5 Provenance and quarantine (TR-1–TR-5)
- **Schema (context step #5):** `structured_memory` and `context_notes` add `origin TEXT DEFAULT 'agent'`, `trust TEXT DEFAULT 'trusted'`, `quarantined INTEGER DEFAULT 0` and `content_sha256 TEXT`.
- **Origin:**
  - The hooks client sends header `X-AST-Origin: hook`. It is honored only from loopback, and checked in `server.go` before dispatch and stored in the request context or args.
  - `store_*` accepts `origin_url` (an optional source URL).
  - Writers set origin explicitly:
    - handoff open/complete: `handoff`
    - extract paths: `extract`, inheriting the source note's trust
    - dashboard pinned: `user`
    - offload: `offload`
  - Store sha256 on every write and edit.
- **TR-2, untrusted detection, new `internal/trust`:**
  - `FromDocs(text string) bool` is true when `origin_url` is given and isn't a local file. It is also true when the text shares at least 2 verbatim spans of 160+ characters with cached `doc_content`, found by an FTS phrase query on the longest lines (const query in `docs`, exposed as `docs.ContainsVerbatim(text)`).
  - Offload notes that contain doc chunks (retrieve with include_docs) are untrusted.
  - `LooksLikeInstruction(text string) bool` uses a const regexp list:
    - imperative openers: always, never, run, execute, install, download, delete, ignore (previous|all), you must
    - URLs
    - shell patterns: `| sh`, `| bash`, `curl `, `wget `, `rm -rf`, `$(`, backticks, `chmod +x`
  - **Quarantine** is untrusted ∧ (procedure kind ∨ `LooksLikeInstruction`).
- **Filters:**
  - Recall, list, search and `fetch_context` exclude quarantined rows unless `include_quarantined=true` (a new argument on `recall_memory`, `search_context`, `list_context` and `fetch_context`). When included, the line is prefixed `[quarantined: untrusted instruction]`.
  - `fetch_context` and `recall` verify sha256 and add `integrity: "mismatch"` (TR-4).
- **TR-5:** an in-memory per-session ring of the last 20 memory-write vectors. More than 10 writes within 60 s with pairwise similarity above 0.9 inserts a `review_items` row of kind `burst` (settings `burst_count`, `burst_window_s`, `burst_similarity`).
- **Table (context step #6):** `review_items(id INTEGER PRIMARY KEY, kind TEXT, ref TEXT, reason TEXT, created_at TEXT, resolved_at TEXT, resolution TEXT)`, with kinds `quarantine`, `stale`, `burst`.

#### 3.6 Code anchors (AN-1–AN-5)
- **Schema (context step #7):**
  - `anchors(id INTEGER PRIMARY KEY, owner_ref TEXT NOT NULL, owner_kind TEXT NOT NULL, project_path TEXT, file_rel TEXT, fqn TEXT, name TEXT, kind TEXT, start_line INTEGER, end_line INTEGER, fingerprint TEXT, inferred INTEGER DEFAULT 0, status TEXT DEFAULT 'fresh', checked_at TEXT)`.
  - Indexes on `(project_path, file_rel)` and `(owner_ref)`.
- **New `internal/anchors`** (imports db, symref, errs and logging only):
  - `Record(ownerRef, ownerKind, projectPath string, a symref.Anchor, inferred bool) error` looks up the symbol and stores its fingerprint.
  - `Reclassify(projectPath, fileRel string)` runs `symref.Classify` for each anchor in the file and updates `status`. Moving to modified, deleted or file_missing inserts a `review_items` row of kind `stale`; moving back to fresh resolves it.
  - `StatusFor(refs []string) map[string]string`.
- **Explicit anchors:** `store_memory` and `store_context` accept `anchor: {file, symbol, project_path}`.
- **Inferred anchors (AN-2):** after a store, scan the text for names in the session's returned-symbol set (`ReturnedKeys(sid)` → `file|name|line`), using whole-word matches. Record up to 3 inferred anchors.
- **Hook:**
  - New `indexer.OnFileIndexed func(filePath, projectPath string)`, invoked after `notifyIndexCommitted` at `indexer.go:656`, and in `prune.go` `PurgeFile`, as an `OnFilePurged`-style variable.
  - Set in `main.go` right after `db.Init` (around 164, *not* in `finishStartup`, which can return early): call `anchors.Reclassify(project, rel)` in a goroutine with a per-file debounce.
- **Output:** recall lines and list index lines get a `⚠stale` flag. The status shows in `expand` output, and the review queue lists the entry. Staleness never invalidates on its own (AN-4).
- **AN-5:** `astcache_anchor_stale_ratio` gauge and a dashboard figure.

#### 3.7 Review queue and health UI (UI-1)
- **API:** `GET /api/dashboard/review` and `POST /api/dashboard/review/{id}` with `{action: approve|keep|reject}`.
  - Approve on quarantine sets trusted and clears the quarantine.
  - Keep on stale re-records the anchor's fingerprint against the current code.
  - Reject invalidates memory with `ReasonQuarantine` or `ReasonStaleAnchor`. For notes it flushes the note, after a confirm dialog.
- **UI:** `ReviewQueueCard` in `MemoryTab.tsx`. `DBHealthCard`, with integrity, snapshots and heal history, goes in the Settings Storage section, plus a banner in `HealthBar` while repairing or read-only.
- Logic in `lib/review.ts` and `lib/dbHealth.ts`, with tests; fixtures and stubs; stories `Memory/ReviewQueue` and `Settings/DbHealth`, added to `verify-stories.mjs`.
- **Realtime:** new `realtime.Review` and `realtime.DBHealth` reasons → panels `review`, `db-health` → `useWebSocket.ts` keys.

### Phase 4: context and agents (5.x)

#### 4.1 Shrink guard (NT-1)
- **`contextnotes/edit.go`:** between 132 and 133, if `before > 0 && after < before*(100-pct)/100 && !in.Force && action != EditRevert`, return `errs.NewCode(CodeShrinkGuard, "edit shrinks note beyond guard", "before", before, "after", after, "pct", pct)`. The setting is `context_shrink_guard_pct` (default 50).
- `EditInput.Force` and the `edit_context` `force` argument.
- `apply_context_fn` passes `Force: true`, because context functions exist to reclaim tokens, and logs a debug line.
- Test: AC26.

#### 4.2 Playbooks (NT-2–NT-5, NT-7)
- **Schema (context step #8):** `playbook_bullets(note_ref TEXT, bullet_id INTEGER, section TEXT, text TEXT, helpful INTEGER DEFAULT 0, harmful INTEGER DEFAULT 0, retired_at TEXT, created_at TEXT, updated_at TEXT, PRIMARY KEY(note_ref, bullet_id))`.
- **`contextnotes/playbook.go`:**
  - `KindPlaybook`.
  - `renderPlaybook(rows) string` produces sections, with `- [b12] text (+3/−1)` per bullet. Note content is that rendered text, materialized on each op through `commitEdit` so FTS, embedding and revisions keep working.
  - **Ops** live in `applyPlaybookOp`, dispatched from `Edit` before `applyEdit` when `note.Kind == KindPlaybook`:
    - `add_bullet {section, text}`
    - `update_bullet {bullet, text}`
    - `vote {bullet, helpful|harmful}`
    - `vote_cited {session_id, outcome}`
    - `merge {bullets:[b1,b2], text}`: keep the first and retire the rest
    - `retire {bullet}`
  - A whole `rewrite` of a playbook needs `force=true`.
- **NT-4:** on `add_bullet`, embed the bullet (embedder with a 300 ms timeout) and compare it with the bullet vectors in the vectors table (`doc_type='bullet'`, `source_file='bullet:<ref>#<id>'`). At similarity ≥ `playbook_dedup_similarity` (0.9), return the existing bullet unless `allow_duplicate`. Without an embedder, fall back to FTS over the note's bullets (AC27).
- **NT-5:** `fetch_context` on a playbook with `token_budget` renders the bullets ranked by (helpful − harmful), then `updated_at`, and reports `withheld`.
- **Citation tracking:**
  - usage table `playbook_citations(session_id, note_ref, bullet_id, cited_at)` (usage step #7).
  - Written when bullet IDs are returned by `fetch_context`, or referenced in `edit_context` and `store_*` calls for that session.
  - `vote_cited` votes on the bullets cited since the session's last `vote_cited`.
- **Flag:** `feature_playbooks` (default on).

#### 4.3 Hooks: compaction, briefing and votes (CH-1–CH-6, NT-6)
- **`internal/hooks/hooks.go`:**
  - New events `pre-compact`, `post-compact`, `post-tool-use-bash` and `post-tool-use-failure-bash`, as consts at 25–31 and cases at 148.
  - `Input` (81–94) adds `Trigger`, `CustomInstructions`, `CompactSummary`, `ToolInput {Command string}`, `ToolResponse` (raw JSON) and `Error string`.
  - Update the unknown-event tests to use a still-unknown name such as `"user-prompt-submit"` (hooks_test.go:497, cli_hook_test.go:65).
- **CH-1, `preCompact`:**
  - Read `transcript_path` from the per-session offset stored in the registry (a new `registry.go` field `TranscriptOffset`).
  - Extract user and assistant text turns, ignoring tool payloads over 2 KB.
  - Call `store_context` with `kind=transcript_archive`, label `transcript <first ts>–<last ts>`, `origin` hook.
  - The bucket is excluded from quotas (1.7) and capped at 2 MB per session (`transcript_archive_max_bytes_session`); older archives are evicted first. TTL is 7 days after last access, through the same purge job with a per-kind TTL map.
  - Never block: always exit 0, with empty output.
- **CH-2, `postCompact`:** `store_context` with `kind=compaction_checkpoint`, content = `compact_summary`, `extract_memory=true`, origin hook.
- **CH-3, `sessionStart`** (164–180): build a briefing within 1,200 tokens (`maxDigestTokens`).
  - For `compact`, it includes the latest checkpoint ref, from a new MCP call `list_context` with `kind` filter. Add a `kind` argument to `list_context`.
  - The pinned note for the project: map `cwd` to a project via a new read-only tool action. Simplest is `recall_memory` plus `list_context` with `project_path=cwd`, since the server normalizes paths.
  - The one-line index from `list_context` (MEM-10) plus `recall_memory` (MEM-10).
  - For `startup` and `resume`, the same briefing without the checkpoint.
  - Handoff re-surfacing on compact stays.
- **CH-6, recitation:**
  - `kind=plan` notes with metadata `recite=true`.
  - New `edit_context` action `set_meta` (a JSON merge of metadata) is used to set `done=true`.
  - `internal/mcp/recite.go` keeps a per-session cache of the active reciting plan, invalidated on note writes. `handleToolCall` appends a second content block for the four search tools: `Goal: <first line, truncated to 30 tokens>` (AC30).
- **NT-6, the votes hooks:**
  - Match the command against `playbook_vote_patterns` (setting; default `go test|make (test|lint|build|race)|npm (test|run (build|lint))|pytest|cargo (test|build|clippy)|golangci-lint|tsc|eslint`).
  - Success means PostToolUse; failure means PostToolUseFailure whose first line is `Exit code N`.
  - Call `edit_context` with action `vote_cited` and the outcome. Gated by `feature_playbook_votes_hook` (default off).
  - **Spike first:** run a failing `go test` under Claude Code with a logging hook to confirm the PostToolUseFailure payload. Save it as `docs/spikes/fixtures/PostToolUseFailure.Bash.json`.
- **Installer (`component_hooks.go`):**
  - `claudeHookSpecs` becomes a function of the flags: compaction specs (PreCompact, PostCompact, plus the existing SessionStart) when `feature_compaction_hooks` is on; vote specs (PostToolUse and PostToolUseFailure, matcher `Bash`) when `feature_playbook_votes_hook` is on.
  - `installed()` (244) compares against the desired set, not the fixed 4.
  - Update the goldens under `testdata/claude_code/*` and `TestHooksAppendKeepsUserHooks` (engine_test.go:234–256).
- **Flags:** `feature_compaction_hooks` and `feature_playbook_votes_hook`, both default off.
- **CH-5, Codex:**
  - **Spike:** confirm the location of Codex's hooks config (`~/.codex/hooks.json` vs `[hooks]` in `config.toml`) and its event payload field names, saved as fixtures under `docs/spikes/fixtures/codex/`.
  - **Implement:**
    - Add a hooks unit to `targets_codex.go:15`, replacing `unsupportedUnit`, using the same JSON or TOML edit machinery as the Codex MCP entry.
    - Commands are `ast-mcp hook --host codex <event>`.
    - `hooks.Input` normalizes Codex field names.
    - Events: SessionStart, PreCompact and PostCompact (compaction flag), PostToolUse (votes flag).
  - **Cursor:** add a documentation section in `docs/host-integration.md` (71) and `docs/handoff.md`, explaining the manual `~/.cursor/hooks.json` setup for `sessionStart`, with `preCompact` marked as undocumented. No installer changes.

#### 4.4 Handoff (HO-1–HO-5)
- **Schema (context step #9):** `scratchpad_entries` adds `topic TEXT`, `choice TEXT` and `conflicting INTEGER DEFAULT 0`.
- **Decisions:**
  - `types.go:72–80` adds `EntryTypeDecision`, and `Valid` at 652–658 accepts it.
  - `scratchpad.go:70–72` allows it, and requires `topic` and `choice`.
  - On post, a const query finds active (not retracted) decisions in the same tree and topic with a different choice. If any exist, `UPDATE … SET conflicting=1` on all of them, in the same transaction.
  - `Read`'s ORDER BY (35–41) becomes `(type='decision') DESC, id`.
  - `collect` includes `conflicts: [{topic, choices[]}]`.
  - Update the `scratchpadTool` enum at `handoff_tools.go:119` and add `topic` and `choice` properties, keeping it under the 1,200-token schema budget test (117–123).
- **HO-3:** `open_handoff` gets `inherit_seen` (boolean). `runOpenHandoff` (handoff_tools.go:232–270) passes it through `OpenRequest`, and `seedChild` (open.go:239–257) runs the manifest seeding for fresh mode when it's set.
- **HO-4:** a test only, asserting that a truncated summary still carries the ref (complete.go:223–225 already does this).
- **HO-5:**
  - `TreeView` (dashboard.go:36–56) adds `TokensTotal` (snapshot + delivered + results, from existing counters) and `SingleAgentEstimate`. The estimate is the parent snapshot tokens plus Σ child `tokens_delivered` plus Σ child search output tokens, from `queries.output_tokens` for the child sessions, summed in Go. Also add a `Method` string.
  - `HandoffTreesCard.tsx` shows both; logic in `lib/handoffTree.ts` with a test.

#### 4.5 Coding-memory eval (EV-1)
- **New `internal/memeval`:** table-driven scenarios run through `callTool`-style helpers against the MCP handler, with the hash embedder. Cases:
  - knowledge update: B supersedes A; recall returns B; history shows both
  - world-time `as_of`
  - restore
  - paraphrase conflict candidates
  - quarantine exclusion
  - staleness after editing an anchored fixture function, then reindex
  - abstention (`no_match` / empty recall) when nothing was stored
- `make eval-memory` runs `TEST_PKGS=./internal/memeval/...`; it is also part of `make test`.

#### 4.6 Docs (all phases, finalized in P4)
- `README.md` (token section, feature list)
- `AGENTS.md`, `CLAUDE.md` instructions, `skills/*` plus `.cursor/skills/*`
- `docs/handoff.md` (hooks events table 370–377, compaction W6)
- `docs/host-integration.md`, `docs/context-edit.md` (shrink guard, playbooks, `set_meta`), `docs/USAGE.md`, `docs/MIGRATING.md`, `docs/stubs.md`

---

## Testing Strategy

**Per phase:**
- `make test`, `make race TEST_PKGS=<touched>`, `make lint`, `make ui-test` and `make verify-stories`, plus `make bench-tokens`, with its table pasted into the PR.
- UI changes rebuild `dist`.
- Every new requirement has a unit or integration test (NF-8).
- **New test infrastructure:**
  - `embedder.NewHashEmbedder` (deterministic vectors, offline).
  - `tokenbench` harness (fixture + scenarios + baseline).
  - DB corruption helper (`internal/db/corrupt_test_helpers.go`, test build only via `_test.go` in package `db`, plus an exported helper in `dbtest` for other packages).
  - `callTool` defaulting to `output=json`.
  - Hook fixtures for PostToolUseFailure and Codex.
- **Patterns:** table-driven testify. `dbtest.Init` for database tests. `newMCPServer` (protocol_test.go:19) for protocol and stream tests. `fakeMCP` for hooks. Goldens updated deliberately. UI logic in `lib/*.test.ts`.

**Edge cases to cover:**
- Ties in every sort.
- Empty index and zero results (no_match vs null).
- Calls without `session_id` (offload synthetic session, dedup off).
- Legacy sessions without streams.
- Toggling a flag mid-call.
- Huge results (offload at exactly the threshold).
- Notes shrinking by exactly 50%.
- Restoring a chain of three.
- TTL expiry at a second boundary.
- `as_of` in RFC3339 vs DateTime.
- FK enforcement on doc removal.
- A heal timing out into read-only.
- Disk-low snapshot skips.
- Anchors to deleted files.
- Quarantine of an untrusted procedure.
- Playbook dedup without an embedder.
- A Codex payload missing fields (hooks must fail open).

**Acceptance criteria → tests:**

| AC | Test (package: name) |
|---|---|
| 1 | search: `TestTopMatchesSortedWhenFewerThanLimit` |
| 2 | tokenbench: determinism assertion; mcp: `TestRetrieveBytesStableNoTimings`, `TestRetrieveTimingsInMeta` |
| 3 | mcp: `TestStatelessListChangedCoalesced`, `TestLegacySessionToolListFrozen` |
| 4 | mcp: `TestCodeToolsTextOutputDefault`, `TestCodeToolsJSONOutput` |
| 5 | context: `TestCapsuleNoMatchOnWeakTop` (+ tokenbench negatives) |
| 6 | context: `TestCollapseTestHits`, `TestCollapseSkippedForTestQueries` |
| 7 | context: `TestEffectiveModeAutoTopThree` |
| 8 | context: `TestEditModeTargetAndCallees` |
| 9 | mcp: `TestResultOffloadHeadAndFetch` |
| 10 | contextnotes: `TestOffloadExpiryTombstone` |
| 11 | tokenbench baseline diff in each phase PR; P1 PR shows a reduction on code-search scenarios |
| 12 | dashboard: `TestDashboardStatsExcludesVirtual` |
| 13 | memory: `TestPruneArchived365`, `TestHistoryAfterSupersede` |
| 14 | memory: `TestRestoreReactivatesAndInvalidatesSuccessor` |
| 15 | memory: `TestWorldTimeAsOf` |
| 16 | memory: `TestRecencyFactorRanking`, `TestProcedureExemptFromDecay` |
| 17 | mcp: `TestStoreMemoryPossibleConflicts` |
| 18 | memory: `TestConflictFallbackFTS` |
| 19 | contextnotes: `TestPinnedCap` |
| 20 | db: `TestHealContextRecoversOrRestores` (corruption helper) |
| 21 | db: `TestHealIndexRebuild` (with a ReindexHook stub) |
| 22 | db: `TestSnapshotRetentionAndQuickCheck` |
| 23 | trust/memory: `TestUntrustedInstructionQuarantined` |
| 24 | contextnotes/memory: `TestIntegrityMismatchFlag` |
| 25 | anchors: `TestReclassifyMarksSuspectAndClears` (fixture + `IndexFile`) |
| 26 | contextnotes: `TestShrinkGuard` |
| 27 | contextnotes: `TestPlaybookAddBulletDedup` |
| 28 | hooks: `TestVotesHookHarmfulOnFailure` (PostToolUseFailure fixture) |
| 29 | hooks: `TestPreCompactArchive`, `TestPostCompactCheckpoint`, `TestSessionStartBriefingBudget` |
| 30 | mcp: `TestPlanRecitationLine` |
| 31 | handoff: `TestConflictingDecisionsFlagged` |
| 32 | memeval: whole suite in CI |
| 33 | db: `TestStepsAreAdditive` + manual: v4.0.11 release binary against a P0+ data dir; `pre-5.0` snapshot exists |

**Manual test plan (each phase PR):**
- Run the phase on both Macs through the Updates button after the release, and watch the dashboard.
- **P1:** Claude Code session with output text, a capsule and an offload, plus the flag toggle warning.
- **P3:** take a copy of a real `context.db`, corrupt it, start, and watch the heal.
- **P4:** a real Claude Code `/compact` with the compaction flag on.

---

## Risks & Open Questions

1. **tiktoken-go o200k availability and speed.** The offline loader may not include o200k; if so, we embed our own vocabulary. BPE speed may miss NF-1 or slow indexing; the mitigations are memoization and `tokens.Approx` for the indexer. Decided per measurements in P0.
2. **Vendored recover linking.** The recover C code must link against mattn's in-package amalgamation symbols on both platforms. The DBPAGE flag must reach every build path: Makefile, goreleaser, CI and Docker. The P3.0 spike is a hard gate, and a failure means asking the user before using the CLI fallback.
3. **FK enforcement** may break existing deletes (`doc_sources`). It is tested in P0, with deletes ordered child-first.
4. **Text output by default** breaks scripts that parse JSON from the four tools. The mitigations are `output=json`, `feature_text_format`, and MIGRATING. `execute_code` users must pass `output=json`, and `code_script_hints` text explains this.
5. **Frozen legacy tool lists.** Users who toggle flags must reconnect their agents. Claude Code's protocol era decides how often this matters; check which era it negotiates against v5. The dashboard warning covers it.
6. **Relevance floor thresholds** across score scales (RRF vs similarity). Tuning depends on how representative the fixture is. Mitigations: conservative defaults, per-call overrides, and the `withheld` counts make dropped results visible.
7. **`mode=edit` callee approximation** is name-based. It can miss method calls through interfaces or pick up same-named functions; it is capped and shown as signatures only.
8. **Untrusted detection heuristics** (doc overlap) can miss paraphrased doc content, or flag legitimate copied snippets. Mitigations: quarantine applies only to instruction-like text, and the review queue shows everything.
9. **Offload without `session_id`** uses a synthetic daily session. Refs are fetchable by anyone with the ref, which is the same as notes today.
10. **Transcript ingest scope** covers all `~/.claude/projects`, because it was not restricted. It is opt-in and stores numbers only. Revisit if a privacy concern comes up.
11. **mcp-local's hard-coded tier list** may still list the removed bundle tools, which needs a cross-repo PR (P1).
12. **Plan size.** P1 and P3 are large. If a phase PR exceeds about 2,500 changed lines (excluding the committed dist, fixtures and vocabulary), split it into two PRs within the phase (for example P1a tool surface + determinism + output; P1b precision + modes + offload + ledgers).
13. **Codex hook format** is unverified; the P4 spike decides it. If Codex's payloads lack the needed fields, CH-5 drops to docs-only (it is a SHOULD).
14. **Claude Code PostToolUseFailure payload** for Bash is unverified. The NT-6 spike decides it; the requirement is a SHOULD and its flag defaults off.

**Deviations from the PRD (recorded):**
- TL-1 moves to P0.
- The memory archive stays in place, with no separate table.
- Offload uses a synthetic session when there is no `session_id`.
- TR-2 is implemented as an explicit `origin_url` plus doc-overlap detection, because no server path moves doc text into notes automatically.
- The retrieve `format` argument keeps its meaning, and the new switch is `output`.
- CH-5: Codex hooks are installable, and Cursor gets docs only.
