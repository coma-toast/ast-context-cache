# Plan: Subagent Handoff Cache, Feature Flags & Safe Installer (v4.0)

Let a parent agent hand a subagent a compact, snapshot-backed brief, and get a ref plus a capped summary back. Siblings coordinate through a shared scratchpad and queued claims. All of it is gated by live feature flags, and it ships with a merge-safe installer for 7 hosts. Along the way the repo moves to `slog`, `errs`, and SQL constants.

- **Date:** 2026-10-05
- **Source PRD:** [prd-subagent-handoff.md](prd-subagent-handoff.md)
- **Related:**
  - Companion PR in `coma-toast/mcp-local`: a stdio bridge, delegating registration to the ast installer, and writer fixes.
  - No ticket. Agents never merge; the user merges both PRs.
- **Delivery:**
  - **One PR** on `ast-context-cache`, built as ordered, independently green commits (one or more per phase), plus one companion PR on `mcp-local`.
  - `VERSION` → `4.0.0`, with a `## Migrating to 4.0` README section.
  - Work happens in a **wtg space** containing both repos.
- **Code references** are as of `d1bb65d` (v3.0.40). Re-verify line numbers once Phase 1's mechanical sweeps land, because they shift almost every file.
- **Exploration notes** are offloaded to `[ctx_27d34dbe6452]`, session `dc405d1d-ab74-4c12-bb22-b5ade72af9b6`.

---

## Context

### Explored

**MCP transport** (`cmd/ast-mcp/main.go:117-137`, `internal/mcp/server.go:66-132`)
- HTTP only, plain JSON POST. No SSE, no `Mcp-Session-Id`.
- `initialize` hard-codes `protocolVersion: "2024-11-05"` and `capabilities: {tools:{}}`.
- `"notifications/initialized"` falls through to `MethodNotFound`, which sends a response to a notification. There is no `ping` handler.
- `GET /mcp` returns the tool list as JSON.
- `tools/list` rebuilds via `FilterTools(srvCfg)` on every call. `srvCfg` (`server.go:27`) is an unsynchronized global.

**Tool plumbing**
- `GetTools()` is at `internal/mcp/tools.go:108-622`, `toolAccess` at `:70-81`, and the deny enum at `:53-61`.
- The handler switch is `server.go:203-555`, and its default branch chains `handleMemoryTool` / `handleContextTool` (`memory_tools.go:15`, `context_tools.go:203`).
- Responses use a top-level `"error"` string. The only structured error is `contextnotes.LimitErrorMap` (`internal/contextnotes/store.go:736-750`).
- Prompts are at `tools.go:649-813`.

**Storage**
- There are three SQLite pools (`internal/db/pools.go:13-60`): index.db (single writer goroutine, `indexwriter.go:83-123`), context.db, and usage.db.
- context.db and usage.db writes autocommit, with no transaction helper and no mutex.
- Schema setup is `CREATE … IF NOT EXISTS` plus unconditional `ALTER … ADD COLUMN` (`internal/db/schema_*.go`). There is no version table.
- New context tables must also be added to `migrate_split.go:17-24`, `purge/purge.go:179-194`, and the dashboard reset-all (`internal/dashboard/api.go:~290-300`).

**Dedup**
- `SymbolDedupKey` (`internal/context/session.go:10`) is `absFile|name|startLine`.
- `GetReturnedSymbolKeys` (`:14`) reads only from the DB. `LogReturned` (`:53`) goes through a 3s write buffer (`internal/db/writebatch.go:240`).
- `retrieve` logs returns *before* trimming to budget (`internal/mcp/retrieve.go:274` vs `:155`).
- Duplicate hits inside one result list are not skipped.

**Query cache** (`internal/cache/query_cache.go`)
- 5-minute TTL; caches final capsule JSON, and only when there is no session (`internal/context/handler.go:60-71`, `:141-145`).
- `ClearProject` (`:106`) never matches anything, because keys are sha256 hashes rather than path-prefixed.
- The key omits `token_budget`.
- `Get` mutates its counters while holding only `RLock`.

**Search paths**
- capsule: `handler.go:28-147`
- search_semantic: inline in `server.go:264-347`, via `PackScoredResults` (`handler.go:150-200`)
- file_context: `internal/mcp/handlers.go:128-231`
- retrieve: `retrieve.go:70-277`

There is no shared finalization step. The per-tool `logToolQuery` calls (`server.go:239,343,375,540`) are the only points where query, session, and result all coexist.

**Symbols**
- `symbols.code` holds only the first line. `embed_hash` is truncated at 480 runes (`internal/indexer/embed_text.go:7-20`). Symbol ids are regenerated on reindex.
- So there is no usable fingerprint today.
- Building blocks exist: `indexer.ReadSourceRange` (`indexer.go:819`), `context.ApplyMode` (`pack.go:80`), `projectlinks.OwningProject` (`links.go:301`), `repokey.SameRepo` (`repokey.go:64`), `projectlinks.ResolveScopeWithRepoSiblings` (`links.go:208`).

**Memory**
- `memory.ExtractFromText` (`internal/memory/extract.go:37`) and `StoreExtracted` (`store.go:193`) are exported.
- `vectorSearch` (`recall.go:273-298`) applies no validity or scope filter and loses rank order. `validityClause` (`:120-128`) has identical branches.

**Background work**
- Background services start in `startBackgroundServices` (`main.go:407-425`) and `db.StartWALCheckpoint` (`db.go:260`).
- `FlushOrphans` and `PruneSuperseded` run only from the dashboard.
- The SIGTERM handler (`main.go:143-151`) calls `os.Exit` without flushing the write buffers.

**Dashboard**
- Plain `ServeMux` (`api.go:37-112`, `react_api.go:20-31`). No auth, no Origin check; the WebSocket accepts any origin (`ws_react.go:110`).
- The settings POST (`api.go:803-968`) writes any key.
- The UI is Vite 8, React 19 and MUI 9, with no state library. The Storybook API is stubbed (`ui/src/storybook/api-stub.ts`). There is no Vitest. `internal/dashboard/ui/dist` is committed.
- Prometheus uses `client_golang` with the `astcache_` prefix (`internal/dashboard/metrics_prom.go:30-93`).

**Installer** (`api.go:970-1169`, `agent_configs` at `schema_usage.go:53`, `db.go:112-151`)
- It overwrites and removes whole files, writes markdown into `~/.claude.json`, and hard-codes port 7821. It has no tests.

**Conventions**
- `log.Printf` (230 sites), `fmt.Errorf`/`errors.New` (157), and inline SQL in 59 files.
- No testify. Tests use `dbtest.Init(t)` (`internal/db/dbtest/dbtest.go:25`), the `callTool` helper (`internal/mcp/memory_tools_test.go:28`), and per-package `TestMain`.
- `make test` needs `-tags sqlite_fts5`. CI runs only `make test`.
- No JSONC/TOML libraries.

**mcp-local** (`~/git/mcp-local`)
- Everything lives under `internal/`, so it can't be imported.
- Its writers drop JSONC comments and are non-atomic. Its Claude Desktop entry `{type:http,url}` is invalid. It has no bridge.
- `fetchToolsList` sends `notifications/initialized` with `id:0` (`cmd/mcp-local/extras.go:1001`).
- `asttools/default_tiers.go` is a hard-coded copy of ast's tiers.

### Key patterns to follow

| Need | Follow |
|---|---|
| New feature package | `internal/contextnotes/` and `internal/memory/` layout (`store.go`, `limits.go`, `stats.go`, `types.go`, `prune.go`) |
| Settings with env override | `contextnotes.envOrSetting` (`limits.go:65`) → generalized into `internal/flags` |
| MCP adapter | `internal/mcp/memory_tools.go` router `handleXTool(name, args) (any, bool, error)`, arg helpers `strArg`/`boolArg`/`parseStringList` (`:191-250`) |
| Token/savings attribution | `context.SavingsMeta` + `logToolQuery` (`server.go:576-593`) |
| Structured error payload | `LimitErrorMap` (`contextnotes/store.go:736`), generalized in `internal/errs` + `handoff.ErrorMap` |
| Tool tests | `callTool` (`memory_tools_test.go:28`), golden lists `tools_example_test.go:17,29,41` |
| DB tests | `dbtest.Init(t)` + `dbtest.WaitFor` |
| Fixture project | `indexedPython()` (`internal/mcp/summary_tool_test.go:18`) |
| Realtime → UI refresh | `realtime.Reason` bit (`internal/realtime/realtime.go:13-31`) → `realtime_bridge.go:41-68` → `ui/src/hooks/useWebSocket.ts:3-14` |
| Prometheus | `metrics_prom.go` `mustRegister`, test in `metrics_prom_test.go` |
| Settings UI section | `SettingsTab.tsx` card pattern, `save()` (`:84-92`) |
| Background loop | `startBackgroundServices` (`main.go:407`), plus STYLEGUIDE §11 loop shape |

### Architectural constraints

- **Cross-DB atomicity is impossible.** All handoff tables live in **context.db**. Reads from usage.db (dedup, trail) happen before the context.db transaction, and child dedup rows are deleted best-effort at expiry.
- **Writes to context.db need serializing for claims (NFR-4).** Add a dedicated one-connection pool opened with `_txlock=immediate` for handoff writes. Never upgrade a DEFERRED transaction from read to write.
- **The query cache must cache *ranked candidates*, not final JSON.** Dedup, mode, budget, and `LogReturned` are woven into the pack loops, and `hitFromScored`/`ApplyMode` mutate `Data` maps, so cached candidates must be deep-copied on read.
- **There is no response middleware.** Annotations and trail capture are explicit calls at four sites. The `claims_granted` notice piggybacks as a second `content` item in `handleToolCall`'s envelope (`server.go:561-573`).
- **The installer must edit JSONC and TOML without losing comments.** That means `hujson` patches and text-level TOML table edits.
- **The UI `dist` is committed.** Every UI change needs a rebuilt dist in the same commit.
- **Clean-room `errs`.** `~/git/common/errs` is Slide company code; this repo is personal and public on GitHub. Re-implement the API shape (`New`, `NewCode`, `Wrap*`, `HasCode`, `Code`, fields) from the STYLEGUIDE's description. **Do not copy source.**

---

## Requirements (from PRD)

| IDs | Priority | Summary |
|---|---|---|
| HO-1–HO-10 | MUST | Create a handoff with an atomic, immutable snapshot (manifest, trail, notes, memory, pointer fingerprints). Prune options; caps; nesting depth 3; 16 children. Snapshot survives the parent flushing. |
| OP-1–OP-12 | MUST | Child opens with ref only; the server mints the child session. Resume; digest ≤1,500 tokens; on-demand expand. Fork seeds dedup. Delivered items marked returned. Staleness; worktree mapping; `parent_trail_match`/`parent_explored`; child recall sees snapshot memory. |
| RT-1–RT-7 | MUST | Result stored as a `handoff_result` note. Summary cap 300 with truncate/derive. Return stub. FACT/RULE lines promoted to parent memory. Claims released. Re-complete supersedes. |
| RT-8 | MAY | Structured `changed_files`/`open_questions`. |
| FI-1–FI-5 | MUST | Collect (budgeted, recursive), list by parent session, abandonment at 30 minutes, read partial work. |
| FI-6 | MAY | `wait_seconds` long-poll (≤60s). |
| SP-1–SP-7 | MUST | Per-tree append-only scratchpad: findings, dead ends, retract, cursor reads, live trail, `sibling_trail_match`. |
| CL-1–CL-4, CL-7, CL-8 | MUST | Advisory claims, FIFO queue, auto-release, auto-grant, documented as advisory, visible everywhere. |
| CL-5, CL-6 | SHOULD | `claims_granted` notice on any tool response; deadlock-cycle rejection. |
| DC-1–DC-4 | MUST | Cache applies with `session_id`; shared, index-aware invalidation; immediate dedup; child dedup expires with the tree. |
| RQ-1–RQ-5 | MUST | 7-day TTL from last access (child notes included); sweeper runs at least hourly; tree flush; caps 64k/300/depth 3/16 children; trail eviction first. |
| TS-1–TS-6 | MUST | Three tools, **all core**; ≤1,200 schema tokens; open-first instruction; structured error codes; golden tests and docs. |
| FF-1–FF-8 | MUST | Flag registry, env > setting > default, API + dashboard toggles, **live + `tools/list_changed`**, initial flags, no data loss, `tools.json` still applies. |
| HI-1, HI-2, HI-4–HI-10 | MUST | Manual path; Claude Code hooks (conditional on the spike); session-id resolution; fail-open; Cursor/Codex/VS Code/JetBrains instructions; workflow and mode docs; skills. |
| HI-3 | SHOULD | Auto-create the handoff from a `PreToolUse` hook on `Agent` (if the host supports it). |
| IN-1–IN-14 | MUST | 7 targets, global only. Components: MCP, skills, rules, hooks. Merge-only, backups, preview, surgical uninstall, correct formats/port, on-disk status, externally-managed detection, markers, canonical source, no env on URL entries, legacy migration, inventory doc. |
| **IN-15** | **MUST (promoted from MAY)** | `ast-mcp install\|uninstall\|verify` CLI; mcp-local delegates to it. |
| BF-1, BF-2, BF-4 | MUST | LIKE scope fix, vector-recall validity and scope, regression tests. |
| BF-3 | SHOULD | Dashboard honors `AST_MCP_TOOLS_CONFIG`. |
| OB-1–OB-6 | MUST | Repeat-search rate; handoff and return tokens saved; tree view; Prometheus; structured lifecycle logs. |
| NFR-1–NFR-11 | MUST | See PRD. NFR-1 latency, NFR-3 token budgets, and NFR-4 concurrency are test-asserted. |

**Plan-level requirements added during planning** (user decisions, same weight as MUST):

| ID | Summary |
|---|---|
| PL-1 | Migrate the **whole repo** to `log/slog` via `internal/logging`. `AST_LOG_FORMAT=text\|json` (default text), `AST_LOG_LEVEL` (default info). Satisfies OB-6. |
| PL-2 | Clean-room `internal/errs` port; migrate the **whole repo** off `fmt.Errorf`/`errors.New`. |
| PL-3 | Hoist **all** SQL in the repo to package-level `const` blocks (STYLEGUIDE §10). |
| PL-4 | Bind MCP and dashboard to `127.0.0.1` by default. `--listen`/`AST_LISTEN` opts out. Docker sets `0.0.0.0`. |
| PL-5 | Origin/Host guard on all mutating dashboard routes, the WebSocket upgrade, and `/mcp`. |
| PL-6 | Streamable HTTP: GET SSE stream, `Mcp-Session-Id`, 202 for notifications, `ping`. Negotiate `2024-11-05`, `2025-06-18`, and `2026-07-28` (the last only if Phase 0 verifies the official spec). `listChanged: true`. |
| PL-7 | Extra fixes: cache correctness (dead `ClearProject`, budget key, counter race); dedup correctness (retrieve budget, in-list dedup); recall (`IncludeHistory` no-op, rank order); shutdown flush. |
| PL-8 | `make race` + CI job, `make lint` (go vet + gofumpt) + CI job, Vitest for UI logic, Storybook stories + `verify-stories`. |
| PL-9 | Resolves HI-4: a `SessionStart` hook tells the agent to use the **host session id** as its ast `session_id`. Manual hosts keep free-form ids. |
| PL-10 | `ast-mcp hook <event>` subcommand implements all Claude Code hooks. No shell or jq dependency, and it fails open. |
| PL-11 | mcp-local companion: `mcp-local bridge <url>` (stdio → Streamable HTTP); Claude Desktop entry uses the bridge, with an `npx mcp-remote` fallback in the ast installer; tier table updated; hujson-safe atomic writes with backups; protocol fixes; ast-context-cache registration delegated to `ast-mcp install`. |

**Out of scope** (PRD Non-Goals):
- Shared model windows
- Inter-session access control (the open model stays)
- LLM summarization in the server
- Remote handoff
- Cursor cloud agents
- Windsurf, Gemini CLI, and Zed targets
- Project-scope installs
- `export_bundle`/`import_bundle`
- TTL for non-tree notes and dedup rows
- Filesystem lock enforcement

**Resolved open questions (user, 2026-10-05):**
1. Hook capabilities: decided by the Phase 0 spike. The agent builds and runs the probes; the user reviews the report before Phase 9.7 builds the hooks.
2. Session-id mapping: the hook injects the host id as `session_id` (PL-9).
3. Fork cache inheritance: affects only the guidance text, which the spike settles.
4. Cursor cloud agents: out of scope.
5. Protocol: negotiate all three versions; `2026-07-28` only after official-spec verification (PL-6).
6. LAN exposure: localhost by default (PL-4), plus an Origin guard (PL-5).
7. Tiers: **all core** (TS-2 kept).
8. Promoted memory **outlives** the tree.
9. Child notes **expire with the tree**.
10. Claude Desktop: `mcp-local bridge`, falling back to `npx mcp-remote`.
11. Host formats: verified in Phase 0.3.
12. Destructive installer: fixed in this PR, no hotfix. It is called out in the PR body.
13. OB-1 threshold of 50%: kept, and revisited after the trial.
14. Additional 4.0 breaking changes: localhost bind, the slog log format, `/api/agent-*` removed, `errs` error strings.
15. Oversized summary is **truncated**.

---

## Approach

The work lands as phased commits in one PR. Each phase compiles, passes `make test lint race`, and leaves the product usable.

0. **Workspace, spikes, spec verification.** Settle the unknowns before writing code that depends on them.
1. **Foundations.** `errs`, `logging`, SQL consts, gofumpt, lint/race targets, Vitest. These are mechanical, so they go first and all later code is written in the final idioms.
2. **Feature-flag registry** (backend only). Phases 3–9 gate on it.
3. **Correctness & caching.** BF-1/2, the candidate cache (DC-1/2), immediate dedup (DC-3), PL-7.
4. **Server hardening & transport.** Listen address, Origin guard, atomic config, version negotiation, Streamable HTTP SSE, flag → `list_changed` (FF-5).
5. **Search trail.** Capture for every session; the prerequisite for snapshots.
6. **Handoff core.** Schema, service, create/open/expand/complete/collect, staleness, annotations, sweeper, tools.
7. **Scratchpad & claims.**
8. **Observability & dashboard.** Metrics, tree view, flags UI, handoff settings.
9. **Installer, CLI & hooks.**
10. **Docs, prompts, release 4.0.**
11. **mcp-local companion PR.**
12. **Validation.** Scenario test, benchmarks, concurrency, installer fixtures, real-host trial.

**Why this order:**
- **Mechanical sweeps first.** They touch about 200 files. Doing them later would conflict with every feature commit and force feature code to be written twice.
- **Flags before the features they gate.** That way no commit ships an ungated tool.
- **Cache and dedup fixes before handoff.** Fork seeding and the "delivered = returned" rule (OP-6) depend on immediate dedup.
- **Transport before the flags UI.** Toggling then has an end-to-end `list_changed` path.
- **Installer last among features.** It installs the hooks, skills, and docs content the earlier phases define.

**Alternatives rejected:**
- **Derive the trail from the `queries` table.** It lacks hit counts and top hits, lags behind writes, has no `session_id` index, and falls back to hourly pseudo-sessions.
- **Clone parent memory lazily at recall time.** Cloning the session-scoped entries into the child session at open means FTS and vector recall work unchanged, and the clones expire with the tree for free.
- **Cache the final response JSON with `session_id` in the key.** No cross-session reuse, and dedup would be baked into the cached result.
- **Hook scripts in shell + jq.** Adds dependencies, and fail-open is harder to guarantee. `ast-mcp hook` is chosen instead.
- **Use mcp-local's writers as a library.** They're `internal/`, lossy, and non-atomic. Delegation runs the other way: mcp-local → `ast-mcp install`.
- **Edit `~/.claude.json` through `claude mcp add`.** It can't produce the preview diff IN-5 requires. Use a hujson patch with re-verify-before-apply instead.

---

## Style Guide Notes

`~/git/STYLEGUIDE.md` (the repo's canonical guide; `~/repos/STYLEGUIDE.md` doesn't exist on this machine) applies in full, following the user's decisions to migrate the whole repo.

**Rules that apply directly:**
- **§1 compact code.** No blank lines inside functions except between logical chunks.
- **§2 errors.** Use `internal/errs`: `errs.WrapMessage("failed to create handoff", err, "parent_session", sid)`. Sentinels are package-level `var errX = errs.New(...)`. Error messages start lowercase (§13).
- **§3 receivers.** Single letter (`s *realService`, `c *CandidateCache`).
- **§4 ID types.** Define `type HandoffRef string` (`hof_…`), `type TreeID string` (`hft_…`), and `type SessionID string` in `internal/handoff/types.go`. Each implements `slog.LogValuer` so it expands in logs.
- **§4 enum constants.** Type-prefixed: `ModeFresh`, `StatusAbandoned`, `SectionPointer`, `EntryTypeDeadEnd`.
- **§5 comments.** Comments say *why*. Every exported type and function gets a godoc comment.
- **§6 imports.** Grouping: stdlib, then third-party, then `github.com/…/ast-context-cache/internal/...`.
- **§9 control flow.** No multi-line `if` conditions; early returns; `switch` on status values.
- **§10 SQL.** Package-level `const` blocks, named `<verb><Thing>Query`. Dynamic filters are built from const fragments.
- **§10 migrations.** New DDL is appended at the **end** of `initContextSchema` / `initUsageSchema`.
- **§10 file order.** Types, then constructor, then exported methods, then handlers, then unexported.
- **§11 service pattern.** `handoff.Service` interface plus `realService`. The constructor `New(ctx, deps)` starts the background loops: sweeper, abandonment check, session-store eviction. Each loop has a separate body method and logs start/stop with `defer`.
- **§12 testing.** Table-driven tests with `t.Run`; `t.Helper()` in helpers; `t.Cleanup`.
- **§13 logging.**
  - Each package logger is `slog.With("tag", "<pkg>")`. Log messages start uppercase and use key/value pairs.
  - **Errors are passed keyed:** `logger.Warn("Failed to expire tree", "error", err)`. *(Changed during implementation: the STYLEGUIDE's bare form `logger.Warn("…", err)` fails go vet's `slog` analyzer, which `go test` runs, so it can't be used with stdlib slog.)* `internal/logging`'s handler expands any error value into `error=<msg>`, `error_codes`, and the `errs` fields, and still rewrites bare `!BADKEY` values defensively. Pass ID types keyed as well (`"handoff", ref`).

**Conflicts and how they're resolved:**

| Style guide says | Repo reality | Resolution |
|---|---|---|
| `errs` from Slide common | Not importable; Slide-owned code | Clean-room `internal/errs` with the same API shape (PL-2) |
| testify `require`/`assert` + mockery | No testify, hand-written stubs | Add `github.com/stretchr/testify` for **new and touched** tests only, using `require`/`assert`. Skip mockery: the handoff tests use a real temp DB, consistent with the repo. |
| `t.Parallel()` | DB pools are package globals | No `t.Parallel()` in DB-backed tests (documented in each `TestMain`). Pure-logic tests (errs, installer edit engines, trail normalization) do use it. |
| `common/util` helpers | Not available | Local helpers in the owning package. Don't create a grab-bag `util` package. |

TypeScript (STYLEGUIDE TS section): import grouping and naming; `interface` for object shapes in `ui/src/api/types.ts`; async/await in `client.ts`.

---

## Detailed Implementation Steps

### Phase 0 — Workspace, spikes, spec verification

#### 0.1 Create the space (spaces/wtg skills)
- **Do:** Create a wtg space `NO-TICKET-v4-subagent-handoff` with worktrees of `ast-context-cache` and `mcp-local` on branch `NO-TICKET-v4-subagent-handoff`. Track phase status in the space's `SPACE.md` and mirror it into the space's Paperclip task (it also shows in Operon), per the spaces skill.
- **Do:** Copy `prd-subagent-handoff.md` and this plan into the ast worktree. They are untracked in `~/git/ast-context-cache` today and are committed in the first commit.

#### 0.2 Claude Code hook spike (decides HI-2/HI-3; the user reviews)
- **Create (scratch, not committed):** `$SCRATCH/hook-probe.sh`, which appends stdin JSON plus the event name to `$SCRATCH/hook-log.jsonl` and echoes a canary `additionalContext`. Register it in a throwaway settings file for `SessionStart`, `SubagentStart`, `SubagentStop`, `PreCompact`, `PreToolUse` (matcher `Agent|Task`), and `PostToolUse` (matcher `Agent|Task`). Use `claude --settings <file>` so the user's real settings are never touched.
- **Run headless scenarios** (`claude -p`):
  - (a) The parent spawns an Explore subagent asked to echo any "canary" text it was given.
  - (b) A fork subagent.
  - (c) The parent runs `/compact`, then continues.
  - (d) `PreToolUse` returns `updatedInput` appending `CANARY-PROMPT` to the Agent prompt.
- **Record** in `docs/spikes/claude-code-hooks.md` (committed), as a capability matrix:
  - which events fire;
  - the input fields (`session_id`, `agent_id`, `agent_type`, prompt, `transcript_path`, `source`);
  - whether `additionalContext` reaches the child;
  - whether `updatedInput` rewrites the prompt;
  - whether the stop hook exposes the final message or transcript;
  - whether forks report a shared prompt cache (from usage data in `--output-format json`).
- **Gate:** the user reviews the matrix. Phase 9.7 builds only the confirmed items. Unsupported items are marked "unsupported" in `docs/host-integration.md`.

#### 0.3 Spec and host-format verification
- **Do:** Per the global ast-context-cache rule, use `fetch_doc` / `add_doc_source` (not WebFetch) to cache:
  - the MCP spec `2025-06-18` transports page;
  - the **official** `2026-07-28` revision. If no official spec page exists, record "unverified" and PL-6 negotiates only `2024-11-05`/`2025-06-18`;
  - Claude Code MCP config (user scope) and settings/hooks docs;
  - OpenCode config schema;
  - Codex `config.toml` `mcp_servers` (whether HTTP transport is supported or a bridge is needed);
  - VS Code `mcp.json` (`servers` key and user path per OS);
  - JetBrains AI Assistant MCP config location;
  - Claude Desktop config.
- **Create:** `docs/host-integration.md` (seeds IN-14), with a target × component table listing the exact path, key, entry shape, and whether HTTP needs the bridge. JetBrains or Codex may turn out to lack a global file-based MCP config. In that case the component is "Unsupported — manual steps", per IN-2's skip-with-reason rule.

### Phase 1 — Foundations (PL-1, PL-2, PL-3, PL-8)

#### 1.1 `internal/errs` (clean-room)
- **File:** `internal/errs/errs.go`, `codes.go`, `errs_test.go`.
- **Add:**
  - `type Error struct{ message string; codes Codes; err error; fields map[string]any }` implementing `Error()`, `Unwrap()`, `Fields() map[string]any`, `Codes() Codes`, and `MarshalJSON`.
  - Constructors: `New(msg, kv...)`, `NewCode(code, msg, kv...)`, `Wrap(err, kv...)`, `WrapCode(code, err, kv...)`, `WrapMessage(msg, err, kv...)`, `WrapCodeMessage(code, msg, err, kv...)`.
  - `HasCode(err, code)`, and `CodeOf(err) Code` (first code found in the chain).
  - `type Code string` and `type Codes []Code` with `Has`.
  - Codes: `CodeInvalidInput`, `CodeNotFound`, `CodeExpired`, `CodeLimitExceeded`, `CodeConflict`, `CodeDisabled`, `CodeUnsupported`, `CodeInternal`.
  - `Error()` format: `msg: wrapped` (fields are not in the string; the logging handler renders them).
- **Tests:** a table test covering wrap chains with `errors.Is`/`errors.As` compatibility, code accumulation, field merging, and JSON.

#### 1.2 `internal/logging`
- **File:** `internal/logging/logging.go`, `handler.go`, `logging_test.go`.
- **Add:**
  - `Setup(w io.Writer)` reads `AST_LOG_FORMAT` (`text` or `json`, default `text`) and `AST_LOG_LEVEL` (`debug`, `info`, `warn`, `error`; default `info`), then calls `slog.SetDefault`.
  - `log.SetOutput` is pointed at `slog.NewLogLogger` so stray stdlib and third-party logs land in the same place.
  - A wrapping `Handler` rewrites `!BADKEY` attrs: an `error` value becomes `error=<msg>` plus its `errs` fields; a `slog.LogValuer` value is resolved.
- **Update:** `cmd/ast-mcp/main.go:69-76`. Keep the existing writer selection (file when stdout is a TTY) and pass it to `logging.Setup`.
- **Tests:** a bare error arg renders as `error=`; fields are expanded; JSON mode emits valid JSON lines.

#### 1.3 Whole-repo slog sweep (~230 sites)
- **Update:**
  - Every package declares `var logger = slog.With("tag", "<pkg>")` at the top, following §10 file order.
  - Convert `log.Printf("pkg: did %s n=%d", a, n)` to `logger.Info("Did thing", "a", a, "n", n)`, choosing the level: Debug for chatty detail, Warn for recovered errors, Error for unexpected ones.
  - Messages start uppercase (§13).
  - `log.Fatal*` in `main.go` becomes `slog.Error` followed by `os.Exit(1)`.
- **Order:** one commit per top-level package directory, so review stays tractable.
- **Tests:** existing tests that assert on log text (grep `log.SetOutput` in `_test.go`) are updated to use a `slog` test handler.

#### 1.4 Whole-repo errs sweep (~157 sites)
- **Update:**
  - `fmt.Errorf("x: %w", err)` becomes `errs.WrapMessage("x", err, kv...)`.
  - `errors.New("x required")` becomes `errs.NewCode(errs.CodeInvalidInput, "x required")`.
  - The `indexer.ErrSymlinkAlias` sentinel (`internal/indexer/prune.go:21`) becomes `errs.New`.
  - `contextnotes.LimitError` keeps its struct but carries `CodeLimitExceeded` (it is wrapped via `errs.WrapCode` where it is returned). `LimitErrorMap` keeps its output shape.
- **Keep:** `errors.Is`/`errors.As` call sites. They still work because `errs.Error` implements `Unwrap`.

#### 1.5 Whole-repo SQL constant hoist (59 files)
- **Update:**
  - Every inline query string (≈178 backtick and ≈87 double-quoted `Exec`/`Query*`/`Prepare` call sites, plus 16 `q := \`…\`` sites) moves to a top-of-file `const (...)` block named `<verb><Thing>Query`.
  - Dynamically built SQL, e.g. `pathPrefixSQLClause` (`internal/search/filters.go:116`) and `scopeClauseFor` (`internal/memory/recall.go:166`), uses const fragments joined at runtime.
  - Schema DDL in `internal/db/schema_*.go` becomes `const createXTable = ...`.
- **Rule:** no behavior change. The commit is mechanical and passes `make test` untouched.

#### 1.6 Tooling targets and CI
- **File:** `Makefile`.
- **Add:**
  - `fmt`: `go run mvdan.cc/gofumpt@v0.7.0 -w ./cmd ./internal`.
  - `lint`: `go vet -tags sqlite_fts5 ./...` plus the gofumpt check (`gofumpt -l` must be empty).
  - `race`: same flags as `test` (`Makefile:181-182`) plus `-race`.
  - `bench`: `go test -tags sqlite_fts5 -run '^$' -bench . ./internal/handoff/... ./internal/cache/...`.
  - `ui-test`: `cd ui && npm test`.
- **Do:** run `make fmt` across the repo in its own commit, formatting only.
- **File:** `.github/workflows/test.yml`. Add jobs `lint`, `race`, and `ui-test` (Node 22 + `npm ci` in `ui/`).
- **File:** `ui/package.json`. Add devDeps `vitest` and `jsdom`, script `"test": "vitest run"`, and `ui/vitest.config.ts` (environment `jsdom`, includes `src/**/*.test.ts`).
- **File:** `go.mod`. Add `github.com/stretchr/testify` (tests only).

### Phase 2 — Feature-flag registry (FF-1–FF-4 backend, FF-6–FF-8)

#### 2.1 `internal/flags`
- **File:** `internal/flags/flags.go`, `registry.go`, `flags_test.go`.
- **Add:**
  - `type Flag struct{ Key, Env, Description string; Default bool; Tools []string; Actions map[string][]string }`.
    - `Tools` lists tools hidden when the flag is off.
    - `Actions` lists per-tool actions disabled when the flag is off, e.g. `feature_handoff_claims` → `scratchpad: [claim, release]`.
  - `registry.go`: the FF-6 table. Env names are `AST_FEATURE_HANDOFF`, `AST_FEATURE_HANDOFF_SCRATCHPAD`, `AST_FEATURE_HANDOFF_CLAIMS`, `AST_FEATURE_HANDOFF_LIVE_TRAIL`, `AST_FEATURE_HANDOFF_HOOKS`, `AST_FEATURE_SHARED_QUERY_CACHE`.
  - `Enabled(key) bool`:
    - Resolution order: env (`strconv.ParseBool`; a non-empty value locks the flag), then `db.GetSetting(key, "")`, then the default.
    - The result is cached in an `atomic.Pointer[map[string]bool]`, which `Set` rebuilds.
    - The master switch implies the children: `feature_handoff=false` disables all handoff flags.
  - `Set(key string, on bool) error` returns `errs.CodeConflict` "flag locked by env" when env-locked. It calls `db.SetSetting`, rebuilds the cache, and fires `OnChange` subscribers.
  - `State() []FlagState{Key, Enabled, Source("env"|"setting"|"default"), Locked, Description}`.
  - `OnChange(func(key string, on bool))`.
  - `ActionEnabled(tool, action) bool`.
- **Tests:** a table test across env, setting, default, the lock, master-switch implication, and OnChange firing. Use `dbtest.Init` and `t.Setenv`.

#### 2.2 Tool gating
- **File:** `internal/mcp/tools.go`.
- **Update:**
  - Add `denyFlag` to the enum (`:53-61`).
  - `toolAccess` (`:70-81`) checks `flags` for any tool listed in a disabled flag's `Tools`, before the tier check.
  - `ToolDenyMessage` (`:84-97`) gets a case returning `feature_disabled: <flag>`.
  - `FilterTools` (`:625`) hides flag-denied tools (FF-8: flag AND `tools.json` AND tier must all allow it).
- **Tests:** `tools_test.go` gets a table case "flag off hides tool" and a `tools.json`-plus-flag interaction case.

### Phase 3 — Correctness & caching (BF-1, BF-2, BF-4, DC-1–DC-3, PL-7)

#### 3.1 BF-1 LIKE fallback scope
- **File:** `internal/contextnotes/store.go:616-627`. The hoisted const becomes `WHERE (label LIKE ? OR content LIKE ?)` followed by the conditional `AND` fragments.
- **Tests:** `store_test.go` `TestSearchLikeRespectsSession`. It must fail on the old SQL (AC26).

#### 3.2 BF-2 + recall fixes
- **File:** `internal/memory/recall.go`.
- **Update:**
  - `vectorSearch` (`:273-298`) re-selects with `validityClause(in)` and `scopeClauseFor(in, paths)` (`:120`, `:166`), then reorders rows to match the vector rank.
  - `validityClause`: when `IncludeHistory` is set, drop the `valid_until`/`superseded_by` predicates.
  - `search.Cache.SearchMemory` (`internal/search/vector.go:413`): stop passing entries with an empty `ProjectPath` unless the caller's scope includes global. The SQL re-select is the authoritative filter.
- **Tests:** `recall_test.go` covers a superseded fact excluded, another session's session-scoped fact excluded, rank order preserved, and IncludeHistory returning superseded rows (AC27).

#### 3.3 Candidate cache (DC-1, DC-2, cache fixes)
- **File:** new `internal/cache/candidate_cache.go`. Delete the final-JSON use of `GlobalCache` in the capsule path. `GlobalCache` stays only if other callers exist; otherwise delete `query_cache.go`.
- **Add:**
  - `type CandidateCache struct{ mu sync.Mutex; entries map[string]*candidateEntry; byProject map[string]map[string]struct{}; ttl time.Duration; max int; hits, misses atomic.Int64 }`.
  - `Key(stage, query, projectPath, filtersKey, docType string, limit int) string`.
  - `Get(key) ([]search.ScoredResult, PipelineMetrics, bool)` returns a **deep copy**: a fresh `Data` map per result, with slices copied.
  - `Set(key, projectPath, results, metrics)`.
  - `ClearProject(projectPath)`, using the `byProject` index.
  - `Stats() (hits, misses int64)`.
  - Defaults: TTL 5 minutes, max 1,000 entries; eviction by oldest timestamp.
- **Update call sites:**
  - capsule: cache `search.HybridSearch(..., 30, filters)` at `internal/context/handler.go:75`.
  - retrieve: cache `HybridSearch(limit*2)` at `internal/mcp/retrieve.go:210`, with `limit` in the key.
  - search_semantic: cache `search.Cache.Search` at `server.go:318`, keyed on `docType`.
  - Use the cache whenever `flags.Enabled("feature_shared_query_cache")`, regardless of `session_id`. Dedup, mode, budget, and `LogReturned` always run per call after `Get`.
- **Invalidation:**
  - Call `cache.Candidates.ClearProject(owningProject)` after a successful commit in `indexer.IndexFile` (`internal/indexer/indexer.go:~645`).
  - Also call it in `indexer.PurgeFile` (`prune.go:57`) and `purge.go:72`, replacing the dead `GlobalCache.ClearProject`.
  - Call `ClearAll` in the dashboard reset-all (`api.go:300`).
  - Linked child projects: clear `projectlinks.ResolveScope(parent)` members too.
- **Tests:**
  - `candidate_cache_test.go`: deep copy isolation, index-commit invalidation, `byProject` correctness, and a `-race` test of concurrent Get/Set/Stats.
  - `internal/context/handler_test.go`: two sessions get the same candidates while each session's dedup still applies (AC23). A reindex causes a miss (AC24).

#### 3.4 Immediate dedup (DC-3) + in-list dedup
- **File:** `internal/context/session_store.go` (new). Update `session.go`.
- **Add:**
  - `type sessionSet struct{ mu sync.Mutex; keys map[string]struct{}; hydrated bool; lastUsed time.Time }`, held in a package `sync.Map`.
  - `ReturnedKeys(sid) map[string]struct{}` hydrates once from the DB query `GetReturnedSymbolKeys` (`:14`), then serves a copy.
  - `MarkReturned(sid string, keys ...returnedKey)` updates the set synchronously, then calls `db.EnqueueSessionReturned`.
  - `SeedReturned(sid, keys)` is used for fork seeding.
  - `StartSessionStoreEviction(ctx)` evicts sets idle for more than 30 minutes, following the §11 loop shape.
- **Update the four pack loops:** `handler.go:92-120`, `handler.go:166-194`, `internal/mcp/handlers.go:152-184`, `retrieve.go:232-275`. Each loop:
  - uses `ReturnedKeys`;
  - adds each emitted key to the local set, so the list itself is deduped;
  - calls `MarkReturned`.
- **Tests:** two back-to-back calls under 1s apart dedup (AC25); a duplicate within one list is skipped; a hydrate-after-restart test.

#### 3.5 retrieve: log only delivered chunks
- **File:** `internal/mcp/retrieve.go`.
- **Update:**
  - Add `StartLine int` and unexported `absFile string \`json:"-"\`` to `RetrieveChunk`.
  - Remove `LogReturned` from `retrieveCode` (`:274`). After `budgetChunks` (`:155`), `MarkReturned` only the code chunks that were kept.
  - Rename stats `deduped_count` → `deduped`, consistent with `ApplyTo`. This is a breaking change, listed in the migration notes.
- **Tests:** `retrieve_test.go`: a budget-trimmed symbol is not deduped on the next call.

#### 3.6 Shutdown flush
- **File:** `cmd/ast-mcp/main.go:143-151`.
- **Update:**
  - Create a root `ctx, cancel := context.WithCancel(context.Background())` in `main`, passed to the background services and the handoff service.
  - On SIGINT or SIGTERM: `cancel()`, then `db.FlushWriteBuffers()` (`writebatch.go:292`) with a 2s timeout, log it, then `os.Exit(0)`.

### Phase 4 — Server hardening & transport (PL-4, PL-5, PL-6, FF-5)

#### 4.1 Listen address
- **File:** `cmd/ast-mcp/main.go:39-55,117-137`.
- **Add:** a `--listen` flag (env `AST_LISTEN`, default `127.0.0.1`) used for both the MCP (`:7821`) and dashboard (`:7830`) servers. Log a Warn when the address is not loopback.
- **Update:** `docker/ast-mcp/Dockerfile:28` and `compose.yml:25` add `AST_LISTEN=0.0.0.0`.

#### 4.2 Origin/Host guard
- **File:** new `internal/httpguard/guard.go`, `guard_test.go`.
- **Add:**
  - `IsLoopbackHost(host string) bool`.
  - `Middleware(next http.Handler) http.Handler`. For `POST`/`PUT`/`PATCH`/`DELETE` and WebSocket upgrades:
    - reject with 403 when an `Origin` header is present and its host is not loopback (`localhost`, `127.0.0.1`, `[::1]`);
    - reject when `Host` is not loopback or the configured listen address (anti DNS-rebinding).
  - A missing `Origin` header is allowed, because CLI and curl clients don't send it.
- **Update:**
  - Wrap the dashboard mux in `api.go` `NewHandler` (`:37`).
  - Wrap `/mcp` in `main.go:117` (the Streamable HTTP spec requires Origin validation).
  - `ws_react.go:110` `CheckOrigin` uses `httpguard`'s predicate.
- **Tests:** a table test of the allowed and rejected origins and hosts. A dashboard `httptest` POST with a foreign Origin returns 403.

#### 4.3 Atomic server config
- **File:** `internal/mcp/server.go:27,58-64`.
- **Update:** `var srvCfg atomic.Pointer[ServerConfig]`. `GetConfig`/`SetConfig` load and store it, and every `srvCfg` read goes through `GetConfig()`. The test helpers' save/restore (`memory_tools_test.go:14-24`, `server_test.go`) are updated to match.

#### 4.4 Protocol negotiation + notification handling
- **File:** new `internal/mcp/protocol.go`. Update `server.go:89-130`.
- **Add:**
  - `supportedVersions = []string{"2026-07-28", "2025-06-18", "2024-11-05"}`, where the first entry is included only if Phase 0.3 verified it.
  - `negotiate(clientVersion) string`: echo the client's version if it is supported, otherwise return the newest supported version.
- **Update `initialize`:**
  - It returns the negotiated version and `capabilities: {tools:{listChanged:true}, prompts:{}}`.
  - For `2025-06-18` and later, it also returns an `Mcp-Session-Id` header.
- **Update other methods:**
  - Any method with the `notifications/` prefix (and the legacy `initialized`) returns HTTP **202** with no body.
  - `ping` returns `{}`.
  - Requests without an `id` are never answered with a body.

#### 4.5 Streamable HTTP SSE stream
- **File:** new `internal/mcp/stream.go`, `stream_test.go`.
- **Add:**
  - `type streamHub struct{ mu sync.Mutex; sessions map[string]*mcpSession; subs map[*subscriber]struct{} }`, where `mcpSession{ id, version, lastSeen }`.
  - **`GET /mcp` with `Accept: text/event-stream`:** register a subscriber, write SSE `event: message` frames, send a heartbeat comment every 25s, and unregister when the client disconnects.
  - **`GET /mcp` without SSE Accept:** keeps the legacy tools-JSON response for back compatibility. This is documented as deprecated.
  - **`DELETE /mcp`** with a session id drops that session.
  - `Broadcast(method string, params any)`.
  - Idle sessions are evicted after 1 hour.
- **For `2026-07-28` (if verified):** accept the per-request `_meta` without requiring a session, and deliver `list_changed` via the mechanism that revision defines (recorded in the 0.3 notes).

#### 4.6 Flag → `list_changed`
- **File:** `internal/mcp/server.go` init (or `mcp.Init`, called from main).
- **Add:** `flags.OnChange(func(key string, _ bool){ if flags.AffectsTools(key) { hub.Broadcast("notifications/tools/list_changed", nil) } })`. Also call `realtime.Notify(realtime.SettingsChanged)` for the dashboard.
- **Tests:** these are the **first HTTP-level MCP tests**, using `httptest.NewServer(mcp.NewHandler())`:
  - initialize negotiation for each version;
  - a notification gets a 202 with an empty body;
  - an SSE client receives `notifications/tools/list_changed` within 1s of `flags.Set` (AC28);
  - a foreign Origin gets 403.

### Phase 5 — Search trail (HO-4 prerequisite, SP-5 source)

#### 5.1 Filter normalization
- **File:** `internal/search/filters.go`.
- **Add:** `(f *SearchFilters) NormalizedKey(projectPath string) string`:
  - kinds lowercased and sorted;
  - language mapped to its canonical name via `languageExtensions` (`:217-242`);
  - path prefix made project-relative and cleaned (`./`, trailing slash, and absolute forms collapse to one).
- **Tests:** a table of equivalent filter pairs producing the same key.

#### 5.2 `internal/trail`
- **File:** `internal/trail/trail.go`, `store.go`, `trail_test.go`. Update `internal/db/schema_usage.go` (append) and `writebatch.go`.
- **Add:**
  - `type Entry struct{ SessionID, Tool, Query, QueryNorm, FiltersKey, Mode, DocType, ProjectPath string; HitCount int; ZeroHit bool; TopHits []string; At time.Time }`.
    - `TopHits` holds up to 5 entries of `relfile#name@line`.
    - `MatchKey()` is `tool|queryNorm|filtersKey|docType`.
  - `normalizeQuery`: lowercase, collapse whitespace, trim.
  - `Record(e Entry)` writes to an in-memory ring per session (capacity 200, in a `sync.Map`) and calls `db.EnqueueTrail(e)`.
  - `ForSession(sid string, limit int) []Entry` merges memory with the DB, deduplicating by `(MatchKey, At)`.
  - `PruneOlderThan(d)`.
  - usage.db table `search_trail(id INTEGER PK, session_id, tool, query, query_norm, filters_key, mode, doc_type, project_path, hit_count, zero_hit, top_hits_json, created_at)` with an index on `(session_id, created_at)`.
  - `writebatch.go` gets a third buffer, `trailBuf`, flushed alongside the sessions buffer (3s / 200 rows) and by `FlushWriteBuffers`.

#### 5.3 Capture at the four tools
- **Update:**
  - Add a `Trail trail.Entry` field to `getContextResult` (`handler.go:17-21`), `fileContextResult` (`handlers.go:119-122`), the `PackScoredResults` return value, and `HandleRetrieve`'s result.
  - `HitCount` is the **pre-dedup** count:
    - capsule: `len(scored)` (`handler.go:75`)
    - search_semantic: the `Cache.Search` length (`server.go:318`)
    - file_context: rows read
    - retrieve: code plus doc candidate counts
  - Add `recordSearch(sessionID string, e trail.Entry)` in `internal/mcp/search_hooks.go`, called at the four `logToolQuery` sites (`server.go:239,343,375,540`).
  - Capture only when `session_id` is non-empty and `flags.Enabled("feature_handoff")`.
- **Tests:** an mcp test checks that a capsule call records an entry with the correct hit count and zero-hit flag, and that it is readable immediately (read-your-writes).

### Phase 6 — Handoff core (HO, OP, RT, FI, RQ, TS, DC-4)

#### 6.1 Schema (context.db, appended at end of `initContextSchema`)
- **File:** `internal/db/schema_context.go`, plus `migrate_split.go:17-24` (`contextTables`), `internal/purge/purge.go:179-194` (delete by project), and dashboard reset-all.
- **Tables:**
  - **`handoff_trees`**: `tree_id PK, root_session_id, project_path, created_at, last_access_at, tokens_used, entries_used`.
  - **`handoffs`**: `ref PK, tree_id, parent_session_id, parent_child_session_id NULL, depth, mode, label, brief, project_path, child_count, created_at, last_access_at`. Indexes on `parent_session_id` and `tree_id`.
  - **`handoff_snapshot_items`**:
    - Columns: `id PK, handoff_ref, section, ord, item_key, label, content, file_rel, fqn, kind, start_line, end_line, fingerprint, token_est`.
    - `section` is one of `manifest`, `trail`, `note`, `memory`, `pointer`.
    - Index on `(handoff_ref, section, ord)`.
  - **`handoff_children`**:
    - Columns: `child_session_id PK, handoff_ref, tree_id, label, status, project_path, opened_at, last_activity_at, result_ref, summary, summary_source, summary_truncated, result_status, search_calls, repeat_calls, tokens_available, tokens_delivered`.
    - Index on `(tree_id, status)`.
  - **`handoff_results`**: `id PK, child_session_id, result_ref, created_at, superseded_at`.
  - **`scratchpad_entries`**: `id INTEGER PK AUTOINCREMENT (cursor), tree_id, author_session_id, type, text, refs_json, token_est, created_at, retracted_at`. Index on `(tree_id, id)`.
  - **`handoff_claims`**: `tree_id, key, holder_session_id, reason, granted_at`, with PK `(tree_id, key)`.
  - **`handoff_claim_queue`**: `id PK, tree_id, key, session_id, reason, enqueued_at`. Index on `(tree_id, key, id)`.
  - **`handoff_claim_grants`**: `id PK, session_id, tree_id, key, granted_at, notified_at NULL`.
- **File:** `internal/db/pools.go`. Add `HandoffWriteDB`, a one-connection pool on context.db opened with `_txlock=immediate`, plus `db.HandoffTx(fn func(*sql.Tx) error) error`. All handoff, scratchpad, and claim writes go through it, which linearizes them (NFR-4/5).

#### 6.2 Types, limits, errors
- **File:** `internal/handoff/types.go`.
  - ID types (`HandoffRef`, `TreeID`, `SessionID`).
  - Enums: `Mode` (`ModeFresh`, `ModeFork`); `Status` (`StatusOpen`, `StatusDone`, `StatusPartial`, `StatusFailed`, `StatusAbandoned`); `Section*`; `EntryType*`.
  - Request and response structs.
- **File:** `internal/handoff/limits.go`. `LoadLimits()` via `envOrSetting`-style resolution, generalized as `settings.Int(key, env, default)` in `internal/db` or a small `internal/settings` helper. Keys and defaults:

  | Setting | Default |
  |---|---|
  | `handoff_ttl_days` | 7 |
  | `handoff_summary_max_tokens` | 300 |
  | `handoff_child_inactive_minutes` | 30 |
  | `handoff_tree_max_tokens` | 64000 |
  | `handoff_tree_max_entries` | 300 |
  | `handoff_max_depth` | 3 |
  | `handoff_max_children` | 16 |
  | `handoff_open_budget_tokens` | 1500 |
- **File:** `internal/handoff/errors.go`.
  - Codes, as `errs.Code` values: `handoff_not_found`, `handoff_expired`, `handoff_depth_exceeded`, `handoff_children_exceeded`, `handoff_tree_limit_exceeded`, `claim_deadlock_risk`, `feature_disabled`.
  - `ErrorMap(err) map[string]any` returns `{"error": code, "message", "details", "suggestions": [...]}`, generalizing `LimitErrorMap` (TS-5).

#### 6.3 Service skeleton
- **File:** `internal/handoff/service.go`.
- **Add:**
  - The `Service` interface: `Create`, `Open`, `Expand`, `Complete`, `Collect`, `List`, `Status`, `Flush`, `Post`, `Read`, `Retract`, `Claim`, `Release`, `Annotate`, `Touch`, `PendingGrants`, `IsTreeSession`.
  - `realService{ emb embedder.Interface; logger *slog.Logger; trees treeIndex; waiters *waitHub }`.
  - `New(ctx, emb) Service` starts `sweepLoop` (hourly expiry) and `abandonLoop` (every minute), following §11.
- **Wiring:**
  - `treeIndex` is an in-memory `sessionID → treeID/handoffRef/mode`, hydrated lazily from `handoff_children` and `handoffs`, so `Annotate`/`Touch` cost nothing for sessions outside a tree (NFR-2).
  - `cmd/ast-mcp/main.go` `startBackgroundServices` (`:407`) constructs it with the root ctx, and `mcp.SetHandoffService(svc)` follows the `SetEmbedder` pattern (`server.go:36`).

#### 6.4 Create + snapshot (HO-1–HO-10)
- **File:** `internal/handoff/create.go`, `snapshot.go`, `fingerprint.go`.
- **Before the transaction** (reads):
  - manifest: `context.ReturnedKeys(parent)`;
  - trail: `trail.ForSession(parent, 200)`, then apply the prune arguments (`exclude_trail` as indexes or `"all"`, `exclude_trail_query` as a substring, `include_manifest` as a bool);
  - notes: copies via the new `contextnotes.Peek(ref)`, an exported, accounting-free read wrapping `noteByRef` (`internal/contextnotes/store.go:205`);
  - memory: the explicit refs plus the parent's active session-scoped entries, via a new `memory.ActiveForSession(sid)`;
  - pointers: resolved through the index (`symbols` by `file`/`name`/`fqn`), with `fingerprint = sha256(indexer.ReadSourceRange(file, start, end))` from the new `context.SymbolFingerprint`.
- **Then** compute section token estimates. If the snapshot exceeds the tree cap, fail with `handoff_tree_limit_exceeded` and the breakdown (HO-7).
- **In one `db.HandoffTx`:**
  - resolve the tree: if the parent is a child, use its tree and `depth = parent depth + 1`, and check the max depth (HO-8); otherwise create an `hft_` tree;
  - insert the handoff and its items;
  - bump the tree counters.
- **Ref:** `hof_` plus 16 hex characters from `crypto/rand`.
- **Return:** `{ref, stub, breakdown}`, where `stub` is formatted `[handoff hof_… ] <label> — call open_handoff first` and checked by a test to be at most 60 tokens (NFR-3).
- **Log:** `logger.Info("Created handoff", ref, tree, "parent_session", …, "tokens", …)` (OB-6).

#### 6.5 Open / resume / expand (OP-1–OP-12, OB-2)
- **File:** `internal/handoff/open.go`, `expand.go`, `staleness.go`.
- **Open** (one tx):
  - Check expiry, then check `child_count` against the cap (HO-9).
  - Mint the child id `<hof_ref>.c<N>`, insert into `handoff_children`, and touch the tree.
  - After the tx:
    - `fork` mode: call `context.SeedReturned(child, manifestKeys)` (OP-5).
    - Clone the snapshot's memory items into the child session via `memory.Store` with `source_ref` set to the original ref (OP-11).
    - Update `treeIndex`.
- **Resume:** the handoff ref plus an existing child id returns the same child (no count bump) and sets status back to `open` (OP-2).
- **Digest:** sections in priority order, each truncated to fit `token_budget`:
  1. brief;
  2. child id and mode;
  3. pointers (key and note only);
  4. notes and memory (ref, label, token count);
  5. trail digest (newest first: query, hits, zero-hit);
  6. scratchpad digest (counts, 3 latest headlines, active claims).

  Overflow returns `truncated: true` and `next: {section, offset}` (OP-3).
- **Expand** (OP-4): `section` plus `items` (ids or `"all"`) plus `mode` (`skeleton`/`auto`/`full` for pointers).
- **Pointer expansion:**
  - **Worktree mapping:** if the child's `project_path` passes `repokey.SameRepo(childProject, snapshotProject)`, resolve `file_rel` under the child's project (OP-8).
  - **Lookup:** find the current symbol by `(owning project, file_rel, fqn)` via `projectlinks.OwningProject`.
  - **Classify:**
    - same fingerprint and same lines → fresh;
    - same fingerprint, different lines → `moved`;
    - different fingerprint → `modified`;
    - no symbol but the file exists → `deleted`;
    - no file → `file_missing`.
  - **Render** current code via `context.ApplyMode` (OP-7).
  - Call `context.MarkReturned(child, deliveredKeys)` for the delivered pointers (OP-6).
- **Savings:** `tokens_available` (the snapshot total at `auto`) and `tokens_delivered` are updated per child. `logToolQuery` gets `SavingsMeta{TokensSaved: available-delivered}` attributed to the `open_handoff` tool (OB-2).

#### 6.6 Search annotations (OP-9, OP-10, OB-1)
- **File:** `internal/handoff/annotate.go`, `internal/mcp/search_hooks.go`.
- **Add:** `Annotate(sid string, e trail.Entry, results []map[string]any) map[string]any`. It returns the response-level fields to merge, after mutating the per-result `parent_explored`:
  - Not a tree session: return nil immediately.
  - `parent_trail_match`: from the snapshot trail items whose `MatchKey` equals `e.MatchKey()`.
  - Fresh mode: set `parent_explored: true` on results whose dedup key is in the manifest.
  - Repeat accounting: a call is a repeat if it matched the trail or if at least 50% of the pre-dedup keys are in the manifest. Increment `search_calls`/`repeat_calls` (batched through the handoff writer).
- **Call sites:** before marshal at `handler.go:125-140`, `server.go:329-340`, `handlers.go:217-228`, and `retrieve.go:166-198`. `RetrieveResult` gains `Handoff map[string]any \`json:"handoff,omitempty"\``. Annotations are never cached.
- **Tests:** an mcp test for AC3, AC4 and AC6.

#### 6.7 Complete (RT-1–RT-7, OB-3)
- **File:** `internal/handoff/complete.go`. Update `internal/contextnotes/store.go`.
- **Store the result** with `contextnotes.Store(child, content, label, project, tags, "handoff_result", meta, emb)`.
- **Update contextnotes:**
  - `FlushOrphans` (`:308`) and `evictSessionLRU` (`:164`) skip `kind='handoff_result'` and every session listed in `handoff_children`. Tree expiry owns them.
  - `Fetch` treats `handoff_result` notes like ordinary notes (the open model).
- **Summary:**
  - Use the child's summary if provided; otherwise derive one: the `FACT:`/`RULE:` lines from `memory.ExtractFromText`, then the leading content lines.
  - Truncate to `handoff_summary_max_tokens` using `db.EstimateTokens`, and set `summary_truncated` and `summary_source` (RT-2, RT-3).
- **Promote memory:** `memory.StoreExtracted(extracted, ScopeSession, sid=parent, source_ref=resultRef)`. These entries outlive the tree (RT-5).
- **In one tx:**
  - supersede the previous `handoff_results` row (RT-7);
  - set the child's `status=result_status`, `result_ref`, and `summary`;
  - release all of the child's claims through `releaseAllTx` from Phase 7, which handles auto-grant (RT-6).
- **After commit:** wake the tree's waiters (FI-6), and log `TokensSaved = estimate(content) - estimate(summary)` attributed to `handoff` (OB-3).
- **Return stub:** `[result ctx_… for hof_…] <status> — <summary>`, at most cap + 40 tokens (RT-4).
- **RT-8 (MAY):** accept optional `changed_files` and `open_questions` arrays, stored in the result note metadata and echoed by collect.

#### 6.8 Collect / list / status / flush (FI-1–FI-6, RQ-3)
- **File:** `internal/handoff/collect.go`, `flush.go`.
- **Collect:** by `handoff` ref, or by `parent session_id` for all of its handoffs. `recursive=true` walks `parent_child_session_id`. Each child entry is about 50 tokens plus its summary, and the response stays within budget.
- **`wait_seconds` (≤60):** block on `waitHub` (`map[TreeID]chan struct{}`, closed and replaced on any child status change) or until the timeout.
- **List:** the parent's handoffs with status counts (FI-2).
- **Status:** a compact view of one handoff or tree.
- **Flush (RQ-3):**
  1. Collect the child ids.
  2. In a tx, delete the tree's rows from every handoff table.
  3. Then, best-effort: `contextnotes.FlushSession(child)` for each child (vectors included); hard-delete the child-session memory with a new `memory.DeleteSession(sid)`; delete the child dedup rows from usage.db `sessions` and the `search_trail` rows; evict the in-memory session sets.

  Child notes expire with the tree (resolved Q9). Memory promoted to the parent is not touched.

#### 6.9 Sweeper, abandonment, activity (FI-3, RQ-1, RQ-2, DC-4)
- **File:** `internal/handoff/sweeper.go`.
- **`abandonLoop`** (every minute): mark open children with `last_activity_at` older than the inactivity window as `abandoned`, and release their claims.
- **`sweepLoop`** (hourly, first run 2 minutes after start): run the flush routine for every tree whose `last_access_at` is older than the TTL, and call `trail.PruneOlderThan(ttl)`.
- **`Touch(sid)`:** called in `handleToolCall` right after project-path normalization (`server.go:~199`). It is an in-memory check against `treeIndex`, coalesces `last_activity_at` and `last_access_at` writes to at most one per session per 10s, and moves an abandoned child back to `open`.

#### 6.10 MCP tools (TS-1–TS-5)
- **File:** `internal/mcp/handoff_tools.go`. Update `tools.go` `GetTools()` and `server.go:545` default chain.
- **Add three `Tool` entries, `TierCore`:**
  - **`handoff`:** `action` (`create`, `complete`, `collect`, `list`, `status`, `flush`) plus `session_id`, `brief`, `label`, `pointers`, `ctx_refs`, `mem_refs`, `mode`, `exclude_trail`, `exclude_trail_query`, `include_manifest`, `handoff`, `content`, `summary`, `status`, `recursive`, `wait_seconds`, `token_budget`, `changed_files`, `open_questions`.
  - **`open_handoff`:** `action` (`open`, `expand`, `resume`) plus `handoff`, `session_id`, `project_path`, `section`, `items`, `mode`, `token_budget`.
  - **`scratchpad`:** `action` (`post`, `read`, `retract`, `claim`, `release`) plus `session_id`, `type`, `text`, `refs`, `since`, `types`, `author`, `include_own`, `key`, `reason`, `entry`, `token_budget`.
  - Descriptions are terse. `open_handoff`'s first sentence is the TS-4 instruction.
- **Router:** `handleHandoffTool(name, args) (any, bool, error)` following `handleMemoryTool`. Action-level gating uses `flags.ActionEnabled`, and errors go through `handoff.ErrorMap`.
- **Tests:** `handoff_tools_test.go`:
  - a TS-3 budget test: `db.EstimateTokens(json.Marshal(the three tool defs)) <= 1200`;
  - registration and tier tests, following `context_tools_test.go`;
  - error-code mapping.

### Phase 7 — Scratchpad & claims (SP, CL, RQ-4/5)

#### 7.1 Scratchpad (SP-1–SP-4, SP-7)
- **File:** `internal/handoff/scratchpad.go`.
- **Post:** checks tree membership via `treeIndex`, caps the entry at 500 tokens, and does the cap accounting in the same tx (RQ-4).
- **Read:** supports `since` (entry id), `types`, `author`, and `include_own` (default false), within budget, and returns `next_cursor`.
- **Retract:** sets `retracted_at`, own entries only.
- **Dead-ends view:** zero-hit trail entries merged with `dead_end` posts (SP-7).
- Snapshot trail items also show up in the dead-ends view for children.

#### 7.2 Live trail sharing (SP-5, SP-6, RQ-5)
- **Update:** `recordSearch` (5.3). If the session is in a tree and `feature_handoff_live_trail` is on, also insert a `trail` scratchpad entry (≤60 tokens).
- When the tree is at its cap, delete the oldest `trail` entries in the same tx before inserting (RQ-5). Explicit writes over the cap fail with `handoff_tree_limit_exceeded`.
- **Annotate:** `sibling_trail_match` comes from tree `trail` entries by other authors with an equal `MatchKey`. Its response is served from the candidate cache (Phase 3).

#### 7.3 Claims (CL-1–CL-8)
- **File:** `internal/handoff/claims.go`.
- **Key normalization:** `filepath.ToSlash(filepath.Clean(rel))` for paths. Symbol keys and other strings are used verbatim.
- **Claim:** in a `HandoffTx`:
  - if the key is free, insert the claim and return `granted`;
  - otherwise run cycle detection: build a wait-for graph from the tree's claims plus the queue, then DFS. If adding `requester → holder` creates a cycle, return `claim_deadlock_risk` naming the cycle (CL-6);
  - otherwise enqueue and return `{queued, position, holder, holder_label}`.
- **Release / releaseAllTx:** delete the claim, pop the FIFO head, insert the grant claim and a `handoff_claim_grants` row, and observe `claim_wait_seconds`.
- **Claims-granted notice:** `PendingGrants(sid)` reads and marks `notified_at`. `handleToolCall` appends a second content item `{"type":"text","text":"[claims_granted] x.go (scratchpad)"}` of at most 30 tokens before marshal (`server.go:561-573`) (CL-4, CL-5).
- Claims appear in the open digest, scratchpad reads, and the tree API (CL-8). The tool descriptions say claims are advisory (CL-7).
- **Tests:** a `-race` test where 16 goroutines claim the same key: exactly 1 holder and 15 queued in FIFO order (AC20). Cycle detection (AC19). Auto-grant plus notice (AC18).

### Phase 8 — Observability & dashboard (OB, FF-3/FF-4 UI)

#### 8.1 Prometheus (OB-5)
- **File:** `internal/handoff/metrics.go` defines the collectors and exports `Collectors() []prometheus.Collector`. `internal/dashboard/metrics_prom.go:30-82` registers them via `mustRegister`.
- **Metrics:**
  - Counters:
    - `astcache_handoffs_created_total`
    - `astcache_handoff_children_opened_total`
    - `astcache_handoff_children_resumed_total`
    - `astcache_handoff_children_completed_total{status}`
    - `astcache_handoff_children_abandoned_total`
    - `astcache_handoff_trees_expired_total`
    - `astcache_handoff_child_searches_total{repeat}`
  - GaugeFuncs:
    - `astcache_handoff_open_trees`
    - `astcache_handoff_open_children`
    - `astcache_handoff_repeat_search_ratio` (24h window)
    - `astcache_query_cache_hit_ratio` (from `CandidateCache.Stats`)
  - Histograms:
    - `astcache_handoff_tree_tokens`
    - `astcache_handoff_claim_wait_seconds`
- **Tests:** extend `metrics_prom_test.go` with the new names.

#### 8.2 Lifecycle logs (OB-6)
- **Do:** emit `slog` events for create, open, resume, complete, abandon, expire, flush, claim-grant and claim-release. Each event carries `tree_id`, `handoff`, `parent_session`, `child_session` and `project_path` through the ID types' `LogValuer`.

#### 8.3 Tree API
- **File:** `internal/dashboard/handoff_api.go`. Register in `react_api.go:20-31`.
- **Add:**
  - `GET /api/dashboard/handoff-trees?limit=20` returns trees with their nested nodes: label, status, mode, depth, children, tokens delivered and saved, repeat rate, claims and queues, last activity.
  - `POST /api/dashboard/handoff-trees/flush {tree_id}`.
- **Realtime:** add `realtime.Handoffs`, notified on every lifecycle change, and wire it through `realtime_bridge.go:41-68` and `ui/src/hooks/useWebSocket.ts:3-14`.

#### 8.4 Flags API + settings knobs
- **File:** `internal/dashboard/flags_api.go`.
- **Add:**
  - `GET /api/dashboard/flags` returns `flags.State()`.
  - `POST /api/dashboard/flags {key, enabled}` calls `flags.Set` and returns 409 when the flag is env-locked.
- **Update:** the generic settings POST (`api.go:805`) rejects flag keys with "use /api/dashboard/flags", so flags have a single write path. The handoff knobs (6.2) get validation in the settings handler: positive integers.

#### 8.5 UI
- **Files:** `ui/src/api/types.ts`, `client.ts`, `ui/src/storybook/api-stub.ts` (every new method), `fixtures.ts`.
- **Add:** `ui/src/components/HandoffTreesCard.tsx`, rendered in `OverviewTab.tsx` after `VirtualContextCard` (`:231-326`).
  - It shows a collapsible MUI `List` tree with status chips, token stats, and a claims/queue badge, plus a flush button with a confirm dialog.
  - Sessions outside trees keep the existing `SessionLine` view.
- **Add:** a "Features" section in `SettingsTab.tsx`, inserted into `SECTIONS` (`:38-48`). Each flag is an MUI `Switch`; env-locked flags are disabled with a tooltip showing the source.
- **Add:** a "Handoff" settings card with the TTL, summary cap, inactivity window, and caps, using `save()`.
- **Add:** `ui/src/lib/handoffTree.ts`, pure functions that build and sort the tree. Tests in `handoffTree.test.ts` (Vitest).
- **Add:** stories `Handoffs.stories.tsx` and `Settings.stories.tsx` (story `Features`). Add their ids to `ui/scripts/verify-stories.mjs:12-23` and `STORY_IDS.md`.
- **Do:** rebuild `internal/dashboard/ui/dist` in the same commit.

### Phase 9 — Installer, CLI & hooks (IN, HI, PL-9, PL-10)

#### 9.1 Dependencies
- **File:** `go.mod`. Add `github.com/tailscale/hujson` (JSONC patching that preserves comments), `github.com/BurntSushi/toml` (validation only), and `github.com/pmezard/go-difflib` (unified diffs).

#### 9.2 Canonical assets (IN-11, HI-10)
- **Files:**
  - `skills/embed.go` (`package skills`, `//go:embed agents/SKILL.md install/SKILL.md usage/SKILL.md operator/SKILL.md`).
  - `rules/cursor/ast-context-cache.mdc`, the new canonical rule, written from `skills/agents/SKILL.md:252-274`, plus `rules/embed.go`.
  - `instructions/agents-block.md`, the shared CLAUDE.md and AGENTS.md block, plus `instructions/embed.go`.
- **Update:**
  - `skills/usage/SKILL.md` gains a "Subagent handoff" section covering W1/W3/W4 and the fork/fresh guidance (HI-8, HI-9).
  - `docs/INSTALL.md:84-97` and `skills/install/SKILL.md:72-85` drop their divergent snippets and point to the installer and CLI.
  - `generateAgentInstructions` (`api.go:1128`) is deleted.

#### 9.3 `internal/installer` engine
- **Files:** `installer.go` (Service: `Targets()`, `Plan`, `Apply`, `Verify`, `Backups`, `Restore`), `change.go`, `jsonedit.go`, `tomledit.go`, `mdblock.go`, `state.go`, `status.go`, `legacy.go`, and `targets_*.go`.
- **`change.go`:**
  - `type FileChange{ Path, Kind(create|modify|remove-block), Before, After []byte; BeforeHash string; Skipped, Reason string }`.
  - `unifiedDiff()`.
  - `atomicWrite(path, data, mode)`: temp file in the same directory, then fsync, then rename, keeping the original file mode.
  - `backup(path)` copies to `~/.astcache/backups/<YYYYMMDD-HHMMSS>/<path with / replaced by %>` and prunes to `installer_backup_keep` (default 5) per path (IN-4).
- **Plans:** held in memory with a 10-minute TTL, keyed by a random `plan_id`. `Apply(plan_id)` re-hashes each target file, and any mismatch returns `errs.CodeConflict` "file changed since preview" (IN-5).
- **`jsonedit.go`:**
  - `hujson.Parse`. A parse error aborts with no write (IN-3).
  - Updates are applied with `Patch` (RFC 6902 `add`/`replace`/`remove` at `/<key>/ast-context-cache`), then `Format` is **not** called, so the original formatting survives.
  - Ownership: the exact entry name, plus the `entry_hash` stored in state. A user-edited entry reports `Modified by user`.
- **`tomledit.go`:**
  - Validate with `toml.Decode`.
  - Locate the `[mcp_servers.ast-context-cache]` header and its span up to the next table header.
  - Replace, append (preceded by `# managed by ast-context-cache vX`), or remove that span only.
- **`mdblock.go`:**
  - Markers `<!-- ast-context-cache:begin v=4.0.0 sha=<hash> -->` … `<!-- ast-context-cache:end -->`.
  - Status: a body hash that doesn't match `sha` means `Modified by user`; an older `v` means `Outdated` (IN-10).
- **`state.go`:** a usage.db table `installer_state(target, component, path, entry_hash, version, installed_at)`, appended to `initUsageSchema`. It replaces the `agent_configs` usage, and `db.go:112-151` is removed after migration.
- **`status.go`:** `Installed`, `Outdated`, `ModifiedByUser`, `Missing`, `NotInstalled`, `ExternallyManaged`, `Unsupported`. Status is computed from disk plus state (IN-8).
- **Externally managed (IN-9):** a skills path that is a symlink resolving outside `~/.astcache` or the repo is reported `ExternallyManaged` and skipped unless `replace_external=true`, which takes a backup first.
- **`legacy.go` (IN-13):**
  - On startup, read `agent_configs` and verify each row on disk.
  - `claude_code` global rows produce a warning that `~/.claude.json` may have been overwritten by v<4, and pointing to the `~/.claude/backups` files that Claude Code itself keeps.
  - Project-scope rows are listed as removable.
  - Then set the setting `installer_legacy_migrated=1` and keep the old table for one release.
- **Targets** (exact paths and shapes come from the 0.3 notes; the MCP URL is `http://127.0.0.1:<configured port>/mcp`, IN-7):

  | Target | MCP | Skills | Rules / instructions | Hooks |
  |---|---|---|---|---|
  | `cursor` | `~/.cursor/mcp.json` `mcpServers.ast-context-cache.url` | Cursor skills dir (0.3) | `~/.cursor/rules/ast-context-cache.mdc` | — |
  | `opencode` | `opencode.jsonc`/`.json` `mcp.ast-context-cache {type:"remote", url, enabled:true}` | — | OpenCode AGENTS.md block (0.3) | — |
  | `claude_code` | `~/.claude.json` `mcpServers.ast-context-cache {type:"http", url}`, user scope, via hujson | `~/.claude/skills/ast-context-cache-<name>/SKILL.md` | `~/.claude/CLAUDE.md` block | `~/.claude/settings.json` `hooks.*` (9.7) |
  | `claude_desktop` | `claude_desktop_config.json` `{command:"mcp-local", args:["bridge", url]}`, or `{command:"npx", args:["-y","mcp-remote", url]}` when mcp-local is absent (the preview warns when npx is missing too) | — | — | — |
  | `codex` | `~/.codex/config.toml` `[mcp_servers.ast-context-cache]` (URL, or bridge command per 0.3) | — | `~/.codex/AGENTS.md` block | — |
  | `vscode` | user `mcp.json` (per-OS path) `servers.ast-context-cache {type:"http", url}` | — | — (Unsupported: no global instructions file) | — |
  | `jetbrains` | per 0.3, or `Unsupported` with manual steps | — | — | — |
- **Tests (AC31–AC38):**
  - Golden fixtures under `internal/installer/testdata/<target>/{empty,existing,upgrade,uninstall}/{in,golden}`, including commented JSONC and TOML, foreign servers, and a user-modified block. This introduces the repo's first `testdata/`, justified by the size of the fixtures.
  - Table tests for jsonedit, tomledit and mdblock using `t.Parallel()` (pure functions).
  - `HOME=t.TempDir()` for engine tests.
  - A port test with a non-default port.

#### 9.4 CLI (IN-15 MUST)
- **File:** `cmd/ast-mcp/main.go`. Dispatch `os.Args[1]` in `{install, uninstall, verify, backups, restore, hook}` before `flag.Parse` (`:51-55`).
- **File:** `cmd/ast-mcp/cli_install.go`.
- **Flags:**
  - `--target` (repeatable, or `all`)
  - `--component mcp,skills,rules,hooks`
  - `--mcp-url`, or `--mcp-port` (default `$AST_MCP_PORT`, then 7821)
  - `--dry-run` prints the unified diff
  - `--yes` applies
  - `--json` gives machine output for mcp-local: `{changes:[...], status:[...], warnings:[...]}`
  - `--replace-external`
- **Exit codes:** 0 OK; 2 confirmation required (neither `--yes` nor `--dry-run`); 3 conflict or parse error; 4 unsupported.
- `verify` prints the status table.
- The CLI runs in-process against the installer package. It does not need the server running, except for `hook`.

#### 9.5 Dashboard API (replaces `/api/agent-*`)
- **Delete:**
  - `api.go:78-80` routes and the handlers `:970-1169`;
  - `settings_data.go:58-78`;
  - `components/settings.go:3-11` and the `Agents` field;
  - client `agentInstall`/`agentUninstall` (`client.ts:175-176`) and their types (`types.ts:305-316`).

  This is a 4.0 breaking change.
- **Add** in `internal/dashboard/installer_api.go`:
  - `GET /api/dashboard/installer`: targets × components with status, plus legacy warnings.
  - `POST /api/dashboard/installer/preview {targets, components, action: install|uninstall, replace_external}` returns `{plan_id, changes:[{path, kind, diff, skipped, reason}], warnings}`.
  - `POST /api/dashboard/installer/apply {plan_id}`.
  - `GET /api/dashboard/installer/backups`.
  - `POST /api/dashboard/installer/restore {backup_id}`.
- The Origin guard (4.2) covers all of these.

#### 9.6 Installer UI
- **Add:** `ui/src/tabs/settings/InstallerSection.tsx`, replacing `SettingsTab.tsx:575-617`.
  - Target cards show per-component status chips and checkboxes. The Hooks checkbox appears only when `feature_handoff_hooks` is on.
  - "Preview" opens a `Dialog` with a monospace diff colored per line; "Apply" sends `plan_id`.
  - Conflicts trigger an automatic re-preview.
  - Backups appear in an accordion with Restore buttons.
  - A legacy-warning `Alert`.
- **Add:** `ui/src/lib/diff.ts` (diff-line classification, status → chip color) with `diff.test.ts`.
- **Add:** the story `Installer.stories.tsx`; update `fixtures.ts:343-362` (it's stale today). Add to `verify-stories`. Rebuild dist.

#### 9.7 `ast-mcp hook` (PL-9, PL-10, HI-2, HI-3, HI-5)
- **Files:** `cmd/ast-mcp/cli_hook.go` and `internal/hooks/hooks.go`, `client.go`, `hooks_test.go`.
- **Client:** a tiny MCP client that POSTs `tools/call` to the local URL with a **2s timeout**. Any error makes the hook exit 0 with no stdout (fail-open, HI-5).
- **Events** (only those the 0.2 spike confirmed):
  - **`session-start`:**
    - Always outputs `additionalContext`: "ast-context-cache: use session_id=<host session_id> for all ast-context-cache tools in this conversation." This resolves HI-4.
    - When `source=compact`, it also calls `handoff list` and appends the open handoffs (HI-2c).
  - **`subagent-start`:** when the prompt or input carries `[handoff hof_…]`, call `open_handoff open` and inject the digest plus the child `session_id` (HI-2a).
  - **`subagent-stop`:** when the child's status is still `open`, call `handoff complete status=partial content=<final message or transcript tail>` (HI-2b).
  - **`pre-tool-use-agent`:** only if `updatedInput` is supported. Create a handoff from the parent session with the Agent `description` as brief, and append the stub to the prompt (HI-3).
- **Installer:** the `claude_code` hooks component writes these entries into `~/.claude/settings.json` using the **absolute** `os.Executable()` path, with `timeout: 3`.
- **Tests:**
  - Table tests with recorded spike payloads as fixtures.
  - A fail-open test: an unreachable URL exits 0 with no output in under 2.5s.
  - Injection-text tests.

### Phase 10 — Docs, prompts, release (TS-6, HI-8/9/10, IN-12, IN-14, BF-3)

#### 10.1 Prompts
- **File:** `internal/mcp/tools.go` `GetPrompts()` (`:649-813`).
- **Update:** add the handoff tools to the `virtual-context-compaction` tool table (`:740-746`) and a handoff bullet to `efficient-context-usage` (`:715-721`).
- **Add:** a new prompt `subagent-handoff` covering W1, W3 and W4, the fork/fresh rule, claims being advisory, and the return-stub contract.

#### 10.2 Golden tests
- **Files:** `internal/mcp/tools_example_test.go:17,29,41` and `context_tools_test.go:13,25-31`. The three tools are inserted after `recall_memory` in `GetTools()` order, in **all three** tier lists, since they're core.

#### 10.3 Docs
- **Update:**
  - `README.md:81-91` features, `:101`, `:143-151` tier table.
  - Fix the broken `#tool-tiers-and-per-tool-overrides` anchor, either by renaming the heading or updating the 5 links.
  - `AGENTS.md:89-136,176-182,220-224`.
  - `CLAUDE.md:55-87`, also adding the missing `diff_impact`, `check_symbol_exists` and `check_deletion_safety`.
  - `docs/USAGE.md:11-55`.
  - `skills/agents/SKILL.md:126-164,191-201,266-269`.
  - `skills/usage/SKILL.md:37-80,102-110,167-182`.
  - `skills/install/SKILL.md:92-101`.
- **Do:**
  - Sync `.cursor/skills/*` per `skills/README.md:13-36`, and fix `ast-operator`'s missing frontmatter.
  - IN-12: remove the `env` blocks on URL entries (`README.md:49-58`, `skills/install`).
- **Add:**
  - `docs/handoff.md`: the user guide, workflows W1–W10, the error-code table, and the knobs.
  - `docs/host-integration.md`, finalized from 0.3 (IN-14).
- **BF-3:** `react_api.go:155-158` uses the `mcp` package's resolved tools-config path instead of hard-coding `~/.astcache/tools.json`.

#### 10.4 Migration notes + version
- **File:** `README.md`. Add `## Migrating to 4.0` after `:153-163`, with a "Change / What to do" table covering:
  - localhost bind and `AST_LISTEN`
  - Origin guard
  - slog format and `AST_LOG_FORMAT`/`AST_LOG_LEVEL`
  - `errs` error strings
  - three new core tools and their flags
  - the query-cache/dedup behavior changes
  - retrieve `deduped_count` → `deduped`
  - `/api/agent-*` removed, replaced by the installer and the CLI
  - re-running the installer to fix legacy installs
  - mcp-local ≥ the companion version for Claude Desktop
- **File:** `VERSION` → `4.0.0`. This is a manual major bump; the merge bot bumps the patch level afterwards.

### Phase 11 — mcp-local companion PR (PL-11)

All paths in `~/git/mcp-local`, on branch `NO-TICKET-v4-subagent-handoff` in the same space.

#### 11.1 `mcp-local bridge <url>`
- **Files:** `internal/bridge/bridge.go`, `bridge_test.go`, and `cmd/mcp-local/bridge.go` (Cobra command).
- **Behavior:**
  - **Input:** newline-delimited JSON-RPC from stdin, POSTed to `url` with `Accept: application/json, text/event-stream` and `Content-Type: application/json`.
  - **Session:** persist `Mcp-Session-Id` from the initialize response.
  - **Responses:** a JSON body is written to stdout as one line. An SSE body is parsed and each `message` event's data is written.
  - **Notifications:** a 202 produces no output.
  - **Server-initiated messages:** after initialize, open a GET SSE stream and forward its messages to stdout.
  - **Shutdown:** exit on stdin EOF; logs go to stderr.
- **Tests:** an `httptest` fake server covering JSON, SSE, a 202 notification, session-header round-trip, and a GET-stream notification forwarded to stdout.

#### 11.2 Writers
- **`internal/mgr/jsonagent/agent.go`:**
  - Parse with hujson and patch only the managed key, so comments, order and formatting survive.
  - Atomic temp-file write plus rename, and a timestamped backup next to `~/.mcp-local/backups/`.
  - `Deregister` tolerates a missing file and removes an entry only on an exact-match owned name.
- **`agents.RegisterAll`/`DeregisterAll`** (`agents/agents.go:31-67`): continue past per-host errors and return an aggregated error.
- **Found flags:** honor them in `cursor.go:44-47` and `claudedesktop.go:52-55`.
- **`internal/mgr/opencode/opencode.go:13-28`:** drop the regex comment stripping in favor of hujson.
- **`claudedesktop`:**
  - HTTP services are written as `{command:"mcp-local", args:["bridge", url]}`.
  - Make the config path lazy (no package-level `var agent`) so tests can redirect HOME.
  - Fix `README.md:36`.
- **Tests:** cover JSONC with inline comments, block comments and trailing commas; preserved foreign entries; a missing-file deregister; and continue-on-error.

#### 11.3 Delegation to `ast-mcp install`
- **File:** `internal/mgr/config` gets a service field `installer: ast-mcp`, auto-set when the command basename is `ast-mcp`.
- **File:** `internal/mgr/agents`. For such services, register and deregister run `<command> install|uninstall --target <opencode|cursor|claude_desktop> --component mcp --yes --json --mcp-url <mcp_url>`.
  - If `ast-mcp --version` is below 4.0.0, fall back to the native writer with a warning.
  - Errors surface per host.
- **Tests:** a fake `ast-mcp` script on PATH records its args.

#### 11.4 Tiers + protocol fixes
- **`internal/mgr/asttools/default_tiers.go:6-20`:** add `handoff`, `open_handoff` and `scratchpad` as core, and reconcile the rest with ast's `GetTools()`.
- **`cmd/mcp-local/extras.go:952-1043` `fetchToolsList`:**
  - Make `ID *int` with `omitempty`, so notifications carry no id.
  - Send `Accept: application/json, text/event-stream`.
  - Parse SSE responses.
  - Persist `Mcp-Session-Id`.
- **`internal/mgr/embedfs/server.go:114-117`:** return 202 for notifications.
- **Docs:** update `AGENTS.md` (`:33,59,161-166`) and the README to describe the bridge and delegation.

### Phase 12 — Validation (AC1–AC42, NFR-1/3/4)

#### 12.1 Scenario test (AC39 plus most functional ACs)
- **File:** `internal/mcp/handoff_scenario_test.go`.
- **Fixture project:** about 12 files, built with the `indexedPython()` pattern (`summary_tool_test.go:18`), extended to Go.
- **Script:**
  1. The parent runs 10 searches.
  2. It creates a fresh handoff and a fork handoff.
  3. Three fresh children and one fork child open them.
  4. One child creates a grandchild handoff.
  5. Children post, claim and complete.
  6. The parent collects with `recursive=true`.
- **Baseline run:** the same scripted child queries under fresh, unlinked sessions.
- **Child policy in the handoff run:** a scripted child skips any planned query that the digest's trail covers, and treats `parent_trail_match` as satisfied. The repeat rate is computed by the OB-1 function.
- **Assert:**
  - a reduction of at least 50%;
  - nonzero handoff and return tokens saved;
  - every status transition correct.
- The test documents that its child policy is synthetic. Real evidence comes from 12.5.

#### 12.2 Benchmarks + latency test (NFR-1)
- **File:** `internal/handoff/bench_test.go`. `BenchmarkCreate` (at cap), `BenchmarkOpen`, `BenchmarkExpandPointer`, `BenchmarkScratchpadPost`, `BenchmarkScratchpadRead`, `BenchmarkClaim`, `BenchmarkCollect16`.
- **File:** `internal/handoff/latency_test.go` `TestLatencyBudgets`. It runs 200 iterations each and asserts p95 against NFR-1, and is skipped under `-short` and `-race`. It runs in `make test` but not in `make race`.

#### 12.3 Concurrency (NFR-4)
- **File:** `internal/handoff/concurrency_test.go`, run via `make race`. Cases:
  - 16 concurrent opens;
  - 16 concurrent claims on one key (AC20);
  - concurrent post and read cursors with no lost entries;
  - concurrent complete and collect.

#### 12.4 Installer fixtures (AC31–AC38)
- Per 9.3.

#### 12.5 Real-host trial (AC41; manual, recorded)
- **Checklist** in the PR body. Record results in `docs/trials/handoff-trial-<date>.md`:
  - Claude Code Agent tool: W1/W2 with hooks installed through the installer.
  - Claude Code fork: W3.
  - A Claude Code workflow with 3 agents: W4.
  - Cursor: W1, manual.
  - Note the dashboard tree view and `/metrics` values (AC42), and screenshot the tree view.

#### 12.6 UI verification
- Run `make verify-stories` for the new stories, and `make dashboard-screenshot` for the PR.

---

## Testing Strategy

Follow existing style:
- `dbtest.Init(t)` for DB-backed tests; `TestMain` per package where pools are shared.
- `callTool` for MCP tool tests.
- Inline golden tests for tool lists.
- testify `require`/`assert` in new and touched tests.
- No `t.Parallel()` in DB tests.
- `make test lint race ui-test` must pass on every commit.

| Area | Tests | Maps to PRD AC |
|---|---|---|
| errs / logging | `internal/errs/errs_test.go`, `internal/logging/logging_test.go` | (PL-1/2) |
| Flags | `internal/flags/flags_test.go`, `internal/mcp/tools_test.go` flag cases, `stream_test.go` list_changed | AC28, AC29, AC30 |
| BF fixes | `contextnotes/store_test.go` LIKE scope; `memory/recall_test.go` vector validity and scope | AC26, AC27 |
| Cache / dedup | `cache/candidate_cache_test.go`, `context/handler_test.go`, `mcp/retrieve_test.go` | AC23, AC24, AC25 |
| Transport / guard | `mcp/stream_test.go`, `mcp/protocol_test.go`, `httpguard/guard_test.go` | AC28 (+PL-4/5/6) |
| Trail | `trail/trail_test.go`, `search/filters_test.go` NormalizedKey | AC1, AC6 |
| Handoff create/open/expand | `handoff/create_test.go`, `open_test.go`, `staleness_test.go`, `mcp/handoff_tools_test.go` | AC1–AC9, AC14, AC15 |
| Complete / collect / recovery | `handoff/complete_test.go`, `collect_test.go` | AC10–AC13 |
| Scratchpad / claims | `handoff/scratchpad_test.go`, `claims_test.go`, `concurrency_test.go` | AC16–AC20 |
| Retention / caps | `handoff/sweeper_test.go` (clock injected via `nowFunc`), `caps_test.go` | AC21, AC22 |
| Installer | `installer/*_test.go` + `testdata/` golden fixtures, `cmd/ast-mcp/cli_install_test.go` | AC31–AC38 |
| Hooks | `hooks/hooks_test.go` (spike payload fixtures, fail-open) | (HI-2/4/5) |
| Metrics / dashboard | `dashboard/metrics_prom_test.go`, `dashboard/handoff_api_test.go`, `flags_api_test.go`, `installer_api_test.go` | AC42 |
| UI logic | `ui/src/lib/handoffTree.test.ts`, `diff.test.ts` (Vitest); stories in `verify-stories` | (OB-4, IN-5) |
| Scenario | `mcp/handoff_scenario_test.go` | AC39 (+AC1–AC22 end to end) |
| Performance | `handoff/bench_test.go`, `latency_test.go` | AC40 |
| Real host | Manual trial report | AC41 |
| mcp-local | `bridge_test.go`, `jsonagent` JSONC/atomic tests, delegation test | (PL-11) |

**Edge cases to cover explicitly:**
- empty parent session (no trail or manifest);
- a snapshot exactly at the cap, and one over it;
- resume after abandonment;
- expand of an expired handoff;
- a pointer whose file was deleted;
- a sibling-worktree child;
- `feature_handoff` off mid-tree (calls return `feature_disabled` and the data survives);
- server restart mid-tree (the claims queue persists, and `treeIndex` re-hydrates);
- a JSONC file with a BOM or CRLF;
- a TOML file with no trailing newline;
- a symlinked skills directory;
- a non-loopback `--listen` warning.

**Manual checklist (PR body):**
- the real-host trial (12.5);
- an installer round-trip on the developer's actual `~/.cursor/mcp.json` with the preview reviewed;
- toggling a flag in the dashboard while Claude Code is connected, confirming that tools refresh;
- `docker compose up` still reachable with `AST_LISTEN=0.0.0.0`.

---

## Risks & Open Questions

| Risk | Mitigation |
|---|---|
| **One very large PR** (sweeps touch ~200 files, plus 3 new subsystems) | Phase-ordered commits, each green. Mechanical sweeps are isolated per package. The PR body has a per-phase reading guide and commit map. The user reviews commit by commit. |
| **Long-lived branch conflicts with other work** on main during the sweeps | Do Phase 1 first and fast. Sync with main via `sync_with_base_branch` (or `git merge origin/main`) after each phase; re-run the gofumpt and SQL hoists on new upstream code. |
| **Hook capabilities unsupported** (0.2) | The manual stub path (HI-1) works on every host. HI-2 items are built only when confirmed; the rest are documented. |
| **`2026-07-28` spec unverifiable** | Negotiate down to `2025-06-18`/`2024-11-05`; leave the version out of `supportedVersions`. |
| **Editing `~/.claude.json` while Claude Code is running**: Claude Code may rewrite it concurrently | The apply step re-hashes and refuses on change (IN-5), and a backup is always taken. The preview notes that it's safest with Claude Code closed. |
| **SQLite contention** from the single immediate-tx handoff writer | Handoff writes are small; activity touches are coalesced (6.9); trail persistence is batched through usage.db. Benchmarks (12.2) catch regressions. |
| **Candidate-cache memory and deep-copy cost** | Cap of 1,000 entries; copy only the `Data` maps; benchmark the capsule path before and after the change. |
| **Synthetic scenario overstates the ≥50% gain** | The trial (12.5) supplies real numbers. OB-1's threshold is revisited afterwards (PRD Q13). |
| **Clean-room `errs` drifts from Slide's API**, causing confusion when switching repos | Mirror the function names and signatures from the STYLEGUIDE; document the differences in the `internal/errs` godoc. |
| **Cross-repo coupling** (mcp-local → `ast-mcp install`) | Version check with fallback to the native writer (11.3); the companion PR is linked and merged after the ast PR. |
| **Localhost bind breaks remote or Docker users** | `AST_LISTEN` override, Docker defaults updated, README migration row. |
| **Committed UI dist churn** | Rebuild only in the UI commits (8.5, 9.6); note it in the PR guide. |
| **JetBrains/Codex lack a file-based global MCP config** | `Unsupported` status with manual steps (IN-2 allows a skip with reason). |
| **Child id format `hof_….cN` with a dot** | No code splits session ids. Add a test that the ids round-trip through every tool. |

**Remaining minor decisions (defaults are fine):**
- Child session id format: `<hof_ref>.c<N>`.
- Tree id prefix: `hft_`.
- Backup location: `~/.astcache/backups/`, keeping 5.
- Plan TTL: 10 minutes.
- Trail ring: 200 entries per session.
- Abandonment sweep: every minute; expiry sweep hourly with a 2-minute initial delay.
- Activity-touch coalescing: 10s.
- Legacy `agent_configs` table: kept one release, then dropped in 4.1.
- testify scope: new and touched tests only.
- Session-store idle eviction: 30 minutes.
- Live-trail entries count toward the tree cap but are evicted first (RQ-5).

---

## Suggested Next Steps

1. **User:** review this plan and the PRD, and flip the PRD's status to `Approved` if satisfied.
2. **Agent:** run `proj-impl` against this plan, starting with Phase 0. It creates the wtg space (`ast-context-cache` + `mcp-local`) via the spaces skill, then runs the hook spike and the spec and format verification. It stops for the user to review `docs/spikes/claude-code-hooks.md`.
3. **Agent:** implement Phases 1–10 as ordered commits, keeping `make test lint race ui-test` green at every commit. Sync with main after each phase.
4. **Agent:** implement Phase 11 in the mcp-local worktree.
5. **Agent + user:** Phase 12 validation, including the real-host trial.
6. **Agent (with this-turn approval only):** open the ast-context-cache PR and the mcp-local companion PR, cross-linked, each linking the PRD and this plan. **The user merges**: ast first, then mcp-local.
