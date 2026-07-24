# Plan: Platform Hardening & Operator Confidence (v3.0)

One-line summary: Ship a single v3.0 PR that hardens embed/WAL reliability, removes legacy dashboard/MCP dead weight, proves agent value on Overview, and adds Prometheus + keep-alive—building on PR #55.

- **Date:** 2026-07-23
- **Source PRD:** [`prd-platform-hardening-and-operator-confidence.md`](prd-platform-hardening-and-operator-confidence.md)
- **Related:** [PR #55](https://github.com/coma-toast/ast-context-cache/pull/55) (aux catch-up + WAL abort restore) — **merge first** (user clicks Merge; agents must not merge). CI currently reports `test` FAILURE — fix/re-run before merge.
- **Delivery:** One big PR; `VERSION` → **3.0.0**; short migration note in README

---

## Context

### Explored
- Embed reliability: `internal/embedqueue/{maintenance,swap,quiet,recovery,workers,aux_workers,runlock,queue}.go`; checkpoint restore bind in `internal/db/checkpoint.go` (`prepCheckpoint`); main hooks in `cmd/ast-mcp/main.go`.
- Dashboard: React SPA is live (`ui/`); legacy `/partials/*` + `*_templ.go` + HTMX/Alpine static still registered in `internal/dashboard/api.go`. WAL APIs exist (`POST /api/wal-checkpoint`, `GET /api/wal-status`); React has `api.walCheckpoint` unused; copy helpers in `components/wal_ui.go`.
- Value metrics: `/api/dashboard/stats`, `/api/context-stats`, `/api/timeseries`, `/api/tools` already power Overview/Activity/Analytics; no Prometheus yet.
- Keep-alive: `scripts/ast-mcp.bash` (start/stop/health, no supervise); no Dockerfile for ast-mcp (only DMR notes under `docker/`).
- Code-mode: richer builtins already under `internal/codescripts/builtin/`; `scripts/code-mode/` is a thin mirror.
- Ghost MCP: `sync_remote` / `reset_project` / `reset_all` — **removed in v3** from MCP dispatch; `internal/mcp/sync.go` deleted. Project reset remains via dashboard APIs only. No tests referenced these tools.

### Key patterns to follow
- Pause/restore: always call existing `PauseAllForMaintenance` / `RestoreAfterMaintenance` (and swap pair)—do not invent parallel pause flags.
- Abnormal exit: reuse `BeginRunLock` / `AbnormalPreviousRun` (`runlock.go`); surface to UI.
- Dashboard tests: httptest + temp HOME + `db.Init` (see `embed_settings_test.go`); embedqueue `TestMain` pattern.
- React: extend `OverviewTab.tsx` / `EmbeddingsPanel.tsx`; keep dark theme in `theme.ts`.
- Logging/errors in this repo: `log.Printf` and `fmt.Errorf` (not Slide `errs` / compact STYLEGUIDE)—match neighboring files.

### Architectural constraints
- Local-only; three SQLite DBs unchanged.
- Breaking removal of templ/partials and ghost tools is intentional for v3.0.
- Auto-recover must not override intentional workers=0 or active maintenance.

---

## Requirements (from PRD)

Preserve MUST/SHOULD/MAY:

| ID | Priority | Summary |
|----|----------|---------|
| A1–A7 | MUST | Aux catch-up; WAL restore; maintenance UX; 5m stuck-worker auto-recover; watchdog + Docker restart; crash banner; intentional pause distinct |
| B8–B12 | MUST | Remove templ/HTMX/partials; collapse duplicate health/settings for React; remove ghost MCP tools; fix stale docs; document memory tools |
| C13–C15 | MUST/SHOULD | First-class embed/WAL docs & visibility; reduce noise; operator failure-mode docs |
| D16–D24 | MUST/SHOULD | Embeddings UX; WAL banner+checkpoint; Overview value metrics; heuristic before/after; per-tool try; weekly digest card; session VC stories; Settings SHOULD; keep dark |
| E25–E26 | MUST | Prometheus `/metrics` on dashboard port; scrape docs |
| F27–F30 | SHOULD/MUST | Memory in skills; thicker code-mode; doc packs; README enough |
| G31–G34 | — | Deferred (remote MCP, cloud, auth, new languages) |

**Out of scope:** remote shared MCP, cloud sync, auth, light theme, email/ntfy digest, greenfield queue rewrite.

**Resolved open questions (user):**
1. Supervise via `ast-mcp` script + Compose restart  
2. Remove `sync_remote` entirely — **done** (v3)  
3. Tokens-first heuristic + optional rounds avoided  
4. Session VC ~30d  
5. Dockerfile+compose in-repo only  
6. Delete templ when Embeddings/WAL/Overview done  

---

## Approach

**Phased work inside one PR branch** (logical commits OK; single PR to `main`):

1. **Prerequisite:** User merges #55 (or rebase this branch onto fixed #55).  
2. **Reliability** (process + queue): stuck-worker watchdog, abnormal-restart banner, supervise/Docker, flush-log noise control.  
3. **Confidence UI:** WAL React parity, Overview digest/heuristic/session stories, embeddings clarity.  
4. **Observability:** `client_golang` `/metrics`.  
5. **Breaking cleanup:** ghost tools, templ/partials/static, API collapse, VERSION 3.0.0, docs/skills.  
6. **SHOULD features:** memory skills, code-mode sync/thicken, doc packs.

**Why this order:** Reliability and Overview value deliver user-visible wins before deleting legacy UI (so React is proven). Cleanup last avoids shipping a binary that still needs HTMX for anything in-scope. Metrics can land mid-stream once Snapshot/stats exist.

**Alternatives rejected:**
- Stacked PRs — user wants one big PR.  
- Hand-rolled Prometheus text — user chose `client_golang`.  
- New Value tab — extend Overview instead.  
- Env-gated `sync_remote` — remove (**done** in v3).

---

## Style Guide Notes

- Canonical Slide `STYLEGUIDE.md` at `/Users/jason/git/STYLEGUIDE.md` applies to Slide services, **not** this repo’s established style.
- **For ast-context-cache:** follow local neighbors—`log.Printf` for ops, `fmt.Errorf` for errors, existing package layout, compact but not Slide-dense unless matching the file.
- React: existing MUI patterns in `ui/src`; no new design system.
- Do not introduce Slide `errs` package here.

---

## Detailed Implementation Steps

### Phase 0 — Branch hygiene

0.1. **User:** Fix CI on #55 if still red, then **click Merge** in GitHub (agents must not merge).  
0.2. Locally: `git fetch origin && git checkout main && git pull` then branch e.g. `NO-TICKET-v3-platform-hardening` from `main` (includes #55).  
0.3. Confirm `auxCanCatchUp`, `prepCheckpoint` restore-on-abort, and recovery flush-on-error are present on `main`.

### Phase 1 — Reliability (MUST A2–A7; A1 assumed from #55)

#### 1.1 Stuck-worker auto-recover (A4)
- **File:** `internal/embedqueue/workers.go` (or new `watchdog.go` in same package).
- **Add:** goroutine started from `Start` / `startPressureBackoff` vicinity; every ~30s evaluate:
  - `WorkerTarget() > 0 && WorkerCount() == 0 && WorkerLive() == 0`
  - `!MaintenancePaused()` / `!SwapPaused()`
  - target > 0 means not intentional pause (intentional pause sets target to 0 via `SetWorkerCount(0)` persist)
  - sustain ≥ **5 minutes** then call restore path: `applyWorkerCountLocked(ThrottledEmbedWorkers(target), false)` and if aux target > 0 ensure aux count restored similarly.
- **Log:** `embedqueue: auto-recovered workers after stuck pause (target=%d)`.
- **Expose:** last recovery time via Snapshot or health DTO for UI (SHOULD).
- **Tests:** `workers_adjust_test.go` / new `watchdog_test.go` — fake clock or inject short threshold in test.

#### 1.2 Abnormal previous run banner (A6)
- **Go:** Thread `embedqueue.AbnormalPreviousRun()` into dashboard health / index-health JSON (`partials_data.go` / `react_api.go` / `handleMCPHealth` as needed). Clear or latch for UI (show until dismissed or until process uptime > N).
- **React:** Banner on Overview or HealthBar: “Restarted after abnormal exit”.
- **Tests:** extend `runlock_test.go`; httptest health if practical.

#### 1.3 Keep-alive: supervise + Docker (A5)
- **Extend** `scripts/ast-mcp.bash` / `.fish`: `supervise` (or `start --supervise`) loop: start binary, wait, restart on nonzero exit with backoff; honor stop via PID file / signal.
- **Add** `docker/ast-mcp/Dockerfile` + `docker-compose.yml` (or `docker/ast-mcp/compose.yml`) running the binary with `restart: unless-stopped`, volume for `~/.astcache`, ports 7821/7830. Do **not** confuse with existing DMR `docker/README.md`—add a clear section.
- **Docs:** `skills/operator/SKILL.md` + README: supervise vs Compose.

#### 1.4 Maintenance UX wiring (A3) — backend already mostly there
- Ensure WAL phase fields already on IndexHealth continue to update over WS.
- No new pause API; React Phase 2 consumes existing fields + checkpoint POST.

#### 1.5 Flush noise (C14 SHOULD)
- **File:** `internal/embedqueue/recovery.go` / `queue.go` — rate-limit or dedupe “flush pending” logs when pending unchanged and in-flight > 0.

### Phase 2 — Confidence UI (MUST D16–D22; A3, A7)

#### 2.1 WAL banner + checkpoint (D17)
- **Port** strings/progress from `internal/dashboard/components/wal_ui.go` into TS helpers (e.g. `ui/src/lib/walUi.ts`).
- **Update** `EmbeddingsPanel.tsx` (and/or `ResourceUtilCard.tsx`): full banner phases, disable button when active, call `api.walCheckpoint()`.
- Optional: `api.walStatus()` if WS snapshot missing fields.

#### 2.2 Embeddings clarity (D16, A7)
- **Update** `EmbeddingsPanel.tsx` / `WorkerControls.tsx`: distinguish intentional pause (target 0) vs stuck/error vs WAL throttle vs maintenance; show primary vs aux live/target/pending/in-flight.
- Surface auto-recover event if Phase 1.1 exposed it.

#### 2.3 Overview value (D18–D19, D21–D22)
- **Backend:** Extend `components.Stats` / `fillVirtualContextStats` / new helpers:
  - Heuristic before/after (tokens-first + optional rounds avoided)—compute from existing query/token aggregates; label `approximate`.
  - Weekly digest rollup (7d tokens saved, VC stored/accessed, embed failures/restarts if logged, top tools)—new SQL or reuse timeseries/tools with `days=7`.
  - Session-level VC stories: query `contextnotes` / usage logs by `session_id` (~30d), return compact list for Overview.
- **API:** Prefer enriching `GET /api/dashboard/stats` and/or small `GET /api/dashboard/weekly-digest` + `GET /api/dashboard/context-sessions` to avoid bloating one payload—keep Overview `loadAll` reasonable.
- **React:** Extend `OverviewTab.tsx` with cards/sections (not a new tab): Tokens + heuristic, VC recoveries/usage, weekly digest, session stories sample.
- **Activity/Analytics (D20 SHOULD):** Emphasize per-tool savings already in Analytics; light tweak to titles/copy.

#### 2.4 Settings (D23 SHOULD — non-blocking)
- Only if time: reduce save-on-blur surprises for embed fields; defer full parity.

### Phase 3 — Prometheus (MUST E25–E26)

- **Dep:** `github.com/prometheus/client_golang`.
- **File:** e.g. `internal/dashboard/metrics_prom.go`; register `mux.Handle("/metrics", promhttp.Handler())` in `NewHandler`.
- **Collectors / gauges** (prefix `astcache_`): process up; embed pending/queued/in_flight; workers target/live primary+aux; embedder state (gauge or labeled); index WAL bytes; `tokens_saved` counter (from query log increments or periodic scrape of totals—prefer counters updated when query log writes if easy); MCP tool latency/counts (hook existing tool stats or middleware).
- **Docs:** operator skill + README scrape example (`localhost:7830/metrics`).
- **Tests:** handler returns 200 and contains expected metric names (httptest).

### Phase 4 — Breaking cleanup (MUST B8–B12)

#### 4.1 Ghost MCP tools (B10) — done
- Removed `case "sync_remote"|"reset_project"|"reset_all"` from `server.go`; deleted `sync.go` (only used by `sync_remote` / `REMOTE_VECTORDB_*`).
- Dashboard project reset/delete APIs unchanged.
- Migration: use local index only; no MCP reset/sync. No unit tests called these tools.

#### 4.2 Remove templ/HTMX path (B8, B9)
- Delete `/partials/*` route registrations and handlers only used by them.
- Remove unused `*_templ.go` / static HTMX/Alpine assets once no Go imports remain; drop `make generate` templ step if obsolete.
- Collapse duplicate health/settings: React uses `/api/dashboard/*`; keep thin `/api/settings` if React still posts keys—document; remove HTML settings partial.
- Update Makefile/`TODO.md` (delete or rewrite stale React checklist).
- Fix `ui/README.md` (React not Preact); `OPENVIKING_COMPARISON.md` memory claims; `docs/USAGE.md` tool tables.

#### 4.3 Agent docs memory (B12, F27)
- Update `AGENTS.md`, `CLAUDE.md`, `skills/usage/SKILL.md`, `.cursor/skills/ast-usage` for `store_memory` / `recall_memory` / `forget_memory` vs `ctx_*`, and `report_kv_repair_event`.

#### 4.4 Version & migration (NFR2)
- Set root `VERSION` to `3.0.0`.
- README section: **Migrating to 3.0** (removed partials/tools, supervise/Docker, `/metrics`, Overview changes).

### Phase 5 — SHOULD features (F28–F30)

#### 5.1 Code-mode
- Sync `scripts/code-mode/` with `internal/codescripts/builtin/` (or document builtins as source of truth and thin scripts README).
- Add 2–4 more useful scripts if gaps (filter/group already in builtins—expose in docs).

#### 5.2 Doc packs
- Curated list (JSON or Go const) of common doc sources; dashboard or MCP helper to “add pack” calling existing `add_doc_source` / `fetch_doc` paths.
- UI: button on Memory/docs area or Settings.

#### 5.3 README polish (F30)
- Local install, supervise, Compose, metrics, v3 migration—enough for another self-hoster.

### Phase 6 — Operator failure-mode docs (C13, C15)

- Update `skills/operator/SKILL.md`: workers=0, WAL throttle, maintenance, embedder error, process down, auto-recover, aux catch-up.

---

## Testing Strategy

Follow existing style (`make test`, package `TestMain`, httptest).

| Area | Tests | Maps to PRD AC |
|------|-------|----------------|
| Aux catch-up / WAL restore | Already in #55 `recovery_test.go`, `checkpoint_test.go` | AC1, AC2 |
| Stuck-worker auto-recover | New unit with short threshold | AC3 |
| Run lock / abnormal flag | Extend `runlock_test.go` | AC4 |
| Ghost tools gone | MCP dispatch test or compile-time absence + optional call returns method not found | AC5 |
| Weekly digest / heuristic / session stats | `contextnotes` or dashboard SQL unit tests | AC6 |
| `/metrics` | httptest contains `astcache_` series | AC7 |
| Docs | Manual checklist in PR description | AC8, AC9 |

**Manual checklist (PR body):** Embeddings panel WAL button; Overview digest; supervise restart; intentional pause not auto-recovered; primary error + aux drain.

**UI:** No mandatory Playwright; visual smoke on dashboard after `make build` / `ast-mcp restart`.

---

## Risks & Open Questions

| Risk | Mitigation |
|------|------------|
| PR #55 CI red | Fix tests before merge; plan blocked on green #55 |
| Templ deletion breaks overlooked HTML consumer | Grep for `/partials` and `templ.` before delete; keep React-only verification |
| Auto-recover fights intentional pause | Key off **persisted target == 0** as intentional |
| `/metrics` + large scrape of SQL | Prefer in-memory Snapshot + counters updated on write paths |
| One huge PR review burden | Logical commits; PR summary by phase |
| Docker build complexity (CGO, onnx, tokenizers) | Start with documented multi-stage Dockerfile; if too heavy, ship compose that mounts prebuilt binary first |

**Remaining minor decisions (defaults OK):**
- Exact backoff for supervise loop (e.g. 1s, 2s, 5s cap).
- Whether weekly digest is separate endpoint vs fields on `/api/dashboard/stats`.

---

## Suggested Next Steps

1. **You:** Fix/merge [PR #55](https://github.com/coma-toast/ast-context-cache/pull/55) in GitHub (do not ask the agent to merge).  
2. Run **`proj-impl`** (or start Phase 0–1 on a new branch from updated `main`).  
3. Keep the PRD and this plan linked in the v3.0 PR description.
