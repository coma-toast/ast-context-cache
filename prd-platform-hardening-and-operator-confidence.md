# PRD: Platform Hardening & Operator Confidence

- **Date:** 2026-07-23
- **Status:** Draft
- **Related:** Follows recent embed-queue / WAL reliability work (e.g. PR #55); supersedes stale React-migration items in `TODO.md`
- **Version intent:** Breaking cleanup acceptable as **v3.0** with a short migration note

---

## Problem Statement

ast-context-cache is useful locally for agents (token-efficient code context, virtual context across compaction) and for the operator (dashboard, embedders, watchers), but day-to-day confidence is uneven. The embed pipeline sometimes sits at zero workers or the process dies and must be restarted by hand. The React dashboard is primary, yet a large legacy templ/HTMX surface and duplicate APIs remain. Value to agents (tokens saved, virtual-context recoveries) is only partly visible, so it is hard to justify running the tool—especially when comparing cloud-agent cost or local-LLM efficiency on desktop hardware.

This effort hardens reliability, removes redundancy, improves operator UX around embeddings and value metrics, and adds light observability—while staying **local-only** (no cloud product, no multi-machine remote MCP host in this scope).

---

## Goals

1. **Fewer operational failures:** Embed workers and the process recover predictably; “stopped” states are visible and self-healing where safe.
2. **Less redundancy and complexity:** One live UI stack, fewer duplicate APIs, no unused MCP “ghost” tools, clearer maintenance of docs/skills.
3. **Operator confidence:** Dashboard makes it obvious that the system is working and *worth* running (tokens saved, virtual-context use, heuristics for speed/cost).
4. **Related product completeness:** Memory tools documented for agents; richer code-mode scripts; starter doc packs; Prometheus metrics for homelab scraping.
5. **Ship as a coherent v3.0:** Cleanup + reliability + confidence UX + metrics delivered—not an open-ended living roadmap.

---

## Non-Goals (Out of Scope)

- Cloud SaaS, accounts, or hosted multi-tenant service
- Running MCP/dashboard as a shared remote host for multiple laptops (deferred; too complex for path/index sharing)
- Changing the product to require network auth (Tailscale / local trust is enough if ever revisited)
- Major visual redesign or light theme (keep current dark GitHub-like look)
- New language ecosystems beyond documenting what already exists (e.g. HCL) unless trivial; Rust/Java expansion deferred
- Rewriting the entire embed queue from scratch (simplify and harden; do not greenfield)
- Email/ntfy weekly digests (dashboard card only)
- Rigorous A/B measurement of agent wall-clock time (heuristic estimates only)

---

## Personas

| Persona | Needs from this work |
|---------|----------------------|
| **Operator (primary)** | Embeddings/workers always understandable; process stays up; WAL/pause never leaves the system wedged; clear value metrics |
| **Coding agents** | Stable MCP; aux catch-up when primary embedder fails; memory + virtual context discoverable in skills/docs |
| **Other self-hosters (secondary)** | README sufficient to install locally; breaking changes called out in v3.0 notes |

---

## Priority Order (from refinement)

1. Reduce redundancy  
2. Reduce complexity (esp. embed queue + WAL—**first-class**)  
3. Enhance UI/UX (confidence + embeddings)  
4. Add features (memory skills, code-mode, doc packs, Prometheus)

---

## Functional Requirements

### A. Reliability — process & embed pipeline (MUST)

1. The system MUST keep the **aux embedder processing work whenever it is configured and healthy**, even if the primary remote embedder is in `error` / unreachable.
2. After WAL / DB maintenance / embedder-swap pause, the system MUST **restore worker pools to their configured targets** and MUST NOT leave `MaintenancePaused` (or equivalent) stuck such that pending cannot flush.
3. The UI MUST show a clear **paused → restoring → resumed** (or equivalent) lifecycle for maintenance so the operator can see the system is not dead.
4. If embed workers remain unexpectedly at effective **0** while the configured target is &gt; 0 for **≥ 5 minutes** (and not in an intentional operator pause), the system MUST **auto-recover** (restore workers and/or clear stuck pause) and SHOULD surface that recovery in the UI/logs.
5. The product MUST provide a **local keep-alive** path: at least a **watchdog** that respawns `ast-mcp` when the process exits unexpectedly, and a **Docker** option with restart policy documented for users who prefer containers.
6. After an abnormal exit, startup MUST continue to reload pending work, and the dashboard MUST show a **“restarted after crash”** (or abnormal-exit) banner with approximate time when that is known.
7. Intentional operator pause (workers explicitly set to 0) MUST remain distinct from failure/stuck states in the UI.

### B. Redundancy removal — v3.0 breaking cleanup (MUST)

8. The product MUST remove the **legacy templ/HTMX dashboard path** from the shipped binary (generated `*_templ.go` partials that React does not use, `/partials/*` routes, and unused Alpine/HTMX static assets), once React coverage for operator workflows in scope is confirmed.
9. The product MUST collapse **duplicate health/settings read models** so the React app and operators rely on `/api/dashboard/*` (and clearly documented exceptions), not parallel HTML partial endpoints.
10. MCP **ghost tools** not listed in `tools/list` (`sync_remote`, `reset_project`, `reset_all`) MUST be **removed from MCP dispatch** (or equivalent: unreachable). Destructive reset MUST remain available only via **dashboard/operator** controls if still needed. *(v3: removed from dispatch; `sync.go` / `REMOTE_VECTORDB_*` gone; dashboard reset APIs kept.)*
11. Stale roadmap/docs that claim React/templ migration is unfinished or that deny agent memory MUST be **updated or deleted** (`TODO.md`, conflicting comparison docs, agent skill tables).
12. Agent-facing docs (`AGENTS.md` / `CLAUDE.md` / usage skills) MUST document **structured memory** tools (`store_memory`, `recall_memory`, `forget_memory`) and KV repair reporting consistently with the live tool list.

### C. Complexity reduction — embed + WAL (SHOULD / MUST where noted)

13. Embed-queue + WAL behavior MUST remain **first-class** in this effort: operator-visible states, recovery rules, and docs MUST match actual code paths (including throttle vs pause vs error vs crash).
14. The system SHOULD reduce accidental complexity where safe (e.g. noisy repeated flush logging, unclear dual worker controls) without removing needed reliability mechanisms (pending persistence, aux pool, quiet-period maintenance).
15. Operator docs MUST describe the failure modes: workers=0, WAL throttle, maintenance pause, embedder error, process down—and what the user should expect for each.

### D. UI/UX — embeddings & confidence (MUST / SHOULD)

16. The Embeddings / workers experience MUST make **pending, queued, in-flight, primary vs aux, embedder state, and maintenance** understandable without reading logs.
17. The dashboard MUST restore **WAL compaction operator UX** parity with documented behavior: visible maintenance/progress and an explicit **checkpoint / compact** action when the API supports it (not chip-only).
18. Overview (or equivalent) MUST prominently show:
    - **Tokens saved** (existing aggregates + clarity on what is/isn’t counted)
    - **Virtual context recoveries** (fetches/searches that return content; tokens returned)
    - Quantification of **how much agents use virtual context** (counts + tokens; orphan / never-fetched rates)
19. The product MUST add a **heuristic before/after estimate** of value (e.g. full-source / naïve context baseline vs tokens actually returned, plus a simple estimate of rounds or cost avoided)—clearly labeled as approximate.
20. The product SHOULD emphasize **per-tool value** in Activity/Analytics (e.g. capsule vs retrieve vs execute_code) enough to try; refine if noisy.
21. The product MUST add a **weekly digest card** on the dashboard (local only)—summarizing tokens saved, virtual-context usage, embed reliability (failures/restarts), and top tools—for the trailing week.
22. The product SHOULD add **session-level** virtual-context stories (e.g. per `session_id`: stored vs fetched, tokens, whether fetch followed store)—enough to see “agents actually recovered after compaction.”
23. Settings UX SHOULD reduce accidental save-on-blur confusion for embed settings and SHOULD clarify when a setting applies immediately vs needs attention; full Settings parity with every backend field is SHOULD, not a blocker for v3.0 if embeddings/value/WAL are done.
24. Keep the **existing dark** visual language; no light-mode requirement.

### E. Observability (MUST)

25. The dashboard HTTP server MUST expose Prometheus **`/metrics`** (same port as the dashboard by default) including at least: process up; embed queue pending / queued / in-flight / worker targets & live (primary + aux); embedder state; index WAL size (or bytes); tokens_saved (counter or rate-friendly series); MCP tool latency or call counts suitable for Grafana.
26. README / operator docs MUST document example scrape config for local/homelab Prometheus.

### F. Features — related completeness (SHOULD)

27. Agent skills / install docs SHOULD teach **memory** tools alongside virtual context (when to use `mem_*` vs `ctx_*`).
28. Code-mode SHOULD ship a **thicker** set of useful built-in or repo scripts beyond a single compact example (still opt-in via complete tier + code mode).
29. The product SHOULD offer **doc packs** (curated or one-click add of common library doc sources) to reduce empty `search_docs` on a fresh install.
30. README MUST be enough for another person to run locally; no requirement for Homebrew in this PRD.

### G. Explicitly deferred (do not implement under this PRD)

31. Remote shared MCP server for multiple machines  
32. Cloud sync of `~/.astcache`  
33. Auth on MCP/dashboard  
34. Expanding primary language set (Rust, etc.) beyond docs for already-supported extras  

---

## Non-Functional Requirements

1. **Local-only:** All durable state remains under the local data dir (e.g. `~/.astcache`); no mandatory external service.
2. **Breaking changes:** Allowed in **v3.0**; MUST include a short migration note (removed routes/tools, how to use dashboard resets, watchdog/Docker keep-alive).
3. **Performance:** Heuristic value metrics and weekly digest MUST NOT meaningfully stall indexing or MCP tool latency under normal desktop load (compute async or from existing aggregates).
4. **Reliability:** Auto-recover at 5 minutes MUST be conservative (must not fight intentional pause or active maintenance).
5. **Security:** No new open remote attack surface in this PRD; `/metrics` is local-trust (document that binding/scraping is operator-controlled).
6. **Observability:** Metrics naming SHOULD follow Prometheus conventions (`astcache_` or similar prefix).
7. **UX:** Critical embed/maintenance states MUST be visible within a few seconds of WebSocket/API refresh under normal load.

---

## User Workflows

### Happy path — operator confidence
1. Operator starts ast-mcp (or watchdog/Docker keeps it running).
2. Opens dashboard Overview: sees tokens saved, virtual-context usage, weekly digest, embedder healthy, workers live.
3. Agents use MCP; Activity shows per-tool contribution.
4. Operator concludes the tool is worth leaving on.

### Primary embedder down
1. Primary remote embedder enters error.
2. Aux workers continue draining pending.
3. UI shows primary error + aux active + pending decreasing.
4. When primary recovers, both pools operate normally.

### WAL / maintenance
1. Quiet or forced maintenance pauses writers.
2. UI shows compacting/progress (and optional manual checkpoint).
3. Workers restore automatically; UI shows resumed.
4. If restore fails and workers stay at 0 for ≥ 5 minutes (non-intentional), auto-recover fires and is visible.

### Process crash
1. Process exits unexpectedly.
2. Watchdog or Docker restart policy brings it back.
3. Pending reloads; banner notes abnormal restart.
4. Embedding resumes without manual `ast-mcp start` (when keep-alive is configured).

### Agent compaction survival
1. Agent `store_context` / uses memory tools per skills.
2. After host compaction, `fetch_context` / `recall_memory`.
3. Dashboard session-level and aggregate VC metrics reflect recoveries.

### Error path — intentional pause
1. Operator sets workers to 0.
2. UI shows paused (not “failed”).
3. Auto-recover does **not** override within the intentional-pause semantics.

---

## Integration Points

| System | Interaction |
|--------|-------------|
| MCP clients (Cursor, etc.) | Existing MCP URL local; tool list after ghost removal |
| Dashboard React SPA | Primary operator UI; WebSocket refresh |
| SQLite (`index` / `context` / `usage`) | Existing; WAL maintenance hooks |
| Embed backends (local onnx, openai-compatible, etc.) | Unchanged product model; aux catch-up rules |
| Prometheus | Scrapes dashboard `/metrics` |
| Docker (optional) | Image + restart policy for keep-alive |
| OS watchdog (optional) | launchd/systemd or project-provided watchdog script/binary |

**Removed / not integrated in this PRD:** remote VectorDB `sync_remote` MCP path (`REMOTE_VECTORDB_*` env) — **removed in v3**; remote multi-client MCP hosting.

---

## Acceptance Criteria

1. **Given** primary embedder is in error and aux is configured, **when** pending exists, **then** aux processes work and pending trends down without requiring primary recovery.
2. **Given** a WAL/maintenance pause completes or aborts, **when** the operation finishes, **then** worker targets are restored and pending can flush without manual restart.
3. **Given** workers are stuck at 0 with target &gt; 0 for ≥ 5 minutes (not intentional pause / not active maintenance), **when** the watchdog timer fires, **then** workers are restored and the event is visible in UI or logs.
4. **Given** keep-alive is configured (watchdog or Docker), **when** the process crashes, **then** it returns without manual start and shows an abnormal-restart affordance.
5. **Given** a v3.0 build, **when** an operator loads the dashboard, **then** legacy HTMX/templ partial UI is not required for any in-scope workflow, and removed MCP ghost tools are not callable.
6. **Given** the Overview tab, **when** the system has traffic, **then** tokens saved, virtual-context usage/recoveries, and weekly digest are visible and labeled clearly (heuristics marked approximate).
7. **Given** Prometheus scrapes dashboard `/metrics`, **when** embed and MCP activity occur, **then** the agreed metric set updates.
8. **Given** agent docs/skills, **when** an agent follows usage guidance, **then** memory vs virtual context guidance is present and matches `tools/list`.
9. **Given** README install steps, **when** a new local user follows them, **then** they can run MCP + dashboard without undocumented templ/SSE steps.
10. **Given** this PRD’s done definition, **when** the above MUST items ship, **then** the effort is complete (not deferred into an open roadmap doc).

---

## Open Questions

| # | Topic | Recommended default |
|---|--------|---------------------|
| 1 | Exact watchdog form (shell loop vs small supervisor vs launchd plist checked into repo) | Ship a simple supervised start in-repo + Docker Compose restart; document launchd as optional |
| 2 | Whether any personal use of `sync_remote` / `REMOTE_VECTORDB_*` should remain as a hidden env-gated tool | **Remove** (done in v3); migration: use local index only |
| 3 | How aggressive the before/after heuristic should be (include estimated tool-round savings vs tokens only) | Tokens-first + optional simple “rounds avoided” estimate; always label approximate |
| 4 | Session-level VC retention window for stories | Align with existing virtual-context retention / 30d stats already used on dashboard |
| 5 | Docker image publish (GHCR) vs Dockerfile-only in repo | Dockerfile + compose in repo for v3.0; publish image optional later |
| 6 | Whether Settings full parity is required before deleting templ | Delete templ when Embeddings/WAL/Overview confidence flows are covered; remaining rare settings can be API/env |

---

## Decisions Captured During Refinement

- Audience: operator, agents, and potential other self-hosters  
- Priority: redundancy → complexity → UI/UX → features  
- Embed queue + WAL: first-class  
- Aux always continues when primary is down  
- Auto-recover stuck workers at **5 minutes**  
- Local only; **no remote MCP host** in this PRD (revert earlier remote-host idea)  
- Tailscale-only would have been enough for remote auth—but remote hosting dropped  
- Breaking cleanup OK as **v3.0**  
- Theme: keep dark  
- Prometheus on **dashboard port**  
- Weekly digest: **dashboard card only**  
- Value: tokens saved, VC recoveries, agent VC usage, heuristic before/after, try per-tool analytics, session-level VC stories  
- Features to include: memory in skills, thicker code-mode, doc packs  
- Done = ship the above—not a living roadmap  

---

## Suggested Next Step

Run **`proj-plan`** against this PRD to produce a phased implementation plan (v3.0 cleanup → reliability → confidence UI → metrics/features).
