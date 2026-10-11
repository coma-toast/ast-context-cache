# PRD: ast-context-cache 5.0, token savings, memory, and repair

- **Date:** 2026-10-09
- **Status:** Draft
- **Related:** research report `reports/Agent memory and token techniques.md` (techniques 1–16, tiers 1 and 2); exploration notes for v4.0.11 (440defe)
- **Target release:** 5.0.0, delivered in phases P0–P4, one PR and release per phase

## Problem statement

ast-context-cache exists to cut the tokens AI coding agents spend on code and context, but several of its own behaviors work against that. It returns more than agents need, in a format that costs extra tokens, and its tool list and responses are not stable enough for hosts' prompt caches. Its "Tokens saved" number mixes unrelated activity and is estimated as bytes ÷ 4. Its memory hard-deletes history, misses paraphrased conflicts, trusts everything equally, and has no integrity checks or backups. External research (context rot, observation masking, ACE, Mem0/Graphiti, memory-poisoning studies, SQLite recovery) points to specific, mostly small changes that fix this.

The users are developers who run ast-context-cache locally for Claude Code, Cursor, Codex and other MCP hosts, and the agents that call its tools.

## Goals

1. Fewer tokens per useful answer: smaller, more precise results, a cheaper format, and large results that stay retrievable without staying in the agent's context.
2. Host prompt caches survive: a stable tool list and byte-stable responses for identical requests.
3. Honest measurement: savings counted with a real tokenizer, split into separate ledgers, against stated baselines, and checked by a repeatable benchmark.
4. Memory that keeps history and stays current: invalidate instead of delete, restore, world time, recency-aware recall, conflict candidates, and staleness tied to code.
5. Data that heals itself: integrity checks, daily snapshots, automatic recovery that never discards data, provenance, and quarantine for untrusted instructions.
6. Context that survives compaction and coordinates subagents: compaction hooks, pinned blocks, playbook notes protected from collapse, and a handoff decision log.

Success means the benchmark harness shows fewer tokens per scenario than the P0 baseline after each phase, the coding-memory eval passes, and no phase regresses an existing test.

## Non-goals

- Research tier 3 techniques: the tiered `ctx_*` cold archive and portable export, embedding epochs and blue-green re-embedding, idle-time consolidation, tool search or a tools-as-code facade, cAST chunking, and enriched references with graph-tool nudges.
- Running an LLM inside ast-context-cache. Every decision that needs judgment (conflict resolution, summaries) is returned to the calling agent.
- An LSP backend.
- Implementing `export_bundle` / `import_bundle`. They are removed from `tools/list` instead (see TS-6).
- Off-machine backup (for example Litestream). Snapshots are local.
- Renaming the project.
- Changing how hosts compact. Only hooks and recipes are in scope.

## Conventions

- **Flags:** every behavior change listed under "Flag" ships behind a feature flag in the existing `internal/flags` registry (env `AST_FEATURE_*` locks it; dashboard Settings → Features toggles it). Default on unless stated.
- **No new tools:** new capabilities are new actions or arguments on existing tools. The only change to the tool list is removing the two unimplemented bundle stubs.
- **Tokens:** "tokens" means the count from the new tokenizer (TS-20) once P1 lands, and the existing estimate before that.
- **Added during refinement:** requirements marked † were added during refinement, not requested directly. Please review them.

## Functional requirements

### P0: Fixes and foundations

**Bug fixes**

- **BF-1** MUST: Vector search results are sorted by score descending whatever the candidate count. Today `topMatches` leaves them unsorted when there are fewer candidates than the limit.
- **BF-2** MUST: Every hybrid, vector, BM25 and retrieve ranking breaks score ties by file path, then start line, then symbol name, so identical inputs give identical order.
- **BF-3** MUST: `recall_memory` returns at most `limit` entries.
- **BF-4** MUST: The `invalidate_previous` argument on `store_memory` takes effect: when false, an exact subject+predicate match does not supersede.
- **BF-5** MUST: All memory and context timestamps compare correctly regardless of format (`YYYY-MM-DD HH:MM:SS` vs RFC 3339), including `as_of` and prune cutoffs, to the second.
- **BF-6** MUST: Deleting a context note by flush, eviction or orphan purge also deletes its revision rows.
- **BF-7** MUST: `lru_session` eviction evicts the least recently accessed note first, not the oldest created.
- **BF-8** MUST: Project- and global-scoped memory stored from one session can be found by the vector path from another session.
- **BF-9** MUST: Memory promoted from `handoff complete` and from the SubagentStop hook is embedded like any other memory.
- **BF-10** MUST: The dashboard "Tokens saved" total counts only compression and dedup savings from search and context-read tools. Writes to virtual context or memory and handoff completions are not counted. Docs and skills match.
- **BF-11** MUST: `recall_memory` ranks FTS hits by text relevance, not by access count alone. Final ranking follows MEM-6.
- **BF-12** MUST: When an embedder is configured, the vector path in `recall_memory` runs alongside FTS, fused like `search_context`, instead of only when FTS and LIKE both return nothing.
- **BF-13** † SHOULD: `forget_memory all=true` respects `scope`, `session_id` and `project_path` when they are given. It invalidates everything only when none are given and `confirm=true` is passed.
- **BF-14** † SHOULD: `get_context_capsule`'s `limit` argument controls the number of candidates instead of the hard-coded 30.

**Database hygiene**

- **DB-1** MUST: Every pooled connection has the intended pragmas applied: WAL, `busy_timeout`, `synchronous=NORMAL`, `cache_size` and `wal_autocheckpoint`. Today only one pooled connection gets the last three.
- **DB-2** MUST: Foreign keys are enforced on every connection, with existing violations reported (not deleted) at first start.
- **DB-3** MUST: Each database records a schema version (`PRAGMA user_version`), and schema changes run as numbered, idempotent steps.
- **DB-4** MUST: All 5.0 schema changes are additive (new tables and columns only), so a 4.x binary still starts and runs against a 5.0 data directory, ignoring the new fields.
- **DB-5** † MUST: Before the first 5.0 schema step runs, a `VACUUM INTO` copy of `context.db` and `usage.db` is written to `<data dir>/snapshots/pre-5.0/`. If it fails, the upgrade does not proceed.

**Measurement**

- **ME-1** MUST: A token benchmark harness (`make bench-tokens`) runs a fixed set of scenarios against a pinned fixture repo and reports tokens returned per scenario, results per scenario, and totals. It runs in CI without an LLM or a network connection. Scenarios cover symbol search, file context, retrieve, recall and handoff open.
- **ME-2** MUST: The harness output is committed as a baseline in P0. Each later phase's PR includes the new numbers and the change from baseline.

### P1: Token savings

**Tool surface**

- **TS-1** MUST: On legacy-era (session) connections, `tools/list` is fixed at session start. Flag, tier or `tools.json` changes reach that session only after it reconnects.
- **TS-2** MUST: On stateless (2026-07-28) connections, flag changes made within a 5-second window send at most one `notifications/tools/list_changed`. Toggling `feature_handoff` sends one frame, not three.
- **TS-3** MUST: Tool order, names, schemas and descriptions are byte-identical across restarts for the same flags, tier and `tools.json`.
- **TS-4** MUST: Settings → Features warns, before a tool-affecting change is saved, that connected agents will lose their prompt cache and that legacy sessions see the change only after reconnecting.
- **TS-5** SHOULD: Tool descriptions contain no counts, dates, project names or other values that change between releases. This excludes documented limits.
- **TS-6** MUST: `export_bundle` and `import_bundle` are removed from `tools/list` until implemented.
- **TS-7** SHOULD: Tool descriptions for the impact tools (`get_impact_graph`, `diff_impact`, `check_deletion_safety`) say when to call them, for example "before changing or deleting an exported symbol".

**Deterministic responses**

- **DR-1** MUST: Two identical requests with identical index, session and memory state return byte-identical result text.
- **DR-2** MUST: Wall-clock timings (`search_time_ms`, per-stage ms), `cache_hit` and `total_vectors` are not in the result text by default. They go in the result's `_meta` and are still logged. `debug=true` puts them back in the text.
- **DR-3** MUST: Stable content (results) comes before variable content (stats, hints) in every response.
- **DR-4** MUST: The `ctx_*`, `mem_*` and `hof_*` stub formats are frozen and documented as stable.
- **DR-5** Flag: `feature_stable_responses`.

**Output format**

- **OF-1** MUST: Code tools (`get_context_capsule`, `search_semantic`, `get_file_context`, `retrieve`) default to a compact text format: one header line per symbol (file:lines, kind, name, score) followed by the raw, unescaped source or skeleton in a fenced block. Response-level metadata goes in a short header line or in `_meta`.
- **OF-2** MUST: `format=json` returns today's JSON structure (minus the DR-2 fields), for scripts and `execute_code`. `code_script_hints` work with `format=json`.
- **OF-3** MUST: `retrieve` returns the assembled context once, with per-chunk metadata (file, lines, score, source) and no repeated chunk content by default. `include_chunks=true` restores chunk content.
- **OF-4** Flag: `feature_text_format`.

**Precision**

- **PR-1** MUST: Code search drops results whose score falls below a relative cutoff from the top score, and every response reports `withheld` (count and tokens) when results were dropped. The cutoff is configurable per tool and overridable per call (`min_relative_score`).
- **PR-2** MUST: When the top score is below a weak-match threshold, code search returns an explicit no-match (`no_match: true`, the best score, and a hint) instead of filling the budget. Docs search keeps its existing floors.
- **PR-3** MUST: Results in test, mock, vendored and generated paths, and same-signature duplicates, are collapsed into one entry with "also N similar in …" (paths listed, no source). Detection uses built-in per-language patterns (for example `*_test.go`, `test_*.py`, `*.spec.ts`, `__mocks__/`, `mocks/`, `vendor/`, `node_modules/`, `testdata/`) plus a configurable glob list.
- **PR-4** MUST: Collapsing (PR-3) is skipped when the query targets tests (contains "test", "spec" or "mock") or when the caller passes `collapse=false`.
- **PR-5** MUST: `output=locations` on code search tools returns only `{file, start_line, end_line, kind, name, score}` per hit, ranked for precision.
- **PR-6** SHOULD: Budget packing skips a result that doesn't fit and keeps trying smaller later results, instead of stopping at the first overflow.
- **PR-7** Flag: `feature_relevance_floor` (covers PR-1, PR-2 and PR-3).

**Modes**

- **MO-1** MUST: `auto` returns full source for the top 3 hits and skeletons for the rest, and never summaries. On `get_file_context`, `auto` behaves like `skeleton`, not `full`.
- **MO-2** MUST: `mode=edit` returns the exact source span of the target symbol plus only the signatures of its direct callees. There is no skeleton padding for the rest of the file and no summaries.
- **MO-3** MUST: Documentation and tool descriptions describe `summary` as navigation-only.
- **MO-4** SHOULD: Session dedup records the mode a symbol was sent in. A later request for `full` or `edit` is not deduped against an earlier skeleton.
- **MO-5** Flag: `feature_mode_v2`.

**Self-offloading results**

- **OL-1** MUST: When a tool result exceeds the offload threshold (default 2,000 tokens; a setting, plus per-call `offload=false`), the full result is stored as a `ctx_*` note with `kind=offload`. The response is a head (the highest-ranked results that fit within the threshold) plus a first line such as `[ctx_… <tool> <target>, <N> tok]`.
- **OL-2** MUST: Offload notes use a separate quota bucket, never evict the agent's own notes, and expire 24 hours after last access or when their session's notes are flushed.
- **OL-3** MUST: `fetch_context` on an expired offload ref returns a clear `expired` result with the original tool and arguments, so the agent can re-run them.
- **OL-4** SHOULD: Docs include host guidance for context-clearing features, for example: keep small ast-context-cache tools such as `recall_memory` and `fetch_context`, and clear bulky results first.
- **OL-5** Flag: `feature_result_offload`.

**Tokenizer and ledger**

- **TL-1** MUST: Token counts use an embedded o200k-style BPE vocabulary in pure Go, offline, with no network access. All counts are labelled as estimates of LLM tokens.
- **TL-2** MUST: The dashboard and `/api/dashboard/stats` report three separate ledgers:
  - compression savings (full-source baseline minus tokens returned);
  - dedup savings;
  - virtual context and memory activity (tokens stored, fetched and recalled).

  Only the first two are "Tokens saved".
- **TL-3** MUST: Each ledger shows its baseline definition. A conservative baseline is shown next to the full-source one, defined as the tokens of the matched files' relevant line ranges an agent would have read after a grep.
- **TL-4** MUST: Historical rows are recounted with the new tokenizer where the original text is still available, for example via stored arguments or a re-render. Rows that can't be recounted are flagged `estimate_method=bytes4` and shown as such.
- **TL-5** SHOULD: An opt-in setting, off by default, reads Claude Code transcript usage blocks under `~/.claude/projects` locally, read-only. The dashboard then shows real input, cache-read and cache-write token totals per day, so periods before and after a change can be compared. Transcript text is never stored; only usage numbers are.
- **TL-6** MUST: The weekly digest and Prometheus metrics use the new ledgers.

### P2: Memory

**History, archive and restore**

- **MEM-1** MUST: Supersession, `forget_memory`, TTL expiry and quarantine rejection invalidate rows with an `invalidated_reason` of `contradiction`, `user_forget`, `ttl`, `stale_anchor` or `quarantine`. They never delete.
- **MEM-2** MUST: Invalidated rows leave default recall, FTS and vector search, and stay queryable through history (MEM-3) for 365 days (setting), after which a daily job deletes them permanently.
- **MEM-3** MUST: `recall_memory` accepts `history=true` with a `ref` or a subject+predicate, and returns the full version chain with reasons and timestamps.
- **MEM-4** MUST: `forget_memory` accepts `action=restore` with refs. Restoring a superseded fact makes it active and invalidates the fact that superseded it, unless that fact has been superseded again, in which case it reports a conflict and changes nothing.

**World time**

- **MEM-5** MUST: `store_memory` accepts an optional `valid_at` and `invalid_at` (when the fact is true in the world). When absent, world time equals system time. `as_of` queries use world time when present, and `as_of_system=true` asks what was known at that time.

**Ranking, TTL and index-first output**

- **MEM-6** MUST: Recall ranking is relevance multiplied by a recency factor clamped to 0.3–1.5x. The factor is built from last access, access count and age, with half-lives of 1 day for session scope, 30 days for project scope and none for global scope (all settings).
- **MEM-7** MUST: Procedural rules (`kind=procedure`) are exempt from decay by default (setting).
- **MEM-8** MUST: `store_memory` accepts `ttl` (duration) or `expires_at`. Expired facts are invalidated with reason `ttl` (MEM-1).
- **MEM-9** SHOULD: `recall_memory debug=true` lists entries that matched but were demoted below the budget, with their scores and the reason.
- **MEM-10** MUST: `recall_memory` and `list_context` default to one line per entry (ref, short label or fact line, trust and staleness flags) under a hard cap. `expand=true` or `fetch_context` returns full bodies.
- **MEM-11** Flag: `feature_memory_v2` (MEM-6 to MEM-10).

**Conflict candidates**

- **MEM-12** MUST: `store_memory` (for facts) searches active facts in the same scope, plus repo siblings for project scope, for semantically similar entries (top 10 by similarity of subject+predicate+object). It returns `possible_conflicts` with refs, fact lines and similarity. It never invalidates on similarity alone.
- **MEM-13** MUST: The similarity lookup embeds synchronously with a 300 ms timeout (setting). On timeout or with no embedder, it falls back to FTS candidates and says so (`conflict_method: fts`).
- **MEM-14** MUST: Exact subject+predicate supersession keeps working as today, subject to BF-4.

**Pinned block**

- **PIN-1** MUST: Each project has at most one pinned note (`store_context kind=pinned`), capped at 500 tokens and versioned. A new store replaces the previous version (revision kept).
- **PIN-2** MUST: The pinned note is returned by `open_handoff` and by the SessionStart hook briefing (CH-3).
- **PIN-3** SHOULD: Tool descriptions and skills tell agents to change the pinned note rarely, because changes move the host's cached prefix.
- **PIN-4** MUST: The dashboard can view and edit the pinned note per project.

### P3: Repair and trust

**Health checks**

- **HC-1** MUST: `PRAGMA quick_check` runs on all three databases at startup and daily. `integrity_check` and `foreign_key_check` run weekly and on demand.
- **HC-2** MUST: The daily check also counts orphans: edges to missing symbols, vectors whose symbol or note is gone, dangling `superseded_by`, revisions without notes, and memory or context anchors to deleted projects.
- **HC-3** MUST: Results appear on a dashboard Health & snapshots card, in `/health` (`db_ok` per database), and as Prometheus gauges (`astcache_db_integrity_ok{db}`, `astcache_db_orphans{kind}`, `astcache_snapshot_age_seconds`, `astcache_disk_free_bytes`).

**Snapshots**

- **SN-1** MUST: A daily `VACUUM INTO` snapshot of `context.db` and `usage.db` is written to `<data dir>/snapshots/` (path is a setting). The newest 7 are kept (setting). Snapshots are not taken when disk space is at the existing "low" level or below.
- **SN-2** MUST: A snapshot is also taken before any schema step and before any restore.
- **SN-3** MUST: Each snapshot passes `quick_check` before it counts toward retention. A snapshot that fails is deleted and reported.
- **SN-4** SHOULD: The dashboard can take a snapshot now and list snapshots with size and age.

**Auto-heal**

- **AH-1** MUST: If `index.db` fails a check, it is moved aside to `snapshots/corrupt-<timestamp>/` and rebuilt from source automatically.
- **AH-2** MUST: If `context.db` or `usage.db` fails a check, healing runs automatically:
  1. MCP writes are paused, and write tools return a clear `repairing` error.
  2. The corrupt file is moved to `snapshots/corrupt-<timestamp>/`, never deleted.
  3. SQLite recovery is attempted into a new file. If the result passes `integrity_check`, it is used.
  4. Otherwise the newest passing snapshot is restored.
  5. Writes resume.
- **AH-3** MUST: After any heal, the system reports what was lost (the snapshot age or recovery gaps, plus counts of notes, memory and handoffs present before vs after, where determinable). The report appears on the dashboard, in the log, and in the next tool response for each active session as a one-line notice.
- **AH-4** MUST: Corrupt copies are kept until the user removes them on the dashboard.

**Provenance and quarantine**

- **TR-1** MUST: Every new `mem_*` and `ctx_*` row records an `origin`: `agent`, `user` (dashboard edits), `handoff`, `hook`, `extract`, `fetched_doc:<url>` or `offload`. It also records a sha256 of its content.
- **TR-2** MUST: Content derived from fetched docs or web content is trust tier `untrusted`. Everything else is `trusted` by default.
- **TR-3** MUST: An untrusted entry whose text looks like an instruction (imperative rule, URL, or shell command) is stored as `quarantined`. It is excluded from default recall and search, returned only with `include_quarantined=true` and a warning label, and listed in a dashboard review queue with approve and reject. Reject invalidates it with reason `quarantine` (MEM-1). It is never silently dropped.
- **TR-4** MUST: `fetch_context` and recall verify the sha256 and flag any row whose content doesn't match (`integrity: mismatch`).
- **TR-5** SHOULD: A burst of near-duplicate memory writes from one session (more than 10 within 60 seconds above 0.9 similarity, settings) is flagged on the review queue.

**Code-anchored staleness**

- **AN-1** MUST: `store_memory` and `store_context` accept an optional `anchor` (file, symbol fqn or name, project). The server records the symbol's line range and content fingerprint at store time.
- **AN-2** MUST: When a stored note or fact names a symbol that was returned to the same session earlier and no explicit anchor is given, an inferred anchor is recorded and marked `inferred`.
- **AN-3** MUST: After the file watcher reindexes a file, anchors into that file are classified as fresh, moved, modified, deleted or file_missing, using the same classification as handoff snapshot pointers. Entries whose anchors are modified, deleted or file_missing are marked `suspect_stale`.
- **AN-4** MUST: `suspect_stale` is advisory. It is shown in recall and list output and on the review queue, and it never invalidates or deletes on its own. Fresh again clears the flag.
- **AN-5** SHOULD: The dashboard reports a staleness rate: the share of anchored entries that are suspect.

**Review queue UI**

- **UI-1** MUST: One dashboard review queue lists quarantined entries, suspect_stale entries and near-duplicate bursts. Each row has approve or keep, reject or invalidate, and open history actions.

### P4: Context and agents

**Notes**

- **NT-1** MUST: Any `rewrite` or edit that would shrink a note by more than 50% (setting) is rejected with `shrink_guard`, unless `force=true` is passed. The response states the before and after sizes.
- **NT-2** MUST: `kind=playbook` notes hold bullets with stable IDs (`ctx_x#b12`), a section (`strategy`, `pitfall`, `command` or custom), and helpful and harmful counters.
- **NT-3** MUST: `edit_context` on playbooks supports `add_bullet`, `update_bullet`, `vote` (helpful or harmful), `merge` and `retire` (soft). Whole-body rewrite of a playbook requires `force=true`.
- **NT-4** MUST: `add_bullet` checks for near-duplicate bullets in the same playbook by embedding similarity, with an FTS fallback. Above the threshold it returns the existing bullet and does not add, unless `allow_duplicate=true`.
- **NT-5** MUST: Budgeted fetches of a playbook return bullets ranked by net votes and recency, and report how many were withheld.
- **NT-6** SHOULD: An optional Claude Code hook on `PostToolUse` / `PostToolUseFailure` for Bash detects test, lint and build commands using configurable patterns. Success comes from PostToolUse; failure comes from PostToolUseFailure's `Exit code N`. The hook records a helpful vote on success, or a harmful vote on failure, on playbook bullets cited in that session since the previous such command. A bullet is cited when its ID appeared in a response or in a store/edit call. The hook is behind `feature_playbook_votes_hook`, default off.
- **NT-7** Flag: `feature_playbooks`. NT-1 is always on.

**Compaction**

- **CH-1** MUST: A Claude Code `PreCompact` hook stores the transcript since the last checkpoint, losslessly, as a `ctx_*` note (`kind=transcript_archive`). It is read from `transcript_path` and contains no secrets beyond what the transcript holds. The hook never blocks compaction.
- **CH-2** MUST: A `PostCompact` hook stores the host's `compact_summary` as `kind=compaction_checkpoint`, with FACT and RULE extraction (origin `hook`).
- **CH-3** MUST: The `SessionStart` hook with `source=compact` injects a briefing within a 1,200-token cap: the latest checkpoint ref, the pinned note, and the one-line index of recent `ctx_*` and `mem_*` entries for the session and project. The same briefing, without the checkpoint, is injected at `source=startup` and `resume`.
- **CH-4** MUST: The installer offers these hooks under the existing hooks component and flag (`feature_handoff_hooks`, or a new `feature_compaction_hooks`; to be settled in planning). They fail open within the existing timeout.
- **CH-5** SHOULD: Equivalent hook recipes ship for Codex (PreCompact, PostCompact, SessionStart) and Cursor (preCompact, sessionStart) where their hook APIs support it. Where a host has no hook API, the docs give manual instructions.
- **CH-6** MUST: `kind=plan` notes accept `recite=true`. While the plan is not marked done, search responses for that session end with one line, at most 30 tokens, stating the current goal.

**Handoff**

- **HO-1** MUST: The scratchpad accepts `post type=decision` with a `topic` and a `choice`. `read` returns decisions first.
- **HO-2** MUST: When two active decisions in the same tree share a topic and differ in choice, both are flagged `conflicting`. The flag is shown to every sibling on read and on the dashboard tree.
- **HO-3** MUST: `open_handoff` accepts `inherit_seen=true` in fresh mode to seed the child's dedup set from the parent's manifest, as fork mode does today.
- **HO-4** MUST: The handoff return stub always includes the result `ctx_*` ref, including when the summary is truncated.
- **HO-5** SHOULD: The dashboard shows tokens per handoff tree (snapshot plus delivered plus results) next to an estimate of a single agent's tokens. The estimate's method is stated.

**Coding-memory eval**

- **EV-1** MUST: A deterministic eval suite (`make eval-memory`), with no LLM, covers knowledge updates (a newer fact supersedes), world-time `as_of` queries, restore, conflict candidates, quarantine exclusion, staleness after code changes, and abstention (no_match when nothing was stored). It runs in CI.

## Non-functional requirements

- **NF-1 Performance:**
  - Token counting adds no more than 2 ms at p95 per response on a 4,000-token result on an Apple Silicon laptop.
  - The conflict lookup adds at most its timeout (MEM-13).
  - Snapshot and check jobs run in the existing maintenance loop and pause or defer when the embed queue is busy or WAL pressure is high.
  - † These targets were added during refinement.
- **NF-2 Storage:**
  - Snapshots skip at low disk.
  - Archived memory and offload notes have the TTLs above.
  - The embedded tokenizer vocabulary adds no more than 5 MB to the binary.
- **NF-3 Safety:**
  - No path deletes user data except the documented TTLs and explicit user actions.
  - Healing never deletes a corrupt file.
  - Every destructive dashboard action asks for confirmation.
- **NF-4 Security:**
  - Quarantine and provenance apply to all write paths.
  - Transcript ingest (TL-5) and transcript archive (CH-1) stay local.
  - The transcript archive respects the existing access-token and trusted-host rules for any remote dashboard access.
- **NF-5 Compatibility:**
  - A 4.x binary runs against a 5.0 data directory (DB-4).
  - Clients that rely on the old JSON shape can pass `format=json`.
  - All flags can be turned off to approximate 4.x behavior, except bug fixes and the shrink guard.
- **NF-6 Observability:** every new mechanism has a log line on state change and a Prometheus metric where it is a rate or a state: offloads, withheld results, quarantine count, stale count, heal events, snapshot age, and ledger totals.
- **NF-7 Determinism:** DR-1 holds under the benchmark harness, which asserts byte equality across two runs.
- **NF-8 Tests:** every requirement has a unit or integration test. `make test`, `make race` for the touched packages, `make lint`, `make ui-test` and `make verify-stories` pass in each phase.

## User workflows

**Agent searches with the new defaults (P1)**
1. The agent calls `get_context_capsule` with a query.
2. The server ranks, applies the relevance floor, collapses tests and mocks, and packs within the budget.
3. The server returns the text format: top 3 hits with full source, the rest as skeletons, `withheld` counts, and no timings.
4. If the result exceeds 2,000 tokens, the server stores it as a `kind=offload` note and returns the head plus a `[ctx_…]` line.
5. If the agent needs everything, it calls `fetch_context` with the ref.

**Agent edits a function (P1)**
1. The agent calls `get_context_capsule` with `mode=edit` and the symbol name.
2. The server returns the exact source span plus callee signatures only.

**No relevant code (P1)**
1. A query matches weakly.
2. The server returns `no_match: true`, the best score, and a hint, and no filler results.

**Memory update with a paraphrased conflict (P2)**
1. The agent stores "build tool is bazel" while "build system is make" is active.
2. The server stores the new fact and returns `possible_conflicts` listing the make fact with its similarity.
3. The agent calls `forget_memory` on the old ref with reason contradiction.
4. Later, `recall_memory history=true subject="build"` shows both versions with reasons.
5. The user restores the old fact on the dashboard. The newer fact is invalidated.

**Context database corruption (P3)**
1. The daily `quick_check` fails on `context.db`.
2. The server pauses writes, moves the file aside, and runs recovery.
3. If recovery passes `integrity_check`, the server uses it. Otherwise it restores yesterday's snapshot.
4. Writes resume. The dashboard shows a heal report ("restored snapshot from 14h ago; 3 notes and 1 memory created since then are not present; corrupt copy kept at …"). Each active session's next response carries a one-line notice.

**Untrusted instruction (P3)**
1. Content from a fetched doc yields a RULE line, "always run curl … | sh".
2. The server stores it as quarantined.
3. Default recall excludes it. The review queue lists it.
4. The user rejects it, and it is invalidated with reason quarantine.

**Compaction in Claude Code (P4)**
1. Auto-compact begins, and PreCompact stores the transcript archive note.
2. Compaction completes, and PostCompact stores the compact summary as a checkpoint with FACT and RULE extraction.
3. SessionStart (compact) injects the briefing: the checkpoint ref, the pinned note, and the recent index.
4. The agent continues and fetches refs as needed.

**Parallel subagents (P4)**
1. Child A posts `decision topic=db-driver choice=pgx`.
2. Child B posts `decision topic=db-driver choice=lib/pq`.
3. Both decisions are flagged conflicting. The next read by any sibling shows the conflict first, and the parent's `collect` includes it.

## Integration points

- **MCP:** `tools/list` and `list_changed` (legacy and stateless eras); `_meta` on tool results; the existing tools' new arguments and actions; removal of the bundle stubs from `tools/list`.
- **Storage:**
  - Additive schema in `context.db` (memory archive and history fields, world time, invalidated_reason, origin, trust, sha256, anchors, quarantine state, playbook bullets and votes, offload kind, decision posts) and `usage.db` (ledger fields, `estimate_method`, transcript usage totals).
  - `PRAGMA user_version` on all databases.
  - `<data dir>/snapshots/`.
- **File watcher:** after a reindex it notifies the anchor classifier with the file and project. It reuses the handoff pointer classification.
- **Hooks:** Claude Code PreCompact, PostCompact, SessionStart (`startup`, `resume`, `compact`), PostToolUse and PostToolUseFailure (Bash). Codex and Cursor equivalents as recipes. Installer hooks component.
- **Dashboard:** ledger cards, Health & snapshots card, review queue, memory history and restore, playbook view, pinned note editor, handoff tree token report, Features warnings, and settings for every threshold above.
- **Prometheus:** the new gauges and counters (HC-3, NF-6).
- **Docs:** README token section, AGENTS.md, skills (usage, operator, agents), MIGRATING.md 5.0 section, host-integration.md hook recipes.

## Acceptance criteria

1. Given a fixture where 3 vector candidates exist and limit is 10, when `search_semantic` runs, then results are in descending score order (BF-1).
2. Given identical index, session and memory state, when the same `retrieve` request is sent twice, then the result text is byte-identical and contains no timing fields, and the timings are in `_meta` (DR-1, DR-2).
3. Given `feature_handoff` is toggled off on a stateless connection, then exactly one `list_changed` frame is sent within 5 seconds. A legacy session's `tools/list` is unchanged until it reconnects (TS-1, TS-2).
4. Given default settings, when `get_context_capsule` returns code, then source appears unescaped in fenced blocks, and `format=json` returns the JSON shape (OF-1, OF-2).
5. Given a query whose top score is below the weak-match threshold, then the response has `no_match: true` and no results (PR-2).
6. Given matches in `foo.go` and `foo_test.go`, when the query does not mention tests, then the test hit is collapsed into "also 1 similar in foo_test.go" (PR-3, PR-4).
7. Given `mode=auto` and 6 hits, then hits 1–3 carry full source, 4–6 carry skeletons, and none carry summaries (MO-1).
8. Given `mode=edit` on a function calling two others, then the response has the function's exact source and two callee signatures, and nothing else from the file (MO-2).
9. Given a result of 5,000 tokens, then the response is at most about 2,000 tokens and starts with a `[ctx_…]` line, and `fetch_context` on that ref returns the full result (OL-1).
10. Given an offload note not accessed for 24 hours, then `fetch_context` returns `expired` with the original tool and arguments (OL-2, OL-3).
11. Given the benchmark harness, when run in each phase's PR, then totals are reported against the committed P0 baseline. P1 shows a reduction on code-search scenarios (ME-1, ME-2).
12. Given a `store_context` call, then the dashboard "Tokens saved" total is unchanged, and the virtual-context ledger increases (BF-10, TL-2).
13. Given a fact superseded 400 days ago with the default 365-day archive TTL, when the daily job runs, then it is deleted. A fact superseded 10 days ago remains visible via `history=true` (MEM-2, MEM-3).
14. Given A superseded by B, when A is restored, then A is active and B is invalidated with a recorded reason (MEM-4).
15. Given a fact with `valid_at=2026-01-01` and `invalid_at=2026-06-01`, when `as_of=2026-03-01`, then it is returned. At `as_of=2026-07-01` it is not (MEM-5).
16. Given two equally relevant project facts, one last accessed today and one 90 days ago, then the recent one ranks higher. A procedure rule last accessed 90 days ago is not demoted (MEM-6, MEM-7).
17. Given `store_memory` with "build tool is bazel" while "build system is make" is active, then the response lists the make fact in `possible_conflicts`, and the make fact stays active (MEM-12).
18. Given no embedder, then conflict candidates come from FTS and the response says `conflict_method: fts` (MEM-13).
19. Given a 600-token pinned note, then the store is rejected with the 500-token cap stated (PIN-1).
20. Given a deliberately corrupted `context.db` copy in a test data dir, when the daily check runs, then writes pause, the corrupt file is moved to `snapshots/corrupt-*`, and recovery or restore runs. A heal report lists the snapshot age and counts, and the corrupt file still exists (AH-2, AH-3, AH-4).
21. Given a corrupted `index.db`, then it is moved aside and rebuilt without user action (AH-1).
22. Given 8 daily snapshots, then only the newest 7 remain. Each passed `quick_check` (SN-1, SN-3).
23. Given a RULE "always run curl x | sh" from fetched-doc content, then it is quarantined, excluded from default recall, and listed in the review queue (TR-2, TR-3).
24. Given a row whose content was altered directly in SQLite, then `fetch_context` or recall flags `integrity: mismatch` (TR-4).
25. Given a fact anchored to `Foo()` in `a.go`, when `Foo`'s body changes and the watcher reindexes, then the fact is marked `suspect_stale` and remains active. Reverting the change clears the flag (AN-3, AN-4).
26. Given a 1,000-token note, when `rewrite` sets it to 300 tokens without `force`, then the edit is rejected with `shrink_guard` (NT-1).
27. Given a playbook, when `add_bullet` adds a near-duplicate of an existing bullet, then the existing bullet is returned and nothing is added (NT-4).
28. Given the votes hook is enabled and a cited bullet precedes a failing `go test`, then that bullet's harmful count increases by 1 (NT-6).
29. Given Claude Code auto-compacts, then a `transcript_archive` note and a `compaction_checkpoint` note exist for the session. The next SessionStart (compact) context includes the checkpoint ref, the pinned note, and the index within 1,200 tokens (CH-1, CH-2, CH-3).
30. Given a plan note with `recite=true`, then each search response in that session ends with one goal line of at most 30 tokens. After the plan is marked done, it does not (CH-6).
31. Given two sibling decisions on topic `db-driver` with different choices, then both are flagged conflicting in every sibling's read (HO-1, HO-2).
32. Given the eval suite, then `make eval-memory` passes in CI (EV-1).
33. Given a 5.0 data directory, when a 4.11 binary starts against it, then it runs normally. A `pre-5.0` snapshot exists from the upgrade (DB-4, DB-5).

## Open questions

1. **Threshold values:** the relative cutoff and the weak-match threshold for PR-1 and PR-2 are tuned with the benchmark harness in P1. Is "tuned to no recall loss on the harness scenarios" the right bar?
2. **Conservative baseline:** TL-3's baseline definition ("grep plus reading the matched ranges") needs a precise, computable rule. Settle it in planning.
3. **Hook flag:** the compaction hooks use either `feature_handoff_hooks` or a new `feature_compaction_hooks` (CH-4). A separate flag seems cleaner.
4. **Transcript archive size and retention (CH-1):** a long session's transcript can be large. Should it use the offload TTL (24h), the ordinary note quotas, or its own (for example 7 days)? Not asked during refinement.
5. **Transcript ingest privacy (TL-5):** reading `~/.claude/projects` touches every project's transcripts. Should ingest be scoped to projects ast-context-cache indexes?
6. **Version-specific hook details:** Claude Code PostToolUse for Bash has no documented exit code; failure detection relies on PostToolUseFailure's `Exit code N` line. Verify with a real failing command before P4.
7. **Hooks enabled by default:** Codex hooks may be on by default and Cursor's preCompact payload is undocumented. Confirm both before writing those recipes.
8. **Auto-heal pause:** with fully automatic healing, an agent mid-task gets `repairing` errors for the duration. Is an upper bound needed, after which the server stays read-only and asks the user? This PRD assumes the heal completes or fails fast, and failure leaves the server read-only with a dashboard alert.
9. **Latency budgets:** these were not selected as a measurement goal. NF-1 sets modest targets anyway; confirm or drop them.
10. **Additive schema vs dedup changes:** changing session dedup to record mode (MO-4) alters how `sessions` rows are read. Confirm it can stay additive.
