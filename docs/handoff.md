# Subagent handoff

When a parent agent delegates to a subagent, the subagent starts with an empty window. It re-runs the parent's searches, re-reads the same code, and returns a long report that bloats the parent's window. Subagent handoff fixes both directions:

- **Parent → child:** the parent snapshots what it already explored into a handoff and puts a stub of at most 60 tokens in the subagent prompt. The child opens it, gets a compact digest (at most 1,500 tokens by default), and expands only what it needs.
- **Child → parent:** the child stores its full result as a note and ends with a return stub: a ref plus a summary capped at 300 tokens.
- **Siblings:** children of one handoff share a scratchpad for findings, dead ends, live search trails, and advisory claims on files.

Everything is local SQLite under `~/.astcache/`, like the rest of ast-context-cache. Nothing leaves the machine.

Contents:
- [Concepts](#concepts)
- [Quick start](#quick-start-manual-any-mcp-host)
- [Fork vs fresh](#fork-vs-fresh)
- [Workflows W1–W10](#workflows)
- [Tool reference](#tool-reference)
- [Search annotations](#search-annotations)
- [Errors](#errors)
- [Settings and limits](#settings-and-limits)
- [Feature flags](#feature-flags)
- [Observability](#observability)
- [Security](#security)
- [Host instructions](#host-instructions)
- [Claude Code hooks](#claude-code-hooks)

## Concepts

| Term | Meaning |
|---|---|
| **Handoff** (`hof_…`) | A package a parent session creates for one or more children. The ref is `hof_` plus 16 hex characters (64 random bits). |
| **Tree** (`hft_…`) | The root session plus every handoff and child session descended from it. Retention, caps, the scratchpad, and claims are per tree. |
| **Snapshot** | An immutable copy of the parent's state at create time: the explored manifest, the search trail, included `ctx_*` notes, included and session-scoped `mem_*` memory, and a fingerprint of each pointer. Later parent activity, including `flush_context` on the parent session, does not change or delete it. |
| **Brief** | Free text from the parent: goal, constraints, acceptance criteria. |
| **Pointer** | A symbol key (`file\|name`, or a qualified name) or a project-relative file path the parent marks as relevant, with an optional note. |
| **Explored manifest** | The symbols already returned to the parent session when the snapshot was taken. |
| **Search trail** | The searches a session ran: tool, normalized query, filters, hit count, top hits, zero-hit flag. Captured automatically. |
| **Child session** | The `session_id` the server mints when a child opens a handoff. The child passes it on every later call to every tool. |
| **Scratchpad** | An append-only log shared by the whole tree: `finding`, `dead_end`, `claim`, and automatic `trail` entries. |
| **Claim** | An advisory, FIFO-queued reservation of a key (normally a file path) by one session. Claims never block an edit. |
| **Mode** | `fresh` (default: the child starts with an empty window) or `fork` (the host spawned a fork that inherited the parent's window). |
| **Result** | The child's full output, stored as a `ctx_*` note with `kind=handoff_result`, plus the capped summary returned to the parent. |

## Quick start (manual, any MCP host)

Parent:

```
handoff(action="create", session_id="parent-uuid", project_path="/abs/repo",
        brief="Find why retry backoff ignores the max delay; propose a fix, don't edit.",
        label="retry backoff",
        pointers=[{"key": "internal/retry/backoff.go", "note": "suspect"}])
→ {"handoff": "hof_3f9c…", "tree_id": "hft_…", "depth": 0,
   "stub": "[handoff hof_3f9c…] retry backoff — call open_handoff first",
   "breakdown": {…tokens per section…}}
```

Put the stub in the subagent prompt. The child:

```
open_handoff(handoff="hof_3f9c…", project_path="/abs/repo")
→ {"session_id": "<child id>", "mode": "fresh", "brief": "…", "pointers": [{"id": 1, "key": "…"}],
   "notes": […], "memory": […], "trail": […], "scratchpad": {…}, "tokens_used": 412}

open_handoff(action="expand", handoff="hof_3f9c…", session_id="<child id>",
             section="pointer", items=[1], mode="auto")
… searches with session_id="<child id>" …

handoff(action="complete", session_id="<child id>", status="done",
        content="<full report>\nFACT: retry.backoff | ignores | max_delay when jitter is on",
        summary="Jitter is added after the clamp; move the clamp below it.")
→ {"result_ref": "ctx_…", "stub": "[result ctx_… for hof_3f9c…] done — Jitter is added after the clamp; …", …}
```

The child outputs that stub as its final message. The parent reads the summary, and calls `fetch_context(refs=["ctx_…"])` only if it needs the full result.

The tool descriptions carry the key instruction for children: **"If your prompt contains `[handoff hof_…]`, call `open_handoff` before any search."** The installed skills, Cursor rule, and AGENTS.md / CLAUDE.md block repeat it.

## Fork vs fresh

| Mode | Use when | Dedup on open |
|---|---|---|
| `fresh` (default) | The subagent starts with an empty window: Claude Code `general-purpose` and other non-fork subagents, Cursor subagents, workflow agents, any manual delegation. | Starts empty. Symbols in the parent's explored manifest are still returned, marked `parent_explored: true`. |
| `fork` | **Only** when the host spawned a fork that inherited the parent's window. | Seeded with the parent's explored manifest, so searches skip symbols already in the inherited window and count them as dedup savings. |

The Claude Code hook spike ([docs/spikes/claude-code-hooks.md](spikes/claude-code-hooks.md), row 8) confirmed that a fork inherits the parent's conversation **and** its warm prompt cache: the fork's first request read exactly the parent's cached prefix, while a non-fork subagent read nothing from cache. Using `fork` for a fresh subagent hides symbols it has never seen, so when in doubt use `fresh`.

## Workflows

### W1: Fresh subagent, manual (Cursor, Codex, VS Code, JetBrains, generic MCP)

1. The parent explores with one `session_id` (searches, notes, memory).
2. The parent calls `handoff` `create` with a brief, pointers, and any `ctx_refs` / `mem_refs`, pruning the trail if needed (`exclude_trail`, `exclude_trail_query`, `include_manifest=false`). It receives a `hof_` ref, the stub, and a per-section token breakdown.
3. The parent spawns the subagent with the stub in its prompt.
4. The child calls `open_handoff` and receives its child `session_id` and the digest.
5. The child expands only the pointers, notes, or memory it needs, and searches with its child `session_id`. A repeat of a parent search returns `parent_trail_match`; parent-explored symbols are marked.
6. The child calls `handoff` `complete` with its full result, a status, and an optional summary, and outputs the returned stub as its final message.
7. The parent sees `[result ctx_… for hof_…] done — <summary>` and fetches the full result only if needed.

### W2: Claude Code with hooks (opt-in)

With the [Claude Code hooks](#claude-code-hooks) installed, the parent just spawns subagents with the `Agent` tool:

1. `SessionStart` gives the parent its Claude Code session id as `session_id`.
2. `PreToolUse` creates a handoff from the parent session and appends the stub to the subagent prompt (W1 steps 2–3).
3. `SubagentStart` opens the handoff for the child and injects its child `session_id` and digest (W1 step 4).
4. The child works and completes as in W1 steps 5–6. If it stops without completing, `SubagentStop` stores its final message as a `partial` result.
5. After the parent compacts, `SessionStart` re-surfaces its handoffs (W6).

### W3: Claude Code fork

1. The parent creates a handoff with `mode="fork"` and spawns a fork with the stub.
2. On open, the child's dedup is seeded with the parent's explored manifest.
3. The child's searches skip symbols already in its inherited window, and those skips count as dedup savings.

### W4: Workflow fan-out (N parallel children)

1. The workflow's parent agent creates **one** handoff.
2. The script launches up to 16 agents with the **same** stub. Each opens it and gets its own child session.
3. Child A posts a finding (`scratchpad` `post`, `type="finding"`); child B sees it on its next `scratchpad` `read` with its last `next_cursor`.
4. Child A's searches are appended to the scratchpad as `trail` entries; child B's identical search returns `sibling_trail_match`, served from the shared query cache.
5. Child A claims `internal/x/file.go`. Child B's claim on the same key is queued at position 1 and names A as the holder.
6. A completes; the claim is granted to B automatically, and B's next tool response carries a `[claims_granted] internal/x/file.go (scratchpad)` notice.
7. The parent calls `handoff` `collect` once and receives N summaries plus result refs.

Error path: a 17th open of the same handoff returns `handoff_children_exceeded`.

### W5: Nested handoff

1. A child (depth 1) creates a handoff for a grandchild (depth 2). It is recorded in the same tree and shares its scratchpad.
2. The grandchild completes, and the child collects it.
3. The parent's `collect` with `recursive=true` shows the whole subtree.

Error path: creating a handoff below depth 3 returns `handoff_depth_exceeded`.

### W6: Parent compaction recovery

1. The host compacts the parent and the `hof_` ref is lost.
2. The parent calls `handoff` `list` with only its `session_id` and sees every handoff it created, with per-status counts.
3. The parent calls `collect` to get the results.

### W7: Child crash or abandonment

1. A child stops issuing calls (crash, interrupt).
2. After 30 minutes without MCP activity it is marked `abandoned`, and its claims are released and granted onward.
3. The parent's `collect` shows `abandoned` with the child's stored-note count, and the parent can read the partial notes with `fetch_context`.
4. If the child is resumed (`open_handoff` `resume`), or makes any call again, it returns to `open`.

### W8: Stale pointer

1. After the snapshot, a sibling edits a function the parent pointed to.
2. The child expands that pointer and receives the **current** code with `stale: true`, `change: "modified"`, and the old and new line ranges.
3. If the function was deleted, it receives `stale: true`, `change: "deleted"` (or `moved`, `file_missing`) and no error.

When the child passes a `project_path` that is a sibling worktree of the snapshot's repo (the rule `recall_memory` uses), pointers resolve inside the child's worktree.

### W9: Safe install / uninstall

1. In the dashboard (Settings → Agent integration) or with `ast-mcp install --target <host>`, the user picks a target and components.
2. The preview shows a per-file diff, plus skipped components with reasons and any externally managed paths.
3. The user confirms (Apply, or `--yes`). The installer backs up each file, merges, and re-verifies; the status shows `installed`.
4. If the user later edits our block by hand, the status shows `modified_by_user`.
5. Uninstall previews, then removes only our entries and blocks. Other servers and content remain.

Error paths: an invalid JSON/JSONC/TOML target aborts with a parse error and nothing is written; a file that changed between preview and apply is re-previewed instead of written. Details: [INSTALL.md](INSTALL.md#connect-your-agents) and [host-integration.md](host-integration.md).

### W10: Live flag toggle

1. The user turns off `feature_handoff_claims` in Settings → Features.
2. The next `tools/list` still shows `scratchpad`, but its `claim` and `release` actions return `feature_disabled`.
3. The user turns off `feature_handoff`. All three tools disappear from `tools/list`, a `notifications/tools/list_changed` notification goes to connected clients, and the data is kept until its TTL. Turning the flag back on restores access.

## Tool reference

All three tools are **core** tier, so they are listed at every `AST_MCP_TIER`. Each is also subject to its [feature flag](#feature-flags) and to `tools.json` overrides: a tool is listed only when the flag, `tools.json`, and the tier all allow it.

### `handoff`

Delegate to a subagent and collect its result. `action` is required.

| Action | Caller | Parameters | Returns |
|---|---|---|---|
| `create` | parent | `session_id`, `project_path`, `brief` (required); `label`; `pointers` (`[{key, note}]` or bare key strings); `ctx_refs`; `mem_refs`; `mode` (`fresh` \| `fork`); `exclude_trail` (trail ids, or `["all"]`); `exclude_trail_query` (drop trail entries matching a substring); `include_manifest` (default `true`) | `handoff`, `tree_id`, `depth`, `stub`, `breakdown` |
| `complete` | child | `session_id` (child), `content` (full result); `status` (`done` \| `partial` \| `failed`); `summary`; `changed_files`; `open_questions` | `result_ref`, `status`, `summary`, `summary_source` (`child` or `derived`), `summary_truncated`, `stub`, `promoted_memory`, `released_claims`, `superseded_ref` |
| `collect` | parent | `session_id`, or `handoff` for one handoff; `recursive`; `wait_seconds` (long-poll, at most 60); `token_budget` | `children[]` (child session, label, status, result ref and summary, last activity, active claims, note count), `tokens_used`, `truncated` |
| `list` | parent | `session_id` | `handoffs[]` with `status_counts` |
| `status` | any | `session_id` or `handoff` | tree usage vs caps, `expires_at`, handoffs, active and queued claims |
| `flush` | root or ref holder | `handoff` (that handoff's whole tree), or the root's `session_id` (every tree it rooted) | counts of deleted handoffs, children, notes, memory |

Completion details:
- A summary over `handoff_summary_max_tokens` (300) is truncated and flagged `summary_truncated: true`; completion never fails because of summary length.
- With no summary, the server derives one: `FACT:` / `RULE:` lines first, then the leading content lines (`summary_source: "derived"`).
- `FACT:` and `RULE:` lines in `content` are promoted to `mem_*` entries scoped to the **parent** session (same parsing as `store_context(extract_memory=true)`), with `source_ref` set to the result note. Promoted memory outlives the tree.
- Completing again (for example after a resume) supersedes the earlier result, which stays fetchable until the tree expires.
- The result note counts against the child session's virtual-context quota, so a very large result can return `context_limit_exceeded`.

### `open_handoff`

The child's entry point. `handoff` is required; `action` defaults to `open`.

| Action | Parameters | Returns |
|---|---|---|
| `open` | `handoff`, `project_path`; `token_budget` (default 1,500); `next` (cursor from a truncated digest) | `session_id` (the new child session), `tree_id`, `mode`, `label`, `brief`, `pointers` (ids, keys, notes; no source), `notes` and `memory` (ref, label, token estimate), `trail` digest, `scratchpad` digest (counts by type, latest headlines, dead ends, active claims), `tokens_used`, `truncated`, `next` |
| `resume` | `handoff`, `session_id` (the existing child id) | the same digest for that child; it does not mint a session or count toward the children cap |
| `expand` | `handoff`, `session_id`; `section` (`pointer` \| `note` \| `memory` \| `trail` \| `manifest`); `items` (ids from the digest, or `["all"]`); `mode` for pointers (`skeleton` \| `auto` \| `full`); `token_budget`; `next` | `items[]` (pointer items carry `stale` and `change`), `tokens_used`, `truncated`, `next` |

Everything delivered through `open` and `expand` is recorded as returned in the child session, so a later search does not resend it. The child's `recall_memory` also sees the snapshot's memory entries.

### `scratchpad`

Notes shared across a handoff tree. `action` and `session_id` are required.

| Action | Parameters | Returns |
|---|---|---|
| `post` | `type` (`finding` \| `dead_end`), `text` (at most 500 tokens); `refs` (files, `ctx_*`, symbol keys) | entry `id`, tree usage |
| `read` | `since` (the last `next_cursor`); `types`; `author`; `include_own` (default `false`); `token_budget` | `entries`, `dead_ends` (zero-hit trail entries plus `dead_end` posts), `claims`, `next_cursor` |
| `retract` | `entry` (your own entry id) | the entry is hidden from reads, not deleted |
| `claim` | `key` (project-relative path, symbol key, or any string); `reason` | `outcome` (`granted` \| `queued` \| `held`), `holder`, `holder_label`, `position` |
| `release` | `key` | the next queued session is granted the key |

Claims are **advisory**: the server never stops an agent from editing a file. A claim is released explicitly, when its holder completes, when the holder is marked abandoned, or when the tree expires. A queued claim that would create a wait cycle (A holds X and waits for Y while B holds Y) is rejected with `claim_deadlock_risk` instead of being queued. A grant made while you were waiting is reported on your next call to any tool as a separate content item: `[claims_granted] <keys> (scratchpad)`.

## Search annotations

Searches by sessions inside a tree (`get_context_capsule`, `search_semantic`, `retrieve`, `get_file_context`) gain these fields. Sessions outside a tree see no change.

| Field | Where | Meaning |
|---|---|---|
| `parent_trail_match` | response | The parent already ran this search (same tool, case- and whitespace-normalized query, equal filters). Carries the parent's hit count and top hits. |
| `sibling_trail_match` | response | Another session in the tree ran this search; names the author session and its hit summary. |
| `parent_explored: true` | result item | In a `fresh` child, the symbol was already returned to the parent. |
| `[claims_granted] …` | extra content item | A claim you were queued for was granted since your last call. |

## Errors

Handoff errors are structured, with a stable code:

```json
{"error": "handoff_children_exceeded", "message": "…", "details": {…}, "suggestions": ["…"]}
```

| Code | Meaning | Suggested next actions |
|---|---|---|
| `handoff_not_found` | Unknown `hof_` ref or session. | Check the ref, or call `handoff(action=list, session_id=<parent>)` to find it. |
| `handoff_expired` | The tree passed its TTL without access. | Ask the parent to create a new handoff; trees expire after `handoff_ttl_days` without access. |
| `handoff_depth_exceeded` | A nested create would go below the maximum depth (3). | Do the work in this session instead of nesting another handoff, or raise `handoff_max_depth`. |
| `handoff_children_exceeded` | The handoff already has 16 children. | Resume an existing child with `open_handoff(action=resume, session_id=<child>)`, or create a new handoff for more children, or raise `handoff_max_children`. |
| `handoff_tree_limit_exceeded` | An explicit write would exceed the tree cap (64,000 tokens / 300 entries). The details carry the per-section breakdown or current usage. | Prune the snapshot (`exclude_trail`, `exclude_trail_query`, `include_manifest=false`, fewer `ctx_refs`), flush finished trees with `handoff(action=flush)`, or raise `handoff_tree_max_tokens` / `handoff_tree_max_entries`. |
| `claim_deadlock_risk` | Queuing this claim would create a wait cycle; the cycle is named. | Release one of your claims before claiming this key, or work on another key. |
| `feature_disabled` | The tool or action is turned off by a flag; `details.flag` names it. | Enable the flag in dashboard Settings → Features, or unset its `AST_FEATURE_*` environment override. |
| `invalid_input` | A required argument is missing or malformed (for example `resume` without `session_id`, or an unknown action). | Check the required arguments in the tool's input schema. |
| `not_found` | A referenced note, entry, or session does not exist. | Check the ref or `session_id`. |
| `context_limit_exceeded` | The result note would exceed the child session's virtual-context quota. | Flush older notes in the child session, or shorten the result. |

## Settings and limits

Every limit resolves as **environment > dashboard setting (Settings → Handoff) > default** and is read per call, so a settings change applies to the next operation without a restart. The environment variable is `AST_` plus the setting key in upper case.

| Setting | Env | Default | Effect |
|---|---|---|---|
| `handoff_ttl_days` | `AST_HANDOFF_TTL_DAYS` | 7 | A tree and everything it owns (snapshots, scratchpad, claims, child sessions' notes and dedup rows) expire this long after the last access to any part of the tree. Expiry runs about 2 minutes after start and then hourly. |
| `handoff_summary_max_tokens` | `AST_HANDOFF_SUMMARY_MAX_TOKENS` | 300 | Cap on the summary returned to the parent. |
| `handoff_child_inactive_minutes` | `AST_HANDOFF_CHILD_INACTIVE_MINUTES` | 30 | A child with no MCP activity for this long is marked `abandoned`. |
| `handoff_tree_max_tokens` | `AST_HANDOFF_TREE_MAX_TOKENS` | 64000 | Per-tree token cap across snapshot copies, scratchpad entries, and results. |
| `handoff_tree_max_entries` | `AST_HANDOFF_TREE_MAX_ENTRIES` | 300 | Per-tree entry cap. |
| `handoff_max_depth` | `AST_HANDOFF_MAX_DEPTH` | 3 | Maximum nesting depth below the root session. |
| `handoff_max_children` | `AST_HANDOFF_MAX_CHILDREN` | 16 | Children per handoff; resumes do not count. |
| `handoff_open_budget_tokens` | `AST_HANDOFF_OPEN_BUDGET_TOKENS` | 1500 | Default `token_budget` of the `open` digest. |

When a tree is at its cap, automatic `trail` entries evict the oldest `trail` entries first. Findings, dead ends, claims, results, and snapshots are never evicted automatically; an explicit write that would exceed the cap returns `handoff_tree_limit_exceeded`. Child sessions keep their own virtual-context quota (50 notes / 32k tokens), and the global quotas still apply.

## Feature flags

Flags live in the settings table and resolve as env > setting > default. A non-empty, parseable env value (`true`, `false`, `1`, `0`, …) **locks** the flag: the dashboard shows it read-only with source `env`. Toggling a flag applies live without a restart; flags that change the tool list trigger `notifications/tools/list_changed` to connected clients. Disabling a flag never deletes data.

| Flag | Env | Default | Gates |
|---|---|---|---|
| `feature_handoff` | `AST_FEATURE_HANDOFF` | on | Master switch: all three tools and the search annotations. While it is off, every `feature_handoff_*` flag reads off too. |
| `feature_handoff_scratchpad` | `AST_FEATURE_HANDOFF_SCRATCHPAD` | on | The `scratchpad` tool and the digest's scratchpad sections. |
| `feature_handoff_claims` | `AST_FEATURE_HANDOFF_CLAIMS` | on | The `claim` and `release` actions. |
| `feature_handoff_live_trail` | `AST_FEATURE_HANDOFF_LIVE_TRAIL` | on | Automatic `trail` entries and `sibling_trail_match`. |
| `feature_handoff_hooks` | `AST_FEATURE_HANDOFF_HOOKS` | off | Whether the installer offers the [Claude Code hooks](#claude-code-hooks) component. Hooks already installed keep running when it is turned off; uninstall them to stop them. |
| `feature_shared_query_cache` | `AST_FEATURE_SHARED_QUERY_CACHE` | on | The search candidate cache shared across sessions (including `session_id` calls). |

Manage them in the dashboard (Settings → Features) or over HTTP:

```bash
curl -s http://127.0.0.1:7830/api/dashboard/flags                 # every flag: enabled, source, locked
curl -s -X POST http://127.0.0.1:7830/api/dashboard/flags \
     -H 'Content-Type: application/json' -d '{"key":"feature_handoff_claims","enabled":false}'
```

A POST to an env-locked flag returns 409. Flags are separate from `tools.json`: `tools.json` hides or re-tiers individual tools and is read at startup; flags switch whole features on and off live. A tool must pass both.

## Observability

- **Tokens saved.** Handoff savings are attributed to the `handoff` and `open_handoff` tools in the dashboard's Tokens saved: snapshot content available to a child minus what was actually delivered through `open` and `expand` (OB-2), and a completed result's size minus the summary returned to the parent (OB-3).
- **Dashboard tree view.** The Overview's **Handoff trees** card lists the 20 newest trees with the 24h child repeat-search rate. Each tree row shows the root session, project, token and entry usage against the caps, tokens delivered and saved, repeat-search rate, last activity, chips for child status counts, active and queued claims, and `expired`, plus a flush button (with confirmation). Expanding a tree shows its handoffs (label, ref, mode, depth) and their children (status, session, label, tokens delivered of available, tokens saved, repeat searches, last activity, result summary, claims), with nested handoffs under the child that created them. The same data is at `GET /api/dashboard/handoff-trees?limit=N` (default 20, max 100; `{"trees", "limits", "repeat_search_ratio_24h"}`), and `POST /api/dashboard/handoff-trees/flush` with `{"tree_id": "hft_…"}` flushes a tree as the `flush` action does.
- **Dashboard settings.** Settings → **Handoff** edits the [limits](#settings-and-limits); a value set by its `AST_HANDOFF_*` env var is shown read-only. Settings → **Features** toggles the [flags](#feature-flags).
- **Prometheus** (`http://127.0.0.1:7830/metrics`, `astcache_` prefix):
  - counters `astcache_handoffs_created_total`, `astcache_handoff_children_opened_total`, `astcache_handoff_children_resumed_total`, `astcache_handoff_children_completed_total{status}`, `astcache_handoff_children_abandoned_total`, `astcache_handoff_trees_expired_total`, `astcache_handoff_child_searches_total{repeat}`;
  - gauges `astcache_handoff_open_trees` (trees accessed within the TTL), `astcache_handoff_open_children`, `astcache_handoff_repeat_search_ratio` (children active in the last 24h), `astcache_query_cache_hit_ratio`;
  - histograms `astcache_handoff_tree_tokens`, `astcache_handoff_claim_wait_seconds`.
- **Repeat-search rate** (OB-1): a child search is a repeat when it matches a parent-trail entry, or when at least half of its pre-dedup results are in the parent's explored manifest.
- **Logs.** Every lifecycle transition (create, open, resume, complete, abandon, expire, flush, claim grant, claim release) is a structured `slog` event carrying the tree id, handoff ref, parent and child session ids, and project path. Set `AST_LOG_FORMAT=json` for machine-readable logs.

## Security

- The trust model is unchanged from virtual context: **anyone who holds a `hof_` ref can open the tree and read it**, and any session in the tree can read the whole scratchpad. Refs carry 64 random bits so they can't be guessed, but they are not credentials.
- **Do not put credentials, tokens, or other secrets in a brief, pointer note, included note, result, or scratchpad post.** Handoff data is stored in plain SQLite under `~/.astcache/` until the tree expires or is flushed.
- The MCP and dashboard servers bind `127.0.0.1` by default and reject foreign `Origin` / `Host` headers. Binding another address with `--listen` / `AST_LISTEN` exposes handoff data to that network.

## Host instructions

The installer ships the same guidance to each host it supports (`ast-mcp install --target <host>`, or Settings → Agent integration): an always-on block or rule, plus the skills where the host has a skills directory. The canonical text lives in [`instructions/agents-block.md`](../instructions/agents-block.md), [`rules/cursor/ast-context-cache.mdc`](../rules/cursor/ast-context-cache.mdc), and [`skills/usage/SKILL.md`](../skills/usage/SKILL.md#subagent-handoff).

| Host | Handoff path | What the installer adds |
|---|---|---|
| Claude Code | W2 with the opt-in hooks; W1 / W3 / W4 by hand | `~/.claude/CLAUDE.md` block, skills in `~/.claude/skills/`, and the hooks in `~/.claude/settings.json` when `feature_handoff_hooks` is on |
| Cursor | W1 (no equivalent hooks) | `~/.cursor/rules/ast-context-cache.mdc` (always-apply rule), skills in `~/.agents/skills/` unless already loaded from `~/.claude/skills/` |
| Codex | W1 | `~/.codex/AGENTS.md` block, skills in `~/.agents/skills/` |
| OpenCode | W1 | `~/.config/opencode/AGENTS.md` block, skills in `~/.agents/skills/` |
| VS Code | W1 | MCP registration only; paste the block below into your instructions |
| JetBrains | W1 | Nothing on disk (UI-only MCP settings); paste the block below into the AI Assistant prompt or project guidelines |
| Claude Desktop | Not applicable (no subagents) | MCP registration only |

### Cursor

Cursor has no subagent hooks, so the always-apply rule carries the manual workflow. When you spawn a Cursor subagent:

1. Call `handoff(action="create", session_id=…, brief=…, pointers=…)` in the parent chat.
2. Paste the returned stub at the top of the subagent's task.
3. The subagent opens it first (the rule and tool description tell it to) and ends with the return stub.

Cursor cloud and background agents that cannot reach a local MCP server are not supported.

### Codex, VS Code, JetBrains (and any other MCP host)

Use W1. If the installer did not place the instruction block for your host, paste this into its instructions (Codex `~/.codex/AGENTS.md`, VS Code custom instructions such as `~/.copilot/copilot-instructions.md`, JetBrains AI Assistant project guidelines or prompt library):

```markdown
- If your prompt contains `[handoff hof_…]`, call `open_handoff` before any search, then pass the child `session_id` it returns on every ast-context-cache call.
- To delegate, call `handoff` with `action=create` (brief, pointers) and put the returned stub in the subagent prompt. Use `mode=fork` only for a host fork that inherited your window.
- A child finishes with `handoff` `action=complete` (full content, status, short summary) and outputs the returned stub as its final message.
- Parallel children share findings and advisory file claims through `scratchpad`. After compaction, `handoff` `action=list` with your `session_id` recovers lost `hof_` refs.
- Never put credentials or secrets in a brief or scratchpad post.
```

## Claude Code hooks

**Status: available, opt-in.** Claude Code can run the W1 steps for you: four hooks hand the session id to the agent, create a handoff for each `Agent` call, open it for the subagent, and save a partial result when a subagent stops without completing. They are built on the behavior the hook spike measured against Claude Code 2.1.278 ([docs/spikes/claude-code-hooks.md](spikes/claude-code-hooks.md)). Without them, Claude Code uses W1 / W3 / W4 by hand.

### Install

1. Turn on the `feature_handoff_hooks` flag (Settings → Features, or `AST_FEATURE_HANDOFF_HOOKS=true`). It is off by default; while it is off the installer reports the `hooks` component as `unsupported` and never writes it.
2. Install the component (or tick **Hooks** for Claude Code in Settings → Agent integration):

   ```bash
   ast-mcp install --target claude_code --component hooks --dry-run   # preview the settings.json diff
   ast-mcp install --target claude_code --component hooks --yes
   ast-mcp verify --target claude_code                                # hooks: installed
   ```

   With the flag on, a plain `ast-mcp install --target claude_code` includes hooks too, since it installs every supported component.
3. Restart Claude Code so it loads the new hooks.

The installer appends one matcher group per event to `~/.claude/settings.json` → `hooks`, next to your own hooks (it never replaces them). Each runs `<absolute path to ast-mcp> hook <event>` with a 3-second timeout; the path is the binary that ran the installer, so re-run the install after moving or renaming the repo (`verify` reports `outdated`). `ast-mcp uninstall --target claude_code --component hooks --yes` removes only those groups, whatever the flag says. Turning the flag off does not remove installed hooks; uninstall them to stop them running.

The hooks reach the server at `http://127.0.0.1:$AST_MCP_PORT/mcp` (default 7821), or `$AST_MCP_URL` verbatim, read from Claude Code's environment. If the server runs on another port, export the same `AST_MCP_PORT` for Claude Code.

### Events

| Event (matcher) | Command | What it does |
|---|---|---|
| `SessionStart` (`startup\|resume\|compact`) | `hook session-start` | Injects `use session_id=<Claude Code session id>` so every ast-context-cache call in the conversation shares one session. After compaction (`source=compact`) it also lists up to 10 handoffs this session created, newest first, with per-status child counts and a pointer to `handoff` `collect` (W6). If that listing fails, the session id is still injected. |
| `SubagentStart` (all) | `hook subagent-start` | Skips forks (`agent_type=fork`), which inherit the parent's window. For any other subagent it looks for the handoff stub in the subagent's transcript (`<parent transcript dir>/<session_id>/subagents/agent-<agent_id>.jsonl`), falling back to the oldest handoff the `PreToolUse` hook queued for this session (FIFO, so parallel spawns pair up in spawn order). It calls `open_handoff` for the child and injects the child `session_id` and the digest (capped at about 1,200 tokens), with a note not to open it again. With no handoff, or if the open fails, it injects the parent's session id and the instruction to call `open_handoff` if the prompt carries a stub. |
| `SubagentStop` (all) | `hook subagent-stop` | For a subagent `subagent-start` opened a handoff for, checks the child's status with `collect`. If the child is still `open` (it never called `complete`), stores its final message (from the payload, else the transcript tail, else a placeholder) as a `partial` result, so the parent's `collect` shows it. The compaction summarizer's stop event (no `agent_type`) is ignored. |
| `PreToolUse` (`Agent`) | `hook pre-tool-use-agent` | Creates a handoff from the calling session, with the Agent `description` as its label and the description plus the first 1,500 characters of the prompt as its brief (the parent's trail and explored manifest are included as usual), then returns `updatedInput` with every original field and the stub appended to the prompt. It queues the ref for `SubagentStart`. Inside a subagent the handoff is created from that subagent's child session, so it nests in the same tree; a subagent the hooks did not open is left alone. It skips forks and prompts that already carry a `[handoff hof_…]` stub. It never sets `permissionDecision`, so your normal permission prompt for the Agent call still applies. |

### Local state

Pending handoffs and the child session opened for each subagent are kept in `~/.astcache/hooks/`, one JSON file per parent session (named by a hash of the session id) with a per-file lock, since several subagents can start at once. Entries no hook consumed (for example a denied Agent call) are dropped after an hour. The hooks never open the databases; they only call the running server's MCP tools.

### Failure behavior

Hooks always **fail open** and always exit 0. A run is capped at 2 seconds (stdin included), under Claude Code's 3-second timeout. If the server is unreachable, slow, or returns an error, or the handoff tools are turned off, a hook falls back to what it can do without the server (`session-start` and `subagent-start` still inject the session id; `pre-tool-use-agent` leaves the prompt unchanged) or writes nothing, and the agent or subagent carries on as if no hook ran. A malformed payload or a panic also writes nothing. Stdout carries only the hook's JSON response.

Set `AST_HOOK_DEBUG=1` in Claude Code's environment to log each hook's decisions and errors to stderr (honoring `AST_LOG_FORMAT`). To try a hook by hand, pipe a payload into it:

```bash
echo '{"session_id":"s1","hook_event_name":"SessionStart","source":"startup"}' | ast-mcp hook session-start
```
