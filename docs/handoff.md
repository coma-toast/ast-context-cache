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

### W2: Claude Code with hooks (planned)

The hooks automate steps 3–6 of W1 and recover from a child that stops without completing. See [Claude Code hooks](#claude-code-hooks). Until they are installed, Claude Code uses W1.

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

Every limit resolves as **environment > dashboard setting > default** and is read per call, so a settings change applies to the next operation without a restart. The environment variable is `AST_` plus the setting key in upper case.

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
| `feature_handoff_hooks` | `AST_FEATURE_HANDOFF_HOOKS` | off | Whether the installer offers the Claude Code hooks component. |
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
- **Dashboard tree view.** The Overview groups sessions into handoff trees: label, status, mode, and depth per node, children, tokens delivered and saved, repeat-search rate, claims and queues, last activity, and a tree flush action. Sessions outside trees appear as before.
- **Prometheus** (`http://127.0.0.1:7830/metrics`, `astcache_` prefix):
  - counters `astcache_handoffs_created_total`, `astcache_handoff_children_opened_total`, `astcache_handoff_children_resumed_total`, `astcache_handoff_children_completed_total{status}`, `astcache_handoff_children_abandoned_total`, `astcache_handoff_trees_expired_total`, `astcache_handoff_child_searches_total{repeat}`;
  - gauges `astcache_handoff_open_trees`, `astcache_handoff_open_children`, `astcache_handoff_repeat_search_ratio`, `astcache_query_cache_hit_ratio`;
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
| Claude Code | W1 / W3 / W4 manually; hooks planned (W2) | `~/.claude/CLAUDE.md` block, skills in `~/.claude/skills/` |
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

**Status: planned (Phase 9.7).** The design below follows the capabilities the hook spike confirmed against Claude Code 2.1.278 ([docs/spikes/claude-code-hooks.md](spikes/claude-code-hooks.md)). Until the hooks are installed, use W1 / W3 / W4 manually.

The installer can register Claude Code hook entries when the `feature_handoff_hooks` flag is on (it is off by default): `ast-mcp install --target claude_code --component hooks`. The entries go into `~/.claude/settings.json` under `hooks`, are appended next to your own hooks (never replacing them), run the absolute path of `ast-mcp` as `ast-mcp hook <event>` with a 3-second timeout, and are removed surgically on uninstall.

| Event (matcher) | Subcommand | Planned behavior |
|---|---|---|
| `SessionStart` (`startup\|resume\|compact`) | `session-start` | Tells the agent to use the Claude Code session id as its ast-context-cache `session_id`. After compaction (`source=compact`) it also lists the parent's open handoffs (W6). |
| `SubagentStart` | `subagent-start` | Injects the open digest, or an instruction to open the handoff, into the subagent's context. Forks are skipped because they inherit the parent's window. |
| `SubagentStop` | `subagent-stop` | If the child never completed, stores its final message as a `partial` result. |
| `PreToolUse` (`Agent`) | `pre-tool-use-agent` | Creates a handoff from the parent session and appends the stub to the subagent prompt. |

Hooks always **fail open**: if the server is unreachable, slow (2-second client timeout), returns an error, or a handler is unavailable, the hook exits 0 with no output and the subagent runs normally.
