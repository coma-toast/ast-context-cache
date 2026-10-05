# PRD: Subagent Handoff Cache, Feature Flags & Safe Host Installer

- **Date:** 2026-10-05
- **Status:** Draft
- **Related:** None (no ticket). Builds on virtual context (`ctx_*`), structured memory (`mem_*`), and session dedup. Supersedes the dashboard "Agent integration" installer.
- **Version intent:** **v4.0 (major)**. Breaking changes are allowed and listed in [§ Breaking changes](#breaking-changes-v40).

---

## Problem Statement

When a parent agent delegates to a subagent (Claude Code Agent tool, forks, workflows, Cursor subagents, or any MCP host), the subagent starts with an empty window. It re-runs searches and re-reads code the parent already explored. Its result comes back as a long report that bloats the parent's window. ast-context-cache already holds the parent's searches, returned symbols, notes, and memory locally, but there is no way to pass that state to a child, link sessions together, or coordinate parallel children. Every delegation pays for its exploration twice.

Separately, the dashboard installer that wires ast-context-cache into editors is destructive. It overwrites `~/.claude.json` with markdown, and uninstall deletes it. Since the handoff hooks depend on that installer, it is reworked here too.

## Background: current state (verified 2026-10-05)

Facts the requirements depend on. File references are for orientation only.

- **No prior art** for handoff, subagents, delegation, or parent/child sessions anywhere in code, docs, skills, or git history.
- **`session_id` is a free-form string the client invents.** Nothing links sessions together.
- **Session dedup** lives in the usage DB `sessions` table and is keyed by `file|name|start_line`.
  - It has no TTL.
  - Writes go through a buffer that flushes every 3s, so a repeat call within ~3s can return duplicates (`internal/db/writebatch.go`).
  - **The query cache is skipped whenever `session_id` is set** (`internal/context/handler.go:60`).
- **`ctx_*` notes:**
  - Limits are 50 notes / 32k tokens per session and 500 notes / 200k tokens globally. Notes have no TTL.
  - `fetch_context` without `session_id` reads any session's note. This is intentional: the code comment says the trust model is a single local user.
  - `FlushOrphans` runs only from the dashboard, not on a schedule.
- **`mem_*` memory:** scopes are `session`, `project`, and `global`. Session-scoped memory is invisible to every other session.
- **Scope-leak bugs** (confirmed by reading the code):
  - `searchNotesLike` builds `label LIKE ? OR content LIKE ? AND session_id = ?`, so its session and project filters are bypassed by operator precedence (`internal/contextnotes/store.go:617`).
  - Memory `vectorSearch` re-selects by `ref IN (...)` with no validity or scope filter, so it can return superseded, forgotten, or out-of-scope entries (`internal/memory/recall.go:273`).
- **Tier and `tools.json` are read once at startup.** There is no feature-flag system and no MCP `tools/list_changed` support.
  - Settings are a key/value table with precedence env > setting > default. Most keys are read live.
- **Installer** (`internal/dashboard/api.go:978-1120`):
  - Targets: Cursor, OpenCode, Claude Code, Claude Desktop.
  - It overwrites whole files and removes whole files on uninstall.
  - The Claude Code global target writes markdown into `~/.claude.json`.
  - The OpenCode format is likely wrong, and the Claude Desktop entry (`"command":"http"`) is invalid.
  - The port is hardcoded to 7821.
  - Project-scope installs resolve against the server's working directory.
  - Installed state comes only from the DB.
  - It installs no skills, rules, or hooks.
- **Hosts:**
  - Fresh subagents get only their prompt plus CLAUDE.md and skills. They return only their final message and do not share the parent's prompt cache.
  - Claude Code **forks** inherit the parent's conversation.
  - No documented MCP header or `_meta` field identifies the calling agent, so parent/child linkage must be explicit.
  - Hook capabilities are only partly verified (see Open Questions).

## Glossary

| Term | Meaning |
|---|---|
| **Handoff** | A package a parent session creates for one or more children. Referenced by `hof_*`. |
| **Handoff tree** | The root session plus every handoff and child session descended from it. |
| **Parent / child session** | The session that created a handoff, and a session minted when a handoff is opened. |
| **Snapshot** | An immutable copy of the parent's relevant session state, captured when the handoff is created. |
| **Brief** | Free text from the parent: goal, constraints, acceptance criteria. |
| **Pointer** | A symbol key or file path the parent marks as relevant, with an optional note. |
| **Explored manifest** | The set of symbol keys already returned to the parent session when the snapshot was taken. |
| **Search trail** | Searches a session ran: tool, query, filters, hit count, top hit keys, zero-hit flag. |
| **Scratchpad** | An append-only shared log for one handoff tree: findings, dead ends, claims, and live trail entries. |
| **Claim** | An advisory, queued reservation of a resource key (for example a file path) by one session in a tree. |
| **Mode** | `fresh` (the child cannot see the parent's window) or `fork` (the child inherited the parent's window). |
| **Result** | The child's full output, stored as a note, plus a short summary returned to the parent. |

---

## Goals

1. **Children stop redoing parent work.** In the reference scenario, the child repeat-search rate drops by **at least 50%** against a no-handoff baseline.
2. **Handoff costs few tokens in both directions.** Parent → child is a stub of at most 60 tokens plus on-demand expansion. Child → parent is a ref plus a summary capped at 300 tokens by default.
3. **Parallel children coordinate without the parent relaying.** They share findings, dead ends, and live search trails, and use queued claims to avoid collisions.
4. **Works on every target host.**
   - Automatic where host hooks allow it.
   - A manual ref is the universal fallback.
   - Covered hosts: Claude Code (Agent tool, forks, workflows), Cursor, and generic MCP clients.
5. **Value is measurable**: dashboard tree view, token-savings attribution, and Prometheus metrics.
6. **A safe installer** that registers the server and ships skills, rules, and hooks for 7 hosts without ever destroying user config.
7. **A general feature-flag framework** in settings, applied live, which gates this feature and future ones.
8. **The scope-leak and cache-bypass defects** that undermine handoff correctness are fixed.

## Non-Goals (Out of Scope)

- **A truly shared or unified model context window.** The server cannot change what a host sends to the model or how prompt caching works. ("Shared window" is approximated by snapshot + scratchpad.)
- **Access control between sessions.** The current open trust model stays: whoever holds a ref can read it. No tree-scoped authorization is added.
- **LLM summarization inside the server.** Summaries are written by the child or derived mechanically.
- **Cross-machine or remote handoff.** Remains local-only.
- **Cursor cloud or background agents that cannot reach a local MCP server.**
- **Installer targets** Windsurf, Gemini CLI, and Zed (deferred).
- **Project-scope installs.** Dropped; the installer is global-only.
- **Implementing `export_bundle` / `import_bundle`.** They remain stubs.
- **TTL for ordinary `ctx_*` notes and dedup rows outside handoff trees.** Unchanged.
- **Enforcing file locks on the filesystem.** Claims are advisory coordination only.
- **Structured result fields** (changed files, open questions) beyond status + summary + ref. Deferred (see RT-8).

---

## Functional Requirements

Priority keywords follow RFC 2119. Items marked **(refinement)** were added during refinement and are not from the original request. Review them.

### HO: Handoff creation (parent side)

- **HO-1 MUST:** A parent session can create a handoff by supplying its `session_id` and the following inputs:
  - a brief (required, free text);
  - an optional label;
  - an optional list of pointers (symbol keys or project-relative file paths, each with an optional note);
  - optional `ctx_*` refs and `mem_*` refs to include;
  - an optional mode (`fresh` by default, or `fork`).
- **HO-2 MUST:** Creating a handoff captures an immutable snapshot of the parent session at that moment. The snapshot contains:
  - (a) the explored manifest;
  - (b) the parent's search trail;
  - (c) copies of the included `ctx_*` note contents;
  - (d) copies of the included `mem_*` entries plus all of the parent's active session-scoped `mem_*` entries;
  - (e) a content fingerprint of each pointed symbol or file, used for staleness detection.
- **HO-3 MUST:** Later changes to the parent's notes, memory, or dedup state do not alter an existing snapshot.
- **HO-4 MUST:** The search trail is captured automatically from the parent's searches. When creating the handoff, the parent can prune it: exclude specific trail entries, exclude by query substring, or include no trail at all. The parent can likewise exclude the explored manifest.
- **HO-5 MUST:** Creation returns:
  - a handoff ref (`hof_` followed by at least 64 bits of randomness, hex-encoded);
  - a ready-to-paste prompt stub of at most 60 tokens, in the form `[handoff hof_… ] <label> — call open_handoff first`;
  - a size breakdown of the snapshot (tokens per section).
- **HO-6 MUST:** Snapshot creation is atomic. Either the full snapshot is stored, or nothing is stored and an error is returned.
- **HO-7 MUST:** If the snapshot would exceed the per-tree cap (RQ-4), creation is rejected with error `handoff_tree_limit_exceeded`. The error includes the per-section breakdown so the parent can prune and retry.
- **HO-8 MUST:** A child session can itself create a handoff. The new handoff is recorded as a nested node in the same tree.
  - Depth counts from the root session (depth 0). The default maximum child depth is 3.
  - Exceeding it returns `handoff_depth_exceeded`.
- **HO-9 MUST:** One handoff can be opened by many children, 16 by default. Exceeding that returns `handoff_children_exceeded`. Resuming an existing child does not count against this limit.
- **HO-10 MUST:** A snapshot survives `flush_context` on the parent session. Tree data is removed only by TTL expiry (RQ-1) or an explicit tree flush (RQ-3).

### OP: Opening a handoff (child side)

- **OP-1 MUST:** A child opens a handoff with only the handoff ref. The server mints a new child `session_id` linked to the handoff, parent session, and tree, and returns it. The child uses it for every later call to any tool.
- **OP-2 MUST:** Re-opening with the handoff ref plus an existing child `session_id` resumes that child session. It does not mint a new one, does not count against HO-9, and returns a refreshed scratchpad digest. This covers hosts that resume a subagent with its history.
- **OP-3 MUST:** The default open response is compact, at most 1,500 tokens (overridable with `token_budget`). It contains:
  - the brief;
  - the child `session_id` and mode;
  - pointers as keys and notes only, no source;
  - included notes and memory as ref, label, and token estimate;
  - a search-trail digest: queries with hit or zero-hit counts, most recent first, capped;
  - a scratchpad digest: entry counts by type, the latest few headlines, and active claims.

  If content exceeds the budget, the response is truncated, flagged `truncated: true`, and says how to page.
- **OP-4 MUST:** A child can expand any snapshot section on demand:
  - pointer source in `skeleton`, `auto`, or `full` mode;
  - an included note's full content;
  - included memory entries;
  - full trail entries;
  - the explored manifest.
- **OP-5 MUST:** Dedup follows mode:
  - **`fresh`:** the child's dedup state starts empty, and the explored manifest is informational only.
  - **`fork`:** the child's dedup state is seeded with the parent's explored manifest, so later searches skip those symbols and count them as dedup savings.
- **OP-6 MUST (refinement):** Everything delivered to the child through open or expand is recorded as returned in the child's session, so a later search does not resend it.
- **OP-7 MUST:** Staleness:
  - When a pointer is expanded and the current content differs from the snapshot fingerprint, the response returns the **current** code with `stale: true`.
  - It also returns a change description: one of `modified`, `moved`, `deleted`, or `file_missing`, plus old and new line ranges where applicable.
  - A deleted or missing symbol returns `stale: true` with its status. This is not an error.
- **OP-8 MUST:** Worktree mapping: if the child supplies a `project_path` that is a repo-sibling worktree of the snapshot's project (the same sibling rule `recall_memory` uses today), pointers resolve inside the child's worktree, and staleness is computed against the snapshot fingerprint.
- **OP-9 MUST:** Repeat warnings. When a child session's search matches an entry in the parent's search trail, the response includes `parent_trail_match`, containing the parent's prior hit count and top hit keys.
  - A match means the same tool, the same normalized query (case- and whitespace-insensitive), and equal filters.
  - SHOULD serve the result from the shared result cache (DC-2).
- **OP-10 MUST:** Search results from a child session mark symbols in the parent's explored manifest with `parent_explored: true` (in `fresh` mode, where they are not deduped).
- **OP-11 MUST (refinement):** The child's `recall_memory` includes the snapshot's memory entries, in addition to the normal global, project, and session scopes.
- **OP-12 MUST:** A child session is an ordinary session for every existing tool: `store_context`, `store_memory`, searches, and so on. Its own quotas apply (RQ-4).

### RT: Completion and return (child → parent)

- **RT-1 MUST:** A child completes by submitting:
  - its full result content (stored as a note in the child session with `kind=handoff_result`);
  - a status of `done`, `partial`, or `failed`;
  - an optional child-written summary.
- **RT-2 MUST:** The summary is capped at 300 tokens by default (setting `handoff_summary_max_tokens`). A child-written summary over the cap is truncated at the cap and flagged `summary_truncated: true`. Completion never fails because of summary length.
- **RT-3 MUST:** If the child gives no summary, the server derives one mechanically: `FACT:`/`RULE:` lines first, then the leading content lines, up to the cap. It is flagged `summary_source: derived`.
- **RT-4 MUST:** Completion returns a return stub of at most (cap + 40) tokens. The child outputs it as its final message, in the form `[result ctx_… for hof_…] <status> — <summary>`.
- **RT-5 MUST:** `FACT:` and `RULE:` lines in the result are promoted to `mem_*` entries, using the same parsing rules as `store_context(extract_memory=true)`.
  - They are session-scoped to the **parent** session, so the parent's `recall_memory` finds them.
  - `source_ref` is set to the result note ref.
- **RT-6 MUST:** Completion releases all of the child's claims, which triggers auto-grant (CL-4).
- **RT-7 MUST:** A child can complete again, for example after being resumed. The new result becomes current; the earlier result is marked superseded but stays fetchable until the tree expires.
- **RT-8 MAY:** Completion accepts optional structured fields (`changed_files`, `open_questions`) for workflow scripts, returned verbatim in fan-in.

### FI: Fan-in, status and recovery (parent side)

- **FI-1 MUST:** The parent can collect results for one handoff, or for all of its handoffs, in a single call. For each child it gets:
  - child `session_id` and label;
  - status: `open`, `done`, `partial`, `failed`, or `abandoned`;
  - current result ref and summary;
  - last-activity time, active claim count, and stored-note count.

  The response respects a token budget, by default enough for 16 children × (summary cap + 50).
- **FI-2 MUST:** The parent can list all handoffs it created using only its `session_id`, with per-handoff status counts. This is how a parent recovers after host compaction loses the ref.
- **FI-3 MUST:** A child with no MCP activity for the inactivity timeout (default 30 minutes, setting `handoff_child_inactive_minutes`) is marked `abandoned`. Any later activity from that child returns it to `open`.
- **FI-4 MUST:** The parent can read an abandoned child's partial work (stored notes and scratchpad posts) through FI-1 and normal `fetch_context`.
- **FI-5 MUST:** Collect at a nested node returns direct children by default. With `recursive=true` it returns the subtree, within the token budget.
- **FI-6 MAY:** Collect accepts `wait_seconds` (at most 60) and long-polls until any child's status changes. This is useful for workflow scripts.

### SP: Sibling scratchpad (one per tree)

- **SP-1 MUST:** Each handoff tree has one append-only scratchpad. Any session in the tree can read it and post to it.
- **SP-2 MUST:** Entry types are:
  - `finding`: free text, at most 500 tokens, with optional refs (`ctx_*`, symbol keys, file paths);
  - `dead_end`: what was tried and why it failed;
  - `claim` (see CL);
  - `trail`: automatic (SP-5).

  Each entry records its author session, timestamp, and id.
- **SP-3 MUST (refinement):** An author can retract their own entry. A retracted entry is marked retracted and hidden by default, not deleted.
- **SP-4 MUST:** Reads support:
  - a `since` cursor, so agents fetch only new entries;
  - filters by type and author;
  - a token budget.

  The caller's own entries are excluded by default.
- **SP-5 MUST:** Live trail sharing. Every search run by any session in the tree is appended automatically as a `trail` entry of at most 60 tokens: tool, normalized query, filters, hit count, top hit keys, and zero-hit flag.
- **SP-6 MUST:** When a session's search matches a `trail` entry from another session in the tree (same matching rule as OP-9), the response includes `sibling_trail_match` with the author session and that entry's hit summary.
- **SP-7 MUST:** Zero-hit `trail` entries and `dead_end` posts are presented together in the dead-ends view of the open digest and scratchpad reads.

### CL: Claims (advisory, queued)

- **CL-1 MUST:** A session in a tree can claim a resource key with an optional reason. The key is normally a project-relative file path, normalized; a symbol key or arbitrary string is also allowed.
- **CL-2 MUST:**
  - If the key is free, the claim is granted.
  - If it is held, the claimant is queued FIFO per key. The response names the holder session and label and gives the claimant's queue position.
- **CL-3 MUST:** A claim is released:
  - explicitly;
  - automatically when the holder completes (RT-6);
  - when the holder is marked abandoned (FI-3);
  - or when the tree expires.
- **CL-4 MUST:** On release, the next queued claimant is granted the claim automatically. The grantee learns of it on its next call to `open_handoff`, `handoff`, or `scratchpad`.
- **CL-5 SHOULD:** Any tool response to a session that has been granted a claim since its last call includes a compact `claims_granted` notice of at most 30 tokens.
- **CL-6 SHOULD (refinement):** When queuing a claim would create a wait cycle (for example A holds X and waits for Y while B holds Y and waits for X), the request is rejected with `claim_deadlock_risk`, naming the cycle, instead of being queued.
- **CL-7 MUST:** Tool descriptions and docs state that claims are advisory: the server does not stop any agent from editing files.
- **CL-8 MUST:** Active claims and queues appear in the open digest, in scratchpad reads, and in the dashboard tree view.

### DC: Dedup and caching (global changes)

- **DC-1 MUST:** The query/result cache is used whether or not `session_id` is set. Per-session dedup is applied after the cache lookup.
  - For a given session, the set of symbols returned versus deduped is identical to today's behavior.
  - Gated by flag `feature_shared_query_cache`, default on.
- **DC-2 MUST:** Cached results are shared across sessions, including siblings in a tree. A cached result is never served after any file it covers, or any file in the queried project scope, has been reindexed since the entry was created.
- **DC-3 MUST (refinement):** A symbol returned to a session is deduped on that session's very next call, regardless of write batching. This closes today's ~3s window, which matters for fork seeding and parallel calls.
- **DC-4 MUST:** Dedup rows belonging to child sessions expire with their tree (RQ-1).

### RQ: Retention and quotas

- **RQ-1 MUST:** All tree data expires 7 days after the last access to any element of the tree (setting `handoff_ttl_days`, default 7). Tree data means:
  - handoffs, snapshots, the scratchpad, claims, and linkage;
  - child sessions' notes, including results;
  - child dedup rows;
  - promoted memory whose `source_ref` is a tree result. Promoted memory stays unless the parent forgets it; see Open Question 8.
- **RQ-2 MUST:** Expiry runs automatically at least hourly, without needing the dashboard open.
- **RQ-3 MUST:** The root session (by `session_id`) or any holder of a tree's handoff ref can flush the whole tree explicitly. The dashboard can do the same.
- **RQ-4 MUST:** Default caps, each overridable in settings:
  - **Per tree:** 64,000 tokens and 300 entries across snapshot copies, scratchpad entries, and results.
  - **Topology:** maximum depth 3 and maximum 16 children per handoff.
  - **Child sessions** keep their own existing per-session `ctx_*` quota (50 notes / 32k tokens).
  - **Global caps** still apply.
- **RQ-5 MUST (refinement):** When a tree is at its cap, automatic `trail` entries evict the oldest `trail` entries first. Findings, dead ends, claims, results, and snapshots are never evicted automatically. Explicit writes that would exceed the cap return `handoff_tree_limit_exceeded` with current usage, cap, and suggested actions.

### TS: Tool surface

- **TS-1 MUST:** At most **3** new MCP tools:
  - **`handoff`**, with actions `create`, `complete`, `collect`, `list`, `status`, and `flush`;
  - **`open_handoff`**, with actions `open`, `expand`, and `resume`;
  - **`scratchpad`**, with actions `post`, `read`, `retract`, `claim`, and `release`.
- **TS-2 MUST:** All three tools are **core** tier, per the user's decision. This deviates from the current convention that core is read-only; see Open Question 7.
- **TS-3 MUST (refinement):** The three tools' combined name, description, and input schema add at most 1,200 tokens to `tools/list`.
- **TS-4 MUST:** The tool descriptions tell agents: "If your prompt contains `[handoff hof_…]`, call `open_handoff` before any search."
- **TS-5 MUST:** Every error is structured, with a stable `code` (for example `handoff_not_found`, `handoff_expired`, `handoff_depth_exceeded`, `handoff_children_exceeded`, `handoff_tree_limit_exceeded`, `claim_deadlock_risk`, `feature_disabled`), a human-readable message, and suggested next actions.
- **TS-6 MUST:** These are all updated to include the new tools:
  - the golden tool-list tests;
  - the tier tables in README, AGENTS.md, CLAUDE.md, and the skills;
  - the virtual-context prompt text.

### FF: Feature-flag framework (general)

- **FF-1 MUST:** A registry of typed boolean flags stored in settings. Each flag has a key, a default, a description, an env-override name, and whether it affects `tools/list`.
- **FF-2 MUST:** Precedence is env > setting > default, matching existing settings. A non-empty env value locks the flag.
- **FF-3 MUST:** The settings API lists every flag with its effective value, its source (`env`, `setting`, or `default`), and an env-locked indicator.
- **FF-4 MUST:** The dashboard Settings tab shows a toggle per flag. Env-locked flags are shown read-only with the reason.
- **FF-5 MUST:** Toggling applies live, without restart. The next `tools/list` reflects it, and the server emits `notifications/tools/list_changed` to connected clients on transports that support it, declaring the `listChanged` capability.
- **FF-6 MUST:** Initial flags:

  | Key | Default | Gates |
  |---|---|---|
  | `feature_handoff` | on | Master switch: all three tools and the handoff search annotations |
  | `feature_handoff_scratchpad` | on | `scratchpad` tool, digest sections, SP-* |
  | `feature_handoff_claims` | on | CL-* (`claim`/`release` actions) |
  | `feature_handoff_live_trail` | on | SP-5/SP-6 automatic trail sharing |
  | `feature_handoff_hooks` | off | Installer offers Claude Code handoff hooks |
  | `feature_shared_query_cache` | on | DC-1/DC-2 |

- **FF-7 MUST:** Disabling a flag never deletes data. Re-enabling it restores access to data that has not expired. A call to a disabled tool or action returns `feature_disabled`.
- **FF-8 MUST:** Existing `tools.json` per-tool overrides still apply. A tool is visible only if its flag is on **and** `tools.json` and the tier allow it.

### HI: Host integration and delivery

- **HI-1 MUST:** The manual path works on every MCP host: the parent pastes the HO-5 stub into the subagent prompt, and the child opens it per TS-4.
- **HI-2 MUST:** The repo ships opt-in Claude Code hook scripts and settings entries. Each item below is required only if the Phase-0 spike (Open Question 1) confirms the host supports it; otherwise it is documented as unsupported.
  - **(a) Subagent start:** when the subagent prompt contains a handoff ref, inject the open digest, or an instruction to open it, into the subagent's initial context.
  - **(b) Subagent stop:** when the child never completed, store its final message as a `partial` result.
  - **(c) After parent compaction:** re-surface the parent's open handoffs (FI-2).
- **HI-3 SHOULD:** If the host lets a hook rewrite the Agent tool input, a hook can create a handoff automatically from the parent session and append the stub to the subagent prompt.
- **HI-4 MUST:** A hook can resolve the agent's ast-context-cache `session_id` from the host's own session identifier. The mechanism is TBD (Open Question 2). Without it, HI-2(c) and HI-3 cannot work.
- **HI-5 MUST:** Hooks fail open. If the server is unreachable, slow (2s timeout), or errors, the subagent is spawned and runs normally, and the hook never blocks.
- **HI-6 MUST:** Cursor gets a rule (`.mdc`) and a skill describing the manual workflow, since it has no equivalent hooks.
- **HI-7 MUST:** Codex, VS Code, and JetBrains get instruction blocks (AGENTS.md or the host's rules mechanism) describing the manual workflow.
- **HI-8 MUST:** Docs include a Claude Code Workflows pattern: one handoff, N agents given the same stub, scratchpad coordination, then collect for fan-in.
- **HI-9 MUST:** Docs give per-mode guidance: use `fork` only when the host spawned a fork (the child inherited the parent's window), and `fresh` otherwise.
- **HI-10 MUST:** Skills and docs are updated with the handoff workflow: `skills/usage`, `skills/agents`, `.cursor/skills/ast-usage`, AGENTS.md, CLAUDE.md, and README.

### IN: Installer rework (dashboard "Agent integration")

- **IN-1 MUST:** Supported targets: **Cursor, OpenCode, Claude Code, Claude Desktop, Codex, VS Code, JetBrains**. Global scope only; the project-scope install option is removed.
- **IN-2 MUST:** For each target, the installer can install:
  - (a) MCP server registration;
  - (b) skills;
  - (c) rules or instruction blocks;
  - (d) handoff hooks (Claude Code only, opt-in, shown only when `feature_handoff_hooks` is on).

  A component the host doesn't support is skipped, and the preview says why.
- **IN-3 MUST:** Merge, never overwrite. The installer adds or updates only its own server entry, its marker-delimited blocks, or its own files.
  - If an existing config file fails to parse, the installer aborts with an error and writes nothing.
  - User content and comments in JSONC, TOML, and markdown are preserved. Formatting SHOULD be preserved.
- **IN-4 MUST:** Before any write, the installer saves a timestamped backup of each file it modifies. It keeps the last 5 per file (setting `installer_backup_keep`). The dashboard can restore any backup.
- **IN-5 MUST:** Preview first. The dashboard shows the exact per-file diff, and applying requires explicit confirmation. On apply, the installer re-checks that each file is unchanged since the preview; if any file changed, it re-previews instead of writing.
- **IN-6 MUST:** Surgical uninstall. It removes only what the installer added and never deletes a file it did not create.
- **IN-7 MUST:** Correct formats and values per host:
  - Claude Code gets a real MCP server registration in its supported user-scope config, never markdown in `~/.claude.json`.
  - OpenCode uses its documented `mcp` schema.
  - Claude Desktop gets a launch config that actually connects to the HTTP server.
  - Codex, VS Code, and JetBrains use their documented formats.
  - Every target uses the server's actual configured MCP port, not a hardcoded 7821.
  - Each format is verified against current host docs during implementation.
- **IN-8 MUST:** Installed state comes from the files on disk, not only the DB. Each component shows one of: `Installed`, `Outdated` (our block or entry differs from the current version), `Modified by user`, `Missing` (recorded but absent), or `Not installed`.
- **IN-9 MUST (refinement):** A skill path that already exists and is externally managed (for example a symlink to a directory outside the repo, as with the user's `~/.claude/skills/ast-context-cache`) is reported as `Externally managed` and skipped unless the user explicitly chooses to replace it, which takes a backup first.
- **IN-10 MUST:** Markdown blocks (CLAUDE.md, AGENTS.md, rules) are delimited by begin/end markers carrying a version stamp. Upgrades replace only the content between the markers.
- **IN-11 MUST:** The installed skills, rules, and instructions come from one canonical source in the repo. This resolves the current drift between `docs/INSTALL.md`, `skills/install`, and the generator text.
- **IN-12 MUST:** The installer never writes `env` blocks onto URL-based server entries (they do nothing there). Docs that suggest otherwise are corrected.
- **IN-13 MUST:** First 4.0 start re-verifies every existing install record on disk.
  - Records from the old Claude Code global target are flagged with a warning that `~/.claude.json` may have been overwritten by an earlier version, plus recovery guidance.
  - Records from removed project-scope installs are reported and can be cleaned up.
- **IN-14 MUST:** Docs include a host-integration inventory: for each target, the exact files and keys written per component.
- **IN-15 MAY:** CLI parity: `ast-mcp install|uninstall --target <host> [--component …] [--dry-run]` with the same preview and safety guarantees.

### BF: Defect fixes (in scope)

- **BF-1 MUST:** The `ctx_*` LIKE-fallback search respects the `session_id` and `project_path` filters exactly as the FTS path does.
- **BF-2 MUST:** The memory vector-recall fallback applies the same validity filters as the FTS path (not superseded, not forgotten, `as_of`) and the same scope filters. Entries with an empty project path are not returned outside their scope.
- **BF-3 SHOULD:** The dashboard MCP-tier view honors `AST_MCP_TOOLS_CONFIG` instead of the hardcoded `~/.astcache/tools.json` path.
- **BF-4 MUST:** Each defect fix comes with a regression test that fails on the current code.

### OB: Observability

- **OB-1 MUST:** A **child repeat-search rate** metric. A child search call is a *repeat* if either:
  - (a) it matches a parent-trail entry (the OP-9 rule); or
  - (b) at least 50% of its pre-dedup result keys are in the parent's explored manifest.

  The rate is repeats ÷ child search calls, per tree and in aggregate.
- **OB-2 MUST:** **Handoff tokens saved**: the token estimate of all snapshot content available to the child (note contents, plus pointer sources at `auto` mode, plus the full trail) minus the tokens actually delivered to the child through open and expand. It is added to the dashboard's Tokens saved, attributed to `handoff`.
- **OB-3 MUST:** **Return tokens saved**: the full result's token estimate minus the summary tokens returned to the parent. It is attributed to `handoff`.
- **OB-4 MUST:** A dashboard **tree view** that groups sessions into handoff trees. For each node it shows:
  - label, status, mode, and depth;
  - children;
  - tokens delivered and tokens saved;
  - repeat-search rate;
  - active claims and queues;
  - last activity.

  It also offers a tree flush action. Sessions outside trees still appear as today.
- **OB-5 MUST:** Prometheus metrics at `/metrics`:
  - handoffs created, opened, resumed, completed (by status), abandoned, and expired;
  - open trees and open children (gauges);
  - tree token-usage histogram;
  - claim wait-time histogram;
  - repeat-search rate;
  - shared query-cache hit ratio.
- **OB-6 MUST:** Structured `slog` events for every lifecycle transition (create, open, resume, complete, abandon, expire, flush, claim grant, claim release). Each event carries the tree id, handoff ref, parent and child session ids, and project path.

---

## Non-Functional Requirements

| ID | Area | Requirement |
|---|---|---|
| NFR-1 | Latency (local, warm index, p95) | `open` ≤ 100 ms; `create` ≤ 250 ms with a snapshot up to the tree cap; `scratchpad` post/read, `claim`/`release`, and `status` ≤ 50 ms; `collect` for 16 children ≤ 150 ms. Verified by Go benchmarks (the repo's first). |
| NFR-2 | Search overhead | Handoff annotations (OP-9, OP-10, SP-6) add ≤ 10 ms p95 to existing search tools, and 0 ms for sessions outside a tree. |
| NFR-3 | Token budgets | Create stub ≤ 60 tokens; default open ≤ 1,500; return stub ≤ summary cap + 40; claims notice ≤ 30; trail entry ≤ 60; tool schemas ≤ 1,200 total. |
| NFR-4 | Concurrency | 16 children opening, posting, claiming, and searching concurrently in one tree produce no lost writes, no duplicate claim grants, and FIFO order per claim key. |
| NFR-5 | Atomicity | Snapshot creation, completion (result + memory promotion + claim release), and tree flush are each all-or-nothing. |
| NFR-6 | Reliability | Hooks fail open (HI-5). A server restart loses no tree state; claims and queues persist. |
| NFR-7 | Storage | Tree data is bounded by RQ-4 and expires per RQ-1 and RQ-2. |
| NFR-8 | Installer safety | Zero destructive writes: across all fixture tests, no file content outside our entries or blocks changes, and every write is preceded by a backup. |
| NFR-9 | Trust model | Unchanged open model (a ref is enough to read). Refs are random (`hof_` ≥ 64 bits). No secrets are stored in handoffs by design; docs warn against putting credentials in briefs. |
| NFR-10 | Compatibility | Sessions that never touch handoff behave exactly as in 3.x, apart from the DC-1 caching change and the BF fixes. |
| NFR-11 | Locality | Everything stays on the local machine. No new network egress. |

### Breaking changes (v4.0)

1. The query cache now applies to `session_id` calls (DC-1). Results are identical, but cache-hit statistics and latency profiles change.
2. Dedup visibility is immediate (DC-3). Calls that previously returned duplicates within ~3s now dedup.
3. LIKE-fallback and vector-recall results shrink to the correct scope (BF-1 and BF-2).
4. The installer API and behavior change: project scope is removed, formats are corrected, and preview/confirm is required. Old DB install records are migrated (IN-13).
5. Three new core-tier tools appear in `tools/list` for every host unless disabled by flag or `tools.json`.

---

## User Workflows

### W1: Fresh subagent, manual (Cursor, Codex, generic MCP)
1. The parent explores with its `session_id` (searches, notes, memory).
2. The parent calls `handoff` `create` with a brief, pointers, and refs, pruning the trail if needed, and receives a `hof_` ref and stub.
3. The parent spawns the subagent with the stub in its prompt.
4. The child calls `open_handoff` `open`, receives its child `session_id` and the digest (≤ 1,500 tokens).
5. The child expands only the pointers or notes it needs, and searches with its child `session_id`. Repeat searches return `parent_trail_match`, and parent-explored symbols are marked.
6. The child calls `handoff` `complete` with its full result and status (and an optional summary), receives the return stub, and outputs it as its final message.
7. The parent sees `[result ctx_… for hof_…] done — <summary>` and fetches the full result only if needed.

### W2: Claude Code with hooks (auto)
1. The user has installed the Claude Code hooks via the installer (`feature_handoff_hooks` on).
2. The parent creates a handoff, or a hook creates one (HI-3, if supported), and the stub is in the Agent prompt.
3. On subagent start, the hook injects the open digest or an open instruction (HI-2a).
4. Steps 5–7 of W1 follow.
5. If the child stops without completing, the stop hook stores its final message as a `partial` result (HI-2b).
- **Error path:** the server is down, so the hook times out after 2s, the subagent runs normally, and nothing is cached.

### W3: Claude Code fork
1. The parent creates a handoff with `mode=fork` and spawns a fork with the stub.
2. On open, the child's dedup is seeded with the parent's explored manifest.
3. The child's searches skip symbols already in its inherited window, and those skips are counted as dedup savings.

### W4: Workflow fan-out (N parallel children)
1. The workflow's parent agent creates one handoff.
2. The script launches N agents (≤ 16) with the same stub. Each opens it and gets its own child session.
3. Child A posts a `finding`, and child B sees it in its next scratchpad read.
4. Child A's searches auto-append `trail` entries, and child B's identical search returns `sibling_trail_match`, served from cache.
5. Child A claims `internal/x/file.go`. Child B's claim on the same key is queued at position 1 and names A as holder.
6. A completes, the claim auto-grants to B, and B sees `claims_granted` on its next call.
7. The parent calls `collect` once and receives N summaries plus refs.
- **Error path:** a 17th open returns `handoff_children_exceeded`.

### W5: Nested handoff
1. A child (depth 1) creates a handoff for a grandchild (depth 2) in the same tree, sharing its scratchpad.
2. The grandchild completes, and the child collects it.
3. The parent's `collect` with `recursive=true` shows the whole subtree.
- **Error path:** creating at depth 3 → 4 returns `handoff_depth_exceeded`.

### W6: Parent compaction recovery
1. The host compacts the parent and the `hof_` ref is lost.
2. The parent calls `handoff` `list` with its `session_id` and sees its open handoffs with status counts (or the HI-2c hook re-surfaces them).
3. The parent calls `collect` to get results.

### W7: Child crash or abandonment
1. A child stops issuing calls (crash, interrupt).
2. After 30 minutes it is marked `abandoned` and its claims are released and auto-granted.
3. The parent's `collect` shows `abandoned` with the child's stored-note count, and the parent fetches the partial notes.
4. If the child is later resumed (OP-2), its status returns to `open`.

### W8: Stale pointer
1. After the snapshot, a sibling edits a function the parent pointed to.
2. The child expands that pointer, receives the current code with `stale: true, change: modified`, and the old and new line ranges.
3. If the function was deleted, it receives `stale: true, change: deleted` and no error.

### W9: Safe install / uninstall
1. In Settings → Agent integration, the user picks a target (for example Claude Code) and components.
2. The dashboard shows the per-file diff, including skipped components with reasons and any externally managed paths.
3. The user confirms. The installer backs up the files, merges, and re-verifies, and the status shows `Installed`.
4. The user later edits our block by hand, and the status shows `Modified by user`.
5. On uninstall, the installer previews, then removes only our entries and blocks. Other servers and content remain.
- **Error path:** the target file is invalid JSON or TOML, so the install aborts with a parse error and nothing is written.
- **Error path:** the file changed between preview and apply, so the dashboard re-previews.

### W10: Live flag toggle
1. The user turns off `feature_handoff_claims` in Settings.
2. The next `tools/list` still shows `scratchpad`, but its `claim` and `release` actions return `feature_disabled`.
3. The user turns off `feature_handoff`. All three tools disappear, a `list_changed` notification is emitted, and the data is retained until TTL.

---

## Integration Points

| System | Contract |
|---|---|
| **MCP clients** (Claude Code, Cursor, Codex, VS Code, JetBrains, OpenCode, Claude Desktop, generic) | 3 new tools (TS-1); new response fields on existing search tools (`parent_trail_match`, `parent_explored`, `sibling_trail_match`, `claims_granted`); `notifications/tools/list_changed` with `listChanged` capability. |
| **Existing tools** | `get_context_capsule`, `search_semantic`, `retrieve`, `get_file_context`: trail capture, annotations, shared cache, immediate dedup. `recall_memory`: snapshot memory for child sessions. `store_context`: result notes (`kind=handoff_result`). `flush_context`: must not delete tree snapshots (HO-10). |
| **Storage** (usage DB, context DB, vector cache) | New tree, handoff, snapshot, scratchpad, and claim data, plus child-session linkage. Must follow existing migration conventions. |
| **Settings** | Flag registry (FF-*), plus new keys: `handoff_ttl_days`, `handoff_summary_max_tokens`, `handoff_child_inactive_minutes`, per-tree caps, `installer_backup_keep`. |
| **Dashboard API/UI** | Settings flag toggles, tree view, tree flush, the reworked installer (preview/apply/restore/verify), and `/metrics` additions. |
| **Host hooks** (Claude Code) | Hook scripts call the local MCP/HTTP endpoint with a 2s timeout and fail open. They need host-session → ast-session resolution (HI-4). |
| **Host config files** | `~/.cursor/mcp.json`, OpenCode config, Claude Code user config, Claude Desktop config, `~/.codex/config.toml`, VS Code and JetBrains MCP configs, skills directories, rules directories, CLAUDE.md, and AGENTS.md. All writes are merge-only with backups. |
| **Prometheus** | New series per OB-5 on the existing `/metrics` endpoint. |

---

## Acceptance Criteria

**Handoff core**
1. Given a parent session that ran 10 searches and was returned 40 symbols, when it creates a handoff with 3 pointers, then the response contains a `hof_` ref, a stub of ≤ 60 tokens, and a breakdown listing the manifest (40 keys), the trail (10 entries), and the pointers (3).
2. Given that snapshot, when the parent then stores a new note and runs 5 more searches, then opening the handoff shows the original 10 trail entries and no new note.
3. Given a `fresh` handoff, when a child opens it and searches for a symbol from the parent manifest, then the symbol is returned (not deduped) with `parent_explored: true`.
4. Given a `fork` handoff, when a child opens it and runs the same search, then the symbol is deduped and counted in the child's `dedup_tokens_saved`.
5. Given a child expanded pointer P at `auto` mode, when it then runs a search that would return P, then P is deduped (OP-6).
6. Given a child search identical (case- and whitespace-normalized, same tool and filters) to a parent-trail query, when it runs, then the response includes `parent_trail_match` with the parent's hit count.
7. Given a pointed function modified after the snapshot, when the child expands it, then the current code is returned with `stale: true, change: modified`. Given it was deleted, the result is `stale: true, change: deleted` and no error.
8. Given a child in a sibling worktree of the parent's repo, when it expands a pointer, then the code comes from the child's worktree.
9. Given a parent that runs `flush_context(session_id=parent)` while a child is open, when the child expands a pointer or included note, then it succeeds.

**Return and fan-in**
10. Given a child completes with a 2,000-token result and a 500-token summary, then the parent-facing summary is ≤ 300 tokens with `summary_truncated: true`, the full result is fetchable by ref, and return tokens saved ≥ 1,700.
11. Given a child completes with no summary and a result containing 2 `FACT:` lines, then the derived summary starts with those facts (`summary_source: derived`), and the parent's `recall_memory` returns both as `mem_*` entries with `source_ref` set to the result ref.
12. Given 16 children of one handoff (10 done, 3 open, 2 failed, 1 inactive > 30 min), when the parent calls `collect`, then it receives 16 entries with correct statuses (`abandoned` for the inactive one) in one response within budget.
13. Given a parent that lost its ref after compaction, when it calls `handoff list` with only its `session_id`, then all its handoffs are listed.
14. Given a 17th open of one handoff, then `handoff_children_exceeded` is returned. Given a resume of an existing child, it succeeds.
15. Given creation at depth 3, then `handoff_depth_exceeded` is returned.

**Scratchpad and claims**
16. Given child A posts a finding, when child B reads with its last cursor, then B receives exactly that entry and not its own entries.
17. Given child A ran `search_semantic("retry backoff")`, when child B runs the same query, then B's response includes `sibling_trail_match` naming A, and the server serves it from cache.
18. Given A holds a claim on `x.go` and B and C claim it in that order, when A completes, then B is granted, C is at position 1, and B's next tool response includes `claims_granted`.
19. Given A holds X and waits for Y while B holds Y, when B claims X, then B receives `claim_deadlock_risk` and is not queued.
20. Given 16 concurrent children claiming the same key, then exactly one holds it and the other 15 are queued in arrival order (NFR-4).

**Retention and quotas**
21. Given a tree not accessed for 7 days, when the hourly expiry runs, then all tree data, child notes, and child dedup rows are gone, and opening the handoff returns `handoff_expired`.
22. Given a tree at 64k tokens, when automatic trail entries arrive, then the oldest trail entries are evicted. When a child posts a finding that would exceed the cap, then `handoff_tree_limit_exceeded` is returned with usage details.

**Caching and fixes**
23. Given two different sessions running the same `get_context_capsule` query on an unchanged index, then the second is a cache hit, and each session's dedup still applies to its own history.
24. Given a file is reindexed, when the same query runs, then it is not served from a pre-reindex cache entry.
25. Given two calls from one session less than 1s apart returning the same symbol, then the second call dedups it.
26. Given notes in sessions S1 and S2 whose labels match a query that FTS misses, when `search_context(session_id=S1)` falls back to LIKE, then only S1's notes are returned (BF-1).
27. Given a superseded fact and a session-scoped fact from another session, when vector-fallback recall runs, then neither is returned (BF-2).

**Flags**
28. Given `feature_handoff` is toggled off in the dashboard, then the next `tools/list` omits all 3 tools without a restart, and a `list_changed` notification is emitted on supporting transports.
29. Given `AST_FEATURE_HANDOFF=false` in the env (exact env name TBD in FF-1), then the dashboard shows the flag locked off, with source `env`.
30. Given a flag is disabled and re-enabled within the TTL, then previously created trees are accessible again.

**Installer**
31. Given `~/.cursor/mcp.json` with 3 other servers and comments, when ast-context-cache is installed, then the 3 servers and the comments remain, our entry is added, and a timestamped backup exists.
32. Given a malformed `~/.codex/config.toml`, when install is attempted, then it aborts with a parse error and the file is byte-identical.
33. Given an installed Claude Code target, when uninstalled, then only our registration, skill files we created, marker blocks, and hook entries are removed, and no file we didn't create is deleted.
34. Given `~/.claude/skills/ast-context-cache` is a symlink outside the repo, then the preview shows `Externally managed` and it is skipped by default.
35. Given the user edits text inside our CLAUDE.md marker block, then the status shows `Modified by user`. Given a newer block version ships, the status shows `Outdated`.
36. Given a target file changed between preview and apply, then apply re-previews instead of writing.
37. Given the server runs on a non-default MCP port, then every installed registration uses that port.
38. Golden-file fixture tests exist for each of the 7 targets × (install into empty, install into existing, upgrade, uninstall), and all pass.

**Measurement and hosts**
39. The scripted Go scenario test (parent → 3 fresh children + 1 fork child → 1 grandchild → fan-in, with a parallel no-handoff baseline) shows a child repeat-search rate reduction of ≥ 50%, and nonzero handoff and return tokens saved.
40. The Go benchmarks meet NFR-1.
41. A manual real-host trial in Claude Code (Agent tool, fork, and a workflow) and in Cursor completes W1, W3, and W4 end to end, and the dashboard tree view shows the trees with token stats. Results are recorded in a short trial report.
42. The dashboard tree view and Prometheus `/metrics` show the OB-1…OB-5 values for the scenario-test tree.

---

## Open Questions

1. **Hook capabilities are unverified (blocks HI-2 and HI-3).** A Phase-0 spike must confirm, against current Claude Code docs and real behavior:
   - whether `SubagentStart` `additionalContext` reaches the subagent itself;
   - whether a subagent-stop hook exists and exposes the final message or transcript;
   - whether `PreToolUse` `updatedInput` can rewrite an Agent tool prompt;
   - whether hook input includes the parent's session id.

   The research agent could not confirm these. Its claim that no `SubagentStop` hook exists is doubtful.
2. **Mapping a host session to an ast session (HI-4).** Agents choose their own `session_id`, but hooks know only the host's id. Options:
   - a convention: agents use the host session id as `session_id`;
   - an alias-registration call;
   - the hook supplies the id via injected context.

   Pick one in planning.
3. **Fork prompt-cache inheritance.** The docs say forks "share the prompt cache," but whether a fork reuses the parent's warm prefix is unverified. This affects only the guidance text in HI-9.
4. **Cursor cloud and background agents** reportedly cannot reach custom or local MCP servers (unverified). If true, they are out of scope. Confirm.
5. **MCP transport version.** A 2026-07-28 stateless revision of the spec reportedly removed `Mcp-Session-Id`; this is confirmed only by third-party and blog sources. Confirm which spec version the server speaks and how `list_changed` is delivered over it (FF-5).
6. **LAN exposure.** The MCP and dashboard servers bind all interfaces (`":port"`). With the open trust model (NFR-9), handoff data (briefs, results, scratchpads) is readable from the local network. Binding to localhost by default may be worth making a separate decision. 4.0 is a natural moment for that breaking change.
7. **All-core tiers (TS-2).** This deviates from "core = read-only," so hosts deliberately limited to core get write tools. Confirm, or consider making `handoff`'s write actions extended while keeping the tool itself core.
8. **Retention of promoted memory.** RT-5 promotes child `FACT:`/`RULE:` lines into the parent's session memory. Should those entries expire with the tree (RQ-1), or outlive it? Currently they outlive it.
9. **Child ordinary notes expire with the tree (RQ-1).** This was decided during refinement because tree-owned data otherwise accumulates forever. Confirm.
10. **Claude Desktop bridge.** A valid HTTP connection likely needs a stdio bridge (for example `npx mcp-remote`), which adds a Node dependency. Acceptable?
11. **Exact config formats** for OpenCode, Codex, VS Code, JetBrains, and Claude Code user-scope registration must be verified against current host docs during implementation (IN-7).
12. **The destructive installer bug is live today.** Per the scoping decision, it is fixed here rather than hot-fixed. Until 4.0 ships, the dashboard's Claude Code install and uninstall (either scope) can destroy `~/.claude.json` or `CLAUDE.md`, and the Cursor and OpenCode installs wipe other servers.
13. **Repeat-search definition (OB-1).** The 50%-overlap threshold for clause (b) was chosen with low confidence. Revisit after the first scenario-test data.
14. **Other 4.0 breaking changes.** Is there anything else to batch into the major version (for example access tightening, a dedup-row TTL for non-tree sessions, or localhost binding from Q6)?
15. **Summary over cap: truncate vs. reject.** RT-2 truncates, because completion must not fail. Confirm this is preferred over forcing the child to rewrite.

---

## Requirement Count

| Priority | Count |
|---|---|
| MUST | 102 |
| SHOULD | 4 |
| MAY | 3 |
| Non-functional (all binding) | 11 |
