## ast-context-cache

ast-context-cache is your code and context memory. Use its MCP tools on every task, before your own file tools. Fall back to grep, find, cat or whole-file reads only for a single known string or file, or when the server is not running.

### Session id

- Pick one stable `session_id` per conversation (a UUID is fine) and pass it on every ast-context-cache call. It deduplicates symbols already returned and scopes notes and memory to the thread.
- Paperclip agents: use the task identifier (for example `JD-123`) as `session_id`, so notes carry over between runs.
- If your prompt contains `[handoff hof_…]`, call `open_handoff` first, before any search, and use the child `session_id` it returns instead.

### Start of work

1. `recall_memory` for rules and prefs, then `list_context` to find notes from earlier runs and `fetch_context` the ones you need.
2. New repo or space: `index_status`; if it is not indexed, `index_files` on the repo path.

### Explore before grep

3. Explore with `get_project_map` (depth=2), `get_context_capsule` (`mode="auto"`), `search_semantic` (intent), `retrieve` (RAG bundle; `include_memory=true` when prefs matter) and `get_file_context` (default `skeleton`) BEFORE grep, find, cat or whole-file reads. Call `check_symbol_exists` before assuming a name exists.
4. Before changing an exported function or type: `get_impact_graph`. For a branch or PR: `diff_impact`. Before deleting code: `check_deletion_safety`.
5. Library or API docs: `search_docs`, then `fetch_doc` on a miss. Don't fetch docs with curl, WebFetch or a browser.
6. If a result includes `code_script_hints`, call `execute_code(script_id, data, project_path)` and use only its `result` instead of reading the raw results.

### Keep context

7. Plans, analysis and findings worth keeping: `store_context`, and keep only the `[ctx_…]` stub in chat (Paperclip agents: put the ref in your task comment).
8. Update a note in place with `edit_context` (`append`/`replace`/`delete`/`rewrite`, undo with `revert`); it keeps the ref and reports `tokens_reclaimed`. Never flush and re-store to make an edit. Pass `expect_revision` when a note is shared across a handoff tree.
9. Facts, prefs and conventions the next run needs: `store_memory` (`mem_*`), read back with `recall_memory`. Keep them out of `ctx_*` notes.
10. If the same edit recurs every turn, define it once with `define_context_fn` and re-apply it with `apply_context_fn` (off by default; dry-run anything wider than a couple of refs).

### Compaction

- Before host compaction or at about 70% context, `store_context` bulky plans, analysis and diffs and keep only the stubs. Never compact just to make room: shrink notes with `edit_context` instead.
- After compaction: `fetch_context(refs)`, or `list_context` / `search_context` if refs were lost; `handoff` `list` with your `session_id` recovers lost `hof_` refs.
- When the thread is done: `flush_context`.

### Delegation

11. To delegate: `handoff` `create` with your `session_id`, a brief and pointers, and put the returned stub (`[handoff hof_…] <label> — call open_handoff first`) at the top of the subagent prompt. Use the default `mode=fresh`; `fork` is only for a host fork that inherited your window.
12. A child calls `open_handoff` first, `expand`s only what it needs, finishes with `handoff` `complete` (full `content`, `status`, short `summary`) and outputs only the return stub `[result ctx_… for hof_…] <status> — <summary>`.
13. Agents sharing one handoff post findings and dead ends with `scratchpad` and `claim` a file before editing it (claims are advisory). The parent fans in with `handoff` `collect`.
14. Never put credentials or secrets in a brief, note or scratchpad post; anyone with the ref can read it.
