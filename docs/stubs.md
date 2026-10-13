# Stub grammars

ast-context-cache hands agents short **stubs**: one-line pointers to content stored on the server, which an agent keeps in chat instead of the content itself. Agents and hooks parse them back, so their shapes are fixed. `internal/mcp/stubs_test.go` checks each producer against the grammars below.

In the grammars, `<hex12>` is 12 lowercase hex characters and `<hex16>` is 16. `—` is an em dash (U+2014).

## Virtual context: `[ctx_…]`

`store_context` returns a ref, `ctx_<hex12>` (producer: `contextnotes.Store`). The stub an agent keeps in chat is the ref in brackets, followed by its label:

```
[ctx_<hex12>] <label>
```

Example: `[ctx_a1b2c3d4e5f6] auth design`. Recover the content with `fetch_context(refs=["ctx_a1b2c3d4e5f6"])`.

## Handoff: `[handoff hof_…]`

`handoff action=create` returns `stub` (producer: `handoff/create.go` `handoffStub`). The label is cut so the stub stays under 60 tokens.

```
[handoff hof_<hex16>] <label> — call open_handoff first
```

A subagent whose prompt holds this stub calls `open_handoff(handoff="hof_…")` before anything else.

## Handoff result: `[result ctx_… for hof_…]`

`handoff action=complete` returns `stub` (producer: `handoff/complete.go` `returnStub`), which the child prints as its final message. `<status>` is `done`, `partial`, `failed` or `abandoned`.

```
[result ctx_<hex12> for hof_<hex16>] <status> — <summary>
```

The parent reads the full result with `fetch_context(refs=["ctx_…"])`, or fans in with `handoff action=collect`.

## Offloaded result: `[ctx_… <tool> <target>, <N> tok — …]`

With `feature_result_offload` on, a code tool result (`get_context_capsule`, `search_semantic`, `get_file_context`, `retrieve`) larger than `result_offload_threshold` tokens (default 2000, `AST_RESULT_OFFLOAD_THRESHOLD`) is stored as an offload note. The response is the stub line followed by a head: the results in rank order that fit under the threshold. Pass `offload=false` to get the whole result instead. Producer: `contextnotes.OffloadStub`.

```
[ctx_<hex12> <tool> <target>, <N> tok — head shown, fetch_context for all]
```

- `<target>` is the query, or the file for `get_file_context`. It is left out when empty.
- `<N>` is the full result's token count: an integer under 1000, otherwise thousands with one decimal and a `k` suffix (`9.4k`, `2k`).
- With `output=json`, the stub is the body's first field, `"offloaded"`, so the head stays valid JSON.

Example: `[ctx_0123456789ab get_context_capsule auth middleware, 9.4k tok — head shown, fetch_context for all]`.

`fetch_context` on the ref returns the full result. Offload notes expire `offload_ttl_hours` (default 24) after their last access; fetching an expired ref returns `status: "expired"` with the `tool` and `args` that produced it, so the call can be re-run. A call without a `session_id` stores its offload under the synthetic session `offload-<UTC date>`.
