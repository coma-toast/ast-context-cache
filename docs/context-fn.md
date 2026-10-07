# Reusable context functions

`define_context_fn` / `apply_context_fn` / `list_context_fns` let an agent register a **named, reusable context transform** and re-invoke it, instead of re-deriving the same edit on every turn.

**Off by default.** Enable with `feature_context_fn` / `AST_FEATURE_CONTEXT_FN`, or Dashboard **Settings → Features**.

## Why

This is the part of [Context Language Models](https://arxiv.org/abs/2609.37725) (arXiv 2609.37725) that [`edit_context`](context-edit.md) deliberately left out. Phase 1 gave the model write access to its own context file. The paper's actual novelty is that the model then **defines its own** context-management functions and reuses them — `compact_turns` appears 37 times in a single trace. That is a different capability: not "the model may edit context" but "the model may author the tool it edits context with."

In the paper that function is Python the model writes into its own context file. Here it is a stored `(pattern, replacement)` pair.

## Why not a general function

Deliberately narrower, because the paper's own Discussion flags model-authored context transforms as a new prompt-injection channel: the more expressive the transform, the more it can be used to rewrite context into something the author did not intend. Concretely, this design:

- **stores no executable code.** A regex transform cannot become a tool call. There is no `exec`, no Python, no shell.
- **compiles the pattern at definition time.** A function that can never match is rejected before it is stored, not rediscovered on every apply.
- **uses Go RE2**, so a stored pattern is linear-time and cannot blow up an apply across 200 notes.
- **bounds the registry.** 50 live functions by default (`AST_CONTEXT_MAX_FNS`), because each one is a transform that can be applied at scale. A note that would be emptied by the pattern is refused by the edit underneath.

What is left is naming, versioning, dry-run and per-note reversibility.

## Tools

| Tool | Tier | Role |
|------|------|------|
| `define_context_fn` | extended | Register or replace a named transform |
| `apply_context_fn` | extended | Invoke it across `refs` or a whole session |
| `list_context_fns` | core | Function metadata and usage counters |

```
define_context_fn(name, pattern, replacement?, description?, project_path?, session_id?, expect_version?)
apply_context_fn(name, refs?, session_id?, max_replacements?, skip_errors?, dry_run?)
list_context_fns(project_path?, limit?)
```

`pattern` is Go RE2. `replacement` supports `$1` group references and may be empty, which deletes the matched text.

```
# define once
define_context_fn(name="compact_turns",
                  pattern="(?ms)^dead ends:.*?(?=\n## )",
                  replacement="",
                  description="drop dead-end candidates as they are found")

# preview the cost before committing a session-wide sweep
apply_context_fn(name="compact_turns", session_id="conv-uuid", dry_run=true)

# then commit
apply_context_fn(name="compact_turns", session_id="conv-uuid")
```

## Every apply is revertible per note

An apply is not a new write path. Each note goes through [`edit_context`](context-edit.md), so the revision log, quota deltas, FTS reindex and vector hygiene are the same code a manual edit takes.

That is the property that makes a session-wide sweep safe to attempt: a bad one is `edit_context(action="revert", ref=...)` per ref, not a restore-from-backup. `apply_context_fn` reports the resulting `revision` and `previous_revision` for exactly that reason.

## Guards

- **`expect_revision` on define.** Two agents defining the same name is the function-level analogue of the lost update `expect_revision` prevents on notes. Redefining bumps `version` and resets the counters, so `call_count` measures the definition actually in force rather than a lifetime total across unrelated rewrites.
- **Registry cap.** 50 live functions by default; a full registry rejects new defines with `context_limit_exceeded` until one is retired. Retiring is a tombstone, not a delete, so the name's history stays auditable and the row's `retired_at` makes `apply_context_fn` refuse to run it.
- **Stops on error by default.** A partial apply that half-succeeded is harder to reason about than one that stopped at the first bad note and reported it in `errors`. `skip_errors=true` opts into continuing.
- **`max_replacements`.** The only brake on a function that matches everything, at session width.
- **`dry_run` does not move the counters.** A preview that looked like it saved 40k tokens would make `list_context_fns` a fiction.

## Did it pay for itself?

Same metric as `edit_context`: tokens. Every apply reports per-ref `tokens_before` / `tokens_after` / `tokens_reclaimed` plus a total, and `list_context_fns` carries lifetime `call_count`, `notes_touched` and `tokens_reclaimed`.

That is the only way to tell a reusable function from a speculative one. A function with `call_count: 0` is dead weight; a function that repeatedly *grows* notes is a cost, not a saving.

## Not included

Materializing a whole session's notes as one editable `context_file` document. It duplicates `list_context` plus per-note editing and adds a cross-note transaction story for little gain at current note counts.