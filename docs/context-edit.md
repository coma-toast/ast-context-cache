# Context editing (context as a file)

`edit_context` lets an agent rewrite stored virtual context **in place**, keeping the same `ctx_*` ref.

## Why

Virtual context is normally **write-once**. `context_notes.content` is a single opaque text blob, and the only `UPDATE` the note path performs bumps `access_count` — so changing a note meant `flush_context` (which invalidates the `ctx_*` stub you already wrote into chat) followed by a fresh `store_context` (new ref, re-debited quota, a second note). For the offload/fetch loop that virtual context exists to support, that is a bad trade: the ref is the only handle the agent has.

This is the [Context Language Models](https://arxiv.org/abs/2609.37725) idea adapted to what this server holds. The paper gives a model write access to a file mirroring its own context and lets it edit that file with general tools, instead of restricting it to a harness-defined action space. Their qualitative traces are almost entirely in-place rewrites: `re.sub` to compact old results, loops to drop irrelevant ones, a whole new `## ORCHESTRATOR STATE` block replacing a stale one.

Here the closest analogue of the paper's context file is a `ctx_*` note: bulky text the agent owns and will read back.

## Tool

| Tool | Tier | Role |
|------|------|------|
| `edit_context` | extended | In-place mutation of one note, with revisions and revert |

```
edit_context(action, ref, session_id?, content?, pattern?, replacement?,
             start_line?, end_line?, max_replacements?, to_revision?,
             expect_revision?, dry_run?)
```

| Action | Required | Effect |
|--------|----------|--------|
| `append` | `content` | Add text to the end of the body |
| `rewrite` | `content` | Replace the whole body |
| `replace` | `pattern` **xor** `start_line`+`end_line`, plus `replacement` | Rewrite the matched region(s) |
| `delete` | `pattern` **xor** `start_line`+`end_line` | Drop the matched region(s) |
| `revert` | `to_revision` (optional) | Restore a prior revision; defaults to the most recent superseded body |

Line ranges are **1-indexed and inclusive**. `replacement` supports `$1` group references and may be empty, which deletes the matched text. Patterns are **Go RE2**: no backreferences or lookaround, and matching is linear in the input, so a model-supplied pattern cannot blow up the call.

```
# keep a running state block current instead of re-storing the whole note
edit_context(action="replace", ref="ctx_a1b2c3d4e5f6", session_id="conv-uuid",
             pattern="(?ms)^## ORCHESTRATOR STATE.*?(?=\n## )",
             replacement="## ORCHESTRATOR STATE\nBudget: 8/100 used. Next: score candidates.")

# drop the noise an earlier sweep left behind, after seeing what it would cost
edit_context(action="delete", ref="ctx_a1b2c3d4e5f6", pattern="(?m)^Searched: .*No relevant results\\.$", dry_run=true)
```

## Every edit is reversible

The paper's own Discussion section flags model-editable context as a new attack surface: self-generated text written into the live context persists across turns, and there are documented cases of a model inserting instructions into its own summary that later changed its behaviour. The mitigation here is that nothing an agent writes is one-way.

- **Revisions.** Every edit stores the body it replaced in `context_note_revisions` and bumps `context_notes.revision`. Retention is bounded — 10 per note by default, `AST_CONTEXT_MAX_REVISIONS` or the `context_max_revisions` setting — because unbounded history would let an edit loop grow `context.db` past what the store quotas allow.
- **`revert`.** Undo is a first-class action. Revision numbers only ever increase, so undoing an undo does not destroy history.
- **`expect_revision`.** Notes can be shared across a handoff tree. Pass the revision you read and an edit that raced someone else fails with `revision_conflict` instead of clobbering it.
- **Edits diff against the stored body**, never a caller-supplied copy, so a stale snapshot cannot be written back wholesale.

## Quotas still apply

`Store` checks caps on insert; an edit can grow a note. Growth is charged against the same caps (`single_note_tokens`, `session_tokens`, `global_tokens`), so repeated appends cannot push a note past what `store_context` would have accepted. `context_session_stats.virtual_tokens_stored` is adjusted by the delta, keeping the dashboard honest.

## Search and vectors are kept in step

Two things that were free while notes were write-once are not anymore:

- **FTS.** `context_notes_fts` is a bare `fts5` table with **no triggers**, so an in-place edit reindexes explicitly. Skipping this leaves `search_context` matching text the agent deliberately removed.
- **Vectors.** Note vectors live in `index.db` under `note:<ref>` and are keyed by row id, not content hash — an upsert alone leaves the pre-edit embedding behind. The old vector is deleted before the row changes, so a failure under index quiesce aborts the edit instead of leaving recall silently serving the old body.

## Did it pay for itself?

The paper's efficiency metric is **prefix-reuse FLOPs**: the cost an edit triggers by invalidating the cached prefix. Our cost is tokens, so every edit reports `tokens_before`, `tokens_after`, and `tokens_reclaimed`.

That number is the whole point of tracking it. A compaction loop that shrinks a note is a win; one that shuffles text around while growing is a cost — and without it, an agent cannot tell which it is doing.

## Feature flag

`feature_context_edit` / `AST_FEATURE_CONTEXT_EDIT`, on by default like `feature_handoff`. Dashboard **Settings → Features** toggles it live; disabling hides the tool and sends `tools/list_changed` without a rebuild.

## Related

[`define_context_fn` / `apply_context_fn`](context-fn.md) build on this: a named reusable transform the agent registers once and re-invokes across notes or a whole session. Off by default (`feature_context_fn`).
