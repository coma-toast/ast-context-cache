# Indexing wtg spaces and worktrees

A wtg space is a directory holding one git worktree per repo, plus `go.work` and often `SPACE.md`:

```
~/git/ast-context-cache            # base clone, already indexed
~/spaces/my-task/
  go.work
  SPACE.md
  ast-context-cache/               # worktree (.git is a file)
  sandbox/                         # worktree
```

Most files in a new worktree are byte-identical to the base clone, so ast-context-cache copies the base clone's index for them instead of parsing and embedding them again.

## Sibling reuse

When a project has no index of its own yet, the first index of it looks for an already-indexed checkout of the same repository (same `git rev-parse --git-common-dir`, see `projectlinks.RepoSiblings`) and picks the one with the most indexed files. For each file:

- **Byte-identical to the sibling's copy** (SHA-256 match), and the sibling's rows were made by the current parser: symbols, import edges and code vectors are copied under the new path. No tree-sitter parse, and no embedding unless the sibling hadn't finished embedding that file yet.
- **Otherwise**: the file is parsed normally and queued for embedding.

Every first-index path goes through `indexer.IndexOrReuse`:

| Path | Trigger |
|------|---------|
| Watcher catch-up | any tool call with a new `project_path`, `open_handoff`, the dashboard's Start watcher, pinned projects at startup |
| Directory index job | `index_files` with a directory |
| Dashboard / CLI | `POST /api/index-project`, `ast-mcp index <dir>` |

The server log shows one line per project, for example:

```
INFO Reused unchanged files from sibling checkout files=784 parsed=3 sibling=/Users/jason/git/ast-context-cache project=/Users/jason/spaces/my-task/ast-context-cache
INFO Catch-up complete project=... reindexed=3 reused=784 removed=0
```

A project that already has an index keeps today's behaviour (incremental catch-up, or a full re-parse for an explicit `index_files`).

## Space roots

A directory is treated as a **space root** when it is not a git checkout itself, has at least one child directory with a `.git` entry, and either has `go.work` or `SPACE.md` or at least one child that is a worktree (`.git` file). A plain folder (a notes vault, a folder of loose scripts, a single repo) is still indexed as one project.

A space root is never indexed or watched as one project. Indexing or watching it (including any tool call that passes it as `project_path`, which is what an agent whose cwd is the space does):

1. Links each child checkout under the root (`project_links`). Linking removes any rows the root already had for files inside that checkout.
2. Watches and indexes each checkout as its own project, with sibling reuse.

Search and analysis tools called with the space root as `project_path` search every linked checkout: they resolve scope with `projectlinks.ResolveScope`, the same mechanism monorepo containers use. `index_status` on the root reports the checkouts under `linked`.

`index_files` with a space root as `project_path`:

- `path` = the root: indexes every checkout.
- `path` inside a checkout (file or directory): indexes it under that checkout's project.
- `path` elsewhere under the root (e.g. `SPACE.md`, a loose `notes/` folder): returns an error. Index such a folder with its own `project_path` if you need it.

**Limits.** Loose files at the space root (plans, `SPACE.md`, folders that are not checkouts) are not indexed. Watchers are refused, as before, for `$HOME`, the wtg spaces root and the discovery root, so those are never fanned out by a watcher. A space with many checkouts starts one watcher per checkout.

## Indexing when a space is created

`ast-mcp index` indexes directories through the running server. It calls the dashboard API (`POST /api/index-project`) instead of opening the databases, because the server owns the index writer. It waits until indexing finishes.

```bash
ast-mcp index ~/spaces/my-task               # every checkout in the space
ast-mcp index ~/spaces/my-task/sandbox       # one checkout
ast-mcp index --json ~/spaces/my-task        # raw server response
```

The server is found at `$AST_DASHBOARD_URL`, else `http://127.0.0.1:$AST_DASHBOARD_PORT` (default 7830).

To index a space as soon as wtg creates it, call it from your wtg `always.run` hook. This is an example snippet, not something the installer writes:

```bash
#!/bin/bash
# ~/.config/wtg/on-event (wtg config: [always] run = "~/.config/wtg/on-event")
case "$WTG_SPACE_EVENT" in
  create)
    # Index every checkout now, copying unchanged files from ~/git/<repo>.
    # Backgrounded so wtg returns immediately; output goes to a log.
    if command -v ast-mcp >/dev/null 2>&1; then
      nohup ast-mcp index "$WTG_SPACE_ROOT" >>"$HOME/.astcache/space-index.log" 2>&1 &
    fi
    ;;
esac
```

If ast-mcp isn't running, the command fails with a "not reachable" message and nothing else happens. The space is still indexed with reuse the first time an agent uses it.

## Cleaning up space roots indexed before this change

Older versions indexed a space root as one project whenever an agent's cwd was the space. `ast-mcp prune-projects --space-roots` lists those projects. It is a dry run unless you pass `--apply`:

```console
$ ast-mcp prune-projects --space-roots
/Users/jason/spaces/devx  files=4562 symbols=48210 checkouts=5  covered
/Users/jason/spaces/old-task  files=812 symbols=9020 checkouts=2  skipped, 1 checkout(s) not indexed yet
  not indexed: /Users/jason/spaces/old-task/sandbox (run: ast-mcp index /Users/jason/spaces/old-task/sandbox)
dry run: re-run with --apply to prune 1 covered space root(s)

$ ast-mcp prune-projects --space-roots --apply
/Users/jason/spaces/devx  files=4562 symbols=48210 checkouts=5  pruned
...
```

A root is **covered** when every checkout in it has an index of its own. `--apply` prunes only covered roots: it links the checkouts under the root, then deletes the root's index rows (symbols, edges, vectors, indexed files, summaries, pending embeds). It does not touch notes, structured memory, handoffs or query history recorded under the root. Searches with the root as `project_path` keep working through the links.

The same API is available at `GET /api/prune-space-roots` (dry run) and `POST /api/prune-space-roots` with `{"apply": true}`.
