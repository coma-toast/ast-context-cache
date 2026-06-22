---
name: create-update-pr
description: Create or update a pull request with branch naming, pre-PR checks, and secrets check. Use when the user wants to open a PR, update an existing PR, or push and create/update a PR.
---

# Create or Update Pull Request

Follow the project workflow to create a new PR or update an existing one. Use MCP when available (GitHub); otherwise use shell commands. Follow the repo's [AGENTS.md](AGENTS.md) for pre-PR conventions.

---

## Target repo (where to run git and PR operations)

All git and PR operations run in the **workspace root** (this repository). Use `git remote get-url origin` for owner/repo; run `git`, `gh`, and MCP PR commands from that directory.

Use the target repo for: repository context, branch state, pre-PR checks, secrets check, push, and PR create/update.

---

## Never merge pull requests

**NEVER merge a PR on the user's behalf** — not `gh pr merge`, GitHub MCP merge, API merge, or auto-merge — **even if the user says "merge PR #…", "merge first", or "merge it".**

When the user asks to merge:

1. Confirm merge-readiness (CI, conflicts, reviews) if relevant.
2. Return the PR URL and status.
3. Tell the user to **click Merge in GitHub** (or run `gh pr merge` themselves).
4. After they merge, you may fetch `main`, merge into a feature branch, re-run tests, or update a follow-up PR.

**Allowed:** create/update PR, push, `git merge origin/main` on a feature branch, babysit until merge-ready.

**Forbidden:** `gh pr merge` and any GitHub action that completes the merge.

---

## Mode: Create PR (new branch / first-time PR)

**Scope:** The branch is the one **established in this conversation** (user-named or created in this chat). Do not default to `git branch --show-current` if the user has not yet specified or created a branch—ask or create it first.

1. **Check for existing PR** for that branch (GitHub MCP `list_pull_requests` with `head: <owner>:<branch>`, `state: open`, or `gh pr list --head "<branch>"`). If an open PR exists, offer to update it instead; stop unless the user wants a new PR on a different branch.
2. **Repository context**: Owner/repo from `git remote get-url origin`; GitHub user from MCP `get_me` or `gh auth status`.
3. **Branch name:** Ask the user for a short description if not already provided. Use one of these patterns:
   - `NO-TICKET-short-description`
   - `feat/short-description`
   - `fix/short-description`
   - `chore/short-description`
4. **Branch (if new task):** Do **not** rename the current branch (no `git branch -m`). Always switch to `main`, pull, then create a **new** branch: `git checkout main && git pull origin main`; then `git checkout -b <branch-name>` (from step 3). Optionally create branch via GitHub MCP `create_branch`. Always branch from `main` unless the user says otherwise.
5. **Pre-PR checks:** Run `make test`. Do not create the PR until tests pass.
6. **Secrets check:** Run the steps in **Secrets check (before any push)** below. Do not push if anything is flagged until the user resolves or confirms.
7. **Push and create PR:** Ensure commits are in; push the branch; create via MCP `create_pull_request` or `gh pr create --base main --title "..." --body "..."` with a **Summary** and **Test plan** section in the body. Return the PR URL.

---

## Mode: Update PR (current branch)

**Scope:** Operate on the **current branch** only (`git branch --show-current`). If current branch is `main` or empty, tell the user to check out a feature branch and stop.

1. **Context:** Current branch, owner/repo from remote, GitHub user (MCP `get_me` or fallback).
2. **PR state:** Find any PR for the current branch (MCP `list_pull_requests` with `head: <owner>:<current-branch>`, `state: all`, or `gh pr list --head "<current-branch>" --state all --json number,state,title,url`).
3. **If open PR exists (Update flow):**
   - Run **Secrets check** (below). Do not push if anything is flagged.
   - Ensure changes are committed; then `git push`.
   - If the user wants title/body updated, call MCP `update_pull_request` or `gh pr edit <number> --title "..." --body "..."`.
   - Return the PR URL.
4. **If merged/closed/no PR (Create flow):**
   - Run `make test`; then **Secrets check**; then ensure committed, push, create PR via MCP or `gh pr create` with Summary + Test plan. Return the new PR URL.

---

## Secrets check (before any push)

Run this before every push (create or update flow). If anything is flagged, do not push until the user resolves or explicitly confirms.

1. **Files to be pushed:** `git diff --name-only main...HEAD` (use the branch in scope—for Create mode that's the chat-established branch; for Update mode that's the current branch). Only tracked files.
2. **Names similar to gitignored secret patterns:** From the repo `.gitignore`, treat as secret-like any pattern that clearly refers to secrets (e.g. `secrets/`, `.env`, `.vault_pass`, `.vault_password`, `.secrets`, `*.tfvars`, `.ghtoken`, `*credentials*`, `*.pem`, `*.key` in a key/cert context). For each file in the diff, check if its path or basename is **similar** (e.g. `.env` → `.env.local`, `.env.production`; `secrets/` → `secrets.yml` or `config/secrets.json`; `.vault_pass` → `vault_pass.txt`). If any file matches, list it and do not push until the user confirms it is safe or removes it.
3. **Secret-like content:** In those files, scan **added** lines (e.g. `git diff main...HEAD -- <file>`) for: `password\s*=`, `api_key\s*=`, `secret\s*=`, `-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----`, AWS key-like strings, or long high-entropy tokens. If found, do not push; report file and line(s); ask the user to remove or redact.

---

## MCP fallbacks

| Feature            | Prefer MCP                         | Fallback |
| ------------------ | ----------------------------------- | -------- |
| Check/list PRs     | `user-github` `list_pull_requests`  | `gh pr list --head <branch>` |
| Create PR          | `user-github` `create_pull_request` | `gh pr create` |
| Update PR          | `user-github` `update_pull_request`  | `gh pr edit <number> ...` |
| Create branch      | `user-github` `create_branch`        | `git checkout -b` + push |
| Current user       | `user-github` `get_me`               | `gh auth status` |

Use MCP first; use fallbacks only when the server is unavailable or the call fails.
