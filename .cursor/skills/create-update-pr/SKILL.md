---
name: create-update-pr
description: Create or update a pull request with branch naming, pre-PR checks, and secrets check. Use when the user wants to open a PR, update an existing PR, or push and create/update a PR.
---

# Create or Update Pull Request

Follow the project workflow to create a new PR or update an existing one. Use MCP when available (GitHub, Atlassian); otherwise use shell commands. Follow the repo's [AGENTS.md](AGENTS.md) for branch naming and pre-PR conventions.

---

## Target repo (where to run git and PR operations)

All git and PR operations must run in a **single repository**. Determine the target repo from the **currently opened folder** (workspace root):

- **If the open folder is not the slide root** (i.e. it is a component repo such as `docs`, `box`, `cloud`, `agent`): the target repo is that folder. Run all `git`, `gh`, and MCP PR commands from that directory (you are already in the target repo).
- **If the open folder is the slide root** (parent of `agent`, `box`, `docs`, etc.): there is no single target repo. Ask the user which repo the PR is for, then run all commands from that repo’s path (e.g. `git -C {slide_root}/docs ...` or `cd {slide_root}/docs` before subsequent commands).

Use the target repo for: repository context (owner/repo from `git remote get-url origin`), branch state, pre-PR checks, secrets check, push, and PR create/update.

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
3. **Ask:** Does this change need a Jira ticket? Use existing ticket id for branch name (e.g. **SLIDE-XXX** or **QA-XXX**—docs repo uses either; other repos may use QA-XXX per AGENTS.md). Create via Atlassian MCP (appropriate project) if needed, or no ticket → `NO-TICKET-short-description` (ask user for the short description).
4. **Branch (if new task):** Do **not** rename the current branch (no `git branch -m`). Always switch to `main`, pull, then create a **new** branch: `git checkout main && git pull origin main`; then `git checkout -b SLIDE-123` or `git checkout -b QA-123` or `git checkout -b NO-TICKET-short-description` (use the ticket or description from step 3). Optionally create branch via GitHub MCP `create_branch`. Always branch from `main` unless the user says otherwise.
5. **Pre-PR checks:** Per repo AGENTS.md (e.g. `make lint && make vulncheck && make test`). Do not create the PR until these pass. If the repo mentions E2E for API/agent/disk changes, remind the user to run E2E before creating the PR.
6. **Secrets check:** Run the steps in **Secrets check (before any push)** below. Do not push if anything is flagged until the user resolves or confirms.
7. **Push and create PR:** Ensure commits are in; push the branch; use [.github/pull_request_template.md](.github/pull_request_template.md) for the body; create via MCP `create_pull_request` or `gh pr create --base main --title "..." --body "..."`. Return the PR URL.
8. **E2E test coverage:** After creating the component PR, check if the `create-tests` skill is installed (look for `create-tests/SKILL.md` in any provider's skills directory at the target repo, e.g. `.cursor/skills/create-tests/SKILL.md`, `.claude/skills/create-tests/SKILL.md`).
   - **If installed:** Ask the user: "Would you like to generate E2E tests in slapi for this PR?" If yes, invoke the `create-tests` skill, passing context: the component repo path, branch name, and PR URL. After the slapi PR is created, update the component PR's body to add the slapi PR URL under **Dependencies**.
   - **If NOT installed:** Notify the user: "The optional `create-tests` skill is available but not installed in this repo. It can auto-generate Playwright E2E tests in the slapi repo for your changes. Would you like to install it?" If the user says yes, run the installer: `{slide_root}/docs/dev/agents/install.sh --target "<target_repo>" --tool "<tool_from_.slide/tool>" --skills create-tests`. After installation, offer to invoke it immediately for the current PR.

---

## Mode: Update PR (current branch)

**Scope:** Operate on the **current branch** only (`git branch --show-current`). If current branch is `main` or empty, tell the user to check out a feature branch and stop.

1. **Context:** Current branch, owner/repo from remote, GitHub user (MCP `get_me` or fallback).
2. **PR state:** Find any PR for the current branch (MCP `list_pull_requests` with `head: <owner>:<current-branch>`, `state: all`, or `gh pr list --head "<current-branch>" --state all --json number,state,title,url`).
3. **If open PR exists (Update flow):**
   - Run **Secrets check** (below). Do not push if anything is flagged.
   - Ensure changes are committed; then `git push`.
   - If the user wants title/body updated, call MCP `update_pull_request` or `gh pr edit <number> --title "..." --body "..."`.
   - **E2E test coverage:** After pushing, check if the `create-tests` skill is installed (same check as Create PR step 8).
     - **If installed:** Ask: "Would you like to update or add E2E tests in slapi for the latest changes?" If yes, invoke the `create-tests` skill. If a slapi PR already exists for this branch, the skill updates it rather than creating a new one.
     - **If NOT installed:** Notify the user: "The optional `create-tests` skill is available but not installed. It can auto-generate Playwright E2E tests in slapi for your changes. Would you like to install it?" If yes, run the installer (same command as Create PR step 8), then offer to invoke it for the current branch.
   - Return the PR URL.
4. **If merged/closed/no PR (Create flow):**
   - Pre-PR checks per AGENTS.md; then **Secrets check**; then ensure committed, push, fill PR template, create PR via MCP or `gh pr create`. Return the new PR URL.

---

## Secrets check (before any push)

Run this before every push (create or update flow). If anything is flagged, do not push until the user resolves or explicitly confirms.

1. **Files to be pushed:** `git diff --name-only main...HEAD` (use the branch in scope—for Create mode that’s the chat-established branch; for Update mode that’s the current branch). Only tracked files.
2. **Names similar to gitignored secret patterns:** From the repo `.gitignore`, treat as secret-like any pattern that clearly refers to secrets (e.g. `secrets/`, `.env`, `.vault_pass`, `.vault_password`, `.secrets`, `*.tfvars`, `.ghtoken`, `*credentials*`, `*.pem`, `*.key` in a key/cert context). For each file in the diff, check if its path or basename is **similar** (e.g. `.env` → `.env.local`, `.env.production`; `secrets/` → `secrets.yml` or `config/secrets.json`; `.vault_pass` → `vault_pass.txt`). If any file matches, list it and do not push until the user confirms it is safe or removes it.
3. **Secret-like content:** In those files, scan **added** lines (e.g. `git diff main...HEAD -- <file>`) for: `password\s*=`, `api_key\s*=`, `secret\s*=`, `-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----`, AWS key-like strings, or long high-entropy tokens. If found, do not push; report file and line(s); ask the user to remove or redact.

---

## MCP fallbacks

| Feature            | Prefer MCP                         | Fallback |
| ------------------ | ----------------------------------- | -------- |
| Check/list PRs     | `user-github` `list_pull_requests`  | `gh pr list --head <branch>` |
| Create PR          | `user-github` `create_pull_request` | `gh pr create` |
| Update PR          | `user-github` `update_pull_request`  | `gh pr edit <number> ...` |
| Create Jira ticket | `user-Atlassian-MCP-Server` (QA)     | User creates in Jira |
| Create branch      | `user-github` `create_branch`        | `git checkout -b` + push |
| Current user       | `user-github` `get_me`               | `gh auth status` |

Use MCP first; use fallbacks only when the server is unavailable or the call fails.
