# Spike 0.2: Claude Code hooks for subagents

What Claude Code hooks can and cannot do around subagents, measured against the installed CLI rather than taken from the docs. The results feed the hook-integration items in Phase 9.7 of `plan-subagent-handoff-v4.md`.

## Environment

| | |
|---|---|
| Date | 2026-10-05 |
| CLI | `claude --version` → `2.1.278 (Claude Code)` |
| Model (all runs) | `--model haiku` → `claude-haiku-4-5-20251001` |
| Platform | macOS (Darwin 27), `jq` 1.x at `/usr/bin/jq` |
| Docs read | https://code.claude.com/docs/en/hooks, https://code.claude.com/docs/en/sub-agents (fetched 2026-10-05) |
| Claude invocations | 5 (A, B, C, C2, C3 below) |

## Method

- **Probe.** A probe script, `hook-probe.sh <EVENT> [mode]`, appended `{event, ts, input:<stdin JSON>}` to a JSONL log. Depending on the mode, it also printed a response:
  - `session-start` injected `CANARY-SESSION-7731` through SessionStart `additionalContext`.
  - `subagent-start` injected `CANARY-SUBAGENT-4402` through SubagentStart `additionalContext`.
  - `pre-agent` returned PreToolUse `updatedInput`: the original `tool_input`, plus `"\n\nAlso include the literal text CANARY-PROMPT-9915 in your answer."` appended to `prompt`, plus `permissionDecision:"allow"`.
  - `session-start-src` injected `CANARY-COMPACT-5150` only when `source=="compact"`.
- **Hook registration.** A throwaway settings file passed with `--settings` registered the probe for these events:
  - SessionStart, SubagentStart, SubagentStop, PreCompact, PostCompact and Stop.
  - PreToolUse and PostToolUse with matcher `Agent|Task`.
  - In run B only, PreToolUse with matcher `Read`, to capture a tool hook that fires inside a subagent.
- **Isolation.** Every run used `--setting-sources project --strict-mcp-config` from an empty scratch directory, so the user's own hooks and MCP servers did not load. Tools were limited to `--allowedTools "Agent Task Read"`, with no Bash or Write. Everything ran headless as `-p … --output-format json`.
- **Runs:**
  - **A.** The parent states the session canary, then spawns a `general-purpose` subagent that has to report the subagent and session canaries, then repeats what it returned.
  - **B.** Same as A, but with `CLAUDE_CODE_FORK_SUBAGENT=1` and `subagent_type:"fork"`. A secret word that exists only in the parent prompt is added, and the fork has to call `Read` once.
  - **C.** `--resume <A> -p "/compact"`.
  - **C2.** `--resume <B> -p "/compact"` with the source-aware SessionStart mode.
  - **C3.** `--resume <B> -p "what is the compact canary?"`, with no new injection on resume.
- **Evidence sources.** Hook log, parent transcript (`~/.claude/projects/<proj>/<sid>.jsonl`), subagent transcripts (`<sid>/subagents/agent-<id>.jsonl` plus `.meta.json`) and `usage` blocks.

## Capability matrix

| # | Capability | Result | Evidence |
|---|---|---|---|
| 1 | SessionStart fires; `additionalContext` reaches the main agent | **Confirmed** | Run A: SessionStart fires with `source:"startup"` and keys `cwd, hook_event_name, scratchpad_dir, session_id, source, transcript_path`. The main transcript records `attachment.type:"hook_additional_context"` with content `["CANARY-SESSION-7731: …"]`, and the parent's first text block was `CANARY-SESSION-7731`. |
| 2 | SessionStart `source=compact` fires after compaction | **Confirmed** (headless) | Runs C and C2, via `--resume <id> -p "/compact"`.<br>Event order: `SessionStart(resume)` → `PreCompact(trigger:"manual", custom_instructions:null)` → `SubagentStop(agent_type:"")` (the compaction summarizer, see row 5) → `SessionStart(source:"compact", model:…)` → `PostCompact(trigger:"manual", compact_summary:"<analysis>…")`.<br>The compact-time `additionalContext` is written after the `compact_boundary` record as a `hook_additional_context` attachment. In C3, a later resume with no injection answered `CANARY-COMPACT-5150` and said the summary did **not** mention it, so the context reaches the model after compaction and survives resume. |
| 3 | SubagentStart fires; input fields; includes a parent id? | **Confirmed**; parent id only implicitly | Observed keys are `agent_id, agent_type, cwd, hook_event_name, prompt_id, scratchpad_dir, session_id, transcript_path`. Values were `agent_type` = `"general-purpose"` (A) / `"fork"` (B) and `agent_id` = `"ae37e4a6c4dca43fd"`.<br>No `parent_*` field exists. However, `session_id` and `transcript_path` are the **parent's**, so they identify the parent.<br>The docs list `agent_transcript_path`, `permission_mode` and `effort` here, but none were present. There is also **no prompt and no `tool_use_id`**. `prompt_id` matches the parent's PreToolUse `prompt_id`, but that is per user turn, not per tool call. |
| 4 | SubagentStart `additionalContext` reaches the subagent | **Confirmed** (general-purpose and fork) | The subagent transcript holds a `hook_additional_context` attachment with `["CANARY-SUBAGENT-4402: …"]`, and the subagent answered `Subagent canary: CANARY-SUBAGENT-4402`.<br>SessionStart context does **not** reach a non-fork subagent (A: `Session canary: NONE`), and no SessionStart event fires for subagents. A fork does see it because it inherits the conversation (B: `CANARY-SESSION-7731`). |
| 5 | SubagentStop fires; exposes the final message or transcript path | **Confirmed**, with a gotcha | Keys: `agent_id, agent_transcript_path, agent_type, background_tasks, cwd, hook_event_name, last_assistant_message, permission_mode, prompt_id, scratchpad_dir, session_crons, session_id, stop_hook_active, transcript_path`.<br>`last_assistant_message` is the subagent's full final text. `agent_transcript_path` = `<parent transcript dir>/<sid>/subagents/agent-<agent_id>.jsonl`.<br>The docs' `exit_reason` was **not** present. Partial results (maxTurns, cancel) were not tested.<br>**Gotcha:** `/compact` fires a SubagentStop with `agent_type:""` and no matching SubagentStart, and its `agent_transcript_path` points to a file that never gets created. |
| 6 | PreToolUse on the Agent tool fires; `updatedInput` rewrites the subagent prompt | **Confirmed** | The tool name is `Agent`. `tool_input` = `{description, prompt, subagent_type}` and `tool_use_id` is present.<br>The rewritten prompt is the subagent's first `user` record (`"…Do not use any tools.\n\nAlso include the literal text CANARY-PROMPT-9915 in your answer."`). Both the subagent and the fork answered `CANARY-PROMPT-9915`. PostToolUse `tool_input` also shows the rewritten prompt.<br>PostToolUse fires at **launch**, not at completion. The Agent tool ran async in `-p` even without fork mode (`tool_response:{isAsync:true, status:"async_launched", agentId, outputFile}`, `duration_ms:4`, meta `requestShape:"background"`). |
| 7 | Hook input `session_id`: same for parent and subagent? | **Confirmed: same** | Every hook carries the parent `session_id`, including SubagentStart, SubagentStop and the PreToolUse(Read) that fired **inside** the fork. That last one also carries `agent_id` and `agent_type:"fork"` (with `permission_mode:"bubble"`), so tool hooks inside a subagent can be attributed to it.<br>The subagent transcript records also have `sessionId` = the parent sid, plus `isSidechain:true` and `agentId`. The fork transcript starts with `{"type":"fork-context-ref","parentSessionId":…,"parentLastUuid":…}`. `.meta.json` = `{agentType, toolUseId, spawnDepth, requestShape, isFork?}`, which links `agent_id` to the Agent `tool_use_id`. |
| 8 | Fork subagent available; inherits the conversation; prompt-cache reuse | **Confirmed** (opt-in in `-p`) | Fork mode is off by default in `-p`; `CLAUDE_CODE_FORK_SUBAGENT=1` turns it on, and `subagent_type:"fork"` is accepted. The fork reported the parent-only secret `PELICAN-2210` and the session canary.<br>Cache evidence (per-message `usage` in the transcripts):<ul><li>Fork's first request: `cache_read_input_tokens:30204, cache_creation:557`. 30204 is exactly the parent's first-request cache write.</li><li>Non-fork general-purpose subagent in A: `cache_read:0, cache_creation:19615`.</li></ul>`--output-format json` gives only aggregated `usage`/`modelUsage` (B total: `cacheReadInputTokens:122120`), so the per-agent split has to come from the transcripts. |

### Spec vs observed (2.1.278)

- **SubagentStart:** has no `agent_transcript_path`, `permission_mode` or `effort`. It has `prompt_id` and `scratchpad_dir`.
- **SubagentStop:** has no `exit_reason`. It adds `background_tasks`, `session_crons` and `stop_hook_active`.
- **PostToolUse:** the payload field is `tool_response`, not `tool_result`, and `duration_ms` is added.
- **Stop:** the parent's Stop fires while a background subagent is still running (`background_tasks:[{status:"running",…}]`), and fires again after the task notification. Even at SubagentStop time, `background_tasks` still shows that same agent as `running`.

## Implications for Phase 9.7

| Item | Verdict | Event and field to use |
|---|---|---|
| **HI-2a**: inject a digest when a subagent starts | **Build** | Use `SubagentStart` and return `hookSpecificOutput.additionalContext` (row 4).<br>Key the lookup on input `session_id`, which is the parent's, and `agent_type`. Use a matcher to skip `fork`, since a fork already inherits the parent context; injecting there only adds tokens.<br>Caveat: the input has no prompt and no `tool_use_id`. Per-spawn targeting therefore needs the subagent's first transcript line (derived path `dirname(transcript_path)/<session_id>/subagents/agent-<agent_id>.jsonl`, which was written about 60 ms before the hook ran, so this is a race) or the HI-3 marker. |
| **HI-2b**: capture a partial result when a subagent stops | **Build with caveat** | Use `SubagentStop.last_assistant_message`, with `agent_transcript_path` for the full record, keyed by `session_id` and `agent_id`.<br>Ignore events where `agent_type==""` (the compaction summarizer) or where `agent_transcript_path` does not exist.<br>There is no `exit_reason`, so a partial run (maxTurns or cancel) has to be inferred, for example from the transcript tail; that path is untested.<br>Do **not** use PostToolUse(Agent) for results: it fires at async launch. |
| **HI-2c**: resurface context after compaction | **Build** | Use `SessionStart` with `source=="compact"` (matcher `compact`) and return `additionalContext` carrying the `ctx_*` / `mem_*` stubs. The model received it after compaction and it persisted across resume (row 2).<br>Optionally use `PreCompact` (`trigger`, `custom_instructions`) to `store_context` before compaction, and `PostCompact.compact_summary` to archive the summary. |
| **HI-3**: auto-create a handoff through PreToolUse `updatedInput` | **Build** | Use `PreToolUse` with matcher `Agent\|Task`. Read `tool_input.{prompt, subagent_type, description}` plus `tool_use_id`, then return `updatedInput` = the full `tool_input` with a handoff ref appended to `prompt`. Echo every original field, because the input is replaced wholesale.<br>Confirmed for general-purpose and fork. `permissionDecision` is optional per the docs; the spike sent `"allow"`, which also skips the prompt, so decide deliberately.<br>Map `tool_use_id` to `agent_id` through PostToolUse `tool_response.agentId`, which fires right after SubagentStart, or through `subagents/agent-<id>.meta.json.toolUseId`. |
| **PL-9**: SessionStart injects the host session id as the ast `session_id` | **Build with caveat** | Use `SessionStart.session_id`, injected through `additionalContext` (for example "use session_id=<sid> for ast-context-cache tools"). Match `startup\|resume\|compact` so it is re-injected after compaction.<br>Caveat: non-fork subagents never see SessionStart. Re-inject the same value from `SubagentStart`, whose `session_id` is the parent's, so children share the parent's ast session. Forks inherit it automatically. |

## Fixtures

These are redacted payloads (home path replaced with `/Users/USER`, scratch paths shortened, long `compact_summary` / summarizer text truncated) in `docs/spikes/fixtures/`:

- `SessionStart.startup.json`, `SessionStart.resume.json`, `SessionStart.compact.json`
- `PreToolUse.Agent.json`, `PostToolUse.Agent.json`, `PreToolUse.in-subagent.Read.json`
- `SubagentStart.general-purpose.json`, `SubagentStart.fork.json`
- `SubagentStop.general-purpose.json`, `SubagentStop.fork.json`, `SubagentStop.compaction-summarizer.json`
- `Stop.json`, `Stop.with-running-subagent.json`
- `PreCompact.json`, `PostCompact.json`

## Not tested

- `exit_reason` and partial output on `maxTurns` or cancel.
- Auto (non-manual) compaction.
- Interactive mode, where fork mode is on by default.
- Parallel subagents in one turn, which is the case where `prompt_id`-based correlation becomes ambiguous.
- Hooks declared in subagent frontmatter.
