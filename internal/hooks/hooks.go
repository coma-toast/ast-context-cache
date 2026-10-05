// Package hooks implements "ast-mcp hook <event>", the Claude Code hook handlers for subagent
// handoffs (PL-9, PL-10, HI-2, HI-3, HI-4). Each run reads one hook payload from stdin, calls the
// running server's MCP tools over HTTP, and writes the hook's JSON response to stdout. Hooks fail
// open (HI-5): any error, timeout, or unreachable server leaves stdout empty, so Claude Code
// carries on as if no hook ran. The package never opens the database or starts a server.
// docs/spikes/claude-code-hooks.md records the payloads and behaviors this relies on.
package hooks

import (
	"context"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/logging"
)

// Hook event names, as the installer writes them into "<ast-mcp> hook <event>".
const (
	EventSessionStart     = "session-start"
	EventSubagentStart    = "subagent-start"
	EventSubagentStop     = "subagent-stop"
	EventPreToolUseAgent  = "pre-tool-use-agent"
	hookNameSessionStart  = "SessionStart"
	hookNameSubagentStart = "SubagentStart"
	hookNamePreToolUse    = "PreToolUse"
)

const (
	// Timeout bounds a whole hook run, stdin included (HI-5). The installer gives Claude Code a
	// 3s hook timeout, so the hook gives up first.
	Timeout = 2 * time.Second
	// maxInputBytes bounds the hook payload read from stdin.
	maxInputBytes = 4 << 20
	// maxBriefPromptRunes is how much of the Agent prompt goes into an auto-created brief.
	maxBriefPromptRunes = 1500
	// maxDigestTokens caps the open digest injected at subagent start.
	maxDigestTokens = 1200
	// maxListedHandoffs caps the handoffs re-surfaced after compaction, keeping it near 400 tokens.
	maxListedHandoffs = 10
	maxLabelRunes     = 60
	// maxTranscriptScanBytes is how much of a transcript the hooks read looking for a stub or a
	// final message.
	maxTranscriptScanBytes = 256 << 10
	// charsPerToken matches the server's rough token estimate.
	charsPerToken = 4

	agentTypeFork   = "fork"
	toolNameAgent   = "Agent"
	toolNameTask    = "Task"
	sourceCompact   = "compact"
	statusOpen      = "open"
	statusPartial   = "partial"
	handoffStubMark = "[handoff hof_"
	// noFinalMessage is the partial result's content when the subagent left no text.
	noFinalMessage = "(the subagent stopped without a final message)"
)

const (
	toolHandoff     = "handoff"
	toolOpenHandoff = "open_handoff"
)

var (
	logger       = logging.Tagged("hooks")
	stubRefRegex = regexp.MustCompile(`\[handoff (hof_[0-9a-f]{16})\]`)
)

// Handler runs hook events against one MCP endpoint and registry.
type Handler struct {
	client   *Client
	registry *Registry
}

// Input is a Claude Code hook payload; each event fills a subset of the fields.
type Input struct {
	SessionID            string                     `json:"session_id"`
	TranscriptPath       string                     `json:"transcript_path"`
	CWD                  string                     `json:"cwd"`
	HookEventName        string                     `json:"hook_event_name"`
	Source               string                     `json:"source"`
	AgentID              string                     `json:"agent_id"`
	AgentType            string                     `json:"agent_type"`
	AgentTranscriptPath  string                     `json:"agent_transcript_path"`
	LastAssistantMessage string                     `json:"last_assistant_message"`
	ToolName             string                     `json:"tool_name"`
	ToolInput            map[string]json.RawMessage `json:"tool_input"`
	ToolUseID            string                     `json:"tool_use_id"`
}

// Output is a hook response.
type Output struct {
	HookSpecificOutput HookSpecificOutput `json:"hookSpecificOutput"`
}

// HookSpecificOutput carries the injected context or the rewritten tool input. It never sets
// permissionDecision: an auto-created handoff must not skip the user's permission prompt.
type HookSpecificOutput struct {
	HookEventName     string                     `json:"hookEventName"`
	AdditionalContext string                     `json:"additionalContext,omitempty"`
	UpdatedInput      map[string]json.RawMessage `json:"updatedInput,omitempty"`
}

// New returns a handler that calls the MCP endpoint at url and keeps its registry in dir.
func New(url, dir string) *Handler {
	return &Handler{client: NewClient(url), registry: NewRegistry(dir)}
}

// Run handles one hook event and writes its response to stdout. It never fails: an error or
// panic is logged and leaves stdout empty (HI-5). The response is written in one piece, only
// once it is complete.
func (h *Handler) Run(ctx context.Context, event string, stdin io.Reader, stdout io.Writer) {
	defer func() {
		if r := recover(); r != nil {
			logger.Error("Hook panicked", "event", event, "panic", r)
		}
	}()
	ctx, cancel := context.WithTimeout(ctx, Timeout)
	defer cancel()
	out, err := h.handle(ctx, event, stdin)
	if err != nil {
		logger.Debug("Hook produced no output", "event", event, "error", err)
		return
	}
	if out == nil {
		return
	}
	data, err := json.Marshal(out)
	if err != nil {
		logger.Debug("Failed to encode hook output", "event", event, "error", err)
		return
	}
	if _, err := stdout.Write(append(data, '\n')); err != nil {
		logger.Debug("Failed to write hook output", "event", event, "error", err)
	}
}

func (h *Handler) handle(ctx context.Context, event string, stdin io.Reader) (*Output, error) {
	in, err := readInput(ctx, stdin)
	if err != nil {
		return nil, err
	}
	switch event {
	case EventSessionStart:
		return h.sessionStart(ctx, in)
	case EventSubagentStart:
		return h.subagentStart(ctx, in)
	case EventSubagentStop:
		return nil, h.subagentStop(ctx, in)
	case EventPreToolUseAgent:
		return h.preToolUseAgent(ctx, in)
	}
	return nil, errs.New("unknown hook event", "event", event)
}

// sessionStart tells the main agent to use the host session id as its ast-context-cache
// session_id (PL-9, HI-4). After compaction it also re-surfaces the session's handoffs (HI-2c);
// if listing them fails the session id is still injected, since that needs no server.
func (h *Handler) sessionStart(ctx context.Context, in *Input) (*Output, error) {
	if in.SessionID == "" {
		return nil, errs.New("session_id missing")
	}
	text := "ast-context-cache: use session_id=" + in.SessionID + " for all ast-context-cache tool calls in this conversation."
	if in.Source == sourceCompact {
		var list listResult
		err := h.client.Call(ctx, toolHandoff, map[string]any{"action": "list", "session_id": in.SessionID}, &list)
		if err != nil {
			logger.Debug("Failed to list handoffs after compaction", "session", in.SessionID, "error", err)
		}
		if err == nil && len(list.Handoffs) > 0 {
			text += "\n" + formatHandoffList(in.SessionID, list.Handoffs)
		}
	}
	return contextOutput(hookNameSessionStart, text), nil
}

// subagentStart gives a non-fork subagent the parent's session id (PL-9), since SessionStart
// never reaches it. When the subagent was handed a handoff, the hook opens it for the child and
// injects the child session id and digest instead (HI-2a). The payload has no prompt, so the
// handoff comes from the stub in the subagent's transcript when it is already written, else from
// the oldest handoff pre-tool-use-agent queued for this session. Forks inherit the parent's
// context, so they get nothing.
func (h *Handler) subagentStart(ctx context.Context, in *Input) (*Output, error) {
	if in.SessionID == "" || in.AgentType == agentTypeFork {
		return nil, nil
	}
	generic := "ast-context-cache: the parent conversation's session_id is " + in.SessionID + ". If your prompt contains " +
		"[handoff hof_…], call open_handoff first and use the child session_id it returns for all ast-context-cache calls."
	ref := stubRefFromTranscript(subagentTranscriptPath(in))
	if ref != "" {
		if err := h.registry.DropPending(ctx, in.SessionID, ref); err != nil {
			logger.Debug("Failed to drop pending handoff", "session", in.SessionID, "handoff", ref, "error", err)
		}
	}
	if ref == "" {
		p, ok, err := h.registry.TakePending(ctx, in.SessionID)
		if err != nil || !ok {
			return contextOutput(hookNameSubagentStart, generic), nil
		}
		ref = p.Ref
	}
	var open openResult
	err := h.client.Call(ctx, toolOpenHandoff, map[string]any{"action": "open", "handoff": ref, "project_path": in.CWD}, &open)
	if err != nil || open.SessionID == "" {
		logger.Debug("Failed to open handoff for subagent", "session", in.SessionID, "handoff", ref, "error", err)
		return contextOutput(hookNameSubagentStart, generic), nil
	}
	if in.AgentID != "" {
		if err := h.registry.SetAgent(ctx, in.SessionID, in.AgentID, AgentHandoff{Ref: ref, SessionID: open.SessionID}); err != nil {
			logger.Debug("Failed to record subagent handoff", "session", in.SessionID, "agent", in.AgentID, "error", err)
		}
	}
	text := "ast-context-cache: this subagent is handoff " + ref + "; open_handoff was already called for you, so do not open it again. " +
		"Use session_id=" + open.SessionID + " for all ast-context-cache calls, open_handoff action expand for more detail, " +
		"and handoff action complete with that session_id to return your result.\n\n" + formatDigest(&open)
	return contextOutput(hookNameSubagentStart, text), nil
}

// subagentStop stores a subagent's final message as a partial result when it stopped without
// completing the handoff the hook opened for it (HI-2b). The compaction summarizer's stop event
// (empty agent_type, a transcript that never exists) is ignored.
func (h *Handler) subagentStop(ctx context.Context, in *Input) error {
	if in.SessionID == "" || in.AgentID == "" || in.AgentType == "" {
		return nil
	}
	if in.AgentTranscriptPath != "" {
		if _, err := os.Stat(in.AgentTranscriptPath); err != nil {
			return nil
		}
	}
	a, ok, err := h.registry.TakeAgent(ctx, in.SessionID, in.AgentID)
	if err != nil || !ok {
		return err
	}
	var col collectResult
	if err := h.client.Call(ctx, toolHandoff, map[string]any{"action": "collect", "handoff": a.Ref}, &col); err != nil {
		return err
	}
	open := false
	for _, c := range col.Children {
		if c.SessionID == a.SessionID {
			open = c.Status == statusOpen
		}
	}
	if !open {
		return nil
	}
	content := strings.TrimSpace(in.LastAssistantMessage)
	if content == "" {
		content = lastAssistantText(in.AgentTranscriptPath)
	}
	if content == "" {
		content = noFinalMessage
	}
	var done completeResult
	return h.client.Call(ctx, toolHandoff, map[string]any{
		"action": "complete", "session_id": a.SessionID, "status": statusPartial, "content": content, "project_path": in.CWD,
	}, &done)
}

// preToolUseAgent creates a handoff from the parent session for an Agent call and appends its
// stub to the subagent prompt (HI-3). updatedInput replaces the tool input wholesale, so every
// original field is echoed. Forks inherit the parent's context and prompts that already carry a
// stub were handed off by the agent itself; both are left alone. Inside a subagent, the handoff
// is created from the child session the hook opened for it, so it nests in the same tree.
func (h *Handler) preToolUseAgent(ctx context.Context, in *Input) (*Output, error) {
	if in.SessionID == "" || in.ToolInput == nil || (in.ToolName != toolNameAgent && in.ToolName != toolNameTask) {
		return nil, nil
	}
	prompt, desc, subagentType := rawString(in.ToolInput["prompt"]), rawString(in.ToolInput["description"]), rawString(in.ToolInput["subagent_type"])
	if subagentType == agentTypeFork || strings.Contains(prompt, handoffStubMark) {
		return nil, nil
	}
	creator := in.SessionID
	if in.AgentID != "" {
		a, ok, err := h.registry.Agent(ctx, in.SessionID, in.AgentID)
		if err != nil || !ok {
			return nil, err
		}
		creator = a.SessionID
	}
	brief := strings.TrimSpace(desc + "\n\n" + truncateRunes(prompt, maxBriefPromptRunes))
	var created createResult
	err := h.client.Call(ctx, toolHandoff, map[string]any{
		"action": "create", "session_id": creator, "brief": brief, "label": desc, "project_path": in.CWD,
	}, &created)
	if err != nil {
		return nil, err
	}
	if created.Ref == "" || created.Stub == "" {
		return nil, errs.New("handoff create returned no stub")
	}
	updated := make(map[string]json.RawMessage, len(in.ToolInput))
	for k, v := range in.ToolInput {
		updated[k] = v
	}
	newPrompt, err := json.Marshal(prompt + "\n\n" + created.Stub)
	if err != nil {
		return nil, errs.WrapMessage("failed to encode prompt", err)
	}
	updated["prompt"] = newPrompt
	// Queued under the root session: SubagentStart carries the root's session_id even for a
	// nested spawn.
	if err := h.registry.AddPending(ctx, in.SessionID, PendingHandoff{Ref: created.Ref, ToolUseID: in.ToolUseID}); err != nil {
		logger.Debug("Failed to queue pending handoff", "session", in.SessionID, "handoff", created.Ref, "error", err)
	}
	return &Output{HookSpecificOutput: HookSpecificOutput{HookEventName: hookNamePreToolUse, UpdatedInput: updated}}, nil
}

// readInput decodes the hook payload. The read runs aside so a host that never closes stdin
// can't hold the hook past its deadline.
func readInput(ctx context.Context, r io.Reader) (*Input, error) {
	type result struct {
		data []byte
		err  error
	}
	ch := make(chan result, 1)
	go func() {
		data, err := io.ReadAll(io.LimitReader(r, maxInputBytes))
		ch <- result{data, err}
	}()
	select {
	case <-ctx.Done():
		return nil, errs.WrapMessage("timed out reading hook input", ctx.Err())
	case res := <-ch:
		if res.err != nil {
			return nil, errs.WrapMessage("failed to read hook input", res.err)
		}
		var in Input
		if err := json.Unmarshal(res.data, &in); err != nil {
			return nil, errs.WrapMessage("failed to decode hook input", err)
		}
		return &in, nil
	}
}

func contextOutput(hookEvent, text string) *Output {
	return &Output{HookSpecificOutput: HookSpecificOutput{HookEventName: hookEvent, AdditionalContext: text}}
}

// subagentTranscriptPath derives the subagent's transcript from the parent's, which
// SubagentStart carries: <parent dir>/<session_id>/subagents/agent-<agent_id>.jsonl.
func subagentTranscriptPath(in *Input) string {
	if in.TranscriptPath == "" || in.AgentID == "" {
		return ""
	}
	return filepath.Join(filepath.Dir(in.TranscriptPath), in.SessionID, "subagents", "agent-"+in.AgentID+".jsonl")
}

// stubRefFromTranscript returns the first handoff ref in the head of a transcript. Claude Code
// writes the subagent's prompt shortly before SubagentStart, so a missing file just means the
// caller falls back to the pending queue.
func stubRefFromTranscript(path string) string {
	if path == "" {
		return ""
	}
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	head, _ := io.ReadAll(io.LimitReader(f, maxTranscriptScanBytes))
	if m := stubRefRegex.FindSubmatch(head); m != nil {
		return string(m[1])
	}
	return ""
}

func rawString(raw json.RawMessage) string {
	var s string
	if len(raw) == 0 || json.Unmarshal(raw, &s) != nil {
		return ""
	}
	return s
}
