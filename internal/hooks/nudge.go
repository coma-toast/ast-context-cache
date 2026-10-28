package hooks

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"io/fs"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// Search-first nudge events. The Claude Code hook listens on PreToolUse for Grep, Glob and Read;
// the Cursor hook on preToolUse for Read, Grep, Glob and MCP tools.
const (
	EventSearchNudge       = "search-nudge"
	EventCursorSearchNudge = "cursor-search-nudge"
	// NudgeEnv overrides feature_search_nudge on the hook side; the server's flag still applies.
	NudgeEnv = "AST_FEATURE_SEARCH_NUDGE"
)

// Nudge kinds: which built-in tool the agent reached for.
const (
	NudgeGrep = "grep"
	NudgeGlob = "glob"
	NudgeRead = "read"
)

const (
	// nudgeMarkerTTL prunes per-session nudge markers; a session idle this long is over.
	nudgeMarkerTTL = 7 * 24 * time.Hour
	// maxNudgeTranscriptBytes is how much of a transcript's tail is scanned for an earlier
	// ast-context-cache call.
	maxNudgeTranscriptBytes = 16 << 20
	nudgeDirName            = "nudge"
	permissionAllow         = "allow"
	permissionDeny          = "deny"
	nudgePrefix             = "ast-context-cache search-first nudge (shown once per session, nothing was run): "
	nudgeSuffix             = " If the built-in tool really is the right one (an exact string, a non-code file, a path you already know), just call it again: this nudge will not block again in this session."
)

var (
	// NudgeMessages are the denial reasons, shared with the OpenCode plugin the installer writes.
	NudgeMessages = map[string]string{
		NudgeGrep: nudgePrefix + "before grepping code, call the ast-context-cache get_context_capsule tool (mode=auto) for symbol names and keywords, " +
			"or search_semantic to search by meaning. Both return ranked symbols with their source in far fewer tokens than grep and file reads." + nudgeSuffix,
		NudgeGlob: nudgePrefix + "to find code, call the ast-context-cache get_project_map tool (depth=2) for the layout, or get_context_capsule (mode=auto) / " +
			"search_semantic to find the files that define what you are looking for." + nudgeSuffix,
		NudgeRead: nudgePrefix + "before reading a whole code file, call the ast-context-cache get_file_context tool (default mode=skeleton) for its signatures, " +
			"then get_file_context mode=edit with symbol=<name>, or Read with offset/limit, for just the part you need." + nudgeSuffix,
	}
	// CodeExtensions are the file types the nudge covers for whole-file reads: the languages the
	// indexer parses into symbols. Markdown, JSON, YAML and the like are read directly.
	CodeExtensions = []string{".go", ".py", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".sh", ".bash", ".fish", ".tf"}
	// ASTToolNames identify an ast-context-cache call when a gateway or host drops the server name
	// from the tool name (Cursor reports MCP:<tool>).
	ASTToolNames = []string{
		"get_context_capsule", "search_semantic", "get_file_context", "get_project_map", "get_impact_graph", "diff_impact",
		"check_symbol_exists", "check_deletion_safety", "open_handoff", "get_pinned_note", "store_context", "fetch_context", "recall_memory",
	}
	// toolUseRegex matches a tool_use block's name in a Claude Code transcript. Anchoring on the
	// block's leading fields skips the tool definitions transcripts also record.
	toolUseRegex = regexp.MustCompile(`"type":\s*"tool_use",\s*"id":\s*"[^"]*",\s*"name":\s*"((?:mcp__|MCP:)[^"]{1,256})"`)
)

// cursorOutput is a Cursor permission hook response.
type cursorOutput struct {
	Permission   string `json:"permission"`
	UserMessage  string `json:"user_message,omitempty"`
	AgentMessage string `json:"agent_message,omitempty"`
}

type healthBody struct {
	SearchNudge *bool `json:"search_nudge"`
}

// searchNudge blocks a session's first Grep, Glob, or whole-file Read of a code file when no
// ast-context-cache tool has been used yet, then never again. It returns nil to allow. The flag
// is read last (env, then the server's /health), so most calls cost a stat and nothing else; an
// unreachable server or older server without the field means no nudge.
func (h *Handler) searchNudge(ctx context.Context, in *Input) (string, error) {
	session := in.SessionID
	if session == "" {
		session = in.ConversationID
	}
	if session == "" {
		return "", nil
	}
	marker := filepath.Join(h.nudgeDir, sessionKey(session+"\x00"+in.AgentID))
	if IsASTTool(in.ToolName) {
		return "", h.markNudged(marker)
	}
	kind := NudgeKind(in.ToolName, in.ToolInput)
	if kind == "" {
		return "", nil
	}
	if _, err := os.Stat(marker); err == nil {
		return "", nil
	}
	if on, ok := envBool(NudgeEnv); ok && !on {
		return "", nil
	}
	if transcriptUsedAST(nudgeTranscriptPath(in)) {
		return "", h.markNudged(marker)
	}
	if !h.nudgeEnabled(ctx) {
		return "", nil
	}
	if err := os.MkdirAll(h.nudgeDir, 0o700); err != nil {
		return "", errs.WrapMessage("failed to create nudge state dir", err, "path", h.nudgeDir)
	}
	f, err := os.OpenFile(marker, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		// A parallel call already nudged this session.
		return "", nil
	}
	_ = f.Close()
	h.pruneNudgeMarkers()
	return NudgeMessages[kind], nil
}

// NudgeKind classifies a built-in tool call: grep, glob, or a whole-file read of a code file.
// Anything else, including a Read with offset or limit, is "".
func NudgeKind(tool string, input map[string]json.RawMessage) string {
	switch strings.ToLower(tool) {
	case "grep":
		return NudgeGrep
	case "glob":
		return NudgeGlob
	case "read":
		for _, k := range []string{"offset", "limit"} {
			if v, ok := input[k]; ok && string(v) != "null" {
				return ""
			}
		}
		for _, k := range []string{"file_path", "filePath", "path", "target_file"} {
			if p := rawString(input[k]); p != "" {
				if slices.Contains(CodeExtensions, strings.ToLower(filepath.Ext(p))) {
					return NudgeRead
				}
				return ""
			}
		}
	}
	return ""
}

// IsASTTool recognizes an ast-context-cache tool by host name: mcp__ast-context-cache__<tool>
// (Claude Code), a gateway-prefixed mcp__<gateway>__<...>_<tool>, or MCP:<tool> (Cursor).
func IsASTTool(name string) bool {
	n := strings.ToLower(name)
	bare, cursor := strings.CutPrefix(n, "mcp:")
	if !cursor && !strings.HasPrefix(n, "mcp__") {
		return false
	}
	if strings.Contains(n, "ast-context-cache") || strings.Contains(n, "ast_context_cache") {
		return true
	}
	for _, t := range ASTToolNames {
		if bare == t || strings.HasSuffix(n, "_"+t) || strings.HasSuffix(n, "-"+t) || strings.HasSuffix(n, "."+t) {
			return true
		}
	}
	return false
}

// nudgeTranscriptPath is the calling agent's own transcript: a subagent's when the hook fires
// inside one, since the parent's ast-context-cache calls never reached the subagent's context.
func nudgeTranscriptPath(in *Input) string {
	if in.AgentID != "" {
		return subagentTranscriptPath(in)
	}
	return in.TranscriptPath
}

// transcriptUsedAST reports whether the transcript's tail records an ast-context-cache tool call.
func transcriptUsedAST(path string) bool {
	if path == "" {
		return false
	}
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()
	if info, err := f.Stat(); err == nil && info.Size() > maxNudgeTranscriptBytes {
		if _, err := f.Seek(-maxNudgeTranscriptBytes, io.SeekEnd); err != nil {
			return false
		}
	}
	data, err := io.ReadAll(io.LimitReader(f, maxNudgeTranscriptBytes))
	if err != nil {
		return false
	}
	for _, m := range toolUseRegex.FindAllSubmatch(data, -1) {
		if IsASTTool(string(m[1])) {
			return true
		}
	}
	return false
}

// nudgeEnabled asks the server for feature_search_nudge through its unauthenticated /health.
func (h *Handler) nudgeEnabled(ctx context.Context) bool {
	if h.healthURL == "" {
		return false
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, h.healthURL, nil)
	if err != nil {
		return false
	}
	resp, err := h.client.http.Do(req)
	if err != nil {
		logger.Debug("Search nudge health check failed", "error", err)
		return false
	}
	defer resp.Body.Close()
	var body healthBody
	if resp.StatusCode != http.StatusOK || json.NewDecoder(io.LimitReader(resp.Body, maxResponseBytes)).Decode(&body) != nil {
		return false
	}
	return body.SearchNudge != nil && *body.SearchNudge
}

func (h *Handler) markNudged(marker string) error {
	if err := os.MkdirAll(h.nudgeDir, 0o700); err != nil {
		return errs.WrapMessage("failed to create nudge state dir", err, "path", h.nudgeDir)
	}
	f, err := os.OpenFile(marker, os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		return errs.WrapMessage("failed to write nudge marker", err, "path", marker)
	}
	return f.Close()
}

// pruneNudgeMarkers drops markers older than nudgeMarkerTTL. It runs once per nudge, so at most
// once per session.
func (h *Handler) pruneNudgeMarkers() {
	entries, err := os.ReadDir(h.nudgeDir)
	if err != nil {
		return
	}
	cutoff := h.registry.now().Add(-nudgeMarkerTTL)
	for _, e := range entries {
		if info, err := e.Info(); err == nil && info.ModTime().Before(cutoff) {
			if err := os.Remove(filepath.Join(h.nudgeDir, e.Name())); err != nil && !errors.Is(err, fs.ErrNotExist) {
				logger.Debug("Failed to prune nudge marker", "name", e.Name(), "error", err)
			}
		}
	}
}

// HealthURL turns the MCP endpoint into its server's /health URL.
func HealthURL(mcpURL string) string {
	u, err := url.Parse(mcpURL)
	if err != nil || u.Host == "" {
		return ""
	}
	u.Path = strings.TrimSuffix(strings.TrimSuffix(u.Path, "/"), "/mcp") + "/health"
	u.RawQuery, u.Fragment = "", ""
	return u.String()
}

// envBool reads a boolean env var the way the flags package does.
func envBool(name string) (bool, bool) {
	v := strings.TrimSpace(os.Getenv(name))
	if v == "" {
		return false, false
	}
	on, err := strconv.ParseBool(v)
	return on, err == nil
}

// nudgeResponse renders the nudge decision for the host. Cursor always gets an explicit
// permission so an empty answer can never read as invalid; Claude Code gets nothing to allow.
func nudgeResponse(event, reason string) any {
	if event == EventCursorSearchNudge {
		if reason == "" {
			return cursorOutput{Permission: permissionAllow}
		}
		return cursorOutput{Permission: permissionDeny, UserMessage: "ast-context-cache: nudged the agent toward its code search tools (once per session)", AgentMessage: reason}
	}
	if reason == "" {
		return nil
	}
	return &Output{HookSpecificOutput: HookSpecificOutput{HookEventName: hookNamePreToolUse, PermissionDecision: permissionDeny, PermissionDecisionReason: reason}}
}
