package mcp

import (
	"os"
	"strings"
)

const (
	JSONRPCVersion = "2.0"
	ParseError     = -32700
	InvalidRequest = -32600
	MethodNotFound = -32601
	InvalidParams  = -32602
	InternalError  = -32603
)

// Tier represents tool access levels. Higher tiers include all lower tiers.
type Tier string

const (
	TierCore     Tier = "core"     // Read-only, safe tools (search, index_status, project_map)
	TierExtended Tier = "extended" // Read-write tools (index_files, cache_summary, impact_graph)
	TierComplete Tier = "complete" // Full access including code execution sandbox
)

// TierIncludes returns true if the active tier grants access to a tool's required tier.
func TierIncludes(active, required Tier) bool {
	order := map[Tier]int{TierCore: 0, TierExtended: 1, TierComplete: 2}
	return order[active] >= order[required]
}

// ParseTier converts a string to Tier, defaulting to TierComplete for unknown values.
func ParseTier(s string) Tier {
	switch strings.ToLower(s) {
	case "core":
		return TierCore
	case "extended":
		return TierExtended
	case "complete":
		return TierComplete
	default:
		return TierComplete
	}
}

// ServerConfig holds runtime configuration for tool filtering and sandbox behavior.
type ServerConfig struct {
	ActiveTier  Tier // Which tier of tools to expose
	CodeMode    bool // Whether execute_code is enabled (requires TierComplete)
	SandboxSecs int  // Timeout for code sandbox execution
}

// DefaultConfig returns config from environment variables with sensible defaults.
func DefaultConfig() ServerConfig {
	tier := TierComplete
	if t := os.Getenv("AST_MCP_TIER"); t != "" {
		tier = ParseTier(t)
	}
	codeMode := true
	if cm := os.Getenv("AST_MCP_CODE_MODE"); strings.EqualFold(cm, "false") || cm == "0" {
		codeMode = false
	}
	return ServerConfig{
		ActiveTier:  tier,
		CodeMode:    codeMode,
		SandboxSecs: 30,
	}
}

type JSONRPCRequest struct {
	JSONRPC string         `json:"jsonrpc"`
	ID      interface{}    `json:"id"`
	Method  string         `json:"method"`
	Params  map[string]any `json:"params,omitempty"`
}

type JSONRPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type JSONRPCResponse struct {
	JSONRPC string        `json:"jsonrpc"`
	ID      interface{}   `json:"id"`
	Result  interface{}   `json:"result,omitempty"`
	Error   *JSONRPCError `json:"error,omitempty"`
}

type Tool struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	InputSchema map[string]interface{} `json:"inputSchema"`
	Tier        Tier                   `json:"-"` // Not exposed to MCP clients
	ReadOnly    bool                   `json:"-"` // Read-only tools are safe at any tier
}

type Prompt struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Prompt      string `json:"prompt"`
}
