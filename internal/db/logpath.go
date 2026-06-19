package db

import (
	"os"
	"path/filepath"
	"strings"
	"time"
)

const legacyServerLogPath = "/tmp/ast-mcp.log"

// McpLocalLogPath is where mcp-local redirects ast-mcp stdout/stderr.
func McpLocalLogPath() string {
	home, _ := os.UserHomeDir()
	if home == "" {
		return ""
	}
	return filepath.Join(home, ".mcp-local", "ast-context-cache.log")
}

// ResolveServerLogPath picks the active server log file for the dashboard Logs tab.
// Order: AST_MCP_LOG_PATH env, then the newest non-empty candidate among default,
// mcp-local, and legacy paths.
func ResolveServerLogPath() string {
	if p := strings.TrimSpace(os.Getenv("AST_MCP_LOG_PATH")); p != "" {
		return p
	}
	best := DefaultLogPath()
	bestMtime := fileModTime(best)
	for _, p := range []string{McpLocalLogPath(), legacyServerLogPath} {
		if p == "" {
			continue
		}
		mt := fileModTime(p)
		if mt.After(bestMtime) {
			best = p
			bestMtime = mt
		}
	}
	return best
}

func fileModTime(path string) time.Time {
	fi, err := os.Stat(path)
	if err != nil || fi.Size() == 0 {
		return time.Time{}
	}
	return fi.ModTime()
}
