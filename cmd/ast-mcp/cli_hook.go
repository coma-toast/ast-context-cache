package main

import (
	"context"
	"io"
	"log/slog"
	"os"

	"github.com/coma-toast/ast-context-cache/internal/hooks"
	"github.com/coma-toast/ast-context-cache/internal/logging"
)

// runHook handles "ast-mcp hook <event>" for Claude Code. It always exits 0 and writes nothing
// but the hook's JSON response to stdout, so a failing hook never blocks the host (HI-5).
// Diagnostics go to stderr only when AST_HOOK_DEBUG=1.
func runHook(args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	setupHookLogging(stderr)
	if len(args) == 0 {
		slog.Debug("Hook event missing")
		return exitOK
	}
	hooks.New(hooks.ResolveURL(), hooks.DefaultRegistryDir()).Run(context.Background(), args[0], stdin, stdout)
	return exitOK
}

// setupHookLogging discards logs unless AST_HOOK_DEBUG=1, which logs everything to w.
func setupHookLogging(w io.Writer) {
	if os.Getenv("AST_HOOK_DEBUG") != "1" {
		slog.SetDefault(slog.New(slog.DiscardHandler))
		return
	}
	slog.SetDefault(slog.New(logging.NewHandler(w, os.Getenv("AST_LOG_FORMAT"), slog.LevelDebug)))
}
