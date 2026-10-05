package main

import (
	"fmt"
	"io"
	"log/slog"
	"os"

	"github.com/coma-toast/ast-context-cache/internal/logging"
	"github.com/coma-toast/ast-context-cache/internal/version"
)

// CLI exit codes. mcp-local and scripts branch on these.
const (
	exitOK          = 0
	exitError       = 1
	exitConfirm     = 2 // a change needs --yes (or --dry-run to only preview)
	exitConflict    = 3 // a file changed since preview, or could not be parsed
	exitUnsupported = 4 // every requested component is unsupported for a named target
)

// runCLI handles subcommands and --version before the server parses its flags. handled is
// false when args name no subcommand, and main goes on to start the server.
func runCLI(args []string) (code int, handled bool) {
	return runCLIWith(args, os.Stdout, os.Stderr)
}

func runCLIWith(args []string, stdout, stderr io.Writer) (int, bool) {
	if len(args) == 0 {
		return exitOK, false
	}
	switch args[0] {
	case "version", "--version", "-version":
		fmt.Fprintln(stdout, "ast-mcp "+version.Version)
		return exitOK, true
	case "install", "uninstall", "verify", "backups", "restore":
		setupCLILogging(stderr)
		return runInstaller(args[0], args[1:], stdout, stderr), true
	case "hook":
		// Hook entries may be installed before the hook handlers ship; exiting 0 with no output
		// keeps them fail-open instead of starting a second server.
		return exitOK, true
	}
	return exitOK, false
}

// setupCLILogging logs warnings and errors to w unless AST_LOG_LEVEL asks for more.
func setupCLILogging(w io.Writer) {
	level := slog.LevelWarn
	if v := os.Getenv("AST_LOG_LEVEL"); v != "" {
		level = logging.ParseLevel(v)
	}
	slog.SetDefault(slog.New(logging.NewHandler(w, os.Getenv("AST_LOG_FORMAT"), level)))
}
