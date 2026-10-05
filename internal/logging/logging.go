// Package logging configures the process-wide slog logger.
//
// AST_LOG_FORMAT selects "text" (default) or "json"; AST_LOG_LEVEL selects debug, info
// (default), warn, or error.
//
// Convention: pass errors keyed, logger.Warn("Failed to flush", "error", err). The STYLEGUIDE's
// bare form (logger.Warn("...", err)) is rejected by go vet's slog analyzer, which go test runs.
// The handler expands any error value into its message, errs codes, and errs fields, and still
// rewrites bare error/LogValuer arguments (slog's "!BADKEY") defensively.
package logging

import (
	"io"
	"log/slog"
	"os"
	"strings"
)

const (
	envFormat = "AST_LOG_FORMAT"
	envLevel  = "AST_LOG_LEVEL"

	// FormatText is the key=value output format.
	FormatText = "text"
	// FormatJSON is the one-JSON-object-per-line output format.
	FormatJSON = "json"
)

// Setup installs a default logger writing to w, configured from AST_LOG_FORMAT and
// AST_LOG_LEVEL. slog.SetDefault also routes the stdlib log package through it.
func Setup(w io.Writer) *slog.Logger {
	logger := slog.New(NewHandler(w, os.Getenv(envFormat), ParseLevel(os.Getenv(envLevel))))
	slog.SetDefault(logger)
	return logger
}

// NewHandler returns a handler in the given format ("json", or text for anything else) at the
// given level, wrapped so bare error and LogValuer arguments render as attributes.
func NewHandler(w io.Writer, format string, level slog.Leveler) slog.Handler {
	opts := &slog.HandlerOptions{Level: level}
	if strings.EqualFold(strings.TrimSpace(format), FormatJSON) {
		return &handler{inner: slog.NewJSONHandler(w, opts)}
	}
	return &handler{inner: slog.NewTextHandler(w, opts)}
}

// ParseLevel parses debug, info, warn/warning, or error (case-insensitive), defaulting to info.
func ParseLevel(s string) slog.Level {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "debug":
		return slog.LevelDebug
	case "warn", "warning":
		return slog.LevelWarn
	case "error":
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}
