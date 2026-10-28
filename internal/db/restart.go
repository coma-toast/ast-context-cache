package db

// RestartProcess, when set, replaces the running ast-mcp with a fresh copy of the
// installed binary so a change that only takes effect at startup (a moved data
// directory, an updated binary) applies without the operator restarting by hand.
//
// It first quiesces everything that holds threads, native resources or files: background
// loops, WAL maintenance, the embed queue, both HTTP servers, file watchers (FSEvents
// streams), the embedder (ONNX sessions, the aux embedder, the ONNX Runtime environment),
// buffered writes and the database pools. It then restarts by AST_RESTART_STRATEGY:
// spawn a detached replacement and exit once it serves /health (the macOS default),
// exec in place (the default elsewhere), or exit for a supervisor to restart.
//
// The in-place exec used to hang on macOS (2026-10-23, 2026-10-28): Darwin's execve has
// to stop every other thread and waits indefinitely on one stuck in an uninterruptible
// kernel wait, and while it waits the Go runtime holds its exec lock, so the process can
// neither finish the exec nor start a thread to recover. Earlier crashes under a busy
// embed queue were the same un-quiesced native threads. Spawning avoids execve in a
// process that has run them at all; the exec path keeps a watchdog that exits 1 if the
// exec has not completed after 20s.
//
// It stays an explicit, user-initiated action (the dashboard's "Restart now") rather
// than something triggered automatically after another operation succeeds: a restart
// drops in-flight MCP streams and the dashboard's connection.
//
// Wired from cmd/ast-mcp/main.go, which owns the *http.Server values db can't import
// directly. Callers must nil-check before calling — it stays nil in tests and any build
// that doesn't wire it up. It does not return when the restart succeeds.
var RestartProcess func()
