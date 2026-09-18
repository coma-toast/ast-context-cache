package db

// RestartProcess, when set, gracefully drains the HTTP servers (no new
// connections; in-flight requests finish), quiesces background work, and
// re-execs the current binary in place — so a fresh db.Init() can pick up a
// change that only takes effect at startup (a moved data directory, an
// updated binary) without the operator restarting ast-mcp by hand.
//
// KNOWN RISK: restarting in place while there's significant concurrent
// background goroutine activity (a busy embed queue, in particular) has been
// observed to crash the process outright instead of restarting it, and the
// exact mechanism isn't understood yet — reordering the drain steps,
// spawning a child process instead of exec'ing in place, and a detached
// watchdog process were all tried and none reliably avoided it. Until this is
// root-caused, callers should treat this as an explicit, user-initiated
// action (a "Restart now" button) rather than something to trigger
// automatically after an unrelated operation succeeds — a failed automatic
// restart is silent downtime, worse than requiring one more click.
//
// Wired from cmd/ast-mcp/main.go, which owns the *http.Server values db
// can't import directly. Callers must nil-check before calling — it stays
// nil in tests and any build that doesn't wire it up.
var RestartProcess func()
