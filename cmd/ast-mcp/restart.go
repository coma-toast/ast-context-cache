package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/embedqueue"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/netlisten"
	"github.com/coma-toast/ast-context-cache/internal/search"
	"github.com/coma-toast/ast-context-cache/internal/sys"
	"github.com/coma-toast/ast-context-cache/internal/watcher"
)

// restartStrategy is how restartProcess replaces the running process once it is quiesced.
//
//   - spawn: start the same binary as a detached child (new session, same args, env,
//     stdout and stderr), wait for it to serve /health, then exit 0. The default on
//     macOS, where an in-place execve has hung: Darwin's execve must stop every other
//     thread and blocks for good if one sits in an uninterruptible kernel wait, and while
//     it waits the Go runtime holds its exec lock, so no new thread can start either.
//   - exec: syscall.Exec in place, keeping the PID. The default elsewhere; systemd and
//     other supervisors that track the main PID keep working. A watchdog exits 1 if the
//     exec has not completed after execWatchdogTimeout.
//   - exit: exit with restartExitCode and let a supervisor (systemd Restart=, launchd
//     KeepAlive) start the new process.
type restartStrategy string

const (
	restartSpawn restartStrategy = "spawn"
	restartExec  restartStrategy = "exec"
	restartExit  restartStrategy = "exit"
)

const (
	// envRestartStrategy overrides the platform default (spawn on darwin, exec elsewhere).
	envRestartStrategy = "AST_RESTART_STRATEGY"
	// envRestartedFrom carries the PID of the process that restarted us, so the new one
	// retries binding its ports while the old sockets finish closing.
	envRestartedFrom = "AST_RESTARTED_FROM_PID"
	// restartExitCode is EX_TEMPFAIL: non-zero, so restart-on-failure policies fire.
	restartExitCode = 75
)

// Restart timing; variables so tests can shorten them.
var (
	childReadyTimeout   = 60 * time.Second
	childReadyPoll      = 250 * time.Millisecond
	execWatchdogTimeout = 20 * time.Second
	bindRetryWindow     = 15 * time.Second
	bindRetryPoll       = 200 * time.Millisecond
)

var (
	restarting atomic.Bool
	// restartedFrom is the previous PID when this process was started by a restart.
	restartedFrom int
)

// readRestartedFrom records and clears envRestartedFrom, so it never leaks into a later
// child that was not started by a restart.
func readRestartedFrom() {
	restartedFrom, _ = strconv.Atoi(strings.TrimSpace(os.Getenv(envRestartedFrom)))
	os.Unsetenv(envRestartedFrom)
}

// selectRestartStrategy returns configured when it names a strategy, otherwise the
// default for goos.
func selectRestartStrategy(goos, configured string) restartStrategy {
	switch s := restartStrategy(strings.ToLower(strings.TrimSpace(configured))); s {
	case restartSpawn, restartExec, restartExit:
		return s
	case "":
	default:
		logger.Warn("Ignoring unknown restart strategy", "env", envRestartStrategy, "value", configured)
	}
	if goos == "darwin" {
		return restartSpawn
	}
	return restartExec
}

// restartStep is one quiesce action. A step that outlives its timeout is logged and left
// running so a wedged subsystem can't stop the restart.
type restartStep struct {
	name    string
	timeout time.Duration
	run     func()
}

// quiesceDeps are the actions restartProcess takes before it replaces the process;
// tests pass fakes to check the order.
type quiesceDeps struct {
	cancelBackground func()                    // ctx loops: tickers, retry, handoff, session eviction
	abortMaintenance func()                    // db.RequestShutdown: abort an in-progress WAL checkpoint
	pauseEmbedQueue  func()                    // stop embed workers and wait for in-flight embeds
	closeListeners   func()                    // extra listeners, so nothing reopens one mid-drain
	shutdownServers  func(ctx context.Context) // drain the MCP and dashboard servers
	stopWatchers     func()                    // FSEvents/inotify streams and their goroutines
	stopEmbedder     func()                    // ONNX sessions, aux embedder, probe, ORT environment
	unloadCaches     func()                    // vector cache, so parent and child don't both hold it
	flushWrites      func()                    // buffered query/session analytics
	endRunLock       func()                    // a clean stop, not an abnormal previous run
	closeDatabases   func()                    // SQLite pools, last: everything above may write
}

// quiesceSteps orders the restart quiesce. db.RequestShutdown comes before the embed
// queue pause because it also runs AfterForceCheckpoint (embedqueue.RestoreAfterMaintenance),
// which would undo a pause taken earlier. Servers drain before watchers and the embedder
// stop, so no request is left using them; databases close last.
func quiesceSteps(d quiesceDeps) []restartStep {
	return []restartStep{
		{"cancel background loops", 5 * time.Second, d.cancelBackground},
		{"abort WAL maintenance", 10 * time.Second, d.abortMaintenance},
		{"pause embed queue", 2*time.Minute + 5*time.Second, d.pauseEmbedQueue},
		{"close extra listeners", 5 * time.Second, d.closeListeners},
		{"shut down HTTP servers", 35 * time.Second, func() {
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			d.shutdownServers(ctx)
		}},
		{"stop file watchers", 15 * time.Second, d.stopWatchers},
		{"stop embedder", 15 * time.Second, d.stopEmbedder},
		{"unload caches", 5 * time.Second, d.unloadCaches},
		{"flush write buffers", shutdownFlushTimeout + time.Second, d.flushWrites},
		{"end run lock", 2 * time.Second, d.endRunLock},
		{"close databases", 15 * time.Second, d.closeDatabases},
	}
}

// runSteps runs steps in order and reports whether every one finished in time.
func runSteps(steps []restartStep) bool {
	ok := true
	for _, s := range steps {
		if s.run == nil {
			continue
		}
		start := time.Now()
		done := make(chan struct{})
		go func() {
			defer close(done)
			s.run()
		}()
		select {
		case <-done:
			logger.Info("Restart step done", "step", s.name, "elapsed", time.Since(start).Round(time.Millisecond))
		case <-time.After(s.timeout):
			ok = false
			logger.Warn("Restart step timed out; continuing", "step", s.name, "timeout", s.timeout)
		}
	}
	return ok
}

func liveQuiesceDeps(mcpSrv, dashSrv *http.Server, cancel context.CancelFunc) quiesceDeps {
	return quiesceDeps{
		cancelBackground: cancel,
		abortMaintenance: db.RequestShutdown,
		pauseEmbedQueue:  func() { embedqueue.PauseAllForMaintenance(2 * time.Minute) },
		closeListeners:   netlisten.CloseAll,
		shutdownServers: func(ctx context.Context) {
			if err := mcpSrv.Shutdown(ctx); err != nil {
				logger.Warn("Failed to shut down MCP server", "error", err)
			}
			if err := dashSrv.Shutdown(ctx); err != nil {
				logger.Warn("Failed to shut down dashboard server", "error", err)
			}
		},
		stopWatchers:   watcher.StopAll,
		stopEmbedder:   embedder.Shutdown,
		unloadCaches:   search.Cache.Unload,
		flushWrites:    func() { flushWriteBuffers(shutdownFlushTimeout) },
		endRunLock:     embedqueue.EndRunLock,
		closeDatabases: db.Close,
	}
}

// restartOps replace the process; tests pass fakes.
type restartOps struct {
	spawn       func(exe string) (pid int, err error)
	adoptPID    func(pid int)
	execInPlace func(exe string) error
	exit        func(code int)
}

// restartProcess returns the hook wired to db.RestartProcess. It quiesces everything
// that holds threads, native resources or files (quiesceSteps), then replaces the
// process by the configured restartStrategy so a fresh main() and db.Init() pick up
// whatever changed on disk (a moved data directory, an updated binary). The target is
// the executable resolved at startup (sys.Executable), not os.Executable: an update has
// renamed the running binary to ast-mcp.prev by now.
func restartProcess(mcpSrv, dashSrv *http.Server, cancel context.CancelFunc) func() {
	return func() {
		if !restarting.CompareAndSwap(false, true) {
			logger.Warn("Restart already in progress")
			return
		}
		// Tell main() not to exit when listenAndServe returns after the dashboard drain.
		exiting.Store(true)
		strategy := selectRestartStrategy(runtime.GOOS, os.Getenv(envRestartStrategy))
		exe, exeErr := sys.Executable()
		logger.Info("Restart requested; quiescing", "strategy", strategy, "exe", exe)
		if !runSteps(quiesceSteps(liveQuiesceDeps(mcpSrv, dashSrv, cancel))) {
			logger.Warn("Restart continuing after a quiesce step timed out", "strategy", strategy)
		}
		if exeErr != nil {
			logger.Error("Failed to resolve executable to restart", "error", exeErr)
			os.Exit(1)
		}
		performRestart(strategy, exe, restartOps{
			spawn:       spawnReplacement,
			adoptPID:    func(pid int) { adoptMCPLocalPIDFile(mcpLocalPIDDir(), os.Getpid(), pid, procIdentity) },
			execInPlace: execWithWatchdog,
			exit:        os.Exit,
		})
	}
}

// performRestart replaces the quiesced process. A spawn that fails, or whose child never
// becomes healthy, falls back to exec in place.
func performRestart(strategy restartStrategy, exe string, ops restartOps) {
	switch strategy {
	case restartExit:
		logger.Info("Exiting for the supervisor to restart", "code", restartExitCode)
		ops.exit(restartExitCode)
		return
	case restartSpawn:
		pid, err := ops.spawn(exe)
		if err == nil {
			logger.Info("Replacement is serving; exiting", "pid", pid)
			ops.adoptPID(pid)
			ops.exit(0)
			return
		}
		logger.Error("RESTART FALLBACK: spawned replacement failed; exec'ing in place instead", "exe", exe, "error", err)
	}
	logger.Info("Restarting", "exe", exe)
	if err := ops.execInPlace(exe); err != nil {
		logger.Error("Failed to restart", "exe", exe, "error", err)
		ops.exit(1)
	}
}

// restartChildEnv is env with envRestartedFrom set to pid.
func restartChildEnv(env []string, pid int) []string {
	out := make([]string, 0, len(env)+1)
	for _, kv := range env {
		if !strings.HasPrefix(kv, envRestartedFrom+"=") {
			out = append(out, kv)
		}
	}
	return append(out, envRestartedFrom+"="+strconv.Itoa(pid))
}

// spawnReplacement starts exe detached with this process's args, env, stdout and stderr,
// and waits until it serves /health on the MCP port. Our ports are closed by now, so the
// child can bind them; a child that exits or misses childReadyTimeout is killed.
func spawnReplacement(exe string) (int, error) {
	cmd := exec.Command(exe, os.Args[1:]...)
	cmd.Env = restartChildEnv(os.Environ(), os.Getpid())
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	// A new session: the child outlives us and gets no SIGHUP from our terminal.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	if err := cmd.Start(); err != nil {
		return 0, errs.WrapMessage("failed to start replacement", err, "exe", exe)
	}
	pid := cmd.Process.Pid
	logger.Info("Spawned replacement; waiting for it to serve", "pid", pid, "url", healthURL(), "timeout", childReadyTimeout)
	exited := make(chan error, 1)
	go func() { exited <- cmd.Wait() }()
	if err := waitForHealthy(healthURL(), pid, exited, childReadyTimeout); err != nil {
		_ = cmd.Process.Kill()
		return 0, err
	}
	return pid, nil
}

// healthURL is the MCP /health endpoint on a loopback-reachable host.
func healthURL() string {
	host := listenAddr
	if ip := net.ParseIP(host); host == "" || (ip != nil && ip.IsUnspecified()) {
		host = defaultListenAddr
	}
	return "http://" + net.JoinHostPort(host, strconv.Itoa(mcpPort)) + "/health"
}

// waitForHealthy polls url until it answers as pid (or as a build whose health has no
// pid), the child exits, or timeout passes.
func waitForHealthy(url string, pid int, exited <-chan error, timeout time.Duration) error {
	client := &http.Client{Timeout: 2 * time.Second}
	deadline := time.After(timeout)
	tick := time.NewTicker(childReadyPoll)
	defer tick.Stop()
	for {
		if got, err := healthPID(client, url); err == nil && (got == pid || got == 0) {
			return nil
		}
		select {
		case err := <-exited:
			return errs.WrapMessage("replacement exited before it was healthy", errors.Join(errs.New("child exited"), err), "pid", pid)
		case <-deadline:
			return errs.New("replacement did not become healthy in time", "pid", pid, "timeout", timeout)
		case <-tick.C:
		}
	}
}

func healthPID(client *http.Client, url string) (int, error) {
	resp, err := client.Get(url)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, errs.New("health not OK", "status", resp.StatusCode)
	}
	var body struct {
		PID int `json:"pid"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return 0, err
	}
	return body.PID, nil
}

// execWithWatchdog execs exe in place under a watchdog. Stacks are captured before the
// exec: once syscall.Exec holds the runtime's exec lock, a live all-goroutine dump stops
// the world, which waits on that lock forever (the 2026-10-28 hang died in
// preemptM -> execLock.rlock), and would leave the watchdog unable to exit.
func execWithWatchdog(exe string) error {
	stacks := allStacks()
	stop := startExecWatchdog(execWatchdogTimeout, stacks, os.Stderr, os.Exit)
	err := syscall.Exec(exe, os.Args, restartChildEnv(os.Environ(), os.Getpid()))
	stop()
	return err
}

// startExecWatchdog exits 1 after timeout unless stop is called first, logging stacks.
// It must not spawn anything: while an exec is stuck the runtime can't start threads. On a
// host whose supervisor restarts on failure (systemd Restart=always or on-failure,
// launchd KeepAlive) the exit brings the service back by itself.
func startExecWatchdog(timeout time.Duration, stacks []byte, w io.Writer, exit func(int)) (stop func()) {
	t := time.AfterFunc(timeout, func() {
		logger.Error("RESTART HUNG: exec did not complete; exiting so a supervisor can restart", "timeout", timeout)
		_, _ = w.Write([]byte("goroutine stacks captured just before exec:\n"))
		_, _ = w.Write(stacks)
		exit(1)
	})
	return func() { t.Stop() }
}

func allStacks() []byte {
	buf := make([]byte, 1<<20)
	for {
		n := runtime.Stack(buf, true)
		if n < len(buf) || len(buf) >= 64<<20 {
			return buf[:n]
		}
		buf = make([]byte, 2*len(buf))
	}
}

// listenRetry is net.Listen that, in a process started by a restart, retries while the
// address is still in use by the previous process's closing sockets.
func listenRetry(addr string) (net.Listener, error) {
	deadline := time.Now().Add(bindRetryWindow)
	for {
		ln, err := net.Listen("tcp", addr)
		if err == nil || restartedFrom == 0 || !errors.Is(err, syscall.EADDRINUSE) || time.Now().After(deadline) {
			return ln, err
		}
		time.Sleep(bindRetryPoll)
	}
}
