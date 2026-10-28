package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSelectRestartStrategy(t *testing.T) {
	cases := []struct {
		goos, configured string
		want             restartStrategy
	}{
		{"darwin", "", restartSpawn},
		{"linux", "", restartExec},
		{"freebsd", "", restartExec},
		{"linux", "spawn", restartSpawn},
		{"darwin", "exec", restartExec},
		{"darwin", " EXIT ", restartExit},
		{"darwin", "bogus", restartSpawn},
		{"linux", "bogus", restartExec},
	}
	for _, c := range cases {
		assert.Equal(t, c.want, selectRestartStrategy(c.goos, c.configured), "%s %q", c.goos, c.configured)
	}
}

func TestQuiesceStepsOrder(t *testing.T) {
	var got []string
	rec := func(name string) func() { return func() { got = append(got, name) } }
	steps := quiesceSteps(quiesceDeps{
		cancelBackground: rec("cancel"),
		abortMaintenance: rec("abort-wal"),
		pauseEmbedQueue:  rec("pause-queue"),
		closeListeners:   rec("close-listeners"),
		shutdownServers: func(ctx context.Context) {
			_, ok := ctx.Deadline()
			assert.True(t, ok, "server shutdown is bounded")
			got = append(got, "servers")
		},
		stopWatchers:   rec("watchers"),
		stopEmbedder:   rec("embedder"),
		unloadCaches:   rec("caches"),
		flushWrites:    rec("flush"),
		endRunLock:     rec("run-lock"),
		closeDatabases: rec("databases"),
	})
	require.True(t, runSteps(steps))
	// RequestShutdown before the pause (it restores the queue as a side effect); servers
	// drain before watchers and the embedder stop; the databases close last.
	assert.Equal(t, []string{
		"cancel", "abort-wal", "pause-queue", "close-listeners", "servers",
		"watchers", "embedder", "caches", "flush", "run-lock", "databases",
	}, got)
}

func TestRunStepsContinuesPastAWedgedStep(t *testing.T) {
	release := make(chan struct{})
	defer close(release)
	var after atomic.Bool
	ok := runSteps([]restartStep{
		{"wedged", 50 * time.Millisecond, func() { <-release }},
		{"nil step", time.Second, nil},
		{"after", time.Second, func() { after.Store(true) }},
	})
	assert.False(t, ok)
	assert.True(t, after.Load(), "a timed-out step must not block the rest")
}

type fakeRestart struct {
	calls    []string
	spawnErr error
	execErr  error
	exitCode int
	adopted  int
}

func (f *fakeRestart) ops() restartOps {
	return restartOps{
		spawn: func(exe string) (int, error) {
			f.calls = append(f.calls, "spawn:"+exe)
			if f.spawnErr != nil {
				return 0, f.spawnErr
			}
			return 4242, nil
		},
		adoptPID:    func(pid int) { f.adopted = pid; f.calls = append(f.calls, "adopt") },
		execInPlace: func(exe string) error { f.calls = append(f.calls, "exec:"+exe); return f.execErr },
		exit:        func(code int) { f.exitCode = code; f.calls = append(f.calls, "exit:"+strconv.Itoa(code)) },
	}
}

func TestPerformRestartSpawnSuccessExitsZero(t *testing.T) {
	f := &fakeRestart{}
	performRestart(restartSpawn, "/opt/ast-mcp", f.ops())
	assert.Equal(t, []string{"spawn:/opt/ast-mcp", "adopt", "exit:0"}, f.calls)
	assert.Equal(t, 4242, f.adopted)
}

func TestPerformRestartSpawnFailureFallsBackToExec(t *testing.T) {
	f := &fakeRestart{spawnErr: errors.New("never healthy")}
	performRestart(restartSpawn, "/opt/ast-mcp", f.ops())
	assert.Equal(t, []string{"spawn:/opt/ast-mcp", "exec:/opt/ast-mcp"}, f.calls)
}

func TestPerformRestartExecFailureExitsOne(t *testing.T) {
	f := &fakeRestart{execErr: errors.New("ENOENT")}
	performRestart(restartExec, "/opt/ast-mcp", f.ops())
	assert.Equal(t, []string{"exec:/opt/ast-mcp", "exit:1"}, f.calls)
}

func TestPerformRestartExitStrategy(t *testing.T) {
	f := &fakeRestart{}
	performRestart(restartExit, "/opt/ast-mcp", f.ops())
	assert.Equal(t, []string{"exit:" + strconv.Itoa(restartExitCode)}, f.calls)
}

func TestRestartChildEnvReplacesMarker(t *testing.T) {
	env := restartChildEnv([]string{"A=1", envRestartedFrom + "=7", "B=2"}, 99)
	assert.Equal(t, []string{"A=1", "B=2", envRestartedFrom + "=99"}, env)
}

func healthServer(t *testing.T, pid *atomic.Int64) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"status": "starting", "pid": pid.Load()})
	}))
	t.Cleanup(srv.Close)
	return srv.URL + "/health"
}

func setReadyPoll(t *testing.T) {
	t.Helper()
	orig := childReadyPoll
	childReadyPoll = 10 * time.Millisecond
	t.Cleanup(func() { childReadyPoll = orig })
}

func TestWaitForHealthyWaitsForTheChildsPID(t *testing.T) {
	setReadyPoll(t)
	var pid atomic.Int64
	pid.Store(1)
	url := healthServer(t, &pid)
	time.AfterFunc(50*time.Millisecond, func() { pid.Store(321) })
	start := time.Now()
	require.NoError(t, waitForHealthy(url, 321, make(chan error), 2*time.Second))
	assert.GreaterOrEqual(t, time.Since(start), 40*time.Millisecond, "another process answering is not the child")
}

func TestWaitForHealthyFailsWhenChildExits(t *testing.T) {
	setReadyPoll(t)
	exited := make(chan error, 1)
	exited <- errors.New("exit status 1")
	err := waitForHealthy("http://127.0.0.1:1/health", 5, exited, 2*time.Second)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "exited")
}

func TestWaitForHealthyTimesOut(t *testing.T) {
	setReadyPoll(t)
	err := waitForHealthy("http://127.0.0.1:1/health", 5, make(chan error), 60*time.Millisecond)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "in time")
}

func TestExecWatchdogExitsWithStacks(t *testing.T) {
	var buf bytes.Buffer
	var mu sync.Mutex
	exited := make(chan int, 1)
	w := writerFunc(func(p []byte) (int, error) { mu.Lock(); defer mu.Unlock(); return buf.Write(p) })
	startExecWatchdog(20*time.Millisecond, []byte("goroutine 1 [running]"), w, func(code int) { exited <- code })
	select {
	case code := <-exited:
		assert.Equal(t, 1, code)
	case <-time.After(2 * time.Second):
		t.Fatal("watchdog did not fire")
	}
	mu.Lock()
	defer mu.Unlock()
	assert.Contains(t, buf.String(), "goroutine 1 [running]")
}

func TestExecWatchdogStop(t *testing.T) {
	var fired atomic.Bool
	stop := startExecWatchdog(30*time.Millisecond, nil, writerFunc(func(p []byte) (int, error) { return len(p), nil }), func(int) { fired.Store(true) })
	stop()
	time.Sleep(80 * time.Millisecond)
	assert.False(t, fired.Load())
}

type writerFunc func([]byte) (int, error)

func (f writerFunc) Write(p []byte) (int, error) { return f(p) }

func TestListenRetryWaitsForThePreviousProcessSockets(t *testing.T) {
	held, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	addr := held.Addr().String()
	origFrom, origPoll := restartedFrom, bindRetryPoll
	t.Cleanup(func() { restartedFrom, bindRetryPoll = origFrom, origPoll })
	bindRetryPoll = 10 * time.Millisecond

	restartedFrom = 0
	_, err = listenRetry(addr)
	require.Error(t, err, "a normal start fails fast on a taken port")

	restartedFrom = 12345
	time.AfterFunc(100*time.Millisecond, func() { held.Close() })
	ln, err := listenRetry(addr)
	require.NoError(t, err)
	ln.Close()
}

func TestAdoptMCPLocalPIDFile(t *testing.T) {
	dir := t.TempDir()
	write := func(name, body string) string {
		p := filepath.Join(dir, name)
		require.NoError(t, os.WriteFile(p, []byte(body), 0o600))
		return p
	}
	jsonFile := write("ast-context-cache.pid", `{"pid":100,"start":"Mon Oct 27 09:00:00 2026","exe":"/opt/ast-mcp"}`)
	plainFile := write("legacy.pid", "100\n")
	otherFile := write("other.pid", `{"pid":200}`)
	info := func(pid int) (string, string, error) { return "Tue Oct 28 10:00:00 2026", "/opt/ast-mcp", nil }

	updated := adoptMCPLocalPIDFile(dir, 100, 300, info)
	assert.ElementsMatch(t, []string{jsonFile, plainFile}, updated)
	for _, p := range []string{jsonFile, plainFile} {
		data, err := os.ReadFile(p)
		require.NoError(t, err)
		var id mcpLocalIdentity
		require.NoError(t, json.Unmarshal(data, &id))
		assert.Equal(t, mcpLocalIdentity{PID: 300, Start: "Tue Oct 28 10:00:00 2026", Exe: "/opt/ast-mcp"}, id)
		fi, err := os.Stat(p)
		require.NoError(t, err)
		assert.Equal(t, os.FileMode(0o600), fi.Mode().Perm())
	}
	data, _ := os.ReadFile(otherFile)
	assert.JSONEq(t, `{"pid":200}`, string(data), "a file naming another process is untouched")
	assert.Empty(t, adoptMCPLocalPIDFile(filepath.Join(dir, "missing"), 100, 300, info))
}

func TestAdoptMCPLocalPIDFileWithoutStart(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "svc.pid")
	require.NoError(t, os.WriteFile(p, []byte(`{"pid":5}`), 0o600))
	adoptMCPLocalPIDFile(dir, 5, 6, func(int) (string, string, error) { return "", "", errors.New("gone") })
	data, _ := os.ReadFile(p)
	assert.JSONEq(t, `{"pid":6}`, string(data))
}

func TestProcIdentityOfSelf(t *testing.T) {
	start, exe, err := procIdentity(os.Getpid())
	if err != nil {
		t.Skipf("process info unavailable: %v", err)
	}
	assert.NotEmpty(t, start)
	assert.NotEmpty(t, exe)
}

func TestHealthPayloadReportsPID(t *testing.T) {
	assert.Equal(t, os.Getpid(), buildHealthPayload()["pid"])
}
