//go:build integration

package main

// Run with:
//
//	make test TEST_PKGS=./cmd/ast-mcp/ TEST_FLAGS="-tags sqlite_fts5,integration -run TestRestartIntegration -v"
//
// MODEL_DIR (default: the repo's model/) lets the parent load a real ONNX session and
// watch a project before the restart, which is the state that hung macOS execs. Without
// the model the embedder points at an unreachable HTTP backend instead. The exec subtest
// is skipped on darwin unless AST_TEST_RESTART_EXEC=1.

import (
	"encoding/json"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type restartInstance struct {
	t        *testing.T
	bin      string
	home     string
	mcpPort  int
	dashPort int
	cmd      *exec.Cmd
	waitErr  chan error
	modelDir string
	pidFile  string
	strategy string
}

// ready is a finished startup; an unreachable HTTP embedder reports degraded.
func ready(status string) bool { return status == "healthy" || status == "degraded" }

func buildRestartBinary(t *testing.T) string {
	t.Helper()
	bin := filepath.Join(t.TempDir(), "ast-mcp")
	build := exec.Command("go", "build", "-tags", "sqlite_fts5", "-o", bin, ".")
	build.Env = append(os.Environ(), "GOWORK=off")
	out, err := build.CombinedOutput()
	require.NoError(t, err, "build: %s", out)
	return bin
}

func repoModelDir() string {
	if d := os.Getenv("MODEL_DIR"); d != "" {
		return d
	}
	_, file, _, _ := runtime.Caller(0)
	d := filepath.Join(filepath.Dir(file), "..", "..", "model")
	if _, err := os.Stat(filepath.Join(d, "model.onnx")); err != nil {
		return ""
	}
	return d
}

func startRestartInstance(t *testing.T, bin, strategy string) *restartInstance {
	t.Helper()
	home := t.TempDir()
	in := &restartInstance{t: t, bin: bin, home: home, mcpPort: freePort(t), dashPort: freePort(t), waitErr: make(chan error, 1), modelDir: repoModelDir(), strategy: strategy}
	in.cmd = exec.Command(bin, "-mcp-port", strconv.Itoa(in.mcpPort), "-dashboard-port", strconv.Itoa(in.dashPort), "-name", "restart-it")
	env := []string{"HOME=" + home, "PATH=" + os.Getenv("PATH"), envRestartStrategy + "=" + strategy, "AST_EMBED_WORKERS=1"}
	if in.modelDir != "" {
		env = append(env, "MODEL_DIR="+in.modelDir, "EMBED_BACKEND=onnx")
		if lib := os.Getenv("ONNXRUNTIME_LIB"); lib != "" {
			env = append(env, "ONNXRUNTIME_LIB="+lib)
		}
	} else {
		env = append(env, "EMBED_BACKEND=http", "EMBED_HTTP_URL=http://127.0.0.1:1/embed")
	}
	in.cmd.Env = env
	// The replacement inherits stdout/stderr; with a pipe, Wait would not return until the
	// child closed it too, so both write to a file in HOME, as under mcp-local.
	logFile, err := os.Create(filepath.Join(home, "service.log"))
	require.NoError(t, err)
	in.cmd.Stdout, in.cmd.Stderr = logFile, logFile
	in.cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	require.NoError(t, in.cmd.Start())
	logFile.Close()
	go func() { in.waitErr <- in.cmd.Wait() }()
	t.Cleanup(in.cleanup)
	return in
}

func (in *restartInstance) logText() string {
	data, _ := os.ReadFile(filepath.Join(in.home, "service.log"))
	return string(data)
}

func (in *restartInstance) health() (map[string]any, error) {
	c := &http.Client{Timeout: 2 * time.Second}
	resp, err := c.Get("http://127.0.0.1:" + strconv.Itoa(in.mcpPort) + "/health")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var body map[string]any
	return body, json.NewDecoder(resp.Body).Decode(&body)
}

func (in *restartInstance) waitHealthy(timeout time.Duration, accept func(pid int, status string) bool) int {
	in.t.Helper()
	var pid int
	require.Eventually(in.t, func() bool {
		h, err := in.health()
		if err != nil {
			return false
		}
		p, _ := h["pid"].(float64)
		s, _ := h["status"].(string)
		pid = int(p)
		return accept(pid, s)
	}, timeout, 200*time.Millisecond, "no healthy instance; log:\n%s", in.logText())
	return pid
}

func (in *restartInstance) post(path, contentType, body string) {
	in.t.Helper()
	resp, err := http.Post("http://127.0.0.1:"+strconv.Itoa(in.dashPort)+path, contentType, strings.NewReader(body))
	require.NoError(in.t, err)
	resp.Body.Close()
	require.Equal(in.t, http.StatusOK, resp.StatusCode, path)
}

// warmUp puts the parent in the state that hung macOS execs: an ONNX session loaded
// and a native file watcher running.
func (in *restartInstance) warmUp() {
	in.t.Helper()
	if in.modelDir != "" {
		resp, err := http.Post("http://127.0.0.1:"+strconv.Itoa(in.mcpPort)+"/embed", "application/json", strings.NewReader(`{"texts":["warm up the onnx session"]}`))
		require.NoError(in.t, err)
		resp.Body.Close()
		require.Equal(in.t, http.StatusOK, resp.StatusCode, "embed; log:\n%s", in.logText())
	}
	project := filepath.Join(in.home, "proj")
	require.NoError(in.t, os.MkdirAll(project, 0o755))
	require.NoError(in.t, os.WriteFile(filepath.Join(project, "main.go"), []byte("package main\n\nfunc main() {}\n"), 0o644))
	body, _ := json.Marshal(map[string]string{"project_path": project})
	in.post("/api/start-watcher", "application/json", string(body))
}

func (in *restartInstance) writeMCPLocalPIDFile(pid int) {
	in.t.Helper()
	dir := filepath.Join(in.home, ".mcp-local", "pids")
	require.NoError(in.t, os.MkdirAll(dir, 0o700))
	start, exe, err := procIdentity(pid)
	require.NoError(in.t, err)
	data, _ := json.Marshal(mcpLocalIdentity{PID: pid, Start: start, Exe: exe})
	in.pidFile = filepath.Join(dir, "ast-context-cache.pid")
	require.NoError(in.t, os.WriteFile(in.pidFile, data, 0o600))
}

func (in *restartInstance) cleanup() {
	if h, err := in.health(); err == nil {
		if p, ok := h["pid"].(float64); ok && int(p) > 1 {
			syscall.Kill(int(p), syscall.SIGTERM)
			deadline := time.Now().Add(10 * time.Second)
			for time.Now().Before(deadline) && syscall.Kill(int(p), 0) == nil {
				time.Sleep(100 * time.Millisecond)
			}
		}
	}
	if in.cmd.Process != nil {
		in.cmd.Process.Kill()
	}
	if in.t.Failed() {
		in.t.Logf("service log:\n%s", in.logText())
	}
}

func TestRestartIntegrationSpawn(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test builds and runs the binary")
	}
	bin := buildRestartBinary(t)
	in := startRestartInstance(t, bin, string(restartSpawn))
	parent := in.cmd.Process.Pid
	in.waitHealthy(60*time.Second, func(pid int, status string) bool { return pid == parent && ready(status) })
	in.warmUp()
	in.writeMCPLocalPIDFile(parent)

	start := time.Now()
	in.post("/api/restart-now", "application/json", "{}")
	child := in.waitHealthy(90*time.Second, func(pid int, _ string) bool { return pid != 0 && pid != parent })
	t.Logf("restart: parent %d -> child %d serving /health after %s", parent, child, time.Since(start).Round(time.Millisecond))

	select {
	case err := <-in.waitErr:
		require.NoError(t, err, "the old process exits 0 once the child serves")
	case <-time.After(30 * time.Second):
		t.Fatalf("old pid %d still running; log:\n%s", parent, in.logText())
	}
	assert.Error(t, syscall.Kill(parent, 0), "old pid %d is gone", parent)
	assert.NoError(t, syscall.Kill(child, 0), "new pid %d is alive", child)

	data, err := os.ReadFile(in.pidFile)
	require.NoError(t, err)
	var id mcpLocalIdentity
	require.NoError(t, json.Unmarshal(data, &id))
	assert.Equal(t, child, id.PID, "mcp-local's PID file follows the restart")
	wantStart, _, err := procIdentity(child)
	require.NoError(t, err)
	assert.Equal(t, wantStart, id.Start)

	log := in.logText()
	for _, line := range strings.Split(log, "\n") {
		if strings.Contains(line, "Restart") || strings.Contains(line, "restart") || strings.Contains(line, "Replacement") || strings.Contains(line, "Spawned") || strings.Contains(line, "PID file") || strings.Contains(line, "shut down") {
			t.Log(line)
		}
	}
	assert.Contains(t, log, "Started by a restart")
	assert.Contains(t, log, "Replacement is serving; exiting")
	assert.NotContains(t, log, "RESTART FALLBACK")
	assert.NotContains(t, log, "timed out; continuing")
}

func TestRestartIntegrationExec(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test builds and runs the binary")
	}
	if runtime.GOOS == "darwin" && os.Getenv("AST_TEST_RESTART_EXEC") != "1" {
		t.Skip("in-place exec is the strategy that hangs on macOS; set AST_TEST_RESTART_EXEC=1 to run it")
	}
	bin := buildRestartBinary(t)
	in := startRestartInstance(t, bin, string(restartExec))
	pid := in.cmd.Process.Pid
	in.waitHealthy(60*time.Second, func(p int, status string) bool { return p == pid && ready(status) })
	in.warmUp()
	in.post("/api/restart-now", "application/json", "{}")
	require.Eventually(t, func() bool { return strings.Contains(in.logText(), "Started by a restart") }, 60*time.Second, 200*time.Millisecond, "log:\n%s", in.logText())
	in.waitHealthy(60*time.Second, func(p int, _ string) bool { return p == pid })
}
