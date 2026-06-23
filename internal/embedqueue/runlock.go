package embedqueue

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

const runLockFile = "ast-mcp.run.lock"

var abnormalPreviousRun bool

// AbnormalPreviousRun reports whether the last ast-mcp run exited without clearing the run lock.
func AbnormalPreviousRun() bool {
	return abnormalPreviousRun
}

// BeginRunLock records this process and returns true when the prior run likely crashed.
func BeginRunLock() bool {
	path := runLockPath()
	abnormalPreviousRun = false
	if data, err := os.ReadFile(path); err == nil {
		pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
		if err == nil && pid > 0 && pid != os.Getpid() && !processRunning(pid) {
			abnormalPreviousRun = true
		}
	}
	_ = os.WriteFile(path, []byte(strconv.Itoa(os.Getpid())), 0o644)
	return abnormalPreviousRun
}

// EndRunLock removes the run lock after a clean shutdown.
func EndRunLock() {
	_ = os.Remove(runLockPath())
}

func runLockPath() string {
	return filepath.Join(filepath.Dir(db.GetDBPath()), runLockFile)
}

func processRunning(pid int) bool {
	p, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	return p.Signal(syscall.Signal(0)) == nil
}
