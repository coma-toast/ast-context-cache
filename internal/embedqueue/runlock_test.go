package embedqueue

import (
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func TestBeginRunLock_stalePID(t *testing.T) {
	useRunLockHome(t)
	path := runLockPath()
	if err := os.WriteFile(path, []byte("999999"), 0o644); err != nil {
		t.Fatal(err)
	}
	if !BeginRunLock() {
		t.Fatal("expected abnormal previous run for stale PID")
	}
	if !AbnormalPreviousRun() {
		t.Fatal("AbnormalPreviousRun should be true")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	got, _ := strconv.Atoi(string(data))
	if got != os.Getpid() {
		t.Fatalf("lock PID = %d, want %d", got, os.Getpid())
	}
}

func TestBeginRunLock_cleanStart(t *testing.T) {
	useRunLockHome(t)
	if BeginRunLock() {
		t.Fatal("expected normal start with no prior lock")
	}
	EndRunLock()
	if BeginRunLock() {
		t.Fatal("expected normal start after clean EndRunLock")
	}
}

// useRunLockHome gives the test a run lock of its own. It doesn't reopen the db
// pools there: the run lock only needs the directory, and TestMain's pools are
// shared with Start's goroutines for the rest of the package.
func useRunLockHome(t *testing.T) {
	t.Helper()
	t.Setenv("HOME", t.TempDir())
	if err := os.MkdirAll(filepath.Dir(runLockPath()), 0o755); err != nil {
		t.Fatal(err)
	}
}

func TestSetStartupWorkers(t *testing.T) {
	if err := db.SetSetting(embedWorkersSetting, "5"); err != nil {
		t.Fatal(err)
	}
	SetStartupWorkers(0)
	if got := loadWorkerCount(); got != 0 {
		t.Fatalf("loadWorkerCount() = %d, want 0 override", got)
	}
	startupWorkerOverride = nil
	if got := loadWorkerCount(); got != 5 {
		t.Fatalf("loadWorkerCount() = %d, want 5 from DB", got)
	}
}
