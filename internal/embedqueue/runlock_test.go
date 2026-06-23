package embedqueue

import (
	"os"
	"strconv"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func TestBeginRunLock_stalePID(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
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
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	if BeginRunLock() {
		t.Fatal("expected normal start with no prior lock")
	}
	EndRunLock()
	if BeginRunLock() {
		t.Fatal("expected normal start after clean EndRunLock")
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
