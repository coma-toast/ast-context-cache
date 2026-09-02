package db

import (
	"os"
	"testing"
)

// TestExportedIndexCallersSurviveQuiesce guards against a real crash: EnsureFTSTriggers,
// GetIndexedFiles, and InvalidateSummariesForFile used to reference the package-level
// IndexDB variable directly instead of going through IndexReader(). quiesceIndexPool
// briefly nils IndexDB out during WAL maintenance, so calling any of them during that
// window panicked with a nil pointer dereference — hit in production by deleting a
// project (EnsureFTSTriggers) and by editing a file (InvalidateSummariesForFile, on
// every re-index) while a checkpoint was quiescing the pool.
func TestExportedIndexCallersSurviveQuiesce(t *testing.T) {
	dir := t.TempDir()
	prev := os.Getenv("HOME")
	os.Setenv("HOME", dir)
	defer os.Setenv("HOME", prev)

	if err := Init(); err != nil {
		t.Fatal(err)
	}
	defer func() {
		indexReadGate.Store(false)
		_ = restoreIndexPool()
		Close()
	}()

	if err := quiesceIndexPool(); err != nil {
		t.Fatal(err)
	}
	if IndexDB != nil {
		t.Fatal("expected nil IndexDB during quiesce")
	}

	// None of these must panic while IndexDB is nil.
	EnsureFTSTriggers()
	if got := GetIndexedFiles("/some/project"); len(got) != 0 {
		t.Fatalf("GetIndexedFiles during quiesce = %v, want empty", got)
	}
	InvalidateSummariesForFile("/some/project/file.go", "/some/project")

	if err := restoreIndexPool(); err != nil {
		t.Fatal(err)
	}
}
