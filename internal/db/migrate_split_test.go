package db

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSplitFreshInit(t *testing.T) {
	tmp := t.TempDir()
	os.Setenv("HOME", tmp)
	defer os.Unsetenv("HOME")
	if err := Init(); err != nil {
		t.Fatal(err)
	}
	defer Close()
	for _, name := range []string{"index.db", "usage.db", "context.db"} {
		p := filepath.Join(tmp, ".astcache", name)
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("missing %s: %v", name, err)
		}
	}
	var n int
	if err := IndexDB.QueryRow(`SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='symbols'`).Scan(&n); err != nil || n != 1 {
		t.Fatalf("symbols table in index.db: n=%d err=%v", n, err)
	}
	if err := DB.QueryRow(`SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='queries'`).Scan(&n); err != nil || n != 1 {
		t.Fatalf("queries table in usage.db: n=%d err=%v", n, err)
	}
	if err := ContextDB.QueryRow(`SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='context_notes'`).Scan(&n); err != nil || n != 1 {
		t.Fatalf("context_notes in context.db: n=%d err=%v", n, err)
	}
}

func TestSessionDedupColumns(t *testing.T) {
	tmp := t.TempDir()
	os.Setenv("HOME", tmp)
	defer os.Unsetenv("HOME")
	if err := Init(); err != nil {
		t.Fatal(err)
	}
	defer Close()
	EnqueueSessionReturned("s1", 0, "foo", 10, "/a.go", "skeleton", 100)
	flushSessionLogBuffer()
	var name string
	var start int
	err := DB.QueryRow(`SELECT symbol_name, start_line FROM sessions WHERE session_id='s1'`).Scan(&name, &start)
	if err != nil {
		t.Fatal(err)
	}
	if name != "foo" || start != 10 {
		t.Fatalf("got name=%q start=%d", name, start)
	}
}
