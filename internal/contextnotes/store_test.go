package contextnotes

import (
	"errors"
	"strings"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func testNotesDB(t *testing.T) {
	t.Helper()
	t.Setenv("HOME", t.TempDir())
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
}

func TestStoreFetchFlush(t *testing.T) {
	testNotesDB(t)
	res, err := Store("sess-1", strings.Repeat("hello ", 100), "greeting", "/tmp/proj", "tag1", nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(res.Ref, "ctx_") {
		t.Fatalf("ref prefix: %s", res.Ref)
	}
	if res.VirtualTokensStored <= 0 {
		t.Fatalf("expected token est > 0")
	}
	fetch, err := Fetch([]string{res.Ref}, "sess-1")
	if err != nil {
		t.Fatal(err)
	}
	if len(fetch.Notes) != 1 {
		t.Fatalf("notes: %d", len(fetch.Notes))
	}
	if v, ok := fetch.Stats["virtual_tokens_returned"].(int); !ok || v <= 0 {
		t.Fatalf("expected access tokens, stats=%v", fetch.Stats)
	}
	list, err := List("sess-1", "", 10)
	if err != nil || list.Total != 1 {
		t.Fatalf("list total=%d err=%v", list.Total, err)
	}
	search, err := Search("hello", "sess-1", "", 5, nil)
	if err != nil {
		t.Fatal(err)
	}
	if search == nil || len(search.Notes) == 0 {
		t.Fatalf("search notes=%d", len(search.Notes))
	}
	flush, err := Flush("sess-1", nil, "", false)
	if err != nil || flush.FlushedRefs != 1 {
		t.Fatalf("flush: %+v err=%v", flush, err)
	}
	_, err = Fetch([]string{res.Ref}, "sess-1")
	if err != nil {
		t.Fatal(err)
	}
}

func TestStoreLimitReject(t *testing.T) {
	testNotesDB(t)
	db.SetSetting("context_max_notes_session", "1")
	db.SetSetting("context_limit_policy", "reject")
	_, err := Store("sess-limit", "first note content", "a", "", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	_, err = Store("sess-limit", "second note should fail", "b", "", nil, nil)
	var le *LimitError
	if !errors.As(err, &le) {
		t.Fatalf("expected LimitError, got %v", err)
	}
}

func TestFetchSessionIsolation(t *testing.T) {
	testNotesDB(t)
	res, err := Store("owner", "secret", "x", "", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	fetch, err := Fetch([]string{res.Ref}, "other-session")
	if err != nil {
		t.Fatal(err)
	}
	if len(fetch.Notes) != 0 {
		t.Fatalf("expected 0 notes for wrong session")
	}
}

func TestLRUEviction(t *testing.T) {
	testNotesDB(t)
	db.SetSetting("context_max_notes_session", "2")
	db.SetSetting("context_limit_policy", "lru_session")
	r1, err := Store("lru-s", "one", "1", "", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := Store("lru-s", "two", "2", "", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	r3, err := Store("lru-s", "three", "3", "", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(r3.EvictedRefs) != 1 || r3.EvictedRefs[0] != r1.Ref {
		t.Fatalf("evicted=%v want %s", r3.EvictedRefs, r1.Ref)
	}
	fetch, _ := Fetch([]string{r1.Ref, r2.Ref, r3.Ref}, "lru-s")
	if len(fetch.Notes) != 2 {
		t.Fatalf("expected 2 notes after eviction, got %d", len(fetch.Notes))
	}
	_ = r2
}

func TestDashboardStats(t *testing.T) {
	testNotesDB(t)
	Store("dash", "content for stats", "lbl", "", nil, nil)
	ds := DashboardStatsFor("", 30)
	if ds.ActiveNotesCount != 1 {
		t.Fatalf("active notes: %d", ds.ActiveNotesCount)
	}
	if ds.Limits["max_notes_session"].(int) != 50 {
		t.Fatalf("limits: %+v", ds.Limits)
	}
}
