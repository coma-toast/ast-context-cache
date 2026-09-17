package memory

import (
	"testing"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

// Superseded facts are soft-deleted (valid_until set) but nothing ever purged
// old rows afterward, unlike contextnotes (a token/count quota) and the
// queries table (age-based retention) — structured_memory grew unbounded.
func TestPruneSupersededDeletesOldRowsOnly(t *testing.T) {
	testMemoryDB(t)

	oldRes, err := Store(StoreInput{
		Kind: KindFact, Scope: ScopeGlobal,
		Subject: "user.editor", Predicate: "is", Object: "vim",
	})
	if err != nil {
		t.Fatal(err)
	}
	recentRes, err := Store(StoreInput{
		Kind: KindFact, Scope: ScopeGlobal,
		Subject: "user.shell", Predicate: "is", Object: "fish",
	})
	if err != nil {
		t.Fatal(err)
	}
	activeRes, err := Store(StoreInput{
		Kind: KindFact, Scope: ScopeGlobal,
		Subject: "user.os", Predicate: "is", Object: "macos",
	})
	if err != nil {
		t.Fatal(err)
	}

	oldCutoff := time.Now().AddDate(0, 0, -100).Format("2006-01-02 15:04:05")
	recentCutoff := time.Now().AddDate(0, 0, -1).Format("2006-01-02 15:04:05")
	if _, err := db.ContextDB.Exec(`UPDATE structured_memory SET valid_until = ? WHERE ref = ?`, oldCutoff, oldRes.Ref); err != nil {
		t.Fatal(err)
	}
	if _, err := db.ContextDB.Exec(`UPDATE structured_memory SET valid_until = ? WHERE ref = ?`, recentCutoff, recentRes.Ref); err != nil {
		t.Fatal(err)
	}
	// activeRes stays untouched (valid_until still NULL — never superseded).

	n, err := PruneSuperseded(90)
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Fatalf("pruned=%d want 1 (only the 100-day-old row)", n)
	}

	assertGone(t, oldRes.Ref)
	assertPresent(t, recentRes.Ref)
	assertPresent(t, activeRes.Ref)
}

func assertGone(t *testing.T, ref string) {
	t.Helper()
	var count int
	db.ContextDB.QueryRow(`SELECT COUNT(*) FROM structured_memory WHERE ref = ?`, ref).Scan(&count)
	if count != 0 {
		t.Fatalf("ref %s should have been pruned, still present", ref)
	}
	db.ContextDB.QueryRow(`SELECT COUNT(*) FROM structured_memory_fts WHERE ref = ?`, ref).Scan(&count)
	if count != 0 {
		t.Fatalf("ref %s's FTS mirror should have been pruned too", ref)
	}
}

func assertPresent(t *testing.T, ref string) {
	t.Helper()
	var count int
	db.ContextDB.QueryRow(`SELECT COUNT(*) FROM structured_memory WHERE ref = ?`, ref).Scan(&count)
	if count != 1 {
		t.Fatalf("ref %s should not have been pruned", ref)
	}
}
