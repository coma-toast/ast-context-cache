package memory

import (
	"testing"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

// TestPruneArchivedDeletesOldRowsOnly tests that PruneArchived deletes rows with
// valid_until set older than the window, regardless of invalidated_reason.
// Rows with valid_until within the window are retained.
func TestPruneArchivedDeletesOldRowsOnly(t *testing.T) {
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

	// Set valid_until for old and recent rows (100 days and 1 day old)
	oldCutoff := time.Now().AddDate(0, 0, -100).Format("2006-01-02 15:04:05")
	recentCutoff := time.Now().AddDate(0, 0, -1).Format("2006-01-02 15:04:05")
	if _, err := db.ContextDB.Exec(`UPDATE structured_memory SET valid_until = ? WHERE ref = ?`, oldCutoff, oldRes.Ref); err != nil {
		t.Fatal(err)
	}
	if _, err := db.ContextDB.Exec(`UPDATE structured_memory SET valid_until = ? WHERE ref = ?`, recentCutoff, recentRes.Ref); err != nil {
		t.Fatal(err)
	}
	// activeRes stays untouched (valid_until still NULL — never superseded).

	// Test PruneSuperseded is now a no-op per MEM-2
	if n, err := PruneSuperseded(90); err != nil {
		t.Fatal(err)
	} else if n != 0 {
		t.Fatalf("PruneSuperseded should be a no-op, got n=%d want 0", n)
	}
	assertPresent(t, oldRes.Ref)   // Still present (not deleted by Superseded)
	assertPresent(t, recentRes.Ref)
	assertPresent(t, activeRes.Ref)

	// Test PruneArchived with a 365-day window: all should be kept (within window)
	if n, err := PruneArchived(365); err != nil {
		t.Fatal(err)
	} else if n != 0 {
		t.Fatalf("PruneArchived with 365-day window: pruned=%d want 0", n)
	}
	assertPresent(t, oldRes.Ref)
	assertPresent(t, recentRes.Ref)
	assertPresent(t, activeRes.Ref)

	// Test PruneArchived with a 2-day window:
	// - oldRes (100 days) should be deleted
	// - recentRes (1 day) should be kept
	// - activeRes (NULL valid_until) should be kept
	if n, err := PruneArchived(2); err != nil {
		t.Fatal(err)
	} else if n != 1 {
		t.Fatalf("PruneArchived with 2-day window: pruned=%d want 1 (the 100-day-old row)", n)
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
