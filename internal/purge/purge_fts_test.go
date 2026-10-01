package purge

import (
	"errors"
	"fmt"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

// A purge removes symbols one of two ways: through the per-row delete triggers
// when the project is a minority of the index, or by dropping the triggers and
// rebuilding both FTS indexes when it is most of it. Every FTS test runs both.
var purgeStrategies = []struct {
	name         string
	otherSymbols int // symbols in an unrelated project that must survive the purge
}{
	{"per-row triggers", 5},
	{"drop and rebuild", 0},
}

// On 2026-10-01 the live server's purge sweep ran while WAL maintenance had index
// reads quiesced. The purge had already dropped the FTS triggers and deleted the
// symbols; its trigram rebuild then failed and EnsureFTSTriggers silently did
// nothing because IndexReader was gated, so symbols was left with no FTS triggers.
// Quiesce at that exact point and require the purge to finish with the triggers
// in place and both indexes matching symbols.
func TestProjectDataKeepsFTSWhenQuiescedMidPurge(t *testing.T) {
	for _, tc := range purgeStrategies {
		t.Run(tc.name, func(t *testing.T) {
			p, other := setupFTSPurge(t, tc.otherSymbols)

			quiesced := make(chan error, 1)
			afterSymbolDelete = func() error {
				go func() { quiesced <- db.QuiesceIndexPoolForTest() }()
				// Carry on only once maintenance has gated index reads, which is when
				// the old purge's EnsureFTSTriggers found IndexReader failing.
				deadline := time.Now().Add(5 * time.Second)
				for !db.IndexReadQuiesced() {
					if time.Now().After(deadline) {
						return errors.New("quiesce never gated index reads")
					}
					time.Sleep(time.Millisecond)
				}
				return nil
			}

			if err := ProjectData(p); err != nil {
				t.Fatal(err)
			}
			select {
			case err := <-quiesced:
				if err != nil {
					t.Fatal(err)
				}
			case <-time.After(10 * time.Second):
				t.Fatal("quiesce never finished")
			}
			if err := db.RestoreIndexPoolForTest(); err != nil {
				t.Fatal(err)
			}

			assertFTSHealthy(t)
			if n := symbolCount(t, p); n != 0 {
				t.Fatalf("purged project still has %d symbols", n)
			}
			if n := symbolCount(t, other); n != tc.otherSymbols {
				t.Fatalf("other project has %d symbols, want %d", n, tc.otherSymbols)
			}
			if tc.otherSymbols > 0 && (ftsHits(t, "symbols_fts", "Keep0") != 1 || ftsHits(t, "symbols_trigram", "Keep0") != 1) {
				t.Fatal("surviving project's symbols dropped out of the FTS indexes")
			}
			// The production symptom: with the triggers gone, new symbols were never indexed.
			insertSymbol(t, other, "WrittenAfterPurge")
			if ftsHits(t, "symbols_fts", "WrittenAfterPurge") != 1 || ftsHits(t, "symbols_trigram", "WrittenAfterPurge") != 1 {
				t.Fatal("a symbol written after the purge is missing from the FTS indexes")
			}
		})
	}
}

// Any failure inside the purge must roll the whole index change back, including
// the trigger drop, rather than leave it half-applied as the old ignored Execs did.
func TestProjectDataRollsBackIndexOnFailure(t *testing.T) {
	for _, tc := range purgeStrategies {
		t.Run(tc.name, func(t *testing.T) {
			p, _ := setupFTSPurge(t, tc.otherSymbols)

			injected := errors.New("injected failure")
			afterSymbolDelete = func() error { return injected }

			if err := ProjectData(p); !errors.Is(err, injected) {
				t.Fatalf("ProjectData error = %v, want %v", err, injected)
			}
			assertFTSHealthy(t)
			if n := symbolCount(t, p); n != 2 {
				t.Fatalf("project has %d symbols after a failed purge, want both rolled back", n)
			}
			if ftsHits(t, "symbols_fts", "PurgeMe") != 1 || ftsHits(t, "symbols_trigram", "PurgeMe") != 1 {
				t.Fatal("rolled-back symbol is missing from the FTS indexes")
			}
		})
	}
}

// setupFTSPurge seeds a project to purge (seedProject's rows plus a named symbol)
// and an unrelated project with otherSymbols symbols, and returns both paths.
func setupFTSPurge(t *testing.T, otherSymbols int) (purged, other string) {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		afterSymbolDelete = nil
		db.SetIndexReadGateForTest(false)
		db.Close()
	})
	db.WaitInitFTSRebuildForTest()

	purged = filepath.Join(home, "git", "purgeme")
	other = filepath.Join(home, "git", "keep")
	seedProject(t, purged)
	insertSymbol(t, purged, "PurgeMe")
	for i := 0; i < otherSymbols; i++ {
		insertSymbol(t, other, fmt.Sprintf("Keep%d", i))
	}
	assertFTSHealthy(t)
	return purged, other
}

func insertSymbol(t *testing.T, projectPath, name string) {
	t.Helper()
	if _, err := db.IndexDB.Exec(`INSERT INTO symbols (name, kind, file, fqn, code, project_path) VALUES (?, 'function', ?, ?, ?, ?)`,
		name, filepath.Join(projectPath, "a.go"), "pkg."+name, "func "+name+"() {}", projectPath); err != nil {
		t.Fatal(err)
	}
}

func ftsHits(t *testing.T, table, term string) int {
	t.Helper()
	var n int
	if err := db.IndexDB.QueryRow(`SELECT count(*) FROM `+table+` WHERE `+table+` MATCH ?`, term).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}

func assertFTSHealthy(t *testing.T) {
	t.Helper()
	h, err := db.MeasureFTSHealth(true)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(h, db.FTSHealth{}) {
		t.Fatalf("FTS unhealthy: %+v", h)
	}
}
