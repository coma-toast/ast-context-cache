package db

import (
	"reflect"
	"testing"
)

// Reproduces the state a purge left on the live server on 2026-10-01: all four
// symbol FTS triggers gone, so later symbol writes skipped both indexes (an insert
// left unindexed, a delete leaving an orphan entry). CheckFTSHealth must see all of
// it, put the triggers back, and rebuild until the indexes match symbols again.
func TestCheckFTSHealthRepairsMissingTriggersAndDrift(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	if err := Init(); err != nil {
		t.Fatal(err)
	}
	defer Close()
	initFTSRebuild.Wait()

	insert := func(name string) int64 {
		t.Helper()
		res, err := IndexDB.Exec(`INSERT INTO symbols (name, kind, file, fqn, code, project_path) VALUES (?, 'function', '/p/a.go', ?, ?, '/p')`,
			name, "pkg."+name, "func "+name+"() {}")
		if err != nil {
			t.Fatal(err)
		}
		id, _ := res.LastInsertId()
		return id
	}
	hits := func(table, term string) int {
		t.Helper()
		var n int
		if err := IndexDB.QueryRow(`SELECT count(*) FROM `+table+` WHERE `+table+` MATCH ?`, term).Scan(&n); err != nil {
			t.Fatal(err)
		}
		return n
	}

	gone := insert("Removed")
	for _, tr := range symbolFTSTriggers {
		if _, err := IndexDB.Exec("DROP TRIGGER " + tr.name); err != nil {
			t.Fatal(err)
		}
	}
	insert("Unindexed")
	if _, err := IndexDB.Exec(`DELETE FROM symbols WHERE id = ?`, gone); err != nil {
		t.Fatal(err)
	}

	h, err := MeasureFTSHealth(true)
	if err != nil {
		t.Fatal(err)
	}
	want := FTSHealth{
		MissingTriggers: []string{"symbols_fts_ins", "symbols_fts_del", "symbols_trigram_ins", "symbols_trigram_del"},
		FTSMissing:      1, FTSOrphans: 1, TrigramMissing: 1, TrigramOrphans: 1,
	}
	if !reflect.DeepEqual(h, want) {
		t.Fatalf("before repair: %+v, want %+v", h, want)
	}

	// measureDrift=false: missing triggers alone must force the drift check.
	if got, err := CheckFTSHealth(false); err != nil {
		t.Fatal(err)
	} else if !reflect.DeepEqual(got, want) {
		t.Fatalf("CheckFTSHealth reported %+v, want %+v", got, want)
	}

	if h, err = MeasureFTSHealth(true); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(h, FTSHealth{}) {
		t.Fatalf("after repair: %+v, want no drift and all triggers", h)
	}
	if hits("symbols_fts", "Unindexed") != 1 || hits("symbols_trigram", "nindex") != 1 {
		t.Fatal("row written while the triggers were missing is still unsearchable after repair")
	}
	insert("Later")
	if hits("symbols_fts", "Later") != 1 || hits("symbols_trigram", "Later") != 1 {
		t.Fatal("recreated triggers don't index new symbols")
	}
}

// A healthy index must come back clean without CheckFTSHealth touching anything.
func TestCheckFTSHealthHealthyIndex(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	if err := Init(); err != nil {
		t.Fatal(err)
	}
	defer Close()
	initFTSRebuild.Wait()

	if _, err := IndexDB.Exec(`INSERT INTO symbols (name, kind, file, project_path) VALUES ('Fine', 'function', '/p/a.go', '/p')`); err != nil {
		t.Fatal(err)
	}
	h, err := CheckFTSHealth(true)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(h, FTSHealth{}) {
		t.Fatalf("healthy index reported %+v", h)
	}
}
