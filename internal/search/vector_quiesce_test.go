package search

import (
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func TestEnsureLoadedSkipsDuringIndexQuiesce(t *testing.T) {
	dir := t.TempDir()
	prev := db.SetHomeForTest(dir)
	defer prev()

	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	defer func() {
		db.SetIndexReadGateForTest(false)
		_ = db.RestoreIndexPoolForTest()
		db.Close()
	}()

	Cache.Unload()
	db.SetIndexReadGateForTest(true)
	Cache.ensureLoaded()
	if Cache.loaded {
		t.Fatal("expected skip load during quiesce")
	}
}
