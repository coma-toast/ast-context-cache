package embedqueue

import (
	"os"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

// TestMain opens the db pools once, for every test in the package. Start's
// goroutines run for the rest of the process and read those pools, so a test
// calling db.Init again after any test has called Start races them.
func TestMain(m *testing.M) {
	if db.IndexDB != nil {
		db.Close()
	}
	tmpHome, err := os.MkdirTemp("", "astcache-embedqueue-home-")
	if err != nil {
		panic(err)
	}
	os.Setenv("HOME", tmpHome)
	if err := db.Init(); err != nil {
		panic(err)
	}
	code := m.Run()
	db.Close()
	os.RemoveAll(tmpHome)
	os.Exit(code)
}
