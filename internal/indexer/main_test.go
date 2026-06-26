package indexer

import (
	"os"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func TestMain(m *testing.M) {
	if db.IndexDB != nil {
		db.Close()
	}
	tmpHome, err := os.MkdirTemp("", "astcache-indexer-home-")
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
