package embedqueue

import (
	"os"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func TestMain(m *testing.M) {
	if db.DB != nil {
		db.DB.Close()
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
	db.DB.Close()
	os.RemoveAll(tmpHome)
	os.Exit(code)
}
