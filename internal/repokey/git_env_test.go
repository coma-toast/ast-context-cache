package repokey

import (
	"os"
	"testing"
)

// TestMain keeps the git fixtures in this package's tests independent of the
// developer's own git config: a global commit-signing setup (e.g. signing through
// a password-manager SSH agent that may be locked) otherwise makes every fixture
// commit fail with "failed to write commit object".
func TestMain(m *testing.M) {
	os.Setenv("GIT_CONFIG_GLOBAL", os.DevNull)
	os.Setenv("GIT_CONFIG_NOSYSTEM", "1")
	os.Exit(m.Run())
}
