package embedqueue

import (
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

// Low free space on the data volume must tighten the embed-worker ceiling even when the
// WAL itself is small, and lift it once space is back. The free-space probe is faked;
// nothing fills a disk. (applyPrimaryCeiling/applyAuxCeiling enforcing a ceiling is
// covered by wal_backpressure_grace_test.go. An end-to-end version that ran the live
// pool through applyWalBackpressure left package-global queue state behind that
// made TestQuietOnWorkersPausedKeepsSwapPause flaky, so it was dropped.)
func TestBackpressureCeilingThrottlesOnLowDisk(t *testing.T) {
	free := uint64(100 << 30)
	restore := db.SetFreeBytesFuncForTest(func(string) (uint64, error) { return free, nil })
	t.Cleanup(restore)
	db.SetWalBackpressureForTest(-1, 0)
	steps := []struct {
		free uint64
		want int
	}{
		{100 << 30, -1}, // plenty of space, small WAL: no ceiling
		{3 << 30, 2},    // low: capped
		{500 << 20, 0},  // critical: paused
		{100 << 30, -1}, // recovered: ceiling lifted
	}
	for _, s := range steps {
		free = s.free
		if got := backpressureCeiling(5); got != s.want {
			t.Fatalf("free=%d MiB: ceiling=%d want %d", s.free>>20, got, s.want)
		}
	}
}
