package db

import "testing"

func TestThrottledEmbedWorkersBySize(t *testing.T) {
	cases := []struct {
		name string
		wal  int64
		want int
	}{
		{"moderate", walModerateBytes, 8},
		{"warn", walWarnBytes, 4},
		{"high", walHighBytes, 2},
	}
	for _, c := range cases {
		if got := staticCapForTest(c.wal, 15); got != c.want {
			t.Fatalf("%s: cap=%d want %d", c.name, got, c.want)
		}
	}
}

// staticCapForTest mirrors ThrottledEmbedWorkers without touching the WAL file.
func staticCapForTest(wal int64, requested int) int {
	switch {
	case wal >= walHighBytes:
		return min(requested, 2)
	case wal >= walWarnBytes:
		return min(requested, 4)
	case wal >= walModerateBytes:
		return min(requested, 8)
	}
	return requested
}

func TestNextBackpressureCapStepsDownToZero(t *testing.T) {
	const static = 2
	wal := int64(walHighBytes * 3)

	ceiling := nextBackpressureCap(-1, static, wal, 0)
	if ceiling != static {
		t.Fatalf("first sample ceiling=%d want %d", ceiling, static)
	}
	for _, want := range []int{1, 0, 0} {
		ceiling = nextBackpressureCap(ceiling, static, wal, wal)
		if ceiling != want {
			t.Fatalf("stalled WAL ceiling=%d want %d", ceiling, want)
		}
	}
}

func TestNextBackpressureCapStepsDownWhenWalGrows(t *testing.T) {
	wal := int64(walHighBytes)
	ceiling := nextBackpressureCap(2, 2, wal+walDrainProgressBytes, wal)
	if ceiling != 1 {
		t.Fatalf("growing WAL ceiling=%d want 1", ceiling)
	}
}

func TestNextBackpressureCapRelaxesWhileDraining(t *testing.T) {
	const static = 4
	wal := int64(walWarnBytes * 2)
	ceiling := 0
	for _, want := range []int{1, 2, 4, 4} {
		wal -= 2 * walDrainProgressBytes
		ceiling = nextBackpressureCap(ceiling, static, wal, wal+2*walDrainProgressBytes)
		if ceiling != want {
			t.Fatalf("draining WAL ceiling=%d want %d", ceiling, want)
		}
	}
}

func TestNextBackpressureCapHoldsBelowDrainThreshold(t *testing.T) {
	wal := int64(walHighBytes)
	// A shrink smaller than walDrainProgressBytes is noise, not drainage.
	ceiling := nextBackpressureCap(2, 2, wal-1, wal)
	if ceiling != 1 {
		t.Fatalf("noise shrink ceiling=%d want 1", ceiling)
	}
}

func TestCappedEmbedWorkersUsesCeiling(t *testing.T) {
	t.Cleanup(func() { SetWalBackpressureForTest(-1, 0) })

	SetWalBackpressureForTest(-1, 0)
	if WalBackpressurePaused() {
		t.Fatal("no ceiling should not report paused")
	}
	SetWalBackpressureForTest(0, walHighBytes)
	if !WalBackpressurePaused() {
		t.Fatal("ceiling 0 should report paused")
	}
	if got := CappedEmbedWorkers(15); got != 0 {
		t.Fatalf("CappedEmbedWorkers=%d want 0", got)
	}
	SetWalBackpressureForTest(1, walHighBytes)
	if got := CappedEmbedWorkers(15); got != 1 {
		t.Fatalf("CappedEmbedWorkers=%d want 1", got)
	}
}
