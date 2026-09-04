package db

import (
	"os"
	"testing"
)

func TestRecordDriveCheckDebouncesBeforeDeclaring(t *testing.T) {
	ResetDriveMonitorForTest()
	// First reachable check establishes the baseline device — never declares.
	if recordDriveCheck(1, true) {
		t.Fatal("first successful check must not declare disconnected")
	}
	// A single miss must not trip it (missesBeforePurge-style debounce).
	if recordDriveCheck(0, false) {
		t.Fatal("a single miss must not declare disconnected")
	}
	// Second consecutive miss reaches the threshold.
	if !recordDriveCheck(0, false) {
		t.Fatal("two consecutive misses should declare disconnected")
	}
}

func TestRecordDriveCheckRecoversFromATransientBlip(t *testing.T) {
	ResetDriveMonitorForTest()
	if recordDriveCheck(1, true) {
		t.Fatal("baseline check must not declare disconnected")
	}
	if recordDriveCheck(0, false) {
		t.Fatal("one miss must not declare disconnected")
	}
	// The path is reachable again on the same device before the threshold is hit —
	// the miss streak must reset, not carry over.
	if recordDriveCheck(1, true) {
		t.Fatal("a recovered check must not declare disconnected")
	}
	if recordDriveCheck(0, false) {
		t.Fatal("miss count should have reset after recovery, so this is only miss 1")
	}
}

func TestRecordDriveCheckDetectsDeviceChangeWithoutUnreachablePath(t *testing.T) {
	ResetDriveMonitorForTest()
	if recordDriveCheck(1, true) {
		t.Fatal("baseline check must not declare disconnected")
	}
	// Path stays statable (e.g. an unmounted mount point left behind as an empty dir
	// on the parent filesystem) but the device id no longer matches the baseline.
	if recordDriveCheck(2, true) {
		t.Fatal("a single device mismatch must not declare disconnected")
	}
	if !recordDriveCheck(2, true) {
		t.Fatal("two consecutive device mismatches should declare disconnected")
	}
}

func TestStatDevReturnsDeviceForRealPath(t *testing.T) {
	dev, ok := statDev(os.TempDir())
	if !ok {
		t.Fatal("statDev on an existing directory should succeed")
	}
	if dev == 0 {
		t.Fatal("expected a non-zero device id for a real path")
	}
}

func TestStatDevMissingPathReturnsNotOK(t *testing.T) {
	if _, ok := statDev("/this/path/does/not/exist/anywhere"); ok {
		t.Fatal("statDev on a nonexistent path should report not ok")
	}
}
