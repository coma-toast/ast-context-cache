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

func TestDeclareDriveDisconnectedGatesReadersAndPoolsReady(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("DB_PATH", "")
	if err := Init(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		ResetDriveMonitorForTest()
		Close()
	})

	if err := SetSetting("host_disks.url", "http://example.local"); err != nil {
		t.Fatal(err)
	}
	if !PoolsReady() {
		t.Fatal("expected PoolsReady before disconnect")
	}
	if _, err := IndexReader(); err != nil {
		t.Fatalf("expected IndexReader before disconnect, got %v", err)
	}
	if got := GetSetting("host_disks.url", "fallback"); got != "http://example.local" {
		t.Fatalf("expected stored setting before disconnect, got %q", got)
	}

	declareDriveDisconnected()

	if !DriveDisconnected() {
		t.Fatal("expected DriveDisconnected to be true")
	}
	if PoolsReady() {
		t.Fatal("expected PoolsReady to be false after disconnect")
	}
	if _, err := IndexReader(); err == nil {
		t.Fatal("expected IndexReader to fail after disconnect")
	}
	if got := GetSetting("host_disks.url", "fallback"); got != "fallback" {
		t.Fatalf("expected fallback from GetSetting after disconnect, got %q", got)
	}
	if got := GetAllSettings(); len(got) != 0 {
		t.Fatalf("expected empty map from GetAllSettings after disconnect, got %v", got)
	}
}

func TestDrivePathOKBlocksUnmountedFilesAndDeviceChange(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("DB_PATH", "")
	if err := Init(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		ResetDriveMonitorForTest()
		Close()
	})

	if err := SetSetting("host_disks.url", "http://example.local"); err != nil {
		t.Fatal(err)
	}

	// Simulate the volume remounting on a new st_dev while pools still hold stale handles.
	origDev := expectedPoolDev.Load()
	expectedPoolDev.Store(origDev + 1)
	if _, err := IndexReader(); err == nil {
		t.Fatal("expected IndexReader to reject read after st_dev change")
	}
	if got := GetSetting("host_disks.url", "fallback"); got != "fallback" {
		t.Fatalf("expected GetSetting fallback after st_dev change, got %q", got)
	}
	if got := GetAllSettings(); len(got) != 0 {
		t.Fatalf("expected empty GetAllSettings after st_dev change, got %v", got)
	}
	expectedPoolDev.Store(origDev)

	// Simulate the USB volume disappearing before the 10s monitor debounce trips.
	if err := os.Remove(indexDBPath()); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(usageDBPath()); err != nil {
		t.Fatal(err)
	}
	if _, err := IndexReader(); err == nil {
		t.Fatal("expected IndexReader to reject read when index.db vanishes")
	}
	if got := GetSetting("host_disks.url", "fallback"); got != "fallback" {
		t.Fatalf("expected GetSetting fallback when usage.db vanishes, got %q", got)
	}
}
