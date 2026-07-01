package db

import "os"

// BeginWALMaintenanceForTest starts WAL maintenance tracking (tests only).
func BeginWALMaintenanceForTest(reason string) { beginWALMaintenance(reason) }

// EndWALMaintenanceForTest ends WAL maintenance tracking (tests only).
func EndWALMaintenanceForTest() { endWALMaintenance() }

// SetIndexReadGateForTest sets the index read gate (tests only).
func SetIndexReadGateForTest(v bool) { indexReadGate.Store(v) }

// RestoreIndexPoolForTest reopens the index pool after quiesce (tests only).
func RestoreIndexPoolForTest() error { return restoreIndexPool() }

// QuiesceIndexPoolForTest closes the index pool for TRUNCATE (tests only).
func QuiesceIndexPoolForTest() error { return quiesceIndexPool() }

// SetHomeForTest points cache paths at dir/home/.astcache; returns restore func.
func SetHomeForTest(dir string) func() {
	prev := os.Getenv("HOME")
	os.Setenv("HOME", dir)
	return func() { os.Setenv("HOME", prev) }
}
