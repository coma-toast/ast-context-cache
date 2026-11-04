//go:build !darwin && !linux

package sys

// DiskHealthInfo returns no disks on platforms without a native implementation.
func DiskHealthInfo(path string) []DiskInfo {
	return nil
}

// DiskHealthNote explains why DiskHealthInfo is empty on this platform.
func DiskHealthNote(path string) string {
	return "SMART monitoring is not available on this platform"
}
