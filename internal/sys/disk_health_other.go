//go:build !darwin && !linux

package sys

// DiskHealthInfo returns no disks on platforms without a native implementation.
func DiskHealthInfo(path string) ([]DiskInfo, string) {
	return nil, "SMART monitoring is not available on this platform"
}
