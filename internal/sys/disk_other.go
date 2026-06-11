//go:build !darwin

package sys

// DiskIORates returns host disk throughput when supported on this platform.
func DiskIORates() DiskIO {
	return DiskIO{}
}

// SSDHealthInfo returns SSD health when supported on this platform.
func SSDHealthInfo() SSDHealth {
	return SSDHealth{}
}
