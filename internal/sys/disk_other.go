//go:build !darwin

package sys

// DiskIORates returns host disk throughput when supported on this platform.
func DiskIORates() DiskIO {
	return DiskIO{}
}

// SSDHealthInfo returns disk health when supported on this platform.
func SSDHealthInfo(path string) SSDHealth {
	return SSDHealth{WearUsedPct: -1, SparePct: -1, DataWrittenTB: -1, TemperatureC: -1}
}
