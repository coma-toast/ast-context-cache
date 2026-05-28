package sys

import "syscall"

// CPUSample captures process user+system CPU time (microseconds).
type CPUSample struct {
	usec int64
}

// SampleCPU reads RUSAGE_SELF for the current process.
func SampleCPU() CPUSample {
	var ru syscall.Rusage
	_ = syscall.Getrusage(syscall.RUSAGE_SELF, &ru)
	return CPUSample{usec: rusageMicros(&ru)}
}

// DeltaMs returns CPU milliseconds consumed between two samples.
func DeltaMs(before, after CPUSample) float64 {
	d := after.usec - before.usec
	if d < 0 {
		return 0
	}
	return float64(d) / 1000.0
}

func rusageMicros(ru *syscall.Rusage) int64 {
	return ru.Utime.Sec*1e6 + int64(ru.Utime.Usec) + ru.Stime.Sec*1e6 + int64(ru.Stime.Usec)
}
