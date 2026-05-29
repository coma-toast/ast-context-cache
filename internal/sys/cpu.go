package sys

import (
	"sync"
	"syscall"
	"time"
)

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

var (
	cpuPctMu    sync.Mutex
	cpuPctLast  CPUSample
	cpuPctAt    time.Time
	cpuPctReady bool
)

// ProcessCPUPercent returns process CPU use since the previous sample (0–100+ on multi-core).
func ProcessCPUPercent() float64 {
	cpuPctMu.Lock()
	defer cpuPctMu.Unlock()
	now := time.Now()
	cur := SampleCPU()
	if !cpuPctReady {
		cpuPctLast = cur
		cpuPctAt = now
		cpuPctReady = true
		return 0
	}
	wallMs := now.Sub(cpuPctAt).Seconds() * 1000
	cpuMs := DeltaMs(cpuPctLast, cur)
	cpuPctLast = cur
	cpuPctAt = now
	if wallMs < 50 {
		return 0
	}
	return cpuMs / wallMs * 100
}
