package sys

import (
	"os"
	"runtime"
	"syscall"
)

// FD usage levels, relative to the soft RLIMIT_NOFILE.
const (
	FDLevelOK       = "ok"
	FDLevelWarning  = "warning"  // at or above FDWarnPct
	FDLevelCritical = "critical" // at or above FDCriticalPct
	FDLevelUnknown  = "unknown"

	FDWarnPct     = 70.0
	FDCriticalPct = 90.0
)

// FDUsage is this process's open file descriptor count against RLIMIT_NOFILE.
//
// Go raises the soft limit to the hard limit at startup (capped at
// kern.maxfilesperproc on macOS), so SoftLimit is the real ceiling: raising it
// further needs root (sysctl), not a setrlimit call.
type FDUsage struct {
	Available bool // false when the open count couldn't be read
	Open      int
	SoftLimit uint64
	HardLimit uint64
}

// Pct is Open as a percentage of the soft limit (0 when either is unknown).
func (u FDUsage) Pct() float64 {
	if !u.Available || u.SoftLimit == 0 {
		return 0
	}
	return float64(u.Open) / float64(u.SoftLimit) * 100
}

// Level buckets Pct into ok / warning / critical.
func (u FDUsage) Level() string {
	if !u.Available || u.SoftLimit == 0 {
		return FDLevelUnknown
	}
	switch p := u.Pct(); {
	case p >= FDCriticalPct:
		return FDLevelCritical
	case p >= FDWarnPct:
		return FDLevelWarning
	}
	return FDLevelOK
}

// FileDescriptorUsage counts this process's open descriptors and reads its limits.
func FileDescriptorUsage() FDUsage {
	var u FDUsage
	var lim syscall.Rlimit
	if syscall.Getrlimit(syscall.RLIMIT_NOFILE, &lim) == nil {
		u.SoftLimit = uint64(lim.Cur)
		u.HardLimit = uint64(lim.Max)
	}
	if n, ok := countOpenFDs(); ok {
		u.Open = n
		u.Available = true
	}
	return u
}

// FDCountSupported reports whether FileDescriptorUsage can count open
// descriptors on this OS at all.
func FDCountSupported() bool { return openFDDir() != "" }

func openFDDir() string {
	switch runtime.GOOS {
	case "linux":
		return "/proc/self/fd"
	case "darwin", "freebsd":
		return "/dev/fd"
	}
	return ""
}

// countOpenFDs lists the per-process descriptor directory. It needs one
// descriptor itself, so it reports unavailable once the process is already at
// its limit — the caller's ok=false is then the signal.
func countOpenFDs() (int, bool) {
	dir := openFDDir()
	if dir == "" {
		return 0, false
	}
	f, err := os.Open(dir)
	if err != nil {
		return 0, false
	}
	names, err := f.Readdirnames(-1)
	f.Close()
	if err != nil {
		return 0, false
	}
	// Don't count the descriptor that was reading the directory.
	return max(len(names)-1, 0), true
}
