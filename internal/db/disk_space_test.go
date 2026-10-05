package db

import (
	"bytes"
	"errors"
	"log/slog"
	"strings"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/logging"
)

func fakeFree(free uint64, err error) func(string) (uint64, error) {
	return func(string) (uint64, error) { return free, err }
}

// Field report #12: nothing checked free space while the WAL grew to ~4 GB.
func TestDiskSpaceCeiling(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	restore := SetFreeBytesFuncForTest(fakeFree(100<<30, nil))
	defer restore()
	for _, c := range []int{-1, 0, 8} {
		if got := DiskSpaceCeiling(c); got != c {
			t.Fatalf("plenty of space: ceiling %d -> %d, want unchanged", c, got)
		}
	}
	if LastDiskSpace().Level != DiskOK {
		t.Fatalf("level=%s want ok", LastDiskSpace().Level)
	}

	SetFreeBytesFuncForTest(fakeFree(3<<30, nil))
	for c, want := range map[int]int{-1: diskLowWorkerCap, 8: diskLowWorkerCap, 1: 1, 0: 0} {
		if got := DiskSpaceCeiling(c); got != want {
			t.Fatalf("low space: ceiling %d -> %d, want %d", c, got, want)
		}
	}
	if s := LastDiskSpace(); s.Level != DiskLow || s.FreeBytes != 3<<30 {
		t.Fatalf("snapshot=%+v want low with 3 GiB", s)
	}

	SetFreeBytesFuncForTest(fakeFree(500<<20, nil))
	if got := DiskSpaceCeiling(-1); got != 0 {
		t.Fatalf("critical space: ceiling -> %d, want 0 (pause)", got)
	}
	if LastDiskSpace().Level != DiskCritical {
		t.Fatalf("level=%s want critical", LastDiskSpace().Level)
	}
}

func TestDiskSpaceProbeErrorNeverThrottles(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	restore := SetFreeBytesFuncForTest(fakeFree(0, errors.New("statfs: no such file")))
	defer restore()
	if got := DiskSpaceCeiling(-1); got != -1 {
		t.Fatalf("probe error: ceiling -> %d, want -1", got)
	}
	if LastDiskSpace().Level != DiskOK {
		t.Fatalf("level=%s want ok on probe error", LastDiskSpace().Level)
	}
}

func TestStatfsFreeBytesReal(t *testing.T) {
	free, err := statfsFreeBytes(t.TempDir())
	if err != nil || free == 0 {
		t.Fatalf("statfs free=%d err=%v", free, err)
	}
}

// The "warn" half of the disk-pressure check: each level change is logged once (not on
// every 30s sample), and recovery is logged too.
func TestSampleDiskSpaceWarnsOnLevelChange(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	free := uint64(100 << 30)
	restore := SetFreeBytesFuncForTest(func(string) (uint64, error) { return free, nil })
	defer restore()
	var buf bytes.Buffer
	prevLogger := slog.Default()
	slog.SetDefault(slog.New(logging.NewHandler(&buf, logging.FormatText, slog.LevelDebug)))
	t.Cleanup(func() { slog.SetDefault(prevLogger) })

	for _, f := range []uint64{100 << 30, 3 << 30, 3 << 30, 500 << 20, 500 << 20, 100 << 30, 100 << 30} {
		free = f
		SampleDiskSpace()
	}
	out := buf.String()
	for msg, want := range map[string]int{
		"Disk space low":       1,
		"Disk space critical":  1,
		"Disk space recovered": 1,
	} {
		if got := strings.Count(out, msg); got != want {
			t.Fatalf("%q logged %d times, want %d; log:\n%s", msg, got, want, out)
		}
	}
	if !strings.Contains(out, "capping embed workers") || !strings.Contains(out, "worker_cap=2") || !strings.Contains(out, "pausing embed workers") {
		t.Fatalf("warnings do not say what is throttled; log:\n%s", out)
	}
}
