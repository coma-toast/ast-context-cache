package sys

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestFileDescriptorUsageCountsOpenFiles(t *testing.T) {
	if openFDDir() == "" {
		t.Skipf("no per-process fd directory on %s", runtime.GOOS)
	}
	before := FileDescriptorUsage()
	if !before.Available {
		t.Fatal("fd count unavailable")
	}
	if before.SoftLimit == 0 {
		t.Fatal("soft RLIMIT_NOFILE not read")
	}
	const extra = 5
	dir := t.TempDir()
	for i := 0; i < extra; i++ {
		f, err := os.Create(filepath.Join(dir, string(rune('a'+i))))
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
	}
	after := FileDescriptorUsage()
	// Other goroutines (the test runner) may open or close a descriptor in
	// between, so allow a little slack either way.
	if got := after.Open - before.Open; got < extra-1 || got > extra+2 {
		t.Fatalf("opened %d files, fd count moved by %d (%d -> %d)", extra, got, before.Open, after.Open)
	}
}

func TestFDUsageLevel(t *testing.T) {
	cases := []struct {
		u    FDUsage
		want string
	}{
		{FDUsage{Available: true, Open: 10, SoftLimit: 100}, FDLevelOK},
		{FDUsage{Available: true, Open: 70, SoftLimit: 100}, FDLevelWarning},
		{FDUsage{Available: true, Open: 95, SoftLimit: 100}, FDLevelCritical},
		{FDUsage{Available: false, Open: 0, SoftLimit: 100}, FDLevelUnknown},
		{FDUsage{Available: true, Open: 5, SoftLimit: 0}, FDLevelUnknown},
	}
	for _, c := range cases {
		if got := c.u.Level(); got != c.want {
			t.Errorf("%+v: Level() = %q, want %q", c.u, got, c.want)
		}
	}
}
