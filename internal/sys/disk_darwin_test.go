//go:build darwin

package sys

import (
	"strings"
	"testing"
)

func TestBlockStorageBytesParse(t *testing.T) {
	sample := `"Statistics" = {"Bytes (Read)"=123456,"Bytes (Write)"=7890}`
	if !reIORead.MatchString(sample) || !reIOWrite.MatchString(sample) {
		t.Fatal("regex should match sample statistics")
	}
	m := reIORead.FindStringSubmatch(sample)
	if m[1] != "123456" {
		t.Fatalf("read=%v", m)
	}
}

func TestDeviceForPathResolvesArbitraryDirectory(t *testing.T) {
	dir := t.TempDir()
	device := deviceForPath(dir)
	if device == "" {
		t.Fatalf("deviceForPath(%q) returned empty, want a device identifier", dir)
	}
	if strings.Contains(device, "/") {
		t.Fatalf("deviceForPath(%q) = %q, want a bare identifier with no /dev/ prefix", dir, device)
	}
}

func TestDeviceForPathUnknownPathReturnsEmpty(t *testing.T) {
	if got := deviceForPath("/this/path/does/not/exist/anywhere"); got != "" {
		t.Fatalf("deviceForPath for a nonexistent path = %q, want empty", got)
	}
}

func TestApplySmartctlNoDiskReturnsFalse(t *testing.T) {
	d := &DiskInfo{}
	if applySmartctl(d, "") {
		t.Fatal("empty wholeDisk should never call out to smartctl")
	}
}

func TestLiveProbeDiskHealth(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping live hardware probe in short mode")
	}
	disks := probeAllDisks("/")
	for _, d := range disks {
		if !d.IsExternal && d.SmartStatus != "" && !strings.EqualFold(d.SmartStatus, "not supported") {
			if d.SmartSource == "" {
				t.Fatalf("expected smart source to be set when smart status is %q", d.SmartStatus)
			}
		}
	}
}
