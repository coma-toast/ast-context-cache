package sys

import (
	"strings"
	"testing"
)

// baseDisk returns a DiskInfo with every unknown attribute set to the model
// sentinel (-1), as real collectors produce.
func baseDisk() DiskInfo {
	return DiskInfo{
		Device:               "/dev/test",
		SmartStatus:          "PASSED",
		WearUsedPct:          -1,
		SparePct:             -1,
		SSDLifeLeft:          -1,
		TemperatureC:         -1,
		PowerOnHours:         -1,
		MediaErrors:          -1,
		ReallocatedSectors:   -1,
		PendingSectors:       -1,
		OfflineUncorrectable: -1,
		UDMACRCErrors:        -1,
		SASGrownDefects:      -1,
		SASUncorrectedErrors: -1,
	}
}

func TestEvaluateDiskWarningsHealthy(t *testing.T) {
	d := baseDisk()
	d.WearUsedPct = 42
	d.SSDLifeLeft = 58
	d.SparePct = 100
	d.TemperatureC = 38
	d.ReallocatedSectors = 0
	level, msg := EvaluateDiskWarnings(d)
	if level != WarningOK {
		t.Fatalf("level = %q, want ok", level)
	}
	if msg != "" {
		t.Fatalf("msg = %q, want empty", msg)
	}
}

func TestEvaluateDiskWarningsUnknownAttributesSkipped(t *testing.T) {
	d := baseDisk()
	d.Device = "/dev/sda"
	d.SmartStatus = "Verified"
	// All numeric attributes are -1 (unknown): no warnings.
	level, msg := EvaluateDiskWarnings(d)
	if level != WarningOK || msg != "" {
		t.Fatalf("got (%q, %q), want (ok, \"\")", level, msg)
	}
}

func TestEvaluateDiskWarningsThresholds(t *testing.T) {
	cases := []struct {
		name    string
		mutate  func(*DiskInfo)
		want    string
		wantSub string
	}{
		{"smart failing", func(d *DiskInfo) { d.SmartStatus = "Failing" }, WarningCritical, "SMART self-assessment failing"},
		{"nvme cw readonly", func(d *DiskInfo) { d.CriticalWarning = 0x04 }, WarningCritical, "NVMe critical warning"},
		{"nvme cw other", func(d *DiskInfo) { d.CriticalWarning = 0x01 }, WarningWarning, "NVMe critical warning"},
		{"wear critical", func(d *DiskInfo) { d.WearUsedPct = 95 }, WarningCritical, "lifetime wear used 95%"},
		{"wear warning", func(d *DiskInfo) { d.WearUsedPct = 80 }, WarningWarning, "lifetime wear used 80%"},
		{"wear ok", func(d *DiskInfo) { d.WearUsedPct = 74 }, WarningOK, ""},
		{"life left critical", func(d *DiskInfo) { d.SSDLifeLeft = 8 }, WarningCritical, "SSD life left 8%"},
		{"life left warning", func(d *DiskInfo) { d.SSDLifeLeft = 15 }, WarningWarning, "SSD life left 15%"},
		{"spare critical", func(d *DiskInfo) { d.SparePct = 5 }, WarningCritical, "available spare 5%"},
		{"spare warning", func(d *DiskInfo) { d.SparePct = 15 }, WarningWarning, "available spare 15%"},
		{"pending critical", func(d *DiskInfo) { d.PendingSectors = 2 }, WarningCritical, "2 pending sector"},
		{"offline uncorrectable critical", func(d *DiskInfo) { d.OfflineUncorrectable = 1 }, WarningCritical, "1 offline uncorrectable"},
		{"reallocated warning", func(d *DiskInfo) { d.ReallocatedSectors = 3 }, WarningWarning, "3 reallocated"},
		{"reallocated critical", func(d *DiskInfo) { d.ReallocatedSectors = 60 }, WarningCritical, "60 reallocated"},
		{"udma crc warning", func(d *DiskInfo) { d.UDMACRCErrors = 9 }, WarningWarning, "9 UDMA CRC"},
		{"media errors warning", func(d *DiskInfo) { d.MediaErrors = 4 }, WarningWarning, "4 media error"},
		{"sas grown defects warning", func(d *DiskInfo) { d.SASGrownDefects = 1 }, WarningWarning, "1 SAS grown defect"},
		{"sas uncorrected warning", func(d *DiskInfo) { d.SASUncorrectedErrors = 2 }, WarningWarning, "2 SAS uncorrected"},
		{"hdd temp warning", func(d *DiskInfo) { d.TemperatureC = 60 }, WarningWarning, "temperature 60"},
		{"hdd temp critical", func(d *DiskInfo) { d.TemperatureC = 70 }, WarningCritical, "temperature 70"},
		{"ssd temp critical", func(d *DiskInfo) { d.SolidState = true; d.TemperatureC = 85 }, WarningCritical, "temperature 85"},
		{"ssd temp ok at hdd warn", func(d *DiskInfo) { d.SolidState = true; d.TemperatureC = 60 }, WarningOK, ""},
		{"zfs degraded", func(d *DiskInfo) { d.ZFSPool = "tank"; d.ZFSHealth = "degraded" }, WarningWarning, `ZFS pool "tank" health: degraded`},
		{"zfs faulted", func(d *DiskInfo) { d.ZFSPool = "tank"; d.ZFSHealth = "FAULTED" }, WarningCritical, `ZFS pool "tank" health: FAULTED`},
		{"zfs capacity critical", func(d *DiskInfo) { d.ZFSPool = "tank"; d.ZFSHealth = "online"; d.ZFSCapacityPct = 92 }, WarningCritical, "92% capacity"},
		{"zfs capacity warning", func(d *DiskInfo) { d.ZFSPool = "tank"; d.ZFSHealth = "online"; d.ZFSCapacityPct = 87 }, WarningWarning, "87% capacity"},
		{"zfs frag warning", func(d *DiskInfo) { d.ZFSPool = "tank"; d.ZFSHealth = "online"; d.ZFSFragPct = 60 }, WarningWarning, "60% fragmentation"},
		{"zfs attrs ignored without pool", func(d *DiskInfo) { d.ZFSCapacityPct = 99; d.ZFSFragPct = 99 }, WarningOK, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := baseDisk()
			tc.mutate(&d)
			level, msg := EvaluateDiskWarnings(d)
			if level != tc.want {
				t.Fatalf("level = %q, want %q (msg=%q)", level, tc.want, msg)
			}
			if tc.wantSub != "" && !strings.Contains(msg, tc.wantSub) {
				t.Fatalf("msg = %q, want substring %q", msg, tc.wantSub)
			}
			if level == WarningOK && msg != "" {
				t.Fatalf("ok level must have empty msg, got %q", msg)
			}
		})
	}
}

func TestEvaluateDiskWarningsMultipleBreaches(t *testing.T) {
	d := baseDisk()
	d.Device = "/dev/sda"
	d.TemperatureC = 58
	d.PendingSectors = 1
	d.ReallocatedSectors = 4
	level, msg := EvaluateDiskWarnings(d)
	if level != WarningCritical {
		t.Fatalf("level = %q, want critical", level)
	}
	if !strings.Contains(msg, "pending sector") || !strings.Contains(msg, "reallocated") || !strings.Contains(msg, "temperature") {
		t.Fatalf("msg = %q, want all three breaches", msg)
	}
	if !strings.Contains(msg, "; ") {
		t.Fatalf("msg = %q, want '; ' separator", msg)
	}
}

func TestEvaluateDiskWarningsCollectorNotePreserved(t *testing.T) {
	d := DiskInfo{Device: "/dev/sdb", WarningMsg: "smartctl open failed: permission denied"}
	level, msg := EvaluateDiskWarnings(d)
	if level != WarningWarning {
		t.Fatalf("level = %q, want warning", level)
	}
	if msg != d.WarningMsg {
		t.Fatalf("msg = %q, want note preserved", msg)
	}
}

func TestApplyDiskWarnings(t *testing.T) {
	d1 := baseDisk()
	d1.Device = "/dev/nvme0n1"
	d1.WearUsedPct = 95

	d2 := baseDisk()
	d2.Device = "/dev/sda"
	d2.TemperatureC = 40

	disks := []DiskInfo{
		d1,
		d2,
		{Device: "/dev/sdb", WarningMsg: "note only"},
		{Device: "/dev/sdc", WarningLevel: WarningCritical, WarningMsg: "collector set both"},
	}
	ApplyDiskWarnings(disks)

	if disks[0].WarningLevel != WarningCritical {
		t.Fatalf("disk 0 level = %q, want critical", disks[0].WarningLevel)
	}
	if disks[1].WarningLevel != WarningOK || disks[1].WarningMsg != "" {
		t.Fatalf("disk 1 = (%q, %q), want (ok, \"\")", disks[1].WarningLevel, disks[1].WarningMsg)
	}
	if disks[2].WarningLevel != WarningWarning || disks[2].WarningMsg != "note only" {
		t.Fatalf("disk 2 = (%q, %q), want (warning, note)", disks[2].WarningLevel, disks[2].WarningMsg)
	}
	if disks[3].WarningLevel != WarningCritical || disks[3].WarningMsg != "collector set both" {
		t.Fatalf("disk 3 = (%q, %q), want untouched", disks[3].WarningLevel, disks[3].WarningMsg)
	}
}
