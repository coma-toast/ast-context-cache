//go:build linux

package sys

import (
	"os"
	"strings"
	"testing"
)

func loadFixture(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatalf("reading fixture %s: %v", name, err)
	}
	return b
}

func TestParseSmartctlScan(t *testing.T) {
	devs := parseSmartctlScan(loadFixture(t, "smartctl_scan.json"))
	if devs == nil {
		t.Fatal("scan fixture: expected devices, got nil")
	}
	if len(devs) != 4 {
		t.Fatalf("expected 4 devices, got %d", len(devs))
	}
	want := []struct {
		name, typ, proto string
	}{
		{"/dev/nvme0n1", "nvme", "NVMe"},
		{"/dev/sda", "auto", "ATA"},
		{"/dev/sdb", "usbsunplus", "SCSI"},
		{"/dev/sdc", "sat", "ATA+SCSI"},
	}
	for i, w := range want {
		if devs[i].Name != w.name || devs[i].Type != w.typ || devs[i].Protocol != w.proto {
			t.Errorf("device %d = %+v, want {%s %s %s}", i, devs[i], w.name, w.typ, w.proto)
		}
	}
	if devs[3].OpenError == "" || !strings.Contains(devs[3].OpenError, "Permission denied") {
		t.Errorf("device 3: expected permission-denied open_error, got %q", devs[3].OpenError)
	}

	if got := parseSmartctlScan([]byte("this is not json")); got != nil {
		t.Errorf("garbage input: expected nil, got %+v", got)
	}
	if got := parseSmartctlScan([]byte(`{"devices": []}`)); got == nil || len(got) != 0 {
		t.Errorf("empty devices: expected non-nil empty slice, got %v", got)
	}
}

func TestApplySmartctlFullNVMe(t *testing.T) {
	di := newDiskInfo("/dev/nvme0n1", "nvme", "NVMe")
	if !applySmartctlFull(&di, loadFixture(t, "smartctl_nvme.json")) {
		t.Fatal("nvme fixture: applySmartctlFull returned false")
	}
	checks := []struct {
		name      string
		got, want string
	}{
		{"Model", di.Model, "Samsung SSD 970 EVO Plus 1TB"},
		{"Serial", di.Serial, "S6EWNX0R900123"},
		{"SmartStatus", di.SmartStatus, "Verified"},
		{"SmartSource", di.SmartSource, "smartctl"},
		{"Capacity", di.Capacity, "931.5 GB"},
		{"Protocol", di.Protocol, "NVMe"},
	}
	for _, c := range checks {
		if c.got != c.want {
			t.Errorf("%s = %q, want %q", c.name, c.got, c.want)
		}
	}
	nums := []struct {
		name string
		got  int
		want int
	}{
		{"WearUsedPct", di.WearUsedPct, 2},
		{"SparePct", di.SparePct, 99},
		{"CriticalWarning", di.CriticalWarning, 0},
		{"PowerOnHours", di.PowerOnHours, 1234},
		{"PowerCycles", di.PowerCycles, 412},
		{"UnsafeShutdowns", di.UnsafeShutdowns, 17},
		{"MediaErrors", di.MediaErrors, 0},
	}
	for _, c := range nums {
		if c.got != c.want {
			t.Errorf("%s = %d, want %d", c.name, c.got, c.want)
		}
	}
	if di.TemperatureC != 33 {
		t.Errorf("TemperatureC = %v, want 33", di.TemperatureC)
	}
	// 28441021 units * 512000 bytes = 14.56 TB
	if di.DataWrittenTB < 14.5 || di.DataWrittenTB > 14.7 {
		t.Errorf("DataWrittenTB = %v, want ~14.6", di.DataWrittenTB)
	}

	// Failing drive: same log, smart_status.passed = false.
	out := strings.Replace(string(loadFixture(t, "smartctl_nvme.json")),
		`"passed": true`, `"passed": false`, 1)
	di2 := newDiskInfo("/dev/nvme0n1", "nvme", "NVMe")
	if !applySmartctlFull(&di2, []byte(out)) {
		t.Fatal("failing nvme: applySmartctlFull returned false")
	}
	if di2.SmartStatus != "Failing" {
		t.Errorf("SmartStatus = %q, want Failing", di2.SmartStatus)
	}
}

func TestApplySmartctlFullSATA(t *testing.T) {
	di := newDiskInfo("/dev/sda", "auto", "ATA")
	if !applySmartctlFull(&di, loadFixture(t, "smartctl_sata_ssd.json")) {
		t.Fatal("sata fixture: applySmartctlFull returned false")
	}
	if di.Model != "Samsung SSD 860 EVO 1TB" {
		t.Errorf("Model = %q", di.Model)
	}
	if di.SmartStatus != "Verified" {
		t.Errorf("SmartStatus = %q, want Verified", di.SmartStatus)
	}
	if !di.SolidState {
		t.Error("SolidState: rotation_rate 0 should set SolidState")
	}
	if !di.TrimSupport {
		t.Error("TrimSupport: trim.supported true should set TrimSupport")
	}
	if di.Capacity != "931.5 GB" {
		t.Errorf("Capacity = %q, want 931.5 GB", di.Capacity)
	}
	if di.TemperatureC != 31 {
		t.Errorf("TemperatureC = %v, want 31", di.TemperatureC)
	}
	if di.PowerOnHours != 5432 || di.PowerCycles != 87 {
		t.Errorf("PowerOnHours/PowerCycles = %d/%d, want 5432/87", di.PowerOnHours, di.PowerCycles)
	}
	attrs := []struct {
		name string
		got  int
		want int
	}{
		{"ReallocatedSectors", di.ReallocatedSectors, 0},
		{"PendingSectors", di.PendingSectors, 0},
		{"OfflineUncorrectable", di.OfflineUncorrectable, 0},
		{"UDMACRCErrors", di.UDMACRCErrors, 3},
		{"SSDLifeLeft", di.SSDLifeLeft, 98},
	}
	for _, c := range attrs {
		if c.got != c.want {
			t.Errorf("%s = %d, want %d", c.name, c.got, c.want)
		}
	}

	// smartmontools 6.x puts the table under ata_smart_read_attributes.
	out := strings.Replace(string(loadFixture(t, "smartctl_sata_ssd.json")),
		`"ata_smart_attributes"`, `"ata_smart_read_attributes"`, 1)
	di2 := newDiskInfo("/dev/sda", "auto", "ATA")
	if !applySmartctlFull(&di2, []byte(out)) {
		t.Fatal("6.x table: applySmartctlFull returned false")
	}
	if di2.UDMACRCErrors != 3 || di2.SSDLifeLeft != 98 {
		t.Errorf("6.x table: UDMA/SSDLife = %d/%d, want 3/98", di2.UDMACRCErrors, di2.SSDLifeLeft)
	}
}

func TestApplySmartctlFullUSBSAT(t *testing.T) {
	di := newDiskInfo("/dev/sdb", "usbsunplus", "SCSI")
	if !applySmartctlFull(&di, loadFixture(t, "smartctl_usb_sat.json")) {
		t.Fatal("usb fixture: applySmartctlFull returned false")
	}
	if di.Model != "Crucial P3 2TB" {
		t.Errorf("Model = %q, want scsi_model_name fallback", di.Model)
	}
	if di.SmartStatus != "Verified" {
		t.Errorf("SmartStatus = %q, want Verified", di.SmartStatus)
	}
	if !di.SolidState {
		t.Error("SolidState: rotation_rate 0 should set SolidState")
	}
	if di.Capacity != "1.8 TB" {
		t.Errorf("Capacity = %q, want 1.8 TB", di.Capacity)
	}
	if di.TemperatureC != 36 {
		t.Errorf("TemperatureC = %v, want 36", di.TemperatureC)
	}
	if di.PowerOnHours != 321 || di.PowerCycles != 14 {
		t.Errorf("PowerOnHours/PowerCycles = %d/%d, want 321/14", di.PowerOnHours, di.PowerCycles)
	}
	// ATA attributes are still readable through the USB-SAT bridge.
	if di.ReallocatedSectors != 0 {
		t.Errorf("ReallocatedSectors = %d, want 0", di.ReallocatedSectors)
	}
	// SCSI counters: zero errors in the fixture.
	if di.SASGrownDefects != 0 || di.SASUncorrectedErrors != 0 {
		t.Errorf("SAS counters = %d/%d, want 0/0", di.SASGrownDefects, di.SASUncorrectedErrors)
	}
}

func TestApplySmartctlFullPermissionDenied(t *testing.T) {
	out := loadFixture(t, "smartctl_permission.json")
	di := newDiskInfo("/dev/sdc", "sat", "ATA+SCSI")
	if applySmartctlFull(&di, out) {
		t.Fatal("permission fixture: applySmartctlFull should return false (no device data)")
	}
	note := smartctlErrorNote(nil, out)
	if !strings.Contains(note, "permission denied") {
		t.Errorf("note = %q, want permission-denied message", note)
	}
}

func TestApplySmartctlFullGarbage(t *testing.T) {
	di := newDiskInfo("/dev/sdx", "auto", "ATA")
	if applySmartctlFull(&di, []byte("")) {
		t.Error("empty output: applySmartctlFull should return false")
	}
	if applySmartctlFull(&di, []byte("not json at all")) {
		t.Error("garbage output: applySmartctlFull should return false")
	}
}

func TestParseLsblkJSON(t *testing.T) {
	out := []byte(`{
		"blockdevices": [
			{
				"name": "nvme0n1", "size": 1000204886016, "rota": 0,
				"tran": "", "model": "Samsung SSD 970 EVO Plus", "removable": 0,
				"type": "disk",
				"children": [ {"name": "nvme0n1p1"}, {"name": "nvme0n1p2"} ]
			},
			{
				"name": "sdb", "size": 2000398934016, "rota": 0,
				"tran": "usb", "model": "Crucial P3", "removable": 1,
				"type": "disk",
				"children": [ {"name": "sdb1"} ]
			},
			{
				"name": "sdb1", "size": 2000388737920, "rota": 0,
				"tran": "usb", "model": "", "removable": 1,
				"type": "part", "mountpoint": "/mnt/usb"
			}
		]
	}`)
	m := parseLsblkJSON(out)
	if len(m) != 2 {
		t.Fatalf("expected 2 whole disks, got %d: %+v", len(m), m)
	}
	nvme := m["nvme0n1"]
	if nvme.Size != 1000204886016 || nvme.Rota != 0 {
		t.Errorf("nvme0n1 = %+v", nvme)
	}
	if len(nvme.ChildNames) != 2 || nvme.ChildNames[1] != "nvme0n1p2" {
		t.Errorf("nvme0n1 children = %v", nvme.ChildNames)
	}
	usb := m["sdb"]
	if usb.Transport != "usb" {
		t.Errorf("sdb tran = %q, want usb", usb.Transport)
	}
}

func TestParseDfOutput(t *testing.T) {
	out := []byte("Filesystem     1K-blocks      Used Available Use% Mounted on\n" +
		"/dev/nvme0n1p2 9315123456 1234567890 7480566670  15% /data\n")
	dev, free := parseDfOutput(out)
	if dev != "/dev/nvme0n1p2" {
		t.Errorf("dev = %q, want /dev/nvme0n1p2", dev)
	}
	// 7480566670 KiB rendered in the standard 1024-base format.
	wantGB := humanBytes(7480566670 * 1024)
	if free != wantGB {
		t.Errorf("free = %q, want %q", free, wantGB)
	}
	if dev, _ := parseDfOutput([]byte("Filesystem\n")); dev != "" {
		t.Errorf("header-only df: dev = %q, want empty", dev)
	}
}

func TestStripPartitionSuffix(t *testing.T) {
	cases := []struct{ in, want string }{
		{"/dev/nvme0n1p2", "/dev/nvme0n1"},
		{"/dev/nvme1n1p3", "/dev/nvme1n1"},
		{"/dev/sda1", "/dev/sda"},
		{"/dev/sdb10", "/dev/sdb"},
		{"/dev/mmcblk0p1", "/dev/mmcblk0"},
		{"/dev/nvme0n1", "/dev/nvme0n1"}, // whole NVMe disk: unchanged
		{"/dev/sda", "/dev/sda"},         // whole disk: unchanged
		{"/dev/sr0", "/dev/sr0"},         // no trailing digit: unchanged
	}
	for _, c := range cases {
		if got := stripPartitionSuffix(c.in); got != c.want {
			t.Errorf("stripPartitionSuffix(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestDataDevMatches(t *testing.T) {
	blk := map[string]lsblkDisk{
		"nvme0n1": {Name: "nvme0n1", ChildNames: []string{"nvme0n1p1", "nvme0n1p2"}},
		"sdb":     {Name: "sdb", ChildNames: []string{"sdb1"}},
	}
	cases := []struct {
		whole, dev string
		blk        map[string]lsblkDisk
		want       bool
	}{
		{"/dev/nvme0n1", "/dev/nvme0n1", blk, true},    // identical
		{"/dev/nvme0n1", "/dev/nvme0n1p2", blk, true},  // lsblk child
		{"/dev/nvme0n1", "/dev/nvme0n1p9", blk, false}, // not a child
		{"/dev/sdb", "/dev/sdb1", blk, true},           // lsblk child
		{"/dev/sdb", "/dev/sdc1", blk, false},          // different disk
		{"/dev/nvme0n1", "/dev/nvme0n1p2", nil, true},  // lsblk missing: suffix strip
		{"/dev/nvme0n1", "/dev/nvme1n1p2", nil, false}, // suffix strip mismatch
		{"/dev/sda", "/dev/sdb1", nil, false},
	}
	for i, c := range cases {
		if got := dataDevMatches(c.whole, c.dev, c.blk); got != c.want {
			t.Errorf("case %d: dataDevMatches(%q, %q) = %v, want %v", i, c.whole, c.dev, got, c.want)
		}
	}
}

func TestEnrichFromLsblk(t *testing.T) {
	blk := map[string]lsblkDisk{
		"sdb": {Name: "sdb", Size: 2000398934016, Rota: 0, Transport: "usb", Model: "Crucial P3"},
		"sda": {Name: "sda", Size: 500107862016, Rota: 1, Model: "WDC WD5000AAKX"},
	}
	di := DiskInfo{Device: "/dev/sdb"}
	enrichFromLsblk(&di, blk)
	if !di.IsExternal {
		t.Error("usb transport should set IsExternal")
	}
	if di.Capacity != "1.8 TB" {
		t.Errorf("Capacity = %q, want 1.8 TB", di.Capacity)
	}
	if di.Model != "Crucial P3" {
		t.Errorf("Model = %q, want Crucial P3", di.Model)
	}
	if !di.SolidState {
		t.Error("rota 0 should set SolidState")
	}

	// smartctl-provided values must not be overwritten.
	di2 := DiskInfo{Device: "/dev/sda", Model: "WDC WD5000AAKX-22FF0B0", SolidState: true}
	enrichFromLsblk(&di2, blk)
	if di2.Model != "WDC WD5000AAKX-22FF0B0" {
		t.Errorf("Model overwritten: %q", di2.Model)
	}
	if di2.IsExternal {
		t.Error("sata disk must not be external")
	}
	if !di2.SolidState {
		t.Error("SolidState should survive (smartctl said solid)")
	}

	// Empty lsblk map: nothing changes.
	di3 := DiskInfo{Device: "/dev/sdz"}
	enrichFromLsblk(&di3, nil)
	if di3.Capacity != "" || di3.IsExternal {
		t.Errorf("nil lsblk should not enrich: %+v", di3)
	}
}

func TestHumanBytes(t *testing.T) {
	cases := []struct {
		n    uint64
		want string
	}{
		{0, "0 B"},
		{512, "512 B"},
		{1024, "1.0 KB"},
		{1000204886016, "931.5 GB"},
		{2000398934016, "1.8 TB"},
	}
	for _, c := range cases {
		if got := humanBytes(c.n); got != c.want {
			t.Errorf("humanBytes(%d) = %q, want %q", c.n, got, c.want)
		}
	}
}

func TestSmartctlErrorNote(t *testing.T) {
	if note := smartctlErrorNote(nil, []byte("Smartctl open device: /dev/sdb failed: Permission denied\n")); !strings.Contains(note, "permission denied") {
		t.Errorf("stderr permission: note = %q", note)
	}
	if note := smartctlErrorNote(nil, []byte("cannot open disk: Operation not permitted")); !strings.Contains(note, "permission denied") {
		t.Errorf("eperm: note = %q", note)
	}
	if note := smartctlErrorNote(nil, []byte("some other weird failure")); !strings.Contains(note, "SMART unreadable") {
		t.Errorf("generic: note = %q", note)
	}
}
