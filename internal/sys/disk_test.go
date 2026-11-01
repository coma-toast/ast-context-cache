package sys

import (
	"encoding/json"
	"testing"
)

func TestParseDiskutilKV(t *testing.T) {
	text := `   Device / Media Name:       APPLE SSD AP0512Z
    Protocol:                  Apple Fabric
    SMART Status:              Verified
    Solid State:               Yes`
	kv := parseDiskutilKV(text)
	if kv["device / media name"] != "APPLE SSD AP0512Z" {
		t.Fatalf("model: %q", kv["device / media name"])
	}
	if kv["smart status"] != "Verified" {
		t.Fatalf("smart: %q", kv["smart status"])
	}
	if kv["solid state"] != "Yes" {
		t.Fatalf("ssd: %q", kv["solid state"])
	}
}

func TestParseDiskutilList(t *testing.T) {
	// Real `diskutil list` output: APFS container header + partition rows + external MBR disk.
	text := `Disk:                    disk0 (APFS)
Size:                    494.4 GB
Partition map schema:   GPT
+:----------:------------------------------------:---------:-------------------------------+
| #          | NAME                               | TYPE    | SIZE                          |
+============+====================================+=========+===============================+
| 0          | Apple_APFS_ISC                     | ISO_CDR | 524.3 MB                        |
| 1          | disk0s2                            | APFS    | 494.4 GB                        |
| 2          | disk0s2s1                          | APFS    | 10.2 GB                         |
+----------:------------------------------------:---------:-------------------------------+
/dev/disk1 (external, physical):
   #:                       TYPE NAME                    SIZE       IDENTIFIER
   0:      FD_partition         *                    500.1 GB   disk1
   1:      FD_partition         *                    512 B      disk1s1
   2:      FD_partition         *                    500.1 GB   disk1s2
`
	got := parseDiskutilList(text)
	want := []string{"disk0", "disk1"}
	if len(got) != len(want) {
		t.Fatalf("got %d disks %v, want %d %v", len(got), got, len(want), want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("disk[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestParseDiskutilListEmpty(t *testing.T) {
	if got := parseDiskutilList(""); len(got) != 0 {
		t.Fatalf("empty input should give 0 disks, got %v", got)
	}
	if got := parseDiskutilList("no disks here"); len(got) != 0 {
		t.Fatalf("no disk rows should give 0 disks, got %v", got)
	}
}

func TestHumanBeforeParen(t *testing.T) {
	cases := map[string]string{
		"263.1 MB (263090176 Bytes) (exactly 513848 512-Byte-Units)": "263.1 MB",
		"500.28 GB":       "500.28 GB",
		"":                "",
		"foo (bar) (baz)": "foo",
	}
	for in, want := range cases {
		if got := humanBeforeParen(in); got != want {
			t.Fatalf("humanBeforeParen(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestSmartUint(t *testing.T) {
	if n, ok := smartUint(float64(7)); !ok || n != 7 {
		t.Fatalf("float64: %d %v", n, ok)
	}
	if n, ok := smartUint(json.Number("42")); !ok || n != 42 {
		t.Fatalf("json.Number: %d %v", n, ok)
	}
	if _, ok := smartUint(float64(-1)); ok {
		t.Fatal("negative float should not convert")
	}
	if _, ok := smartUint("not a number"); ok {
		t.Fatal("string should not convert")
	}
}

func TestApplySmartWear(t *testing.T) {
	d := DiskInfo{}
	applySmartWear(&d, map[string]uint64{
		"PERCENTAGE_USED":      12,
		"AVAILABLE_SPARE":      98,
		"DATA_UNITS_WRITTEN_0": 1_000_000,
		"DATA_UNITS_WRITTEN_1": 0,
		"TEMPERATURE":          325,
	})
	if d.WearUsedPct == nil || *d.WearUsedPct != 12 {
		t.Fatalf("wear=%v", d.WearUsedPct)
	}
	if d.SparePct == nil || *d.SparePct != 98 {
		t.Fatalf("spare=%v", d.SparePct)
	}
	if d.DataWrittenTB == nil || *d.DataWrittenTB <= 0 {
		t.Fatalf("written tb=%v", d.DataWrittenTB)
	}
	if d.TemperatureC == nil || *d.TemperatureC != 32.5 {
		t.Fatalf("temp=%v", d.TemperatureC)
	}
}

func TestApplySmartWearDoesNotOverwrite(t *testing.T) {
	d := DiskInfo{}
	d.WearUsedPct = intPtr(5)
	applySmartWear(&d, map[string]uint64{"PERCENTAGE_USED": 12})
	if *d.WearUsedPct != 5 {
		t.Fatalf("wear should not be overwritten: %d", *d.WearUsedPct)
	}
}

func TestApplySmartctlJSONNoData(t *testing.T) {
	d := &DiskInfo{}
	if applySmartctlJSON(d, nil) {
		t.Fatal("empty output should not report found")
	}
	if applySmartctlJSON(d, []byte("not json")) {
		t.Fatal("invalid JSON should not report found")
	}
	if applySmartctlJSON(d, []byte(`{"smartctl":{"exit_status":2}}`)) {
		t.Fatal("response with no usable fields should not report found")
	}
}

func TestApplySmartctlJSONFillsOnlyUnknownFields(t *testing.T) {
	nvmeJSON := []byte(`{
		"smart_status": {"passed": true},
		"temperature": {"current": 42},
		"model_name": "Samsung SSD 980 PRO",
		"serial_number": "S5EWNX0R",
		"power_on_time": {"hours": 1234},
		"power_cycle_count": 56,
		"nvme_smart_health_information_log": {
			"percentage_used": 3,
			"available_spare": 100,
			"data_units_written": 2000000,
			"critical_warning": 0,
			"media_errors": 2,
			"unsafe_shutdowns": 7
		}
	}`)

	// Nothing known yet: smartctl fills everything.
	d := &DiskInfo{}
	if !applySmartctlJSON(d, nvmeJSON) {
		t.Fatal("expected applySmartctlJSON to report found")
	}
	if d.SmartStatus != "Verified" || d.SmartSource != "smartctl" {
		t.Fatalf("smart status=%q source=%q", d.SmartStatus, d.SmartSource)
	}
	if d.Model != "Samsung SSD 980 PRO" {
		t.Fatalf("model=%q", d.Model)
	}
	if d.Serial != "S5EWNX0R" {
		t.Fatalf("serial=%q", d.Serial)
	}
	if d.TemperatureC == nil || *d.TemperatureC != 42 {
		t.Fatalf("temp=%v", d.TemperatureC)
	}
	if d.WearUsedPct == nil || *d.WearUsedPct != 3 {
		t.Fatalf("wear=%v", d.WearUsedPct)
	}
	if d.SparePct == nil || *d.SparePct != 100 {
		t.Fatalf("spare=%v", d.SparePct)
	}
	if d.DataWrittenTB == nil || *d.DataWrittenTB <= 0 {
		t.Fatalf("data written tb=%v", d.DataWrittenTB)
	}
	if d.PowerOnHours == nil || *d.PowerOnHours != 1234 {
		t.Fatalf("power on hours=%v", d.PowerOnHours)
	}
	if d.PowerCycles == nil || *d.PowerCycles != 56 {
		t.Fatalf("power cycles=%v", d.PowerCycles)
	}
	if d.CriticalWarning == nil || *d.CriticalWarning != 0 {
		t.Fatalf("critical warning=%v", d.CriticalWarning)
	}
	if d.MediaErrors == nil || *d.MediaErrors != 2 {
		t.Fatalf("media errors=%v", d.MediaErrors)
	}
	if d.UnsafeShutdowns == nil || *d.UnsafeShutdowns != 7 {
		t.Fatalf("unsafe shutdowns=%v", d.UnsafeShutdowns)
	}

	// diskutil already had a verdict and a model: smartctl must not override them.
	d2 := &DiskInfo{SmartStatus: "Verified", Model: "Existing Model", TemperatureC: float64Ptr(55)}
	applySmartctlJSON(d2, nvmeJSON)
	if d2.SmartStatus != "Verified" || d2.SmartSource != "" {
		t.Fatalf("existing smart status/source should be preserved: status=%q source=%q", d2.SmartStatus, d2.SmartSource)
	}
	if d2.Model != "Existing Model" {
		t.Fatalf("existing model should be preserved: %q", d2.Model)
	}
	if *d2.TemperatureC != 55 {
		t.Fatalf("existing temperature should be preserved: %v", *d2.TemperatureC)
	}
	if d2.WearUsedPct == nil || *d2.WearUsedPct != 3 {
		t.Fatalf("wear should have been filled: %v", d2.WearUsedPct)
	}
}

func TestApplySmartctlJSONSATAAttributes(t *testing.T) {
	sataJSON := []byte(`{
		"smart_status": {"passed": true},
		"model_name": "WD Blue 1TB",
		"smart_attributes": {
			"table": [
				{"id": 5, "name": "Reallocated_Sector_Ct", "raw": {"value": 3}},
				{"id": 197, "name": "Current_Pending_Sector", "raw": {"value": 0}},
				{"id": 198, "name": "Offline_Uncorrectable", "raw": {"value": 1}},
				{"id": 199, "name": "UDMA_CRC_Error_Count", "raw": {"value": 0}},
				{"id": 233, "name": "Percentage_Used", "raw": {"value": 45}}
			]
		}
	}`)
	d := &DiskInfo{}
	applySmartctlJSON(d, sataJSON)
	if d.ReallocatedSectors == nil || *d.ReallocatedSectors != 3 {
		t.Fatalf("realloc=%v", d.ReallocatedSectors)
	}
	if d.PendingSectors == nil || *d.PendingSectors != 0 {
		t.Fatalf("pending=%v", d.PendingSectors)
	}
	if d.OfflineUncorrectable == nil || *d.OfflineUncorrectable != 1 {
		t.Fatalf("offline=%v", d.OfflineUncorrectable)
	}
	if d.UDMACRCErrors == nil || *d.UDMACRCErrors != 0 {
		t.Fatalf("udma=%v", d.UDMACRCErrors)
	}
	if d.SSDLifeLeft == nil || *d.SSDLifeLeft != 45 {
		t.Fatalf("life left=%v", d.SSDLifeLeft)
	}
}

func TestApplySmartctlJSONSASAttributes(t *testing.T) {
	sasJSON := []byte(`{
		"smart_status": {"passed": true},
		"model_name": "SEAGATE Exos 10E1000",
		"smart_attributes": {
			"table": [
				{"id": 201, "name": "SAS_Grown_Defect", "raw": {"value": 2}},
				{"id": 253, "name": "SAS_Uncorrected", "raw": {"value": 0}}
			]
		}
	}`)
	d := &DiskInfo{}
	applySmartctlJSON(d, sasJSON)
	if d.SASGrownDefects == nil || *d.SASGrownDefects != 2 {
		t.Fatalf("grown defects=%v", d.SASGrownDefects)
	}
	if d.SASUncorrectedErrors == nil || *d.SASUncorrectedErrors != 0 {
		t.Fatalf("uncorrected=%v", d.SASUncorrectedErrors)
	}
}

func TestSSDHealthFromDisk(t *testing.T) {
	d := DiskInfo{
		Available:     true,
		Device:        "disk0",
		Model:         "APPLE SSD",
		SmartStatus:   "Verified",
		SmartSource:   "diskutil",
		Protocol:      "Apple Fabric",
		Capacity:      "494 GB",
		FreeSpace:     "250 GB",
		IsExternal:    false,
		TrimSupport:   true,
		WearUsedPct:   intPtr(5),
		SparePct:      intPtr(95),
		DataWrittenTB: float64Ptr(1.2),
		TemperatureC:  float64Ptr(42.5),
	}
	ss := SSDHealthFromDisk(d)
	if !ss.Available || ss.Device != "disk0" || ss.Model != "APPLE SSD" {
		t.Fatalf("basic: %+v", ss)
	}
	if ss.WearUsedPct != 5 || ss.SparePct != 95 {
		t.Fatalf("wear=%d spare=%d", ss.WearUsedPct, ss.SparePct)
	}
	if ss.DataWrittenTB != 1.2 || ss.TemperatureC != 42.5 {
		t.Fatalf("tb=%v temp=%v", ss.DataWrittenTB, ss.TemperatureC)
	}

	// Nil pointers become -1.
	d2 := DiskInfo{Available: true, Device: "disk1"}
	ss2 := SSDHealthFromDisk(d2)
	if ss2.WearUsedPct != -1 || ss2.SparePct != -1 {
		t.Fatalf("nil int ptrs should be -1: wear=%d spare=%d", ss2.WearUsedPct, ss2.SparePct)
	}
	if ss2.DataWrittenTB != -1 || ss2.TemperatureC != -1 {
		t.Fatalf("nil float ptrs should be -1: tb=%v temp=%v", ss2.DataWrittenTB, ss2.TemperatureC)
	}
}

func TestApplyDiskWarnings(t *testing.T) {
	// No warnings.
	d := &DiskInfo{SmartStatus: "Verified"}
	applyDiskWarnings(d)
	if d.WarningLevel != "" || d.WarningMsg != "" {
		t.Fatalf("clean disk should have no warning: %q %q", d.WarningLevel, d.WarningMsg)
	}

	// SMART failing → critical.
	d2 := &DiskInfo{SmartStatus: "Failing"}
	applyDiskWarnings(d2)
	if d2.WarningLevel != "critical" {
		t.Fatalf("failing SMART should be critical: %q", d2.WarningLevel)
	}

	// Pending sectors → warning.
	d3 := &DiskInfo{SmartStatus: "Verified", PendingSectors: intPtr(4)}
	applyDiskWarnings(d3)
	if d3.WarningLevel != "warning" {
		t.Fatalf("pending sectors should be warning: %q", d3.WarningLevel)
	}
	if d3.WarningMsg == "" {
		t.Fatal("warning message should be non-empty")
	}

	// Critical warning + media errors → critical with both.
	d4 := &DiskInfo{CriticalWarning: intPtr(1), MediaErrors: intPtr(3)}
	applyDiskWarnings(d4)
	if d4.WarningLevel != "critical" {
		t.Fatalf("critical warning should be critical: %q", d4.WarningLevel)
	}
}

func intPtr(v int) *int { return &v }

func float64Ptr(v float64) *float64 { return &v }
