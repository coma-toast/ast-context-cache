//go:build darwin

package sys

import (
	"encoding/json"
	"strings"
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

func TestApplySmartWear(t *testing.T) {
	h := SSDHealth{}
	applySmartWear(&h, map[string]uint64{
		"PERCENTAGE_USED":      12,
		"AVAILABLE_SPARE":      98,
		"DATA_UNITS_WRITTEN_0": 1_000_000,
		"DATA_UNITS_WRITTEN_1": 0,
		"TEMPERATURE":          325,
	})
	if h.WearUsedPct != 12 || h.SparePct != 98 {
		t.Fatalf("wear=%d spare=%d", h.WearUsedPct, h.SparePct)
	}
	if h.DataWrittenTB <= 0 {
		t.Fatalf("written tb=%v", h.DataWrittenTB)
	}
	if h.TemperatureC != 32.5 {
		t.Fatalf("temp=%v", h.TemperatureC)
	}
}

func TestDeviceForPathResolvesArbitraryDirectory(t *testing.T) {
	// diskutil info rejects an ordinary subdirectory outright ("Could not find disk")
	// — this is the exact regression that made SSDAvailable silently go false for the
	// default ~/.astcache data directory once probeDiskHealth stopped assuming disk0.
	// t.TempDir() is nested several directories deep, same shape as a real data dir.
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

func TestApplySmartctlJSONNoData(t *testing.T) {
	h := &SSDHealth{WearUsedPct: -1, SparePct: -1, DataWrittenTB: -1, TemperatureC: -1}
	if applySmartctlJSON(h, nil) {
		t.Fatal("empty output should not report found")
	}
	if applySmartctlJSON(h, []byte("not json")) {
		t.Fatal("invalid JSON should not report found")
	}
	// A response with none of the fields we look for (e.g. a permission-denied stub)
	// must not claim success either.
	if applySmartctlJSON(h, []byte(`{"smartctl":{"exit_status":2}}`)) {
		t.Fatal("response with no usable fields should not report found")
	}
}

func TestApplySmartctlJSONFillsOnlyUnknownFields(t *testing.T) {
	nvmeJSON := []byte(`{
		"smart_status": {"passed": true},
		"temperature": {"current": 42},
		"model_name": "Samsung SSD 980 PRO",
		"nvme_smart_health_information_log": {
			"percentage_used": 3,
			"available_spare": 100,
			"data_units_written": 2000000
		}
	}`)

	// Nothing known yet: smartctl fills everything.
	h := &SSDHealth{WearUsedPct: -1, SparePct: -1, DataWrittenTB: -1, TemperatureC: -1}
	if !applySmartctlJSON(h, nvmeJSON) {
		t.Fatal("expected applySmartctlJSON to report found")
	}
	if h.SmartStatus != "Verified" || h.SmartSource != "smartctl" {
		t.Fatalf("smart status=%q source=%q", h.SmartStatus, h.SmartSource)
	}
	if h.Model != "Samsung SSD 980 PRO" {
		t.Fatalf("model=%q", h.Model)
	}
	if h.TemperatureC != 42 || h.WearUsedPct != 3 || h.SparePct != 100 {
		t.Fatalf("temp=%v wear=%d spare=%d", h.TemperatureC, h.WearUsedPct, h.SparePct)
	}
	if h.DataWrittenTB <= 0 {
		t.Fatalf("data written tb=%v", h.DataWrittenTB)
	}

	// diskutil already had a verdict and a model: smartctl must not override them,
	// even though it also has an opinion — only genuinely unknown fields get filled.
	h2 := &SSDHealth{SmartStatus: "Verified", Model: "Existing Model", WearUsedPct: -1, SparePct: -1, DataWrittenTB: -1, TemperatureC: 55}
	applySmartctlJSON(h2, nvmeJSON)
	if h2.SmartStatus != "Verified" || h2.SmartSource != "" {
		t.Fatalf("existing smart status/source should be preserved: status=%q source=%q", h2.SmartStatus, h2.SmartSource)
	}
	if h2.Model != "Existing Model" {
		t.Fatalf("existing model should be preserved: %q", h2.Model)
	}
	if h2.TemperatureC != 55 {
		t.Fatalf("existing temperature should be preserved: %v", h2.TemperatureC)
	}
	// Wear/spare/written were still unknown (-1), so those should still be filled.
	if h2.WearUsedPct != 3 || h2.SparePct != 100 {
		t.Fatalf("wear=%d spare=%d should have been filled", h2.WearUsedPct, h2.SparePct)
	}
}

func TestApplySmartctlNoDiskReturnsFalse(t *testing.T) {
	h := &SSDHealth{WearUsedPct: -1, SparePct: -1, DataWrittenTB: -1, TemperatureC: -1}
	if applySmartctl(h, "") {
		t.Fatal("empty wholeDisk should never call out to smartctl")
	}
}

func TestSmartUint(t *testing.T) {
	if n, ok := smartUint(float64(7)); !ok || n != 7 {
		t.Fatalf("float64: %d %v", n, ok)
	}
	if n, ok := smartUint(json.Number("42")); !ok || n != 42 {
		t.Fatalf("json.Number: %d %v", n, ok)
	}
}
