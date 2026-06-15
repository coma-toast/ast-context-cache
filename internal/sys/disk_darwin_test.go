//go:build darwin

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
		"PERCENTAGE_USED":       12,
		"AVAILABLE_SPARE":       98,
		"DATA_UNITS_WRITTEN_0":  1_000_000,
		"DATA_UNITS_WRITTEN_1": 0,
		"TEMPERATURE":           325,
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

func TestSmartUint(t *testing.T) {
	if n, ok := smartUint(float64(7)); !ok || n != 7 {
		t.Fatalf("float64: %d %v", n, ok)
	}
	if n, ok := smartUint(json.Number("42")); !ok || n != 42 {
		t.Fatalf("json.Number: %d %v", n, ok)
	}
}
