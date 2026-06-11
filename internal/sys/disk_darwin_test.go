//go:build darwin

package sys

import "testing"

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
