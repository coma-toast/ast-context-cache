package projectmeta

import "testing"

func TestDisplayNameOverride(t *testing.T) {
	SetDisplayNameOverrideFunc(func(path string) string {
		if path == "/tmp/demo" {
			return "Custom Demo"
		}
		return ""
	})
	t.Cleanup(func() { SetDisplayNameOverrideFunc(nil) })

	Invalidate("/tmp/demo")
	info := Enrich("/tmp/demo")
	if info.Label != "Custom Demo" {
		t.Fatalf("label=%q want Custom Demo", info.Label)
	}
}
