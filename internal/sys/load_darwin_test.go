//go:build darwin

package sys

import "testing"

func TestHostLoadAverageDarwin(t *testing.T) {
	la := HostLoadAverage()
	if !la.Available {
		t.Fatal("expected load average on darwin")
	}
	if la.Load1 < 0 || la.Load5 < 0 || la.Load15 < 0 {
		t.Fatalf("negative load: %+v", la)
	}
}
