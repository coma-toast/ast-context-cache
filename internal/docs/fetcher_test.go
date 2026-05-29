package docs

import (
	"testing"
	"time"
)

func TestSourceNeedsRefresh(t *testing.T) {
	if !SourceNeedsRefresh("") {
		t.Fatal("empty last_updated should refresh")
	}
	if !SourceNeedsRefresh("not-a-date") {
		t.Fatal("bad timestamp should refresh")
	}
	fresh := time.Now().Add(-24 * time.Hour).Format(time.RFC3339)
	if SourceNeedsRefresh(fresh) {
		t.Fatal("1 day old should not refresh")
	}
	stale := time.Now().Add(-8 * 24 * time.Hour).Format(time.RFC3339)
	if !SourceNeedsRefresh(stale) {
		t.Fatal("8 days old should refresh")
	}
}
