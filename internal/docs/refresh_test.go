package docs

import "testing"

func TestIsRefreshing(t *testing.T) {
	// Add to (not replace) the live map and remove the fake id afterwards, so refreshes
	// other tests started stay tracked and the map drains (see waitDocRefreshIdle).
	refreshMu.Lock()
	refreshing[42] = struct{}{}
	refreshMu.Unlock()
	t.Cleanup(func() {
		refreshMu.Lock()
		delete(refreshing, 42)
		refreshMu.Unlock()
	})
	if !IsRefreshing(42) {
		t.Fatal("expected id 42 refreshing")
	}
	if IsRefreshing(99) {
		t.Fatal("expected id 99 not refreshing")
	}
}

func TestTryQuietRefreshSkipsWhenNoStale(t *testing.T) {
	ResetQuietRefreshForTest()
	// With empty/no DB sources ListSources may error or return empty — should not panic.
	TryQuietRefresh("test")
}
