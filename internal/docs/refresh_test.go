package docs

import "testing"

func TestIsRefreshing(t *testing.T) {
	refreshMu.Lock()
	refreshing = map[int]struct{}{}
	refreshing[42] = struct{}{}
	refreshMu.Unlock()
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
