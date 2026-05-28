package context

import "testing"

func TestSymbolDedupKey(t *testing.T) {
	k := SymbolDedupKey("/proj/a.go", "Foo", 10)
	if k != "/proj/a.go|Foo|10" {
		t.Fatalf("key = %q", k)
	}
}
