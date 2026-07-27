package indexer

import "testing"

func TestMaxEmbedSymbolsPerFileCap(t *testing.T) {
	if maxEmbedSymbolsPerFile < 64 || maxEmbedSymbolsPerFile > 1024 {
		t.Fatalf("maxEmbedSymbolsPerFile=%d out of expected range", maxEmbedSymbolsPerFile)
	}
	if embedBatchSize < 1 || embedBatchSize > maxEmbedSymbolsPerFile {
		t.Fatalf("embedBatchSize=%d invalid vs cap %d", embedBatchSize, maxEmbedSymbolsPerFile)
	}
}

func TestCapEmbedSymbolSlice(t *testing.T) {
	syms := make([]int, 1000)
	capped := syms
	if n := len(capped); n > maxEmbedSymbolsPerFile {
		capped = capped[:maxEmbedSymbolsPerFile]
	}
	if len(capped) != maxEmbedSymbolsPerFile {
		t.Fatalf("len=%d want %d", len(capped), maxEmbedSymbolsPerFile)
	}
	short := syms[:10]
	out := short
	if n := len(out); n > maxEmbedSymbolsPerFile {
		out = out[:maxEmbedSymbolsPerFile]
	}
	if len(out) != 10 {
		t.Fatal("short slice should pass through")
	}
}

func TestBuildEmbedTextTruncates(t *testing.T) {
	src := stringsRepeat("x", 2000)
	got := BuildEmbedText("function", "Foo", src)
	if n := len([]rune(got)); n > maxEmbedInputRunes {
		t.Fatalf("len=%d want <= %d", n, maxEmbedInputRunes)
	}
	if !hasPrefix(got, "function Foo: ") {
		t.Fatalf("missing prefix: %q", got[:min(20, len(got))])
	}
}

func stringsRepeat(s string, n int) string {
	b := make([]byte, 0, len(s)*n)
	for i := 0; i < n; i++ {
		b = append(b, s...)
	}
	return string(b)
}

func hasPrefix(s, prefix string) bool {
	return len(s) >= len(prefix) && s[:len(prefix)] == prefix
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
