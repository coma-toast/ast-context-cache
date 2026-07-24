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
