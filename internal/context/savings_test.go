package context

import (
	"encoding/json"
	"testing"
)

func TestCacheHasSavingsMeta(t *testing.T) {
	if CacheHasSavingsMeta(map[string]interface{}{"results": []any{}}) {
		t.Fatal("results alone should not count")
	}
	if !CacheHasSavingsMeta(map[string]interface{}{"tokens_saved": float64(1)}) {
		t.Fatal("tokens_saved should count")
	}
}

func TestCoerceInt(t *testing.T) {
	if coerceInt(float64(42)) != 42 {
		t.Fatal("float64")
	}
	if coerceInt(json.Number("7")) != 7 {
		t.Fatal("json.Number")
	}
}

func TestComputeSavings(t *testing.T) {
	m := ComputeSavings(100, 500, 800, 50)
	if m.TokensSaved != 450 {
		t.Fatalf("TokensSaved=%d want 450", m.TokensSaved)
	}
	if m.SavingsVsFiles != 750 {
		t.Fatalf("SavingsVsFiles=%d want 750", m.SavingsVsFiles)
	}
	m2 := ComputeSavings(600, 500, 800, 0)
	if m2.TokensSaved != 0 {
		t.Fatalf("negative mode savings should clamp: got %d", m2.TokensSaved)
	}
}

func TestParseSavingsMeta(t *testing.T) {
	parsed := map[string]interface{}{
		"tokens_used":            float64(120),
		"symbol_baseline_tokens": float64(400),
		"file_baseline_tokens":   float64(900),
		"dedup_tokens_saved":     float64(30),
		"deduped":                float64(2),
	}
	m := ParseSavingsMeta(parsed, "auto", false)
	if m.TokensUsed != 120 || m.SymbolBaseline != 400 || m.DedupedCount != 2 {
		t.Fatalf("parse: %+v", m)
	}
	if m.TokensSaved != 310 {
		t.Fatalf("TokensSaved=%d want 310", m.TokensSaved)
	}
}

func TestApplyToRoundTrip(t *testing.T) {
	m := ComputeSavings(80, 300, 600, 20)
	m.Mode = "skeleton"
	resp := map[string]interface{}{"mode": "skeleton"}
	m.ApplyTo(resp)
	b, _ := json.Marshal(resp)
	var back map[string]interface{}
	json.Unmarshal(b, &back)
	parsed := ParseSavingsMeta(back, "skeleton", false)
	if parsed.TokensSaved != m.TokensSaved {
		t.Fatalf("round trip TokensSaved=%d want %d", parsed.TokensSaved, m.TokensSaved)
	}
}
