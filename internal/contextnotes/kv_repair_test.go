package contextnotes

import (
	"strings"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func TestStoreKvRepairMetadata(t *testing.T) {
	testNotesDB(t)
	meta := map[string]interface{}{
		"model_id":     "llama-3.1-8b",
		"kv_quant":     "q4_0",
		"token_count":  4096,
		"trigger_hint": RepairProactive,
	}
	res, err := Store("kv-sess", "golden prompt archive text", "repair archive", "/tmp/p", TagKvRepair, KindKvRepair, meta, nil)
	if err != nil {
		t.Fatal(err)
	}
	n, err := noteByRef(res.Ref)
	if err != nil {
		t.Fatal(err)
	}
	if n.Kind != KindKvRepair {
		t.Fatalf("kind: %q", n.Kind)
	}
	parsed, err := ParseKvRepairMetadata(n)
	if err != nil {
		t.Fatal(err)
	}
	if parsed.ModelID != "llama-3.1-8b" || parsed.KvQuant != "q4_0" || parsed.TokenCount != 4096 {
		t.Fatalf("metadata: %+v", parsed)
	}
}

func TestFetchKvRepairByRef(t *testing.T) {
	testNotesDB(t)
	body := "exact archive bytes\nline two"
	res, err := Store("kv-sess", body, "x", "", TagKvRepair, KindKvRepair, map[string]interface{}{"kv_quant": "q8_0"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	fetch, err := Fetch([]string{res.Ref}, "kv-sess", RepairManual)
	if err != nil {
		t.Fatal(err)
	}
	if len(fetch.Notes) != 1 || fetch.Notes[0].Content != body {
		t.Fatalf("content mismatch: %+v", fetch.Notes)
	}
	if fetch.Stats["repair_reason"] != RepairManual {
		t.Fatalf("stats: %+v", fetch.Stats)
	}
}

func TestSearchKvRepair(t *testing.T) {
	testNotesDB(t)
	_, err := Store("kv-sess", "needle model llama q4 archive", "llama q4", "", TagKvRepair, KindKvRepair, map[string]interface{}{"model_id": "llama", "kv_quant": "q4_0"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	search, err := Search("llama q4", "kv-sess", "", 5, nil)
	if err != nil || len(search.Notes) == 0 {
		t.Fatalf("search: notes=%d err=%v", len(search.Notes), err)
	}
	if !IsKvRepairNote(search.Notes[0]) {
		t.Fatalf("expected kv_repair note")
	}
}

func TestKvRepairSessionIsolation(t *testing.T) {
	testNotesDB(t)
	res, err := Store("owner", "secret archive", "x", "", TagKvRepair, KindKvRepair, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	fetch, err := Fetch([]string{res.Ref}, "other-session", RepairManual)
	if err != nil {
		t.Fatal(err)
	}
	if len(fetch.Notes) != 0 {
		t.Fatalf("expected 0 notes for wrong session")
	}
}

func TestKvRepairQuota(t *testing.T) {
	testNotesDB(t)
	db.SetSetting("context_max_notes_session", "1")
	db.SetSetting("context_limit_policy", "reject")
	_, err := Store("kv-quota", strings.Repeat("a", 40), "first", "", TagKvRepair, KindKvRepair, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	_, err = Store("kv-quota", strings.Repeat("b", 40), "second", "", TagKvRepair, KindKvRepair, nil, nil)
	if err == nil {
		t.Fatal("expected quota error")
	}
}

func TestKvRepairObservability(t *testing.T) {
	testNotesDB(t)
	res, err := Store("obs", "archive", "lbl", "", TagKvRepair, KindKvRepair, map[string]interface{}{"model_id": "m", "kv_quant": "q4_0"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := ReportKvRepairEvent(ReportEventInput{Reason: RepairCacheMiss, SessionID: "obs", Ref: res.Ref, ModelID: "m", KvQuant: "q4_0"}); err != nil {
		t.Fatal(err)
	}
	_, err = Fetch([]string{res.Ref}, "obs", RepairCacheMiss)
	if err != nil {
		t.Fatal(err)
	}
	stats := KvRepairDashboardStatsFor("", 30)
	if stats.ArchivesActive < 1 || stats.RepairsTotal30d < 1 {
		t.Fatalf("stats: %+v", stats)
	}
	if stats.RepairsByReason[RepairCacheMiss] < 1 {
		t.Fatalf("by reason: %+v", stats.RepairsByReason)
	}
	if stats.CacheMissSignals30d < 1 {
		t.Fatalf("signals: %d", stats.CacheMissSignals30d)
	}
}
