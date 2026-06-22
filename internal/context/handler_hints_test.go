package context_test

import (
	"encoding/json"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/codescripts"
)

func TestAttachHintsOnResponse(t *testing.T) {
	results := make([]map[string]interface{}, 12)
	for i := range results {
		results[i] = map[string]interface{}{
			"name": "fn", "kind": "function", "file": "internal/x.go",
		}
	}
	resp := map[string]interface{}{"query": "auth", "results": results}
	codescripts.AttachHints(resp, "get_context_capsule", "auth", "", results)
	raw, err := json.Marshal(resp)
	if err != nil {
		t.Fatal(err)
	}
	var parsed map[string]interface{}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		t.Fatal(err)
	}
	hints, ok := parsed["code_script_hints"].([]interface{})
	if !ok || len(hints) == 0 {
		t.Fatalf("expected code_script_hints in cached-style JSON, got %v", parsed["code_script_hints"])
	}
}
