package mcp

import (
	"encoding/json"
	"testing"
)

func TestHandleExecuteCodeScriptID(t *testing.T) {
	data := `[{"name":"Foo","kind":"function","file":"a.go","start_line":1,"end_line":5}]`
	out := handleExecuteCodeWithMeta(map[string]interface{}{
		"script_id": "compact-symbol-list",
		"data":      data,
	}).Result
	if _, ok := out["error"]; ok {
		t.Fatalf("unexpected error: %v", out)
	}
	if out["script_id"] != "compact-symbol-list" {
		t.Fatalf("script_id: %v", out["script_id"])
	}
	saved, _ := out["tokens_saved"].(int)
	if saved < 0 {
		t.Fatalf("tokens_saved: %v", saved)
	}
	baseline, _ := out["data_baseline_tokens"].(int)
	used, _ := out["tokens_used"].(int)
	if baseline == 0 || used == 0 {
		t.Fatalf("expected baseline/used tokens, got baseline=%v used=%v", baseline, used)
	}
	result, ok := out["result"].([]interface{})
	if !ok || len(result) != 1 {
		t.Fatalf("result: %#v", out["result"])
	}
}

func TestHandleExecuteCodeCustomCode(t *testing.T) {
	out := handleExecuteCodeWithMeta(map[string]interface{}{
		"code": `return DATA.length;`,
		"data": `[1,2,3]`,
	}).Result
	switch v := out["result"].(type) {
	case int64:
		if v != 3 {
			t.Fatalf("got %v", v)
		}
	case float64:
		if v != 3 {
			t.Fatalf("got %v", v)
		}
	default:
		t.Fatalf("unexpected result type %T %v", out["result"], out["result"])
	}
	b, _ := json.Marshal(out)
	if len(b) == 0 {
		t.Fatal("empty response")
	}
}
