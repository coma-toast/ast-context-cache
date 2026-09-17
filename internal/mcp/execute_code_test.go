package mcp

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
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

// On timeout, the handler used to give up waiting and return, but the
// goroutine running the script kept executing in the background forever
// (an infinite loop ran forever, orphaned and unobservable). vm.Interrupt
// now stops it, and the handler blocks only long enough for that to happen.
func TestHandleExecuteCodeTimeoutInterruptsScript(t *testing.T) {
	done := make(chan map[string]interface{}, 1)
	go func() {
		out := handleExecuteCodeWithMeta(map[string]interface{}{
			"code":    `for(;;){}`,
			"data":    `[]`,
			"timeout": float64(1),
		}).Result
		done <- out
	}()
	select {
	case out := <-done:
		errMsg, _ := out["error"].(string)
		if !strings.Contains(errMsg, "timeout") {
			t.Fatalf("expected a timeout error, got %v", out)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("handleExecuteCodeWithMeta did not return within 5s of a 1s timeout — vm.Interrupt likely isn't stopping the infinite loop")
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
