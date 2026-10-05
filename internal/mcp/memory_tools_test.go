package mcp

import (
	"encoding/json"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/memory"
)

func setupMemoryToolTest(t *testing.T) {
	t.Helper()
	t.Setenv("HOME", t.TempDir())
	t.Setenv("AST_MCP_TIER", "complete")
	origCfg := srvCfg
	srvCfg = DefaultConfig()
	t.Cleanup(func() { srvCfg = origCfg })
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
}

// callTool runs a tools/call through the JSON-RPC handler and returns the
// decoded tool payload plus the MCP isError flag.
func callTool(t *testing.T, name string, arguments map[string]interface{}) (map[string]interface{}, bool) {
	t.Helper()
	req := JSONRPCRequest{
		JSONRPC: "2.0", ID: 1, Method: "tools/call",
		Params: map[string]any{"name": name, "arguments": arguments},
	}
	rec := httptest.NewRecorder()
	handleToolCall(rec, req)
	var resp JSONRPCResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal response: %v (body=%s)", err, rec.Body.String())
	}
	result := resp.Result.(map[string]interface{})
	text := result["content"].([]interface{})[0].(map[string]interface{})["text"].(string)
	var out map[string]interface{}
	if err := json.Unmarshal([]byte(text), &out); err != nil {
		t.Fatalf("unmarshal tool result: %v (text=%s)", err, text)
	}
	isErr, _ := result["isError"].(bool)
	return out, isErr
}

func storeSessionRules(t *testing.T, sessionID string, n int) []string {
	t.Helper()
	var refs []string
	for i := 0; i < n; i++ {
		res, err := memory.Store(memory.StoreInput{
			Kind: memory.KindProcedure, Scope: memory.ScopeSession,
			SessionID: sessionID, Rule: "rule " + string(rune('a'+i)),
		})
		if err != nil {
			t.Fatal(err)
		}
		refs = append(refs, res.Ref)
	}
	return refs
}

func toStrings(v interface{}) []string {
	var out []string
	if arr, ok := v.([]interface{}); ok {
		for _, x := range arr {
			out = append(out, x.(string))
		}
	}
	return out
}

// Issue #8: every call shape from the field report must invalidate all named
// refs; only "single ref + scope + session_id" used to work.
func TestForgetMemoryFieldReportShapes(t *testing.T) {
	setupMemoryToolTest(t)
	const sess = "a5c68d1d-42e7-413b-873c-b1d2f8d5fcd1"
	cases := []struct {
		name  string
		n     int
		build func(refs []string) map[string]interface{}
	}{
		{"comma-separated", 3, func(r []string) map[string]interface{} {
			return map[string]interface{}{"refs": strings.Join(r, ",")}
		}},
		{"comma-separated with spaces", 3, func(r []string) map[string]interface{} {
			return map[string]interface{}{"refs": strings.Join(r, ", ")}
		}},
		{"json array", 3, func(r []string) map[string]interface{} {
			arr := make([]interface{}, len(r))
			for i, x := range r {
				arr[i] = x
			}
			return map[string]interface{}{"refs": arr}
		}},
		{"json array as string", 3, func(r []string) map[string]interface{} {
			b, _ := json.Marshal(r)
			return map[string]interface{}{"refs": string(b)}
		}},
		{"json array + scope session + session_id", 3, func(r []string) map[string]interface{} {
			arr := make([]interface{}, len(r))
			for i, x := range r {
				arr[i] = x
			}
			return map[string]interface{}{"refs": arr, "scope": "session", "session_id": sess}
		}},
		{"single ref no scope/session", 1, func(r []string) map[string]interface{} {
			return map[string]interface{}{"refs": r[0]}
		}},
		{"single ref + scope + session_id", 1, func(r []string) map[string]interface{} {
			return map[string]interface{}{"refs": r[0], "scope": "session", "session_id": sess}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			refs := storeSessionRules(t, sess, tc.n)
			out, isErr := callTool(t, "forget_memory", tc.build(refs))
			if isErr {
				t.Fatalf("isError: %v", out)
			}
			if got := int(out["invalidated_refs"].(float64)); got != tc.n {
				t.Fatalf("invalidated_refs=%d want %d (out=%v)", got, tc.n, out)
			}
			if got := toStrings(out["invalidated"]); !reflect.DeepEqual(got, refs) {
				t.Fatalf("invalidated=%v want %v", got, refs)
			}
		})
	}
}

func TestForgetMemoryReportsNotFound(t *testing.T) {
	setupMemoryToolTest(t)
	refs := storeSessionRules(t, "s-nf", 1)
	out, isErr := callTool(t, "forget_memory", map[string]interface{}{"refs": refs[0] + ",mem_000000000000"})
	if isErr || int(out["invalidated_refs"].(float64)) != 1 {
		t.Fatalf("partial match: isErr=%v out=%v", isErr, out)
	}
	if got := toStrings(out["not_found"]); !reflect.DeepEqual(got, []string{"mem_000000000000"}) {
		t.Fatalf("not_found=%v", got)
	}
	out, isErr = callTool(t, "forget_memory", map[string]interface{}{"refs": []interface{}{"mem_nope1", "mem_nope2"}})
	if !isErr || out["error"] == nil {
		t.Fatalf("nothing matched should be an error, got isErr=%v out=%v", isErr, out)
	}
	if got := toStrings(out["not_found"]); !reflect.DeepEqual(got, []string{"mem_nope1", "mem_nope2"}) {
		t.Fatalf("not_found=%v", got)
	}
}

func TestParseStringList(t *testing.T) {
	cases := map[string]struct {
		in   interface{}
		want []string
	}{
		"single":       {"mem_a", []string{"mem_a"}},
		"comma":        {"mem_a,mem_b, mem_c", []string{"mem_a", "mem_b", "mem_c"}},
		"json string":  {`["mem_a", "mem_b"]`, []string{"mem_a", "mem_b"}},
		"bracket list": {`[mem_a, mem_b]`, []string{"mem_a", "mem_b"}},
		"array":        {[]interface{}{"mem_a", " mem_b ", ""}, []string{"mem_a", "mem_b"}},
		"empty":        {"  ", nil},
	}
	for name, tc := range cases {
		if got := parseStringList(tc.in); !reflect.DeepEqual(got, tc.want) {
			t.Errorf("%s: got %v want %v", name, got, tc.want)
		}
	}
}

// Issue #7 end to end: store_context(extract_memory=true) on a note mixing
// headings, prose, path:line text, and 2 FACT + 2 RULE lines stores exactly 4.
func TestStoreContextExtractMemoryOnlyMarkedLines(t *testing.T) {
	setupMemoryToolTest(t)
	note := `## BROKEN NOW
bonsai/plugin.py:33 SERVICE_MODEL=Ternary-Bonsai is hardcoded here
## WRONG BEHAVIOR
Sync overwrites manual edits on every run.
FACT: bonsai.service_model | is_set_at | bonsai/plugin.py:33 SERVICE_MODEL=Ternary-Bonsai
FACT: litellm_sync.py:120 drops api_base
RULE: Never hardcode SERVICE_MODEL
RULE: Run a dry-run before applying sync
FACT: too-short`
	out, isErr := callTool(t, "store_context", map[string]interface{}{
		"content": note, "session_id": "s-extract", "extract_memory": true,
	})
	if isErr {
		t.Fatalf("store_context error: %v", out)
	}
	extracted, _ := out["memory_extracted"].([]interface{})
	var lines []string
	for _, e := range extracted {
		lines = append(lines, e.(map[string]interface{})["line"].(string))
	}
	want := []string{
		"bonsai.service_model is_set_at bonsai/plugin.py:33 SERVICE_MODEL=Ternary-Bonsai",
		"litellm_sync.py:120 drops api_base",
		"PROC: Never hardcode SERVICE_MODEL",
		"PROC: Run a dry-run before applying sync",
	}
	if !reflect.DeepEqual(lines, want) {
		t.Fatalf("memory_extracted lines=%q\nwant %q", lines, want)
	}
	if got := toStrings(out["memory_skipped"]); !reflect.DeepEqual(got, []string{"FACT: too-short"}) {
		t.Fatalf("memory_skipped=%v", got)
	}
	var n int
	if err := db.ContextDB.QueryRow(`SELECT COUNT(*) FROM structured_memory WHERE session_id = 's-extract'`).Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n != 4 {
		t.Fatalf("structured_memory rows=%d want 4", n)
	}
}
