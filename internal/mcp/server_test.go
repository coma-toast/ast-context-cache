package mcp

import (
	"encoding/json"
	"net/http/httptest"
	"testing"
)

func TestResultIsError(t *testing.T) {
	cases := []struct {
		json string
		want bool
	}{
		{`{"error":"path not found"}`, true},
		{`{"error":""}`, false},
		{`{"indexed":5}`, false},
		{`{"results":[{"error":"nested, not top-level"}]}`, false},
		{`not even json`, false},
	}
	for _, c := range cases {
		if got := resultIsError([]byte(c.json)); got != c.want {
			t.Fatalf("resultIsError(%s)=%v want %v", c.json, got, c.want)
		}
	}
}

// Every failure short of tier/access denial used to report isError:false with
// the error buried in content[].text, indistinguishable from a real empty
// result to a caller checking only the protocol-level flag.
func TestHandleToolCallSetsIsErrorOnFailure(t *testing.T) {
	origCfg := srvCfg
	srvCfg = DefaultConfig()
	t.Cleanup(func() { srvCfg = origCfg })

	req := JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/call",
		Params: map[string]any{
			"name":      "index_files",
			"arguments": map[string]interface{}{},
		},
	}
	rec := httptest.NewRecorder()
	handleToolCall(rec, req)

	var resp JSONRPCResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal response: %v (body=%s)", err, rec.Body.String())
	}
	result, ok := resp.Result.(map[string]interface{})
	if !ok {
		t.Fatalf("result is not an object: %#v", resp.Result)
	}
	if isErr, _ := result["isError"].(bool); !isErr {
		t.Fatalf("isError=%v want true for a missing-required-param failure: %#v", result["isError"], result)
	}
}

// export_bundle/import_bundle are not yet implemented and must say so as a
// real failure (isError:true), not a success-shaped {"message": "..."} a
// caller could mistake for a completed export/import.
func TestExportImportBundleReportIsError(t *testing.T) {
	origCfg := srvCfg
	srvCfg = DefaultConfig()
	t.Cleanup(func() { srvCfg = origCfg })

	for _, tc := range []struct {
		tool string
		args map[string]interface{}
	}{
		{"export_bundle", map[string]interface{}{"project_path": "/tmp/proj", "output_path": "/tmp/out.astbundle"}},
		{"import_bundle", map[string]interface{}{"bundle_path": "/tmp/out.astbundle"}},
	} {
		req := JSONRPCRequest{
			JSONRPC: "2.0",
			ID:      1,
			Method:  "tools/call",
			Params:  map[string]any{"name": tc.tool, "arguments": tc.args},
		}
		rec := httptest.NewRecorder()
		handleToolCall(rec, req)

		var resp JSONRPCResponse
		if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
			t.Fatalf("%s: unmarshal response: %v (body=%s)", tc.tool, err, rec.Body.String())
		}
		result, ok := resp.Result.(map[string]interface{})
		if !ok {
			t.Fatalf("%s: result is not an object: %#v", tc.tool, resp.Result)
		}
		if isErr, _ := result["isError"].(bool); !isErr {
			t.Fatalf("%s: isError=%v want true (not yet implemented)", tc.tool, result["isError"])
		}
	}
}

func TestHandleToolCallLeavesIsErrorFalseOnSuccess(t *testing.T) {
	origCfg := srvCfg
	srvCfg = DefaultConfig()
	t.Cleanup(func() { srvCfg = origCfg })

	req := JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/call",
		Params: map[string]any{
			"name":      "index_status",
			"arguments": map[string]interface{}{"project_path": t.TempDir()},
		},
	}
	rec := httptest.NewRecorder()
	handleToolCall(rec, req)

	var resp JSONRPCResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal response: %v (body=%s)", err, rec.Body.String())
	}
	result, ok := resp.Result.(map[string]interface{})
	if !ok {
		t.Fatalf("result is not an object: %#v", resp.Result)
	}
	if isErr, _ := result["isError"].(bool); isErr {
		t.Fatalf("isError=%v want false for a successful index_status call: %#v", result["isError"], result)
	}
}
