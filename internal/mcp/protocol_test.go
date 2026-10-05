package mcp

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// These are HTTP-level tests of the /mcp endpoint. No t.Parallel: the config, feature
// flags, stream hub, and db pools are package globals.

func newMCPServer(t *testing.T) *httptest.Server {
	t.Helper()
	origCfg := GetConfig()
	SetConfig(DefaultConfig())
	srv := httptest.NewServer(NewHandler())
	t.Cleanup(func() {
		srv.Close()
		SetConfig(origCfg)
	})
	return srv
}

// rpcCall POSTs body with headers and returns the response and its decoded JSON body,
// which is nil when the body is empty.
func rpcCall(t *testing.T, srv *httptest.Server, headers map[string]string, body any) (*http.Response, map[string]any) {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, srv.URL, bytes.NewReader(mustJSON(body)))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	if len(raw) == 0 {
		return resp, nil
	}
	var out map[string]any
	require.NoError(t, json.Unmarshal(raw, &out), string(raw))
	return resp, out
}

func rpcRequest(id any, method string, params map[string]any) map[string]any {
	msg := map[string]any{"jsonrpc": "2.0", "method": method}
	if id != nil {
		msg["id"] = id
	}
	if params != nil {
		msg["params"] = params
	}
	return msg
}

func initialize(t *testing.T, srv *httptest.Server, version string) (*http.Response, map[string]any) {
	t.Helper()
	return rpcCall(t, srv, nil, rpcRequest(1, "initialize", map[string]any{"protocolVersion": version, "capabilities": map[string]any{}, "clientInfo": map[string]any{"name": "test", "version": "1"}}))
}

// modernCall sends a well-formed 2026-07-28 request: _meta plus the mirrored headers.
// override replaces or, with an empty value, removes headers.
func modernCall(t *testing.T, srv *httptest.Server, id any, method string, params map[string]any, override map[string]string) (*http.Response, map[string]any) {
	t.Helper()
	if params == nil {
		params = map[string]any{}
	}
	params[metaKey] = map[string]any{metaProtocolVersion: ModernProtocolVersion, metaClientCapabilities: map[string]any{}, "io.modelcontextprotocol/clientInfo": map[string]any{"name": "test", "version": "1"}}
	headers := map[string]string{headerProtocolVersion: ModernProtocolVersion, headerMethod: method}
	if name := bodyName(JSONRPCRequest{Params: params}); name != "" {
		headers[headerName] = name
	}
	for k, v := range override {
		headers[k] = v
	}
	return rpcCall(t, srv, headers, rpcRequest(id, method, params))
}

func resultOf(t *testing.T, body map[string]any) map[string]any {
	t.Helper()
	require.Nil(t, body["error"], "unexpected error: %v", body["error"])
	result, ok := body["result"].(map[string]any)
	require.True(t, ok, "result is not an object: %v", body)
	return result
}

func errorOf(t *testing.T, body map[string]any) (code int, rerr map[string]any) {
	t.Helper()
	rerr, ok := body["error"].(map[string]any)
	require.True(t, ok, "no error in %v", body)
	return int(rerr["code"].(float64)), rerr
}

func toolNames(t *testing.T, result map[string]any) []string {
	t.Helper()
	var names []string
	for _, tool := range result["tools"].([]any) {
		names = append(names, tool.(map[string]any)["name"].(string))
	}
	return names
}

func TestNegotiate(t *testing.T) {
	t.Parallel()
	tests := []struct{ client, want string }{
		{"2025-11-25", "2025-11-25"},
		{"2025-06-18", "2025-06-18"},
		{"2025-03-26", "2025-03-26"},
		{"2024-11-05", "2024-11-05"},
		{ModernProtocolVersion, "2025-11-25"},
		{"2099-01-01", "2025-11-25"},
		{"", "2025-11-25"},
	}
	for _, tt := range tests {
		t.Run(tt.client, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.want, negotiate(tt.client))
		})
	}
}

func TestLegacyInitialize(t *testing.T) {
	srv := newMCPServer(t)
	tests := []struct {
		client, want string
		session      bool
	}{
		{client: "2025-11-25", want: "2025-11-25", session: true},
		{client: "2025-06-18", want: "2025-06-18", session: true},
		{client: "2025-03-26", want: "2025-03-26", session: true},
		{client: "2024-11-05", want: "2024-11-05"},
		{client: "1999-01-01", want: "2025-11-25", session: true},
	}
	for _, tt := range tests {
		t.Run(tt.client, func(t *testing.T) {
			resp, body := initialize(t, srv, tt.client)
			require.Equal(t, http.StatusOK, resp.StatusCode)
			result := resultOf(t, body)
			assert.Equal(t, tt.want, result["protocolVersion"])
			assert.Equal(t, map[string]any{"tools": map[string]any{"listChanged": true}, "prompts": map[string]any{}}, result["capabilities"])
			assert.Equal(t, serverName, result["serverInfo"].(map[string]any)["name"])
			assert.Equal(t, tt.session, resp.Header.Get(headerSessionID) != "", "Mcp-Session-Id presence")
		})
	}
	r1, _ := initialize(t, srv, "2025-06-18")
	r2, _ := initialize(t, srv, "2025-06-18")
	assert.NotEqual(t, r1.Header.Get(headerSessionID), r2.Header.Get(headerSessionID), "each initialize mints a new session")
}

func TestNotificationsGet202(t *testing.T) {
	srv := newMCPServer(t)
	modern := map[string]string{headerProtocolVersion: ModernProtocolVersion}
	tests := []struct {
		name    string
		headers map[string]string
		body    map[string]any
	}{
		{name: "initialized notification", body: rpcRequest(nil, "notifications/initialized", nil)},
		{name: "pre-2025 initialized", body: rpcRequest(nil, "initialized", nil)},
		{name: "cancelled", body: rpcRequest(nil, "notifications/cancelled", map[string]any{"requestId": 3})},
		{name: "notification method with id", body: rpcRequest(7, "notifications/initialized", nil)},
		{name: "client response", body: map[string]any{"jsonrpc": "2.0", "id": 5, "result": map[string]any{}}},
		{name: "modern notification", headers: modern, body: rpcRequest(nil, "notifications/cancelled", nil)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, body := rpcCall(t, srv, tt.headers, tt.body)
			assert.Equal(t, http.StatusAccepted, resp.StatusCode)
			assert.Nil(t, body, "a notification never gets a body")
		})
	}
}

func TestLegacyMethods(t *testing.T) {
	srv := newMCPServer(t)
	resp, body := rpcCall(t, srv, nil, rpcRequest(2, "ping", nil))
	require.Equal(t, http.StatusOK, resp.StatusCode)
	assert.Equal(t, map[string]any{}, resultOf(t, body))
	assert.Equal(t, float64(2), body["id"])

	resp, body = rpcCall(t, srv, nil, rpcRequest(3, "no/such", nil))
	assert.Equal(t, http.StatusOK, resp.StatusCode, "legacy keeps reporting unknown methods in a 200")
	code, _ := errorOf(t, body)
	assert.Equal(t, MethodNotFound, code)

	// A legacy client after initialize sends the negotiated version header; an unknown
	// session (here, as after a restart) is still served.
	resp, body = rpcCall(t, srv, map[string]string{headerProtocolVersion: "2025-06-18", headerSessionID: "unknown"}, rpcRequest(4, "tools/list", nil))
	require.Equal(t, http.StatusOK, resp.StatusCode)
	result := resultOf(t, body)
	assert.Contains(t, toolNames(t, result), "get_context_capsule")
	assert.NotContains(t, result, "resultType", "legacy results are unchanged")

	req, err := http.NewRequest(http.MethodPost, srv.URL, bytes.NewReader([]byte("{not json")))
	require.NoError(t, err)
	raw, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	raw.Body.Close()
	assert.Equal(t, http.StatusBadRequest, raw.StatusCode)
}

func TestLegacyGetAndDelete(t *testing.T) {
	srv := newMCPServer(t)
	do := func(method string, headers map[string]string) *http.Response {
		t.Helper()
		req, err := http.NewRequest(method, srv.URL, nil)
		require.NoError(t, err)
		for k, v := range headers {
			req.Header.Set(k, v)
		}
		resp, err := http.DefaultClient.Do(req)
		require.NoError(t, err)
		t.Cleanup(func() { resp.Body.Close() })
		return resp
	}
	plain := do(http.MethodGet, nil)
	require.Equal(t, http.StatusOK, plain.StatusCode, "deprecated plain GET still lists tools")
	var listing map[string]any
	require.NoError(t, json.NewDecoder(plain.Body).Decode(&listing))
	assert.Contains(t, toolNames(t, resultOf(t, listing)), "get_context_capsule")

	sse := map[string]string{"Accept": sseContentType}
	assert.Equal(t, http.StatusMethodNotAllowed, do(http.MethodGet, sse).StatusCode, "no stream without a session")
	assert.Equal(t, http.StatusNotFound, do(http.MethodGet, map[string]string{"Accept": sseContentType, headerSessionID: "nope"}).StatusCode)
	assert.Equal(t, http.StatusMethodNotAllowed, do(http.MethodDelete, nil).StatusCode)

	resp, _ := initialize(t, srv, "2025-11-25")
	sid := resp.Header.Get(headerSessionID)
	require.NotEmpty(t, sid)
	assert.Equal(t, http.StatusNoContent, do(http.MethodDelete, map[string]string{headerSessionID: sid}).StatusCode)
	assert.Equal(t, http.StatusNotFound, do(http.MethodDelete, map[string]string{headerSessionID: sid}).StatusCode)
	assert.Equal(t, http.StatusNotFound, do(http.MethodGet, map[string]string{"Accept": sseContentType, headerSessionID: sid}).StatusCode)
}

func TestModernDiscover(t *testing.T) {
	srv := newMCPServer(t)
	resp, body := modernCall(t, srv, "d1", "server/discover", nil, map[string]string{headerSessionID: "ignored"})
	require.Equal(t, http.StatusOK, resp.StatusCode)
	assert.Empty(t, resp.Header.Get(headerSessionID), "modern era never mints or echoes a session")
	assert.Equal(t, "d1", body["id"])
	result := resultOf(t, body)
	assert.Equal(t, "complete", result["resultType"])
	assert.Equal(t, []any{"2026-07-28", "2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"}, result["supportedVersions"])
	assert.Equal(t, map[string]any{"listChanged": true}, result["capabilities"].(map[string]any)["tools"])
	assert.Equal(t, serverName, result[metaKey].(map[string]any)[metaServerInfo].(map[string]any)["name"])
	assert.Equal(t, float64(listTTLMs), result["ttlMs"])
	assert.Equal(t, cacheScopePrivate, result["cacheScope"])
}

func TestModernResults(t *testing.T) {
	srv := newMCPServer(t)
	_, body := modernCall(t, srv, 1, "tools/list", nil, nil)
	result := resultOf(t, body)
	assert.Equal(t, "complete", result["resultType"])
	assert.Equal(t, float64(listTTLMs), result["ttlMs"])
	assert.Equal(t, cacheScopePrivate, result["cacheScope"])
	assert.Contains(t, result[metaKey], metaServerInfo)
	names := toolNames(t, result)
	assert.Contains(t, names, "get_context_capsule")
	_, again := modernCall(t, srv, 2, "tools/list", nil, nil)
	assert.Equal(t, names, toolNames(t, resultOf(t, again)), "tools/list order is deterministic")

	_, body = modernCall(t, srv, 3, "prompts/list", nil, nil)
	result = resultOf(t, body)
	assert.Equal(t, cacheScopePrivate, result["cacheScope"])
	assert.NotEmpty(t, result["prompts"])

	// Mcp-Name arrives base64 encoded when the client chooses the sentinel form.
	prompt := GetPrompts()[0].Name
	encoded := base64Prefix + base64.StdEncoding.EncodeToString([]byte(prompt)) + base64Suffix
	resp, body := modernCall(t, srv, 4, "prompts/get", map[string]any{"name": prompt}, map[string]string{headerName: encoded})
	require.Equal(t, http.StatusOK, resp.StatusCode)
	result = resultOf(t, body)
	assert.Equal(t, "complete", result["resultType"])
	assert.NotContains(t, result, "ttlMs", "only list results carry cache hints")
	assert.NotEmpty(t, result["prompt"])

	_, body = modernCall(t, srv, 5, "tools/call", map[string]any{"name": "no_such_tool", "arguments": map[string]any{}}, nil)
	result = resultOf(t, body)
	assert.Equal(t, "complete", result["resultType"])
	assert.Equal(t, true, result["isError"])
	assert.NotEmpty(t, result["content"])
}

func TestModernValidation(t *testing.T) {
	srv := newMCPServer(t)
	tests := []struct {
		name       string
		method     string
		params     map[string]any
		override   map[string]string
		meta       map[string]any
		wantStatus int
		wantCode   int
	}{
		{name: "missing Mcp-Method", method: "tools/list", override: map[string]string{headerMethod: ""}, wantStatus: 400, wantCode: HeaderMismatch},
		{name: "Mcp-Method mismatch", method: "tools/list", override: map[string]string{headerMethod: "prompts/list"}, wantStatus: 400, wantCode: HeaderMismatch},
		{name: "Mcp-Name mismatch", method: "tools/call", params: map[string]any{"name": "index_status"}, override: map[string]string{headerName: "retrieve"}, wantStatus: 400, wantCode: HeaderMismatch},
		{name: "missing Mcp-Name", method: "prompts/get", params: map[string]any{"name": "x"}, override: map[string]string{headerName: ""}, wantStatus: 400, wantCode: HeaderMismatch},
		{name: "bad base64 Mcp-Name", method: "prompts/get", params: map[string]any{"name": "x"}, override: map[string]string{headerName: "=?base64?***?="}, wantStatus: 400, wantCode: HeaderMismatch},
		{name: "missing version header", method: "tools/list", override: map[string]string{headerProtocolVersion: ""}, wantStatus: 400, wantCode: HeaderMismatch},
		{name: "version header mismatch", method: "tools/list", override: map[string]string{headerProtocolVersion: "2025-06-18"}, wantStatus: 400, wantCode: HeaderMismatch},
		{name: "unsupported version", method: "tools/list", meta: map[string]any{metaProtocolVersion: "2099-01-01", metaClientCapabilities: map[string]any{}}, override: map[string]string{headerProtocolVersion: "2099-01-01"}, wantStatus: 400, wantCode: UnsupportedProtocolVersion},
		{name: "unsupported version header only", method: "tools/list", meta: map[string]any{}, override: map[string]string{headerProtocolVersion: "2099-01-01"}, wantStatus: 400, wantCode: UnsupportedProtocolVersion},
		{name: "missing _meta", method: "tools/list", meta: map[string]any{}, wantStatus: 400, wantCode: InvalidParams},
		{name: "missing clientCapabilities", method: "tools/list", meta: map[string]any{metaProtocolVersion: ModernProtocolVersion}, wantStatus: 400, wantCode: InvalidParams},
		{name: "unknown method", method: "no/such", wantStatus: 404, wantCode: MethodNotFound},
		{name: "ping removed", method: "ping", wantStatus: 404, wantCode: MethodNotFound},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.meta == nil {
				resp, body := modernCall(t, srv, 9, tt.method, tt.params, tt.override)
				checkRPCError(t, resp, body, tt.wantStatus, tt.wantCode)
				return
			}
			params := map[string]any{metaKey: tt.meta}
			if len(tt.meta) == 0 {
				params = nil
			}
			headers := map[string]string{headerProtocolVersion: ModernProtocolVersion, headerMethod: tt.method}
			for k, v := range tt.override {
				headers[k] = v
			}
			resp, body := rpcCall(t, srv, headers, rpcRequest(9, tt.method, params))
			checkRPCError(t, resp, body, tt.wantStatus, tt.wantCode)
			if tt.wantCode == UnsupportedProtocolVersion {
				data := body["error"].(map[string]any)["data"].(map[string]any)
				assert.Equal(t, "2099-01-01", data["requested"])
				assert.Contains(t, data["supported"], ModernProtocolVersion)
			}
		})
	}
}

// checkRPCError asserts the HTTP status, the JSON-RPC code, and that the error answers
// request 9. An empty override value in these cases sends the header empty, which the
// server treats as missing.
func checkRPCError(t *testing.T, resp *http.Response, body map[string]any, status, code int) {
	t.Helper()
	assert.Equal(t, status, resp.StatusCode)
	got, _ := errorOf(t, body)
	assert.Equal(t, code, got)
	assert.Equal(t, float64(9), body["id"])
}

func TestModernGetDelete405(t *testing.T) {
	srv := newMCPServer(t)
	for _, method := range []string{http.MethodGet, http.MethodDelete} {
		req, err := http.NewRequest(method, srv.URL, nil)
		require.NoError(t, err)
		req.Header.Set(headerProtocolVersion, ModernProtocolVersion)
		req.Header.Set("Accept", sseContentType)
		req.Header.Set(headerSessionID, "anything")
		resp, err := http.DefaultClient.Do(req)
		require.NoError(t, err)
		resp.Body.Close()
		assert.Equal(t, http.StatusMethodNotAllowed, resp.StatusCode, method)
	}
}

func TestDecodeHeaderValue(t *testing.T) {
	t.Parallel()
	tests := []struct {
		in, want string
		ok       bool
	}{
		{"get_weather", "get_weather", true},
		{"=?base64?SGVsbG8sIOS4lueVjA==?=", "Hello, 世界", true},
		{"=?base64?PT9iYXNlNjQ/bGl0ZXJhbD89?=", "=?base64?literal?=", true},
		{"=?base64?!!?=", "", false},
		{"=?BASE64?eA==?=", "=?BASE64?eA==?=", true},
	}
	for _, tt := range tests {
		t.Run(tt.in, func(t *testing.T) {
			t.Parallel()
			got, ok := decodeHeaderValue(tt.in)
			assert.Equal(t, tt.ok, ok)
			assert.Equal(t, tt.want, got)
		})
	}
}
