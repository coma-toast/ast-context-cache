package mcp

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"slices"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/version"
)

// One /mcp endpoint serves two protocol eras (docs/spikes/mcp-protocol-versions.md):
//
//   - Legacy (2024-11-05 through 2025-11-25): an initialize handshake negotiates the version,
//     2025-03-26 and later get an Mcp-Session-Id, and list_changed goes out on a GET SSE stream.
//   - Modern (2026-07-28): stateless. Every request declares its version in
//     params._meta and the MCP-Protocol-Version header, there are no sessions, and
//     list_changed goes out on a subscriptions/listen stream.
//
// eraOf picks the era per request, so both kinds of client can share the endpoint.

// ModernProtocolVersion is the stateless protocol revision this server speaks.
const ModernProtocolVersion = "2026-07-28"

// LegacyProtocolVersions are the handshake revisions initialize negotiates, newest first.
// 2024-11-05 stays because the plain JSON POST response already serves those clients.
var LegacyProtocolVersions = []string{"2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"}

// Header names. Go canonicalizes header keys, so the spellings here only matter for output.
const (
	headerProtocolVersion = "MCP-Protocol-Version"
	headerMethod          = "Mcp-Method"
	headerName            = "Mcp-Name"
	headerSessionID       = "Mcp-Session-Id"
)

// Reserved _meta keys from the 2026-07-28 revision.
const (
	metaKey                = "_meta"
	metaProtocolVersion    = "io.modelcontextprotocol/protocolVersion"
	metaClientCapabilities = "io.modelcontextprotocol/clientCapabilities"
	metaServerInfo         = "io.modelcontextprotocol/serverInfo"
	metaSubscriptionID     = "io.modelcontextprotocol/subscriptionId"
)

// JSON-RPC error codes the 2026-07-28 revision defines.
const (
	HeaderMismatch             = -32020
	UnsupportedProtocolVersion = -32022
)

const (
	serverName = "ast-context-cache"
	// sessionVersionMin is the first revision with Mcp-Session-Id; 2024-11-05 clients
	// would not echo one back.
	sessionVersionMin = "2025-03-26"
	// listTTLMs is the modern cache hint for list results. Kept short because flags can
	// change the tool list; listening clients hear about that sooner via list_changed.
	listTTLMs = 60_000
	// cacheScopePrivate keeps shared intermediaries from caching a local server's lists.
	cacheScopePrivate = "private"
	base64Prefix      = "=?base64?"
	base64Suffix      = "?="
)

type era int

const (
	eraLegacy era = iota
	eraModern
)

// supportedVersions is every version this server accepts, for discovery and errors.
func supportedVersions() []string {
	return append([]string{ModernProtocolVersion}, LegacyProtocolVersions...)
}

// negotiate implements the legacy handshake rule: echo the client's version when it is
// supported, otherwise offer the newest legacy version and let the client decide.
func negotiate(clientVersion string) string {
	if slices.Contains(LegacyProtocolVersions, clientVersion) {
		return clientVersion
	}
	return LegacyProtocolVersions[0]
}

// eraOf decides which protocol a POST speaks. initialize always selects legacy, as the
// spec requires of dual-era servers. Otherwise modern _meta, or an MCP-Protocol-Version
// header that is not a legacy version, selects the modern path, where an unknown version
// gets a -32022 listing what is supported. Requests with neither (curl, pre-2025-06-18
// clients) stay legacy.
func eraOf(r *http.Request, req JSONRPCRequest) era {
	if req.Method == "initialize" {
		return eraLegacy
	}
	if _, ok := requestMeta(req)[metaProtocolVersion]; ok {
		return eraModern
	}
	if isModernHeader(r) {
		return eraModern
	}
	return eraLegacy
}

// isModernHeader reports whether the MCP-Protocol-Version header names anything other
// than a legacy version, which also covers GET and DELETE from modern clients.
func isModernHeader(r *http.Request) bool {
	v := r.Header.Get(headerProtocolVersion)
	return v != "" && !slices.Contains(LegacyProtocolVersions, v)
}

// isNotification reports whether msg needs no reply body: a notification (no id, or a
// notifications/* method) or a JSON-RPC response from the client (an id but no method).
func isNotification(req JSONRPCRequest) bool {
	return req.ID == nil || req.Method == "" || req.Method == "initialized" || strings.HasPrefix(req.Method, "notifications/")
}

func requestMeta(req JSONRPCRequest) map[string]any {
	meta, _ := req.Params[metaKey].(map[string]any)
	return meta
}

// validateModern applies the 2026-07-28 request checks, returning the error to send and its
// HTTP status. The version is checked first so a client on a revision we lack learns the
// supported list before anything else.
func validateModern(r *http.Request, req JSONRPCRequest) (*JSONRPCError, int) {
	meta := requestMeta(req)
	metaVersion, _ := meta[metaProtocolVersion].(string)
	hdrVersion := r.Header.Get(headerProtocolVersion)
	requested := metaVersion
	if requested == "" {
		requested = hdrVersion
	}
	if requested != ModernProtocolVersion {
		return &JSONRPCError{Code: UnsupportedProtocolVersion, Message: "Unsupported protocol version", Data: map[string]any{"supported": supportedVersions(), "requested": requested}}, http.StatusBadRequest
	}
	if metaVersion == "" {
		return &JSONRPCError{Code: InvalidParams, Message: "Missing required _meta field " + metaProtocolVersion}, http.StatusBadRequest
	}
	if _, ok := meta[metaClientCapabilities]; !ok {
		return &JSONRPCError{Code: InvalidParams, Message: "Missing required _meta field " + metaClientCapabilities}, http.StatusBadRequest
	}
	if msg := headerMismatch(headerProtocolVersion, hdrVersion, metaVersion); msg != "" {
		return &JSONRPCError{Code: HeaderMismatch, Message: msg}, http.StatusBadRequest
	}
	if msg := headerMismatch(headerMethod, r.Header.Get(headerMethod), req.Method); msg != "" {
		return &JSONRPCError{Code: HeaderMismatch, Message: msg}, http.StatusBadRequest
	}
	if !namedMethod(req.Method) {
		return nil, 0
	}
	name, ok := decodeHeaderValue(r.Header.Get(headerName))
	if !ok {
		return &JSONRPCError{Code: HeaderMismatch, Message: "Header mismatch: invalid " + headerName + " encoding"}, http.StatusBadRequest
	}
	if msg := headerMismatch(headerName, name, bodyName(req)); msg != "" {
		return &JSONRPCError{Code: HeaderMismatch, Message: msg}, http.StatusBadRequest
	}
	return nil, 0
}

// headerMismatch describes why header value got fails to mirror the body value want, or
// returns "" when it matches.
func headerMismatch(header, got, want string) string {
	switch {
	case got == "":
		return "Header mismatch: missing required " + header + " header"
	case got != want:
		return "Header mismatch: " + header + " header value '" + got + "' does not match body value '" + want + "'"
	}
	return ""
}

// namedMethod reports whether the method must mirror params.name or params.uri in Mcp-Name.
func namedMethod(method string) bool {
	return method == "tools/call" || method == "prompts/get" || method == "resources/read"
}

func bodyName(req JSONRPCRequest) string {
	if name, ok := req.Params["name"].(string); ok {
		return name
	}
	uri, _ := req.Params["uri"].(string)
	return uri
}

// decodeHeaderValue undoes the spec's =?base64?…?= sentinel for values that are not plain
// header-safe ASCII. ok is false when an encoded value isn't valid base64.
func decodeHeaderValue(v string) (string, bool) {
	if !strings.HasPrefix(v, base64Prefix) || !strings.HasSuffix(v, base64Suffix) || len(v) < len(base64Prefix)+len(base64Suffix) {
		return v, true
	}
	raw, err := base64.StdEncoding.DecodeString(v[len(base64Prefix) : len(v)-len(base64Suffix)])
	if err != nil {
		return "", false
	}
	return string(raw), true
}

func serverCapabilities() map[string]any {
	return map[string]any{"tools": map[string]any{"listChanged": true}, "prompts": map[string]any{}}
}

func serverInfo() map[string]any {
	return map[string]any{"name": serverName, "version": version.Version}
}

// initializeResult is the legacy handshake reply for the negotiated version.
func initializeResult(negotiated string) map[string]any {
	return map[string]any{"protocolVersion": negotiated, "capabilities": serverCapabilities(), "serverInfo": serverInfo()}
}

// discoverResult answers the modern server/discover, which the spec makes cacheable.
func discoverResult() map[string]any {
	return map[string]any{
		"resultType":        "complete",
		"supportedVersions": supportedVersions(),
		"capabilities":      serverCapabilities(),
		metaKey:             map[string]any{metaServerInfo: serverInfo()},
		"ttlMs":             listTTLMs,
		"cacheScope":        cacheScopePrivate,
	}
}

// cacheableMethod reports whether the modern result must carry ttlMs and cacheScope.
func cacheableMethod(method string) bool {
	return method == "tools/list" || method == "prompts/list"
}

// modernize rewrites a legacy-shaped JSON-RPC response body for the 2026-07-28 revision:
// every result gets resultType and _meta.serverInfo, and list results get the cache hints.
// The result is edited as raw JSON so tool output passes through byte for byte. Error
// responses and bodies it can't parse are returned unchanged.
func modernize(body []byte, method string) []byte {
	var env map[string]json.RawMessage
	if json.Unmarshal(body, &env) != nil || env["result"] == nil {
		return body
	}
	var result map[string]json.RawMessage
	if json.Unmarshal(env["result"], &result) != nil {
		return body
	}
	meta := map[string]json.RawMessage{}
	if raw, ok := result[metaKey]; ok {
		_ = json.Unmarshal(raw, &meta)
	}
	meta[metaServerInfo] = mustJSON(serverInfo())
	result[metaKey] = mustJSON(meta)
	result["resultType"] = mustJSON("complete")
	if cacheableMethod(method) {
		result["ttlMs"] = mustJSON(listTTLMs)
		result["cacheScope"] = mustJSON(cacheScopePrivate)
	}
	env["result"] = mustJSON(result)
	return append(mustJSON(env), '\n')
}

// mustJSON marshals values built here from JSON-safe types, so it cannot fail.
func mustJSON(v any) json.RawMessage {
	b, _ := json.Marshal(v)
	return b
}

// responseBuffer captures a handler's output so the modern path can rewrite it.
type responseBuffer struct {
	header http.Header
	status int
	body   bytes.Buffer
}

func newResponseBuffer() *responseBuffer {
	return &responseBuffer{header: http.Header{}, status: http.StatusOK}
}

func (b *responseBuffer) Header() http.Header         { return b.header }
func (b *responseBuffer) Write(p []byte) (int, error) { return b.body.Write(p) }
func (b *responseBuffer) WriteHeader(status int)      { b.status = status }

// writeRPC writes one JSON-RPC message with the given HTTP status.
func writeRPC(w http.ResponseWriter, status int, resp JSONRPCResponse) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		logger.Debug("Failed to write MCP response", "error", err)
	}
}

func rpcResult(id, result any) JSONRPCResponse {
	return JSONRPCResponse{JSONRPC: JSONRPCVersion, ID: id, Result: result}
}

func rpcError(id any, rerr *JSONRPCError) JSONRPCResponse {
	return JSONRPCResponse{JSONRPC: JSONRPCVersion, ID: id, Error: rerr}
}
