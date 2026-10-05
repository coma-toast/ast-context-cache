package hooks

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

const (
	// defaultMCPPort matches the server's default -mcp-port and installer.DefaultMCPPort. It is
	// repeated here so the hook binary path doesn't pull in the installer.
	defaultMCPPort = 7821
	// maxResponseBytes bounds what a hook reads from the server.
	maxResponseBytes = 4 << 20
)

// Client calls MCP tools on the local server over Streamable HTTP. It sends legacy-era requests
// without an initialize handshake, which the server answers statelessly.
type Client struct {
	url  string
	http *http.Client
}

type rpcRequest struct {
	JSONRPC string     `json:"jsonrpc"`
	ID      int        `json:"id"`
	Method  string     `json:"method"`
	Params  callParams `json:"params"`
}

type callParams struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

type rpcResponse struct {
	Result *struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		IsError bool `json:"isError"`
	} `json:"result"`
	Error *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

// NewClient returns a client for the MCP endpoint at url. Calls are bounded by the caller's
// context; the HTTP client adds the same cap so a client built without a deadline still fails open.
func NewClient(url string) *Client {
	return &Client{url: url, http: &http.Client{Timeout: Timeout}}
}

// ResolveURL is the MCP endpoint: $AST_MCP_URL verbatim, else http://127.0.0.1:<port>/mcp with
// the port from $AST_MCP_PORT or the default.
func ResolveURL() string {
	if u := strings.TrimSpace(os.Getenv("AST_MCP_URL")); u != "" {
		return u
	}
	port, _ := strconv.Atoi(strings.TrimSpace(os.Getenv("AST_MCP_PORT")))
	if port <= 0 {
		port = defaultMCPPort
	}
	return "http://127.0.0.1:" + strconv.Itoa(port) + "/mcp"
}

// Call runs tool with args and decodes the JSON text of its first content item into out. A
// JSON-RPC error or an isError result is returned as an error.
func (c *Client) Call(ctx context.Context, tool string, args map[string]any, out any) error {
	body, err := json.Marshal(rpcRequest{JSONRPC: "2.0", ID: 1, Method: "tools/call", Params: callParams{Name: tool, Arguments: args}})
	if err != nil {
		return errs.WrapMessage("failed to encode tool call", err, "tool", tool)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.url, bytes.NewReader(body))
	if err != nil {
		return errs.WrapMessage("failed to build tool call request", err, "tool", tool)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	resp, err := c.http.Do(req)
	if err != nil {
		return errs.WrapMessage("failed to call tool", err, "tool", tool)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return errs.New("tool call returned an http error", "tool", tool, "status", resp.StatusCode)
	}
	var rpc rpcResponse
	if err := json.NewDecoder(io.LimitReader(resp.Body, maxResponseBytes)).Decode(&rpc); err != nil {
		return errs.WrapMessage("failed to decode tool call response", err, "tool", tool)
	}
	switch {
	case rpc.Error != nil:
		return errs.New("tool call failed", "tool", tool, "code", rpc.Error.Code, "detail", rpc.Error.Message)
	case rpc.Result == nil || len(rpc.Result.Content) == 0:
		return errs.New("tool call returned no content", "tool", tool)
	case rpc.Result.IsError:
		return errs.New("tool returned an error", "tool", tool, "detail", rpc.Result.Content[0].Text)
	}
	if err := json.Unmarshal([]byte(rpc.Result.Content[0].Text), out); err != nil {
		return errs.WrapMessage("failed to decode tool result", err, "tool", tool)
	}
	return nil
}
