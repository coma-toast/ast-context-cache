package mcp

import (
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	astcontext "github.com/coma-toast/ast-context-cache/internal/context"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/flags"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/coma-toast/ast-context-cache/internal/search"
	"github.com/coma-toast/ast-context-cache/internal/tokens"
	"github.com/coma-toast/ast-context-cache/internal/watcher"
)

// setupCodeToolTest indexes and embeds src as project/name with the hash embedder, so all four
// code tools (search_semantic needs an embedder) answer through handleToolCall.
func setupCodeToolTest(t *testing.T, name, src string) (project, file string) {
	t.Helper()
	t.Setenv("AST_MCP_TIER", "complete")
	origCfg := GetConfig()
	SetConfig(DefaultConfig())
	t.Cleanup(func() { SetConfig(origCfg) })
	dbtest.Init(t)
	// The relevance floor's VectorMin default is tuned for real embedders; 0.22 suits the hash one.
	require.NoError(t, db.SetSetting("relevance_vector_min", "0.22"))
	project = t.TempDir()
	file = filepath.Join(project, name)
	require.NoError(t, os.WriteFile(file, []byte(src), 0o644))
	// An mtime before the index time keeps the watcher's catch-up from reindexing the file
	// (and dropping its vectors) on the first tool call.
	past := time.Now().Add(-time.Hour)
	require.NoError(t, os.Chtimes(file, past, past))
	_, _, _, err := indexer.IndexFile(file, project)
	require.NoError(t, err)
	t.Cleanup(func() { watcher.DeleteWatcher(project) })
	t.Cleanup(db.FlushWriteBuffers)
	emb := embedder.NewHashEmbedder(search.VectorDims)
	prevEmb, prevCtxEmb := GetEmbedder(), astcontext.Emb
	SetEmbedder(emb)
	astcontext.Emb = emb
	t.Cleanup(func() {
		SetEmbedder(prevEmb)
		astcontext.Emb = prevCtxEmb
	})
	indexer.EmbedDirectorySymbols(emb, project, project)
	return project, file
}

// codeToolCall is a raw tools/call result: content[0].text, _meta, and isError.
type codeToolCall struct {
	text  string
	meta  map[string]any
	isErr bool
}

func callCodeTool(t *testing.T, name string, arguments map[string]any) codeToolCall {
	t.Helper()
	rec := httptest.NewRecorder()
	handleToolCall(rec, JSONRPCRequest{JSONRPC: "2.0", ID: 1, Method: "tools/call", Params: map[string]any{"name": name, "arguments": arguments}})
	var resp struct {
		Result struct {
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
			IsError bool           `json:"isError"`
			Meta    map[string]any `json:"_meta"`
		} `json:"result"`
	}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp), rec.Body.String())
	require.NotEmpty(t, resp.Result.Content, rec.Body.String())
	return codeToolCall{text: resp.Result.Content[0].Text, meta: resp.Result.Meta, isErr: resp.Result.IsError}
}

// codeToolArgs are each code tool's arguments against the twoClientsPy fixture.
func codeToolArgs(project, file string) map[string]map[string]any {
	return map[string]map[string]any{
		"get_context_capsule": {"query": "load_model", "project_path": project},
		"search_semantic":     {"query": "load_model", "project_path": project},
		"get_file_context":    {"file": file, "project_path": project},
		"retrieve":            {"query": "load_model", "project_path": project, "include_docs": false},
	}
}

func withArgs(base map[string]any, kv ...any) map[string]any {
	out := make(map[string]any, len(base)+len(kv)/2)
	for k, v := range base {
		out[k] = v
	}
	for i := 0; i+1 < len(kv); i += 2 {
		out[kv[i].(string)] = kv[i+1]
	}
	return out
}

var locationLine = regexp.MustCompile(`^clients\.py:\d+-\d+ \w+ \S+( -?[0-9.e+-]+)?$`)

// AC4: text, json and locations for all four code tools; text is the default.
func TestCodeToolOutputs(t *testing.T) {
	project, file := setupCodeToolTest(t, "clients.py", twoClientsPy)
	resultsKey := map[string]string{"get_context_capsule": "results", "search_semantic": "results", "get_file_context": "symbols", "retrieve": "chunks"}
	firstKey := map[string]string{"get_context_capsule": "results", "search_semantic": "results", "get_file_context": "symbols", "retrieve": "context"}
	for tool, args := range codeToolArgs(project, file) {
		t.Run(tool, func(t *testing.T) {
			target := "load_model"
			if tool == "get_file_context" {
				target = "clients.py"
			}
			text := callCodeTool(t, tool, withArgs(args, "output", outputText))
			require.False(t, text.isErr, text.text)
			assert.True(t, strings.HasPrefix(text.text, tool+" \""+target+"\" · "), text.text)
			assert.NotContains(t, text.text, "code_script_hints")
			if tool == "retrieve" {
				assert.Contains(t, text.text, "# Retrieved Context")
				assert.Regexp(t, `(?m)^clients\.py:\d+-\d+ method \S+load_model `, text.text)
			} else {
				assert.Regexp(t, `(?m)^### clients\.py:\d+-\d+ \w+ \S+`, text.text)
				assert.Contains(t, text.text, "```python\n")
			}
			assert.Equal(t, text.text, callCodeTool(t, tool, args).text, "text is the default output")

			js := callCodeTool(t, tool, withArgs(args, "output", outputJSON))
			require.False(t, js.isErr, js.text)
			assert.True(t, strings.HasPrefix(js.text, `{"`+firstKey[tool]+`":`), "results come first: %s", js.text)
			var body map[string]any
			require.NoError(t, json.Unmarshal([]byte(js.text), &body))
			assert.NotEmpty(t, body[resultsKey[tool]])

			locs := callCodeTool(t, tool, withArgs(args, "output", outputLocations))
			require.False(t, locs.isErr, locs.text)
			lines := strings.Split(strings.TrimRight(locs.text, "\n"), "\n")
			require.Greater(t, len(lines), 1, locs.text)
			assert.True(t, strings.HasPrefix(lines[0], tool+" \""+target+"\" · "), lines[0])
			for _, l := range lines[1:] {
				assert.Regexp(t, locationLine, l)
			}
			assert.Less(t, tokens.Count(locs.text), tokens.Count(text.text)+1, "locations are no larger than text")
		})
	}
}

// testLastQueryTokensQuery reads the newest query log row's output and used tokens for a tool.
const testLastQueryTokensQuery = `SELECT output_tokens, tokens_used, result_chars FROM queries WHERE tool_name = ? ORDER BY id DESC LIMIT 1`

// The query log counts the text each code tool actually returned, not its JSON body.
func TestCodeToolQueryLogCountsRenderedText(t *testing.T) {
	project, file := setupCodeToolTest(t, "clients.py", twoClientsPy)
	for tool, args := range codeToolArgs(project, file) {
		t.Run(tool, func(t *testing.T) {
			for _, output := range []string{outputText, outputLocations, outputJSON} {
				got := callCodeTool(t, tool, withArgs(args, "output", output))
				require.False(t, got.isErr, got.text)
				db.FlushWriteBuffers()
				var outTokens, used, chars int
				require.NoError(t, db.DB.QueryRow(testLastQueryTokensQuery, tool).Scan(&outTokens, &used, &chars))
				assert.Equal(t, tokens.Count(got.text), outTokens, output)
				assert.Equal(t, outTokens, used, output)
				assert.Equal(t, len(got.text), chars, output)
			}
		})
	}
}

func TestCodeToolOutputJSONWhenTextFormatOff(t *testing.T) {
	project, file := setupCodeToolTest(t, "clients.py", twoClientsPy)
	setFlagEnvs(t, map[string]string{"AST_FEATURE_TEXT_FORMAT": "false"})
	out := callCodeTool(t, "get_file_context", codeToolArgs(project, file)["get_file_context"])
	assert.True(t, strings.HasPrefix(out.text, `{"symbols":`), out.text)
}

func TestCodeToolRejectsUnknownOutput(t *testing.T) {
	project, file := setupCodeToolTest(t, "clients.py", twoClientsPy)
	out := callCodeTool(t, "get_file_context", withArgs(codeToolArgs(project, file)["get_file_context"], "output", "yaml"))
	assert.True(t, out.isErr)
	assert.Contains(t, out.text, "output must be text, json or locations")
}

func TestCodeToolHintsInMetaForText(t *testing.T) {
	body := map[string]any{"query": "q", "results": []any{}, "code_script_hints": []any{map[string]any{"script_id": "x"}}}
	data, err := json.Marshal(body)
	require.NoError(t, err)
	text, meta := renderCodeResult("get_context_capsule", nil, "", "", codeCall{output: outputText}, data, nil, nil)
	assert.NotContains(t, text, "code_script_hints")
	assert.Equal(t, body["code_script_hints"], meta[metaHintsKey])
	text, meta = renderCodeResult("get_context_capsule", nil, "", "", codeCall{output: outputJSON}, data, nil, nil)
	assert.Contains(t, text, `"code_script_hints":`)
	assert.Nil(t, meta[metaHintsKey])
}

// AC2: identical retrieve calls return identical bytes, with no timings in the body.
func TestRetrieveBytesStableNoTimings(t *testing.T) {
	project, file := setupCodeToolTest(t, "clients.py", twoClientsPy)
	for _, output := range []string{outputText, outputJSON} {
		args := withArgs(codeToolArgs(project, file)["retrieve"], "output", output)
		first, second := callCodeTool(t, "retrieve", args), callCodeTool(t, "retrieve", args)
		require.False(t, first.isErr, first.text)
		assert.Equal(t, first.text, second.text, output)
		for _, k := range []string{"search_time_ms", "code_retrieve_ms", "dedup_budget_ms", "bm25_candidates", "timings"} {
			assert.NotContains(t, first.text, k, output)
		}
	}
	capsuleArgs := withArgs(codeToolArgs(project, file)["get_context_capsule"], "output", outputJSON)
	first := callCodeTool(t, "get_context_capsule", capsuleArgs)
	assert.Equal(t, first.text, callCodeTool(t, "get_context_capsule", capsuleArgs).text)
	assert.NotContains(t, first.text, `"pipeline"`)
	assert.Contains(t, first.meta[metaStatsKey], "pipeline")
}

// AC2: retrieve timings go to _meta, or back into the body as timings with debug=true or
// feature_stable_responses off.
func TestRetrieveTimingsInMeta(t *testing.T) {
	project, file := setupCodeToolTest(t, "clients.py", twoClientsPy)
	args := withArgs(codeToolArgs(project, file)["retrieve"], "output", outputJSON)
	out := callCodeTool(t, "retrieve", args)
	stats, ok := out.meta[metaStatsKey].(map[string]any)
	require.True(t, ok, "%v", out.meta)
	for _, k := range []string{"search_time_ms", "code_retrieve_ms", "docs_retrieve_ms", "dedup_budget_ms", "bm25_candidates"} {
		assert.Contains(t, stats, k)
	}

	debug := callCodeTool(t, "retrieve", withArgs(args, "debug", true))
	assert.NotContains(t, debug.meta, metaStatsKey)
	var body map[string]any
	require.NoError(t, json.Unmarshal([]byte(debug.text), &body))
	assert.Contains(t, body["timings"], "search_time_ms")
	assert.Contains(t, body["stats"], "bm25_candidates")

	setFlagEnvs(t, map[string]string{"AST_FEATURE_STABLE_RESPONSES": "false"})
	off := callCodeTool(t, "retrieve", args)
	assert.NotContains(t, off.meta, metaStatsKey)
	assert.Contains(t, off.text, `"timings":{"search_time_ms":`)
}

// manyFunctionsPy is a module of n small functions, about 40 tokens each in full mode.
func manyFunctionsPy(n int) string {
	var b strings.Builder
	for i := 0; i < n; i++ {
		fmt.Fprintf(&b, "def handler_%03d(request, retries=3):\n    \"\"\"Handle request number %d.\"\"\"\n    return request.process(%d, retries)\n\n\n", i, i, i)
	}
	return b.String()
}

var offloadStubLine = regexp.MustCompile(`^\[(ctx_[0-9a-f]{12}) get_file_context big\.py, [0-9.]+k? tok — head shown, fetch_context for all\]$`)

// AC9: a ~6k-token result comes back as a head under the threshold with the stub, and
// fetch_context returns the full result.
func TestResultOffloadHeadAndFetch(t *testing.T) {
	project, _ := setupCodeToolTest(t, "big.py", manyFunctionsPy(150))
	require.True(t, flags.Enabled(flags.KeyResultOffload))
	file := filepath.Join(project, "big.py")
	args := map[string]any{"file": file, "project_path": project, "mode": "full"}
	full := callCodeTool(t, "get_file_context", withArgs(args, "offload", false))
	require.False(t, full.isErr, full.text)
	require.Greater(t, tokens.Count(full.text), 5000)

	out := callCodeTool(t, "get_file_context", args)
	require.False(t, out.isErr, out.text)
	stub, head, _ := strings.Cut(out.text, "\n")
	m := offloadStubLine.FindStringSubmatch(stub)
	require.NotNil(t, m, stub)
	assert.LessOrEqual(t, tokens.Count(head), defaultResultOffloadThreshold-offloadStubReserve)
	assert.LessOrEqual(t, tokens.Count(out.text), defaultResultOffloadThreshold)
	assert.True(t, strings.HasPrefix(head, `get_file_context "big.py" · `), head)
	assert.Contains(t, head, "### big.py:1-3 function handler_000 (full)")
	assert.NotContains(t, head, "handler_149")

	fetched := mustCall(t, "fetch_context", map[string]any{"refs": []any{m[1]}})
	notes := resultMaps(fetched["notes"])
	require.Len(t, notes, 1, "%v", fetched)
	assert.Equal(t, strings.TrimSpace(full.text), notes[0]["content"])
	assert.Equal(t, "offload-", notes[0]["session_id"].(string)[:len("offload-")], "no session_id uses the synthetic offload session")

	js := callCodeTool(t, "get_file_context", withArgs(args, "output", outputJSON, "session_id", "s-offload"))
	require.True(t, strings.HasPrefix(js.text, `{"offloaded":"[ctx_`), js.text)
	var body map[string]any
	require.NoError(t, json.Unmarshal([]byte(js.text), &body))
	assert.NotEmpty(t, body["symbols"])
	assert.Less(t, len(resultMaps(body["symbols"])), 150)
	assert.LessOrEqual(t, tokens.Count(js.text), defaultResultOffloadThreshold)

	t.Setenv("AST_RESULT_OFFLOAD_THRESHOLD", "100000")
	assert.Equal(t, full.text, callCodeTool(t, "get_file_context", args).text, "under the threshold nothing is offloaded")
}

func TestOrderedBodyKeyOrder(t *testing.T) {
	data, err := json.Marshal(orderedBody(map[string]any{"zeta": 1, "tokens_used": 2, "results": []int{}, "query": "q", "alpha": 3, "code_script_hints": nil}, codeJSONKeyOrder))
	require.NoError(t, err)
	assert.Equal(t, `{"results":[],"query":"q","tokens_used":2,"code_script_hints":null,"alpha":3,"zeta":1}`, string(data))
}
