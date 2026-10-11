package mcp

import (
	"encoding/json"
	"strconv"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/context"
	"github.com/coma-toast/ast-context-cache/internal/contextnotes"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/flags"
	"github.com/coma-toast/ast-context-cache/internal/render"
	"github.com/coma-toast/ast-context-cache/internal/tokens"
)

const (
	// metaStatsKey holds a code tool's run-to-run variable stats in the result's _meta (DR-2).
	metaStatsKey = "ast-context-cache/stats"
	// metaHintsKey holds code_script_hints in _meta when the output is not JSON.
	metaHintsKey = "hints"

	outputText      = "text"
	outputJSON      = "json"
	outputLocations = "locations"

	defaultResultOffloadThreshold = 2000
	// offloadStubReserve is the headroom kept under the threshold for the offload stub line.
	offloadStubReserve = 80
)

// toolMeta is the MCP result's _meta object.
type toolMeta map[string]any

// codeTools are the tools whose responses are rendered as text, JSON or locations, and may be
// offloaded.
var codeTools = map[string]bool{"get_context_capsule": true, "search_semantic": true, "get_file_context": true, "retrieve": true}

// metaStats are the body keys each code tool moves into _meta with feature_stable_responses.
var metaStats = map[string][]string{
	"get_context_capsule": {"pipeline", "cache_hit", "input_tokens", "output_tokens"},
	"search_semantic":     {"total_vectors", "cache_hit"},
}

// retrieveMetaStats are the retrieve stats keys that move into _meta.
var retrieveMetaStats = []string{"bm25_candidates", "vector_candidates", "hybrid_after_fuse"}

// codeCall is a code tool call's rendering arguments.
type codeCall struct {
	output         string
	debug, offload bool
}

// parseCodeCall reads output (default text with feature_text_format, json otherwise), debug
// and offload (default true).
func parseCodeCall(args map[string]any) (codeCall, error) {
	cc := codeCall{output: strArg(args, "output"), debug: boolArg(args, "debug"), offload: true}
	if off, ok := args["offload"].(bool); ok {
		cc.offload = off
	}
	switch cc.output {
	case "":
		cc.output = outputJSON
		if flags.Enabled(flags.KeyTextFormat) {
			cc.output = outputText
		}
	case outputText, outputJSON, outputLocations:
	default:
		return cc, errs.NewCode(errs.CodeInvalidInput, "output must be text, json or locations", "output", cc.output)
	}
	return cc, nil
}

// resultOffloadThreshold is the token count above which a code tool result is offloaded.
func resultOffloadThreshold() int {
	return db.SettingInt("result_offload_threshold", "AST_RESULT_OFFLOAD_THRESHOLD", defaultResultOffloadThreshold)
}

// codeResult is a code tool's response ready to render: the parsed body and its ranked results.
type codeResult struct {
	tool, target, root, resultsKey string
	body                           map[string]any
	results                        []map[string]any
	// retrieve is set for retrieve, whose context is reassembled for a head.
	retrieve *retrieveView
}

// renderCodeResult turns a code tool's JSON body into the content text and _meta: variable
// stats move to _meta (feature_stable_responses, unless debug), the body renders as
// cc.output, and a rendering over the offload threshold (feature_result_offload) is stored as
// an offload note and replaced by a head plus its stub. timings and view are retrieve's.
func renderCodeResult(tool string, args map[string]any, projectPath, sid string, cc codeCall, body []byte, timings *retrieveTimings, view *retrieveView) (string, toolMeta) {
	var parsed map[string]any
	if json.Unmarshal(body, &parsed) != nil {
		return string(body), nil
	}
	meta := toolMeta{}
	if stats := moveStats(tool, parsed, cc.debug, timings); len(stats) > 0 {
		meta[metaStatsKey] = stats
	}
	if hints, ok := parsed["code_script_hints"]; ok && cc.output != outputJSON {
		meta[metaHintsKey] = hints
		delete(parsed, "code_script_hints")
	}
	r := newCodeResult(tool, projectPath, parsed, view)
	text := r.render(cc.output, len(r.results), "")
	if cc.offload && flags.Enabled(flags.KeyResultOffload) {
		text = r.offload(text, cc.output, args, projectPath, sid)
	}
	if len(meta) == 0 {
		return text, nil
	}
	return text, meta
}

// moveStats moves tool's variable stats out of body and returns them; with debug or
// feature_stable_responses off they stay, and retrieve's timings go into the body.
func moveStats(tool string, body map[string]any, debug bool, timings *retrieveTimings) map[string]any {
	if debug || !flags.Enabled(flags.KeyStableResponses) {
		if timings != nil {
			body["timings"] = timings
		}
		return nil
	}
	stats := map[string]any{}
	for _, k := range metaStats[tool] {
		if v, ok := body[k]; ok {
			stats[k] = v
			delete(body, k)
		}
	}
	if s, ok := body["stats"].(map[string]any); ok {
		for _, k := range retrieveMetaStats {
			if v, ok := s[k]; ok {
				stats[k] = v
				delete(s, k)
			}
		}
	}
	if timings != nil {
		stats["search_time_ms"], stats["code_retrieve_ms"] = timings.SearchTimeMs, timings.CodeRetrieveMs
		stats["docs_retrieve_ms"], stats["dedup_budget_ms"] = timings.DocsRetrieveMs, timings.DedupBudgetMs
	}
	return stats
}

func newCodeResult(tool, projectPath string, body map[string]any, view *retrieveView) codeResult {
	r := codeResult{tool: tool, root: projectPath, resultsKey: "results", body: body, retrieve: view}
	r.target, _ = body["query"].(string)
	switch tool {
	case "get_file_context":
		r.resultsKey = "symbols"
		r.target, _ = body["file"].(string)
	case "retrieve":
		r.resultsKey = "chunks"
	}
	r.results = resultMaps(body[r.resultsKey])
	if view != nil && len(view.chunks) != len(r.results) {
		r.retrieve = nil
	}
	return r
}

// render renders the first n results as output; stub is set on a JSON head.
func (r codeResult) render(output string, n int, stub string) string {
	switch {
	case output == outputJSON:
		return r.json(n, stub)
	case r.tool == "retrieve":
		return r.retrieveText(output, n)
	case output == outputLocations:
		return render.Locations(r.response(n))
	default:
		return render.Text(r.response(n))
	}
}

// json is the ordered JSON body with only the first n results, plus the offload stub if set.
func (r codeResult) json(n int, stub string) string {
	body := make(map[string]any, len(r.body)+1)
	for k, v := range r.body {
		body[k] = v
	}
	if n < len(r.results) {
		body[r.resultsKey] = r.results[:n]
		if r.retrieve != nil {
			body["context"] = r.retrieve.context(n)
		}
	}
	if stub != "" {
		body["offloaded"] = stub
	}
	data, err := json.Marshal(orderedBody(body, codeJSONKeyOrder))
	if err != nil {
		logger.Warn("Failed to marshal code tool result", "tool", r.tool, "error", err)
		return "{}"
	}
	return string(data)
}

// retrieveText is the summary line, the assembled context (text only), then one line per chunk.
func (r codeResult) retrieveText(output string, n int) string {
	resp := r.response(n)
	lines := render.LocationLines(resp)
	if output == outputLocations {
		return render.Header(resp, lines) + lines
	}
	ctx, _ := r.body["context"].(string)
	if n < len(r.results) && r.retrieve != nil {
		ctx = r.retrieve.context(n)
	}
	text := strings.TrimRight(ctx, "\n") + "\n"
	if lines != "" {
		text += "\n" + lines
	}
	return render.Header(resp, text) + text
}

// response is the render.Response for the first n results.
func (r codeResult) response(n int) render.Response {
	resp := render.Response{Tool: r.tool, Query: r.target, Root: r.root, Total: intOf(r.body["total"])}
	file, _ := r.body["file"].(string)
	for _, res := range r.results[:n] {
		resp.Results = append(resp.Results, renderable(r.tool, file, res))
	}
	var withheld context.Withheld
	if decodeField(r.body["withheld"], &withheld) {
		resp.Withheld, resp.WithheldTokens = withheld.Count, withheld.Tokens
	}
	switch nm := r.body["no_match"].(type) {
	case bool:
		if nm {
			hint, _ := r.body["hint"].(string)
			resp.NoMatch = &render.NoMatch{Hint: hint}
		}
	case map[string]any:
		var noMatch render.NoMatch
		if decodeField(nm, &noMatch) {
			resp.NoMatch = &noMatch
		}
	}
	decodeField(r.body["collapsed"], &resp.Collapsed)
	if d := intOf(r.body["deduped"]); d > 0 {
		resp.Notes = append(resp.Notes, "deduped "+strconv.Itoa(d)+" already returned this session")
	} else if s, ok := r.body["stats"].(map[string]any); ok && intOf(s["deduped"]) > 0 {
		resp.Notes = append(resp.Notes, "deduped "+strconv.Itoa(intOf(s["deduped"]))+" already returned this session")
	}
	if h, ok := r.body["handoff"]; ok {
		if data, err := json.Marshal(h); err == nil {
			resp.Notes = append(resp.Notes, "handoff: "+string(data))
		}
	}
	return resp
}

// renderable adapts a result map for the renderer: file context symbols get the response's
// file, and doc hits map their path, title and content onto file, name and source.
func renderable(tool, file string, res map[string]any) map[string]any {
	if title, ok := res["title"].(string); ok && res["name"] == nil {
		return map[string]any{"file": res["path"], "kind": "doc", "name": title, "source": res["content"], "mode": "doc", "score": res["score"]}
	}
	if tool != "get_file_context" || file == "" {
		return res
	}
	out := make(map[string]any, len(res)+1)
	for k, v := range res {
		out[k] = v
	}
	out["file"] = file
	return out
}

// offload stores text as an offload note when it is over the threshold, returning a head of
// the results in rank order that fits under it, with the stub line (a JSON head carries the
// stub as "offloaded"). On a store failure text is returned whole.
func (r codeResult) offload(text, output string, args map[string]any, projectPath, sid string) string {
	threshold := resultOffloadThreshold()
	total := tokens.Count(text)
	if total <= threshold {
		return text
	}
	session := sid
	if session == "" {
		session = "offload-" + time.Now().UTC().Format(time.DateOnly)
	}
	stored, err := contextnotes.StoreOffload(session, projectPath, text, r.tool, args, r.target)
	if err != nil {
		logger.Warn("Failed to offload tool result", "tool", r.tool, "tokens", total, "error", err)
		return text
	}
	stub := contextnotes.OffloadStub(stored.Ref, r.tool, r.target, total)
	jsonStub := ""
	if output == outputJSON {
		jsonStub = stub
	}
	limit := threshold - offloadStubReserve
	head := r.render(output, 0, jsonStub)
	for n := 1; n <= len(r.results); n++ {
		next := r.render(output, n, jsonStub)
		if tokens.Count(next) > limit {
			break
		}
		head = next
	}
	if output == outputJSON {
		return head
	}
	return stub + "\n" + head
}

// decodeField decodes a parsed JSON value into out, reporting whether it was present and valid.
func decodeField(v any, out any) bool {
	if v == nil {
		return false
	}
	data, err := json.Marshal(v)
	if err != nil {
		return false
	}
	return json.Unmarshal(data, out) == nil
}

func intOf(v any) int {
	f, _ := v.(float64)
	return int(f)
}
