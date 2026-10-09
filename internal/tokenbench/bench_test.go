package tokenbench

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"

	astcontext "github.com/coma-toast/ast-context-cache/internal/context"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/handoff"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/coma-toast/ast-context-cache/internal/mcp"
	"github.com/coma-toast/ast-context-cache/internal/search"
	"github.com/coma-toast/ast-context-cache/internal/tokens"
	"github.com/coma-toast/ast-context-cache/internal/watcher"
)

const (
	scenariosFile = "scenarios.yaml"
	baselineFile  = "baseline.json"
	// projectToken and homeToken replace the temp project root and HOME in responses before
	// counting tokens and comparing runs, so the numbers do not depend on the temp path.
	projectToken = "<project>"
	homeToken    = "<home>"
)

// handleToolCall starts a watcher on project_path; stop it before dbtest closes the pools.
func init() {
	dbtest.WaitFor(watcher.StopAll)
}

// scenario is one benchmark query. Strings in args may use ${project}, ${session} (the run's
// session id), ${setup_session}, and variables saved by setup calls. AST_TOKENBENCH_DUMP=<dir>
// writes each scenario's two normalized responses there.
type scenario struct {
	Name     string         `yaml:"name"`
	Tool     string         `yaml:"tool"`
	Args     map[string]any `yaml:"args"`
	Expect   []string       `yaml:"expect"`
	Negative bool           `yaml:"negative"`
	Setup    []setupCall    `yaml:"setup"`
}

// setupCall runs once before a scenario's measured calls, under ${setup_session}. save maps a
// variable name to a top-level string field of the response.
type setupCall struct {
	Tool string            `yaml:"tool"`
	Args map[string]any    `yaml:"args"`
	Save map[string]string `yaml:"save"`
}

// outcome is one scenario's measurement, and one row of baseline.json.
type outcome struct {
	Name          string  `json:"name"`
	Tool          string  `json:"tool"`
	Tokens        int     `json:"tokens"`
	Results       int     `json:"results"`
	Recall        float64 `json:"recall"`
	Deterministic bool    `json:"deterministic"`
	// NoMatch is set on negative scenarios: whether the query came back with no results.
	NoMatch *bool `json:"no_match,omitempty"`
}

type baseline struct {
	Scenarios []outcome `json:"scenarios"`
}

// bench is the server and fixture the scenarios run against.
type bench struct {
	t       *testing.T
	srv     *httptest.Server
	project string
	home    string
	nextID  int
	// tag prefixes session ids, so repeated runs of a scenario (the sweep) start undeduped.
	tag string
}

func TestTokenBench(t *testing.T) {
	scenarios := loadScenarios(t)
	b := newBench(t)
	base := loadBaseline(t)
	if os.Getenv("AST_RELEVANCE_SWEEP") == "1" {
		b.sweep(scenarios, base)
		return
	}
	var got []outcome
	for _, sc := range scenarios {
		got = append(got, b.run(sc))
	}
	if os.Getenv("AST_TOKENBENCH_VERBOSE") == "1" || testing.Verbose() {
		printTable(got, base)
	}
	// Updating accepts the new numbers, recall drops included.
	if os.Getenv("AST_TOKENBENCH_UPDATE") == "1" {
		raw, err := json.MarshalIndent(baseline{Scenarios: got}, "", "  ")
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(baselineFile, append(raw, '\n'), 0o644))
		return
	}
	for _, o := range got {
		if prev, ok := base[o.Name]; ok {
			assert.GreaterOrEqual(t, o.Recall, prev.Recall-1e-9, "scenario %s: recall dropped below the baseline", o.Name)
		}
	}
}

func loadScenarios(t *testing.T) []scenario {
	t.Helper()
	raw, err := os.ReadFile(scenariosFile)
	require.NoError(t, err)
	var scenarios []scenario
	require.NoError(t, yaml.Unmarshal(raw, &scenarios))
	seen := map[string]bool{}
	for _, sc := range scenarios {
		require.False(t, seen[sc.Name], "duplicate scenario %s", sc.Name)
		seen[sc.Name] = true
	}
	return scenarios
}

func loadBaseline(t *testing.T) map[string]outcome {
	t.Helper()
	out := map[string]outcome{}
	raw, err := os.ReadFile(baselineFile)
	if os.IsNotExist(err) {
		return out
	}
	require.NoError(t, err)
	var bl baseline
	require.NoError(t, json.Unmarshal(raw, &bl))
	for _, o := range bl.Scenarios {
		out[o.Name] = o
	}
	return out
}

// newBench copies the fixture to a temp dir, indexes and embeds it with the hash embedder,
// and starts the MCP server.
func newBench(t *testing.T) *bench {
	t.Helper()
	home := dbtest.Init(t)
	search.Cache.Unload()
	t.Cleanup(search.Cache.Unload)
	root := filepath.Join(t.TempDir(), "fixture")
	require.NoError(t, os.CopyFS(root, os.DirFS("testdata/fixture")))
	root, err := filepath.EvalSymlinks(root)
	require.NoError(t, err)
	// A fixed mtime, older than the index time, keeps the watcher's catch-up from re-indexing
	// the copy while scenarios run.
	fixed := time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC)
	require.NoError(t, filepath.WalkDir(root, func(path string, _ os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		return os.Chtimes(path, fixed, fixed)
	}))
	home, err = filepath.EvalSymlinks(home)
	require.NoError(t, err)
	prevEmb := mcp.GetEmbedder()
	emb := embedder.NewHashEmbedder(search.VectorDims)
	prevCtxEmb := astcontext.Emb
	mcp.SetEmbedder(emb)
	astcontext.Emb = emb
	t.Cleanup(func() {
		mcp.SetEmbedder(prevEmb)
		astcontext.Emb = prevCtxEmb
	})
	_, err = indexer.IndexDirectory(root, root)
	require.NoError(t, err)
	indexer.EmbedDirectorySymbols(emb, root, root)
	ctx, cancel := context.WithCancel(context.Background())
	prevHandoff := handoff.Default()
	handoff.Start(ctx, emb)
	t.Cleanup(func() {
		cancel()
		handoff.SetDefault(prevHandoff)
	})
	prevCfg := mcp.GetConfig()
	mcp.SetConfig(mcp.DefaultConfig())
	srv := httptest.NewServer(mcp.NewHandler())
	t.Cleanup(func() {
		srv.Close()
		mcp.SetConfig(prevCfg)
	})
	return &bench{t: t, srv: srv, project: root, home: home}
}

// run measures one scenario: setup once, then the call twice under different session ids so
// session dedup cannot change the second response.
func (b *bench) run(sc scenario) outcome {
	vars := map[string]string{"${project}": b.project, "${setup_session}": "tb-" + b.tag + sc.Name + "-setup"}
	for _, call := range sc.Setup {
		out, isErr := b.call(call.Tool, expand(call.Args, vars))
		require.False(b.t, isErr, "%s setup %s: %s", sc.Name, call.Tool, out)
		saved := decodeObject(out)
		for name, field := range call.Save {
			v, _ := saved[field].(string)
			require.NotEmpty(b.t, v, "%s setup %s: no %q in %s", sc.Name, call.Tool, field, out)
			vars["${"+name+"}"] = v
		}
		if call.Tool == "store_memory" {
			ref, _ := saved["ref"].(string)
			b.waitMemoryVector(ref)
		}
	}
	var texts [2]string
	for i, suffix := range []string{"a", "b"} {
		sid := "tb-" + b.tag + sc.Name + "-" + suffix
		vars["${session}"] = sid
		out, isErr := b.call(sc.Tool, expand(sc.Args, vars))
		require.False(b.t, isErr, "%s: %s", sc.Name, out)
		texts[i] = b.normalize(out, sid)
	}
	if dir := os.Getenv("AST_TOKENBENCH_DUMP"); dir != "" {
		require.NoError(b.t, os.WriteFile(filepath.Join(dir, sc.Name+".a.txt"), []byte(texts[0]), 0o644))
		require.NoError(b.t, os.WriteFile(filepath.Join(dir, sc.Name+".b.txt"), []byte(texts[1]), 0o644))
	}
	o := outcome{Name: sc.Name, Tool: sc.Tool, Tokens: tokens.Count(texts[0]), Deterministic: texts[0] == texts[1]}
	hits, n := b.hits(texts[0])
	o.Results = n
	if sc.Negative {
		noMatch := n == 0
		o.NoMatch = &noMatch
	}
	o.Recall = recall(sc.Expect, hits, texts[0])
	return o
}

// call sends a tools/call over HTTP and returns content[0].text and isError.
func (b *bench) call(tool string, args map[string]any) (string, bool) {
	b.nextID++
	body, err := json.Marshal(map[string]any{"jsonrpc": "2.0", "id": b.nextID, "method": "tools/call", "params": map[string]any{"name": tool, "arguments": args}})
	require.NoError(b.t, err)
	resp, err := http.Post(b.srv.URL, "application/json", bytes.NewReader(body))
	require.NoError(b.t, err)
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	require.NoError(b.t, err)
	var rpc struct {
		Result struct {
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
			IsError bool `json:"isError"`
		} `json:"result"`
		Error any `json:"error"`
	}
	require.NoError(b.t, json.Unmarshal(raw, &rpc), string(raw))
	require.Nil(b.t, rpc.Error, "%s: %s", tool, raw)
	require.NotEmpty(b.t, rpc.Result.Content, "%s: %s", tool, raw)
	return rpc.Result.Content[0].Text, rpc.Result.IsError
}

// waitMemoryVector waits for store_memory's background embed of ref, so recall scenarios see
// the same vectors on every run.
func (b *bench) waitMemoryVector(ref string) {
	conn, err := db.IndexReader()
	require.NoError(b.t, err)
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		var n int
		require.NoError(b.t, conn.QueryRow("SELECT COUNT(*) FROM vectors WHERE source_file = ?", "mem:"+ref).Scan(&n))
		if n > 0 {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	b.t.Fatalf("memory %s was never embedded", ref)
}

// randomRef matches the random hex part of mem_, hof_, hft_, and ctx_ refs.
var randomRef = regexp.MustCompile(`\b(mem|hof|hft|ctx)_[0-9a-f]+`)

// normalize replaces the temp paths and the run's session id with fixed tokens, and zeroes
// random ref ids (keeping their length), so token counts do not drift between runs.
func (b *bench) normalize(text, sid string) string {
	text = strings.NewReplacer(b.project, projectToken, b.home, homeToken, sid, "${session}").Replace(text)
	return randomRef.ReplaceAllStringFunc(text, func(ref string) string {
		prefix, hex, _ := strings.Cut(ref, "_")
		return prefix + "_" + strings.Repeat("0", len(hex))
	})
}

// hits returns "relpath:name" keys for the response's results, and how many results it has.
// Results are the first array found under results, chunks, symbols, or lines.
func (b *bench) hits(text string) (map[string]bool, int) {
	obj := decodeObject(text)
	out := map[string]bool{}
	for _, key := range []string{"results", "chunks", "symbols", "lines"} {
		arr, ok := obj[key].([]any)
		if !ok {
			continue
		}
		parentFile, _ := obj["file"].(string)
		for _, item := range arr {
			m, ok := item.(map[string]any)
			if !ok {
				continue
			}
			file, _ := m["file"].(string)
			if file == "" {
				file = parentFile
			}
			file = strings.TrimPrefix(strings.TrimPrefix(file, projectToken), "/")
			for _, nameKey := range []string{"name", "qualified_name"} {
				if name, _ := m[nameKey].(string); name != "" {
					out[file+":"+name] = true
				}
			}
		}
		return out, len(arr)
	}
	return out, 0
}

// recall is the share of expect found. "text:<s>" expects match a substring of the response;
// others are "relpath:symbol" result keys. No expectations count as full recall.
func recall(expect []string, hits map[string]bool, text string) float64 {
	if len(expect) == 0 {
		return 1
	}
	found := 0
	for _, e := range expect {
		if s, ok := strings.CutPrefix(e, "text:"); ok {
			if strings.Contains(text, s) {
				found++
			}
			continue
		}
		if hits[e] {
			found++
		}
	}
	return float64(found) / float64(len(expect))
}

func decodeObject(text string) map[string]any {
	var obj map[string]any
	if json.Unmarshal([]byte(text), &obj) != nil {
		return map[string]any{}
	}
	return obj
}

// expand substitutes vars into every string in args, recursively.
func expand(args map[string]any, vars map[string]string) map[string]any {
	pairs := make([]string, 0, len(vars)*2)
	for k, v := range vars {
		pairs = append(pairs, k, v)
	}
	r := strings.NewReplacer(pairs...)
	var walk func(v any) any
	walk = func(v any) any {
		switch x := v.(type) {
		case string:
			return r.Replace(x)
		case map[string]any:
			out := make(map[string]any, len(x))
			for k, val := range x {
				out[k] = walk(val)
			}
			return out
		case []any:
			out := make([]any, len(x))
			for i, val := range x {
				out[i] = walk(val)
			}
			return out
		}
		return v
	}
	out := make(map[string]any, len(args))
	for k, v := range args {
		out[k] = walk(v)
	}
	return out
}

func printTable(got []outcome, base map[string]outcome) {
	fmt.Printf("\n%-38s %-20s %7s %7s %7s %5s %6s %4s %s\n", "scenario", "tool", "tokens", "base", "delta", "res", "recall", "det", "neg")
	total, baseTotal := 0, 0
	for _, o := range got {
		baseCol, delta := "-", "new"
		if prev, ok := base[o.Name]; ok {
			baseCol, delta = fmt.Sprint(prev.Tokens), fmt.Sprintf("%+d", o.Tokens-prev.Tokens)
			baseTotal += prev.Tokens
		}
		total += o.Tokens
		neg := ""
		if o.NoMatch != nil {
			neg = "miss"
			if *o.NoMatch {
				neg = "ok"
			}
		}
		fmt.Printf("%-38s %-20s %7d %7s %7s %5d %6.2f %4v %s\n", o.Name, o.Tool, o.Tokens, baseCol, delta, o.Results, o.Recall, o.Deterministic, neg)
	}
	fmt.Printf("%-38s %-20s %7d %7d %+7d\n\n", "TOTAL", "", total, baseTotal, total-baseTotal)
}

// sweepGrid is the relevance threshold grid AST_RELEVANCE_SWEEP=1 runs.
var sweepGrid = struct{ minRelative, vectorMin, coverageMin []float64 }{
	minRelative: []float64{0.35, 0.5, 0.6, 0.7, 0.75, 0.8},
	vectorMin:   []float64{0.05, 0.1, 0.2, 0.22, 0.25, 0.45},
	coverageMin: []float64{0.34, 0.5, 1},
}

// sweepTools are the tools the relevance floor applies to.
var sweepTools = map[string]bool{"get_context_capsule": true, "search_semantic": true, "retrieve": true}

// sweep runs the code-search scenarios once per grid setting and prints, per setting, the
// expected hits found, scenarios whose recall fell below the baseline (lost), negatives
// flagged as no-match, and total tokens. Pick the setting with lost 0, then the fewest tokens.
func (b *bench) sweep(scenarios []scenario, base map[string]outcome) {
	var code []scenario
	for _, sc := range scenarios {
		if sweepTools[sc.Tool] && len(sc.Setup) == 0 {
			code = append(code, sc)
		}
	}
	fmt.Printf("\n%-8s %-8s %-8s %9s %5s %9s %7s\n", "min_rel", "vec_min", "cov_min", "expected", "lost", "negatives", "tokens")
	run := 0
	for _, minRel := range sweepGrid.minRelative {
		for _, vecMin := range sweepGrid.vectorMin {
			for _, covMin := range sweepGrid.coverageMin {
				b.t.Setenv("AST_RELEVANCE_MIN_RELATIVE", fmt.Sprint(minRel))
				b.t.Setenv("AST_RELEVANCE_VECTOR_MIN", fmt.Sprint(vecMin))
				b.t.Setenv("AST_RELEVANCE_COVERAGE_MIN", fmt.Sprint(covMin))
				run++
				b.tag = fmt.Sprintf("sweep%d-", run)
				found, expected, lost, flagged, negatives, total := 0.0, 0, 0, 0, 0, 0
				for _, sc := range code {
					o := b.run(sc)
					total += o.Tokens
					if sc.Negative {
						negatives++
						if *o.NoMatch {
							flagged++
						}
						continue
					}
					found += o.Recall * float64(len(sc.Expect))
					expected += len(sc.Expect)
					if prev, ok := base[sc.Name]; ok && o.Recall < prev.Recall-1e-9 {
						lost++
					}
				}
				fmt.Printf("%-8g %-8g %-8g %5.0f/%-3d %5d %5d/%-3d %7d\n", minRel, vecMin, covMin, found, expected, lost, flagged, negatives, total)
			}
		}
	}
}
