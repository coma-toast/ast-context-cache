package mcp

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/handoff"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/coma-toast/ast-context-cache/internal/trail"
)

// scenarioFiles is the AC39 fixture project: twelve small Python and Go files, one topic each,
// so a query by symbol name lands in one file.
var scenarioFiles = map[string]string{
	"auth.py":      "def verify_token(token):\n    return token.startswith(\"tok_\")\n\n\ndef issue_token(user):\n    return \"tok_\" + user\n",
	"loader.py":    "class ModelLoader:\n    def load_model(self, name):\n        return name\n\n    def unload_model(self, name):\n        return None\n",
	"retry.py":     "def retry_backoff(attempt):\n    return 2 ** attempt\n\n\ndef jitter_delay(delay):\n    return delay / 2\n",
	"cache.py":     "class LRUCache:\n    def get_entry(self, key):\n        return None\n\n    def put_entry(self, key, value):\n        pass\n",
	"config.py":    "def load_config(path):\n    return {}\n\n\ndef merge_defaults(cfg):\n    return cfg\n",
	"metrics.py":   "def record_latency(ms):\n    pass\n\n\ndef export_metrics():\n    return []\n",
	"server.go":    "package app\n\nfunc StartServer(addr string) error {\n\treturn nil\n}\n\nfunc handleHealth() string {\n\treturn \"ok\"\n}\n",
	"router.go":    "package app\n\nfunc NewRouter() int {\n\treturn 0\n}\n\nfunc routeRequest(path string) string {\n\treturn path\n}\n",
	"queue.go":     "package app\n\nfunc Enqueue(job string) {\n}\n\nfunc Dequeue() string {\n\treturn \"\"\n}\n",
	"storage.go":   "package app\n\nfunc SaveBlob(b []byte) error {\n\treturn nil\n}\n\nfunc LoadBlob(id string) []byte {\n\treturn nil\n}\n",
	"scheduler.go": "package app\n\nfunc ScheduleJob(name string) int {\n\treturn 0\n}\n\nfunc CancelJob(id int) {\n}\n",
	"logger.go":    "package app\n\nfunc NewLogger() int {\n\treturn 0\n}\n\nfunc RotateLogs(n int) int {\n\treturn n\n}\n",
}

// scenarioStep is one planned search: a query, or for get_file_context the project-relative file.
type scenarioStep struct {
	tool  string
	query string
}

func capsuleStep(q string) scenarioStep  { return scenarioStep{tool: "get_context_capsule", query: q} }
func semanticStep(q string) scenarioStep { return scenarioStep{tool: "search_semantic", query: q} }
func fileStep(rel string) scenarioStep   { return scenarioStep{tool: "get_file_context", query: rel} }
func retrieveStep(q string) scenarioStep { return scenarioStep{tool: "retrieve", query: q} }

// coverKey is what a child compares a planned step against a trail entry by: tool and
// normalized query. The open digest carries no filters, so they are left out.
func (s scenarioStep) coverKey() string {
	return s.tool + "|" + trail.NormalizeQuery(s.query)
}

func (s scenarioStep) args(project, sid string) map[string]any {
	a := map[string]any{"project_path": project, "session_id": sid}
	switch s.tool {
	case "get_file_context":
		a["file"] = filepath.Join(project, s.query)
	case "retrieve":
		a["query"], a["include_docs"] = s.query, false
	default:
		a["query"] = s.query
	}
	return a
}

// scenarioPlans are the parent's searches and each child's planned searches. Children repeat
// some parent searches exactly (with case and whitespace changes, OP-9 normalization), reread
// files the parent explored under a different query (OB-1 rule b), and search new ground;
// child 3 also repeats a sibling's search (SP-6).
var (
	scenarioParentPlan = []scenarioStep{
		capsuleStep("load_model"), capsuleStep("retry_backoff"), capsuleStep("verify_token"), capsuleStep("LRUCache"),
		fileStep("loader.py"), fileStep("retry.py"), retrieveStep("load_config"), retrieveStep("StartServer"),
		capsuleStep("Enqueue"), fileStep("server.go"), semanticStep("token verification"),
	}
	scenarioChildPlans = [][]scenarioStep{
		{capsuleStep("Load_Model"), capsuleStep("  retry_backoff  "), fileStep("loader.py"), capsuleStep("SaveBlob"), capsuleStep("LoadBlob"), fileStep("storage.go")},
		{capsuleStep("verify_token"), retrieveStep("load_config"), fileStep("auth.py"), capsuleStep("ScheduleJob"), fileStep("scheduler.go"), semanticStep("Token  Verification")},
		{capsuleStep("LRUCache"), retrieveStep("StartServer"), fileStep("server.go"), capsuleStep("NewRouter"), fileStep("router.go"), capsuleStep("SaveBlob")},
		{capsuleStep("Enqueue"), fileStep("retry.py"), capsuleStep("load_model"), capsuleStep("NewLogger"), fileStep("logger.go"), fileStep("queue.go")},
	}
)

// searchLog collects the trail entries recorded per session, with their pre-dedup candidate
// keys, which the trail never stores (OB-1 rule b needs them for the offline baseline).
type searchLog struct {
	mu        sync.Mutex
	bySession map[string][]trail.Entry
}

var (
	scenarioTrail     atomic.Pointer[searchLog]
	scenarioTrailOnce sync.Once
)

// captureSearches routes every recorded search to a new log until the test ends. trail has no
// unsubscribe, so one subscription forwards to the current log.
func captureSearches(t *testing.T) *searchLog {
	t.Helper()
	scenarioTrailOnce.Do(func() {
		trail.Subscribe(func(e trail.Entry) {
			if l := scenarioTrail.Load(); l != nil {
				l.add(e)
			}
		})
	})
	l := &searchLog{bySession: map[string][]trail.Entry{}}
	scenarioTrail.Store(l)
	t.Cleanup(func() { scenarioTrail.CompareAndSwap(l, nil) })
	return l
}

func (l *searchLog) add(e trail.Entry) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.bySession[e.SessionID] = append(l.bySession[e.SessionID], e)
}

func (l *searchLog) entries(sid string) []trail.Entry {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]trail.Entry(nil), l.bySession[sid]...)
}

// parentView is the parent's snapshot as OB-1 matches against it: trail match keys and the
// explored manifest as trail hit refs.
type parentView struct {
	trail    map[string]bool
	manifest map[string]bool
}

func parentViewOf(t *testing.T, ref string) parentView {
	t.Helper()
	pv := parentView{trail: map[string]bool{}, manifest: map[string]bool{}}
	rows, err := db.ContextDB.Query(`SELECT section, item_key, COALESCE(label, ''), COALESCE(file_rel, ''), COALESCE(start_line, 0)
		FROM handoff_snapshot_items WHERE handoff_ref = ? AND section IN ('trail', 'manifest')`, ref)
	require.NoError(t, err)
	defer rows.Close()
	for rows.Next() {
		var section, key, label, file string
		var line int
		require.NoError(t, rows.Scan(&section, &key, &label, &file, &line))
		if section == string(handoff.SectionTrail) {
			pv.trail[key] = true
			continue
		}
		pv.manifest[trail.HitRef(file, label, line)] = true
	}
	require.NoError(t, rows.Err())
	return pv
}

// repeats counts searches that are repeats by OB-1: the search matches a parent trail entry, or
// at least half its pre-dedup candidates are in the parent's manifest.
func (pv parentView) repeats(entries []trail.Entry) int {
	n := 0
	for _, e := range entries {
		in := 0
		for _, k := range e.CandidateHits {
			if pv.manifest[k] {
				in++
			}
		}
		if pv.trail[e.MatchKey()] || (len(e.CandidateHits) > 0 && in*2 >= len(e.CandidateHits)) {
			n++
		}
	}
	return n
}

// runRate is one run's child search count and repeats.
type runRate struct {
	calls, repeats int
}

func (r runRate) rate() float64 {
	if r.calls == 0 {
		return 0
	}
	return float64(r.repeats) / float64(r.calls)
}

// setupScenario indexes the fixture project, turns every handoff flag on, and starts a handoff
// service as the Default.
func setupScenario(t *testing.T) string {
	t.Helper()
	t.Setenv("AST_MCP_TIER", "complete")
	origCfg := GetConfig()
	SetConfig(DefaultConfig())
	t.Cleanup(func() { SetConfig(origCfg) })
	dbtest.Init(t)
	project := t.TempDir()
	for name, src := range scenarioFiles {
		file := filepath.Join(project, name)
		require.NoError(t, os.WriteFile(file, []byte(src), 0o644))
		_, _, _, err := indexer.IndexFile(file, project)
		require.NoError(t, err, name)
	}
	t.Cleanup(db.FlushWriteBuffers)
	setFlagEnvs(t, map[string]string{
		"AST_FEATURE_HANDOFF": "true", "AST_FEATURE_HANDOFF_SCRATCHPAD": "true",
		"AST_FEATURE_HANDOFF_CLAIMS": "true", "AST_FEATURE_HANDOFF_LIVE_TRAIL": "true",
	})
	ctx, cancel := context.WithCancel(context.Background())
	prev := handoff.Default()
	handoff.Start(ctx, nil)
	t.Cleanup(func() {
		cancel()
		handoff.SetDefault(prev)
	})
	return project
}

// runnable drops search_semantic steps when no embedder is loaded (as in unit tests).
func runnable(plan []scenarioStep) []scenarioStep {
	if GetEmbedder() != nil {
		return plan
	}
	out := make([]scenarioStep, 0, len(plan))
	for _, s := range plan {
		if s.tool != "search_semantic" {
			out = append(out, s)
		}
	}
	return out
}

func runStep(t *testing.T, project, sid string, s scenarioStep) map[string]any {
	t.Helper()
	return mustCall(t, s.tool, s.args(project, sid))
}

// scenarioChild is a scripted child agent in the handoff run.
type scenarioChild struct {
	sid     string
	ref     string
	covered map[string]bool
	cursor  float64
	ran     int
	skipped int
}

// openScenarioChild opens ref and seeds the child's covered set from the digest's trail.
func openScenarioChild(t *testing.T, project, ref string) *scenarioChild {
	t.Helper()
	opened := mustCall(t, toolOpenHandoff, map[string]any{"handoff": ref, "project_path": project})
	c := &scenarioChild{ref: ref, covered: map[string]bool{}}
	c.sid, _ = opened["session_id"].(string)
	require.NotEmpty(t, c.sid)
	for _, e := range resultMaps(opened["trail"]) {
		tool, _ := e["tool"].(string)
		query, _ := e["query"].(string)
		c.covered[scenarioStep{tool: tool, query: query}.coverKey()] = true
	}
	return c
}

// run executes plan under the synthetic child policy. This policy is a stand-in for an agent
// that reads its digest, not evidence of real agent behavior (that is the AC41 real-host trial):
// before each planned search the child reads its siblings' new live trail entries (SP-5) and
// skips the search when the digest's parent trail or a sibling already covered it (same tool and
// normalized query); a search answered with parent_trail_match or sibling_trail_match covers
// its key for the rest of the run.
func (c *scenarioChild) run(t *testing.T, project string, plan []scenarioStep) {
	t.Helper()
	for _, s := range plan {
		read := mustCall(t, toolScratchpad, map[string]any{"action": "read", "session_id": c.sid, "since": c.cursor, "types": []any{"trail"}})
		for _, e := range resultMaps(read["entries"]) {
			if k := liveTrailCoverKey(e["text"]); k != "" {
				c.covered[k] = true
			}
		}
		c.cursor, _ = read["next_cursor"].(float64)
		if c.covered[s.coverKey()] {
			c.skipped++
			continue
		}
		out := runStep(t, project, c.sid, s)
		c.ran++
		if ann, ok := out["handoff"].(map[string]any); ok {
			if ann["parent_trail_match"] != nil || ann["sibling_trail_match"] != nil {
				c.covered[s.coverKey()] = true
			}
		}
	}
}

// liveTrailCoverKey parses a live trail entry's text, "<tool>: <normalized query> (<n> hits)".
func liveTrailCoverKey(v any) string {
	text, _ := v.(string)
	tool, rest, ok := strings.Cut(text, ": ")
	if !ok {
		return ""
	}
	i := strings.LastIndex(rest, " (")
	if i < 0 {
		return ""
	}
	return scenarioStep{tool: tool, query: rest[:i]}.coverKey()
}

func completeChild(t *testing.T, sid, status, summary string, words int) map[string]any {
	t.Helper()
	content := "FACT: " + sid + " finished its slice\n" + strings.Repeat("Detailed findings from the child's investigation. ", words)
	return mustCall(t, toolHandoff, map[string]any{"action": "complete", "session_id": sid, "content": content, "summary": summary, "status": status})
}

func sumQueryTokensSaved(t *testing.T, tool string) int {
	t.Helper()
	var n int
	require.NoError(t, db.DB.QueryRow(`SELECT COALESCE(SUM(tokens_saved), 0) FROM queries WHERE tool_name = ?`, tool).Scan(&n))
	return n
}

// TestHandoffScenario is AC39 end to end: a parent searches and hands off to three fresh
// children and one fork child, one child nests a handoff for a grandchild, the children share
// the scratchpad and a claim, everyone completes, and the parent collects the whole tree. The
// same child query plans run first under fresh unlinked sessions as the no-handoff baseline,
// and the child repeat-search rate (OB-1) must drop by at least half with handoffs.
func TestHandoffScenario(t *testing.T) {
	project := setupScenario(t)
	log := captureSearches(t)
	parent := "scenario-parent"
	parentPlan := runnable(scenarioParentPlan)
	for _, s := range parentPlan {
		out := runStep(t, project, parent, s)
		assert.NotContains(t, out, "handoff", "no annotations before the parent is in a tree")
	}
	require.Len(t, log.entries(parent), len(parentPlan))

	fresh := mustCall(t, toolHandoff, map[string]any{
		"action": "create", "session_id": parent, "project_path": project, "brief": "Map the storage, scheduling, and routing code",
		"label": "explore", "pointers": []any{map[string]any{"key": "loader.py|load_model", "note": "entry point"}},
	})
	fork := mustCall(t, toolHandoff, map[string]any{
		"action": "create", "session_id": parent, "project_path": project, "brief": "Check the queue and logging code", "label": "queue", "mode": "fork",
	})
	freshRef, _ := fresh["handoff"].(string)
	forkRef, _ := fork["handoff"].(string)
	require.Equal(t, fresh["tree_id"], fork["tree_id"], "one tree per root session")
	pv := parentViewOf(t, freshRef)
	require.Len(t, pv.trail, len(parentPlan))
	require.NotEmpty(t, pv.manifest)
	assert.Equal(t, pv, parentViewOf(t, forkRef), "both handoffs snapshot the same parent state")

	// Baseline: the same child plans in fresh sessions outside any tree, scored offline against
	// the parent's snapshot.
	var baseline runRate
	for i, plan := range scenarioChildPlans {
		sid := "scenario-baseline-c" + string(rune('1'+i))
		for _, s := range runnable(plan) {
			out := runStep(t, project, sid, s)
			assert.NotContains(t, out, "handoff", "baseline sessions are outside every tree")
		}
		entries := log.entries(sid)
		baseline.calls += len(entries)
		baseline.repeats += pv.repeats(entries)
	}

	// Handoff run: three fresh children and one fork child, run in turn under the child policy.
	children := make([]*scenarioChild, len(scenarioChildPlans))
	for i := range scenarioChildPlans {
		ref := freshRef
		if i == 3 {
			ref = forkRef
		}
		children[i] = openScenarioChild(t, project, ref)
		assert.Len(t, children[i].covered, len(parentPlan), "the digest lists every parent search")
	}
	c1, c2, c3, c4 := children[0], children[1], children[2], children[3]
	assert.Equal(t, []string{freshRef + ".c1", freshRef + ".c2", freshRef + ".c3", forkRef + ".c1"}, []string{c1.sid, c2.sid, c3.sid, c4.sid})
	for i, c := range children {
		c.run(t, project, runnable(scenarioChildPlans[i]))
	}
	assert.Equal(t, 4, c3.skipped, "child 3 skips 3 parent searches and its sibling's SaveBlob search")
	expanded := mustCall(t, toolOpenHandoff, map[string]any{"action": "expand", "handoff": freshRef, "session_id": c1.sid, "section": "pointer", "items": "all"})
	require.Len(t, expanded["items"], 1)

	// Scratchpad and claims between siblings (SP-1, SP-4, CL-2, CL-4, CL-5).
	mustCall(t, toolScratchpad, map[string]any{"action": "post", "session_id": c1.sid, "type": "finding", "text": "SaveBlob never checks the size", "refs": []any{"storage.go"}})
	mustCall(t, toolScratchpad, map[string]any{"action": "post", "session_id": c3.sid, "type": "dead_end", "text": "router.go has no middleware"})
	read := mustCall(t, toolScratchpad, map[string]any{"action": "read", "session_id": c2.sid, "types": []any{"finding", "dead_end"}})
	entries := resultMaps(read["entries"])
	require.Len(t, entries, 2)
	assert.Equal(t, c1.sid, entries[0]["author"])
	assert.Equal(t, c3.sid, entries[1]["author"])
	claim := mustCall(t, toolScratchpad, map[string]any{"action": "claim", "session_id": c1.sid, "key": "storage.go"})
	assert.Equal(t, "granted", claim["outcome"])
	claim = mustCall(t, toolScratchpad, map[string]any{"action": "claim", "session_id": c2.sid, "key": "./storage.go", "reason": "add a size check"})
	assert.Equal(t, "queued", claim["outcome"])
	assert.Equal(t, float64(1), claim["position"])
	release := mustCall(t, toolScratchpad, map[string]any{"action": "release", "session_id": c1.sid, "key": "storage.go"})
	assert.Equal(t, c2.sid, release["granted_to"])
	texts, isErr := callToolTexts(t, "index_status", map[string]any{"project_path": project, "session_id": c2.sid})
	require.False(t, isErr)
	require.Len(t, texts, 2)
	assert.Equal(t, "[claims_granted] storage.go (scratchpad)", texts[1])

	// Child 2 nests a handoff; the grandchild opens it, searches, and completes (HO-8, FI-5).
	nested := mustCall(t, toolHandoff, map[string]any{"action": "create", "session_id": c2.sid, "brief": "Trace CancelJob callers", "label": "cancel"})
	assert.Equal(t, float64(2), nested["depth"])
	assert.Equal(t, fresh["tree_id"], nested["tree_id"])
	nestedRef, _ := nested["handoff"].(string)
	grandchild := openScenarioChild(t, project, nestedRef)
	assert.NotEmpty(t, grandchild.covered, "the nested digest lists child 2's searches")
	grandchild.run(t, project, []scenarioStep{capsuleStep("CancelJob"), capsuleStep("ScheduleJob")})
	assert.Equal(t, 1, grandchild.ran, "child 2 already searched ScheduleJob")
	gcDone := completeChild(t, grandchild.sid, "done", "CancelJob has no callers", 60)
	assert.Equal(t, "done", gcDone["status"])
	assert.Contains(t, gcDone["stub"], nestedRef)

	// Everyone completes; child 2 still holds storage.go, which completion releases (RT-6).
	returnSaved := 0
	for _, c := range []struct {
		child   *scenarioChild
		status  string
		summary string
	}{
		{c1, "done", "storage writes are unchecked"},
		{c2, "done", "scheduling is fine; added a size check"},
		{c3, "partial", "routing mapped, middleware not found"},
		{c4, "done", "queue and logger are trivial"},
	} {
		done := completeChild(t, c.child.sid, c.status, c.summary, 80)
		assert.Equal(t, c.status, done["status"])
		saved, _ := done["tokens_saved"].(float64)
		assert.Positive(t, saved, "OB-3 for %s", c.child.sid)
		returnSaved += int(saved)
		if c.child == c2 {
			assert.Equal(t, []any{"storage.go"}, done["released_claims"])
		}
	}
	saved, _ := gcDone["tokens_saved"].(float64)
	returnSaved += int(saved)

	collected := mustCall(t, toolHandoff, map[string]any{"action": "collect", "session_id": parent, "recursive": true})
	type node struct {
		status string
		depth  float64
	}
	got := map[string]node{}
	for _, c := range resultMaps(collected["children"]) {
		sid, _ := c["session_id"].(string)
		status, _ := c["status"].(string)
		depth, _ := c["depth"].(float64)
		got[sid] = node{status, depth}
		assert.NotEmpty(t, c["result_ref"], sid)
		assert.NotEmpty(t, c["summary"], sid)
	}
	assert.Equal(t, map[string]node{
		c1.sid: {"done", 1}, c2.sid: {"done", 1}, c3.sid: {"partial", 1}, c4.sid: {"done", 1}, grandchild.sid: {"done", 2},
	}, got)
	status := mustCall(t, toolHandoff, map[string]any{"action": "status", "session_id": parent})
	assert.Equal(t, parent, status["root_session_id"])
	assert.Len(t, status["handoffs"], 3)
	assert.Zero(t, status["active_claims"])
	assert.Zero(t, status["queued_claims"])

	// OB-1: the handoff run scored offline, and the service's own per-child counters agree.
	var withHandoff runRate
	for _, c := range children {
		entries := log.entries(c.sid)
		require.Len(t, entries, c.ran, c.sid)
		withHandoff.calls += len(entries)
		withHandoff.repeats += pv.repeats(entries)
	}
	var calls, repeats int
	require.NoError(t, db.ContextDB.QueryRow(`SELECT SUM(search_calls), SUM(repeat_calls) FROM handoff_children WHERE child_session_id IN (?, ?, ?, ?)`,
		c1.sid, c2.sid, c3.sid, c4.sid).Scan(&calls, &repeats))
	assert.Equal(t, withHandoff, runRate{calls, repeats}, "handoff_children counters match OB-1 computed offline")
	var gcCalls int
	require.NoError(t, db.ContextDB.QueryRow(`SELECT search_calls FROM handoff_children WHERE child_session_id = ?`, grandchild.sid).Scan(&gcCalls))
	assert.Equal(t, 1, gcCalls)

	// OB-2 and OB-3 as the dashboard sums them from the query log.
	db.FlushWriteBuffers()
	handoffSaved := sumQueryTokensSaved(t, toolOpenHandoff)
	var available int
	require.NoError(t, db.ContextDB.QueryRow(`SELECT SUM(tokens_available - tokens_delivered) FROM handoff_children`).Scan(&available))
	assert.Positive(t, handoffSaved, "OB-2 handoff tokens saved")
	assert.Equal(t, available, handoffSaved, "logged open/expand savings sum to available − delivered")
	assert.Positive(t, returnSaved)
	assert.Equal(t, returnSaved, sumQueryTokensSaved(t, toolHandoff), "OB-3 return tokens saved")

	reduction := 1 - withHandoff.rate()/baseline.rate()
	t.Logf("AC39 repeat-search rate: baseline %d/%d = %.1f%%, handoff %d/%d = %.1f%% (reduction %.1f%%); "+
		"child searches skipped %d; handoff tokens saved %d; return tokens saved %d",
		baseline.repeats, baseline.calls, 100*baseline.rate(), withHandoff.repeats, withHandoff.calls, 100*withHandoff.rate(),
		100*reduction, c1.skipped+c2.skipped+c3.skipped+c4.skipped, handoffSaved, returnSaved)
	require.Positive(t, baseline.rate())
	assert.GreaterOrEqual(t, reduction, 0.5, "AC39: the child repeat-search rate drops by at least half")
}
