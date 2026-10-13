package mcp

import (
	"database/sql"
	"encoding/json"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/memory"
)

func setupMemoryToolTest(t *testing.T) {
	t.Helper()
	t.Setenv("AST_MCP_TIER", "complete")
	origCfg := GetConfig()
	SetConfig(DefaultConfig())
	t.Cleanup(func() { SetConfig(origCfg) })
	dbtest.Init(t)
}

// callTool runs a tools/call through the JSON-RPC handler and returns the
// decoded tool payload plus the MCP isError flag. The code tools default to text, so it asks
// them for output=json unless the call sets output.
func callTool(t *testing.T, name string, arguments map[string]interface{}) (map[string]interface{}, bool) {
	t.Helper()
	if _, ok := arguments["output"]; codeTools[name] && !ok {
		withJSON := make(map[string]interface{}, len(arguments)+1)
		for k, v := range arguments {
			withJSON[k] = v
		}
		withJSON["output"] = outputJSON
		arguments = withJSON
	}
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

// BF-5: as_of is normalized to SQLite's datetime form, and a bad value is an error rather than
// a text comparison that silently matches the wrong rows.
func TestRecallMemoryAsOf(t *testing.T) {
	setupMemoryToolTest(t)
	storeSessionRules(t, "s-asof", 1)
	out, isErr := callTool(t, "recall_memory", map[string]interface{}{"session_id": "s-asof", "as_of": "not a time"})
	assert.True(t, isErr)
	assert.Contains(t, out["error"], "invalid timestamp")
	out, isErr = callTool(t, "recall_memory", map[string]interface{}{"session_id": "s-asof", "as_of": "2999-01-01T00:00:00Z"})
	require.False(t, isErr, out)
	assert.Len(t, out["lines"], 1)
	out, isErr = callTool(t, "recall_memory", map[string]interface{}{"session_id": "s-asof", "as_of": "2000-01-01"})
	require.False(t, isErr, out)
	assert.Empty(t, out["lines"])
}

// MEM-5: store_memory accepts optional valid_at and invalid_at world-time anchors.
// When valid_at and invalid_at are set on a fact, as_of queries use those world-time
// columns instead of system valid_from/valid_until. as_of_system=true uses system time.
func TestRecallMemoryAsOfWorldTime(t *testing.T) {
	setupMemoryToolTest(t)

	// Store a fact with world-time anchors using RFC3339 format.
	out1, isErr := callTool(t, "store_memory", map[string]interface{}{
		"kind":       "fact",
		"session_id": "s-world",
		"subject":    "world-time-fact",
		"predicate":  "is",
		"object":     "active now",
		"valid_at":   "2026-01-01T00:00:00Z",
		"invalid_at": "2026-06-01T00:00:00Z",
		"expires_at": "30",
	})
	require.False(t, isErr, out1)

	// as_of within the window (March 1 2026, between Jan 1 and June 1) → should return the fact
	out, isErr := callTool(t, "recall_memory", map[string]interface{}{
		"session_id": "s-world",
		"as_of":      "2026-03-01",
	})
	require.False(t, isErr, out)
	assert.Len(t, out["lines"], 1, "as_of within [valid_at, invalid_at] should return the fact")

	// as_of after the window (July 1 2026, after invalid_at) → should not return
	out, isErr = callTool(t, "recall_memory", map[string]interface{}{
		"session_id": "s-world",
		"as_of":      "2026-07-01",
	})
	require.False(t, isErr, out)
	assert.Empty(t, out["lines"], "as_of after invalid_at should not return the fact")

	// as_of_system=true uses system time (valid_from/valid_until) instead of world time.
	// The fact was just stored, so valid_from is now and valid_until is empty.
	// With as_of=2999 (future), the fact's valid_from <= as_of → returned.
	out, isErr = callTool(t, "recall_memory", map[string]interface{}{
		"session_id":   "s-world",
		"as_of":        "2999-01-01T00:00:00Z",
		"as_of_system": true,
	})
	require.False(t, isErr, out)
	assert.Len(t, out["lines"], 1, "as_of_system=true should use system time and return the fact")
}

// BF-4: store_memory supersedes by default and keeps both facts when told not to.
func TestStoreMemoryInvalidatePreviousDefault(t *testing.T) {
	setupMemoryToolTest(t)
	fact := func(object string, extra map[string]interface{}) map[string]interface{} {
		args := map[string]interface{}{"kind": "fact", "session_id": "s-inv", "subject": "editor", "object": object}
		for k, v := range extra {
			args[k] = v
		}
		out, isErr := callTool(t, "store_memory", args)
		require.False(t, isErr, out)
		return out
	}
	first := fact("vim", nil)
	out := fact("emacs", nil)
	assert.Equal(t, []string{first["ref"].(string)}, toStrings(out["invalidated_refs"]))
	out = fact("helix", map[string]interface{}{"invalidate_previous": false})
	assert.Empty(t, toStrings(out["invalidated_refs"]))
}

// BF-13: forget_memory all=true needs a scope or confirm=true; the server's default project
// does not count as a scope.
func TestForgetMemoryAllRequiresScopeOrConfirm(t *testing.T) {
	setupMemoryToolTest(t)
	mine := storeSessionRules(t, "s-all-mine", 2)
	other := storeSessionRules(t, "s-all-other", 1)
	out, isErr := callTool(t, "forget_memory", map[string]interface{}{"all": true})
	assert.True(t, isErr)
	assert.Contains(t, out["error"], "all=true without scope requires confirm=true")
	out, isErr = callTool(t, "forget_memory", map[string]interface{}{"all": true, "session_id": "s-all-mine"})
	require.False(t, isErr, out)
	assert.Equal(t, float64(len(mine)), out["invalidated_refs"])
	out, isErr = callTool(t, "forget_memory", map[string]interface{}{"all": true, "confirm": true})
	require.False(t, isErr, out)
	assert.Equal(t, float64(len(other)), out["invalidated_refs"])
}

// TestParseTTLTable drives parseTTL with a table of inputs and expected outputs/errors
// using the errs package (MEM-8).
func TestParseTTLTable(t *testing.T) {
	cases := []struct {
		name                             string
		input                            string
		wantDays, wantHours, wantMinutes int
		wantErr                          bool
	}{
		{"empty string", "", 0, 0, 0, false},
		{"plain number (days)", "42", 42, 0, 0, false},
		{"with d suffix", "42d", 42, 0, 0, false},
		{"with h suffix", "42h", 0, 42, 0, false},
		{"with m suffix", "42m", 0, 0, 42, false},
		{"combination 1h30m", "1h30m", 0, 1, 30, false},
		{"combination 2d3h", "2d3h", 2, 3, 0, false},
		{"unknown unit", "42x", 0, 0, 0, true},
		{"negative days", "-5d", -5, 0, 0, false},
		{"zero hours", "0h", 0, 0, 0, false},
		{"one minute", "1m", 0, 0, 1, false},
		{"trailing whitespace", " 5d ", 5, 0, 0, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gotDays, gotHours, gotMinutes, err := parseTTL(tc.input)
			if tc.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.Equal(t, tc.wantDays, gotDays)
			assert.Equal(t, tc.wantHours, gotHours)
			assert.Equal(t, tc.wantMinutes, gotMinutes)
		})
	}
}

// Issue #3: MEM-3 history test for make→bazel→ninja chain
func TestHistoryMakeBazelNinjaChain(t *testing.T) {
	setupMemoryToolTest(t)

	// Store the initial fact: build tool is make
	resMake, err := memory.Store(memory.StoreInput{
		Kind:               memory.KindFact,
		Scope:              memory.ScopeSession,
		SessionID:          "history-test-session",
		Subject:            "build tool",
		Predicate:          "is",
		Object:             "make",
		InvalidatePrevious: false,
	})
	require.NoError(t, err)

	// Store the second fact: build tool is bazel (supersedes make)
	resBazel, err := memory.Store(memory.StoreInput{
		Kind:               memory.KindFact,
		Scope:              memory.ScopeSession,
		SessionID:          "history-test-session",
		Subject:            "build tool",
		Predicate:          "is",
		Object:             "bazel",
		InvalidatePrevious: true, // This will supersede the active fact
	})
	require.NoError(t, err)

	// Store the third fact: build tool is ninja (supersedes bazel)
	resNinja, err := memory.Store(memory.StoreInput{
		Kind:               memory.KindFact,
		Scope:              memory.ScopeSession,
		SessionID:          "history-test-session",
		Subject:            "build tool",
		Predicate:          "is",
		Object:             "ninja",
		InvalidatePrevious: true, // This will supersede the active fact
	})
	require.NoError(t, err)

	// Verify the invalidation chain was set up correctly via direct SQL
	var makeSupersededBy, makeReason sql.NullString
	err = db.ContextDB.QueryRow(`SELECT superseded_by, invalidated_reason FROM structured_memory WHERE ref = ?`, resMake.Ref).Scan(&makeSupersededBy, &makeReason)
	require.NoError(t, err)
	assert.Equal(t, resBazel.Ref, makeSupersededBy.String, "make should be superseded by bazel")
	assert.Equal(t, memory.ReasonContradiction, makeReason.String, "make should be invalidated by contradiction")

	var bazelSupersededBy, bazelReason sql.NullString
	err = db.ContextDB.QueryRow(`SELECT superseded_by, invalidated_reason FROM structured_memory WHERE ref = ?`, resBazel.Ref).Scan(&bazelSupersededBy, &bazelReason)
	require.NoError(t, err)
	assert.Equal(t, resNinja.Ref, bazelSupersededBy.String, "bazel should be superseded by ninja")
	assert.Equal(t, memory.ReasonContradiction, bazelReason.String, "bazel should be invalidated by contradiction")

	// ninja should be active (not superseded)
	var ninjaSupersededBy, ninjaReason sql.NullString
	err = db.ContextDB.QueryRow(`SELECT superseded_by, invalidated_reason FROM structured_memory WHERE ref = ?`, resNinja.Ref).Scan(&ninjaSupersededBy, &ninjaReason)
	require.NoError(t, err)
	assert.Empty(t, ninjaSupersededBy.String, "ninja should be active (not superseded)")
	assert.Empty(t, ninjaReason.String, "ninja should be active (no invalidation reason)")

	// Test history by ref starting from the first fact (make)
	hist, err := memory.History(memory.HistoryInput{
		Ref:          resMake.Ref,
		IncludeWorld: true,
	})
	require.NoError(t, err)
	require.NotNil(t, hist)

	// Should have 3 entries in the chain: make → bazel → ninja
	assert.Len(t, hist.Entries, 3, "history chain should have 3 entries")

	// Entries should be in chronological order (oldest first)
	assert.Equal(t, resMake.Ref, hist.Entries[0].Ref, "first entry should be make")
	assert.Equal(t, resBazel.Ref, hist.Entries[1].Ref, "second entry should be bazel")
	assert.Equal(t, resNinja.Ref, hist.Entries[2].Ref, "third entry should be ninja")

	// Verify the invalidation reasons and superseded_by links
	assert.Equal(t, memory.ReasonContradiction, hist.Entries[0].InvalidatedReason, "make should be invalidated by contradiction")
	assert.Equal(t, resBazel.Ref, hist.Entries[0].SupersededBy, "make should point to bazel as superseded_by")

	assert.Equal(t, memory.ReasonContradiction, hist.Entries[1].InvalidatedReason, "bazel should be invalidated by contradiction")
	assert.Equal(t, resNinja.Ref, hist.Entries[1].SupersededBy, "bazel should point to ninja as superseded_by")

	assert.Empty(t, hist.Entries[2].InvalidatedReason, "ninja should be active (no invalidation reason)")
	assert.Empty(t, hist.Entries[2].SupersededBy, "ninja should be active (no superseded_by)")

	// Test recall_memory with history=true via the tool interface
	out, isErr := callTool(t, "recall_memory", map[string]interface{}{
		"history":    true,
		"ref":        resMake.Ref,
		"session_id": "history-test-session",
	})
	require.False(t, isErr, "recall_memory with history should not error")
	require.NotNil(t, out["entries"], "should have entries in response")

	// The tool should return the same chain
	entries := out["entries"].([]interface{})
	assert.Len(t, entries, 3, "tool history should return 3 entries")

	// Check that we got the refs in order
	assert.Equal(t, resMake.Ref, entries[0].(map[string]interface{})["ref"], "first tool entry should be make")
	assert.Equal(t, resBazel.Ref, entries[1].(map[string]interface{})["ref"], "second tool entry should be bazel")
	assert.Equal(t, resNinja.Ref, entries[2].(map[string]interface{})["ref"], "third tool entry should be ninja")
}

// Issue #4: MEM-2 dashboard 90-day prune behavior test
// Tests that PruneSuperseded (90-day default) retains rows with invalidated_reason
// and deletes legacy rows (without reason) older than 90 days.
func TestPruneSuperseded90DayBehavior(t *testing.T) {
	setupMemoryToolTest(t)

	now := time.Now().UTC()

	// Row A: legacy row (no invalidated_reason), invalidated 100 days ago → retained by no-op prune
	old100 := now.AddDate(0, 0, -100).Format("2006-01-02 15:04:05")
	resA, err := memory.Store(memory.StoreInput{
		Kind:      memory.KindFact,
		Scope:     memory.ScopeSession,
		SessionID: "prune-test-session",
		Subject:   "legacy",
		Predicate: "is",
		Object:    "old",
	})
	require.NoError(t, err)
	_, err = db.ContextDB.Exec(`UPDATE structured_memory SET valid_until = ?, invalidated_reason = NULL WHERE ref = ?`, old100, resA.Ref)
	require.NoError(t, err)

	// Row B: invalidated row with reason, 10 days ago → retained
	old10 := now.AddDate(0, 0, -10).Format("2006-01-02 15:04:05")
	resB, err := memory.Store(memory.StoreInput{
		Kind:      memory.KindFact,
		Scope:     memory.ScopeSession,
		SessionID: "prune-test-session",
		Subject:   "invalidated",
		Predicate: "is",
		Object:    "with reason",
	})
	require.NoError(t, err)
	_, err = db.ContextDB.Exec(`UPDATE structured_memory SET valid_until = ?, invalidated_reason = ? WHERE ref = ?`, old10, memory.ReasonUserForget, resB.Ref)
	require.NoError(t, err)

	// Row C: legacy row (no invalidated_reason), invalidated 400 days ago → retained by no-op prune
	old400 := now.AddDate(0, 0, -400).Format("2006-01-02 15:04:05")
	resC, err := memory.Store(memory.StoreInput{
		Kind:      memory.KindFact,
		Scope:     memory.ScopeSession,
		SessionID: "prune-test-session",
		Subject:   "legacy2",
		Predicate: "is",
		Object:    "very old",
	})
	require.NoError(t, err)
	_, err = db.ContextDB.Exec(`UPDATE structured_memory SET valid_until = ?, invalidated_reason = NULL WHERE ref = ?`, old400, resC.Ref)
	require.NoError(t, err)

	// Row D: invalidated row with reason, 20 days ago → retained
	old20 := now.AddDate(0, 0, -20).Format("2006-01-02 15:04:05")
	resD, err := memory.Store(memory.StoreInput{
		Kind:      memory.KindFact,
		Scope:     memory.ScopeSession,
		SessionID: "prune-test-session",
		Subject:   "invalidated2",
		Predicate: "is",
		Object:    "recent reason",
	})
	require.NoError(t, err)
	_, err = db.ContextDB.Exec(`UPDATE structured_memory SET valid_until = ?, invalidated_reason = ? WHERE ref = ?`, old20, memory.ReasonUserForget, resD.Ref)
	require.NoError(t, err)

	// Verify rows before prune: all 4 rows exist
	var cnt int
	err = db.ContextDB.QueryRow(`SELECT COUNT(*) FROM structured_memory WHERE session_id = 'prune-test-session'`).Scan(&cnt)
	require.NoError(t, err)
	assert.Equal(t, 4, cnt, "should have 4 rows before prune")

	// Run PruneSuperseded with default 90-day window — now a no-op per MEM-2
	nPruned, err := memory.PruneSuperseded(0) // 0 means default 90
	require.NoError(t, err)
	assert.Equal(t, int64(0), nPruned, "PruneSuperseded is a no-op, should prune 0 rows")

	// Verify all rows remain after no-op prune
	var cntAfter int
	err = db.ContextDB.QueryRow(`SELECT COUNT(*) FROM structured_memory WHERE session_id = 'prune-test-session'`).Scan(&cntAfter)
	require.NoError(t, err)
	assert.Equal(t, 4, cntAfter, "all 4 rows should remain after no-op prune")

	// Specifically check: A, B, C, D should all still be present
	var refA string
	err = db.ContextDB.QueryRow(`SELECT ref FROM structured_memory WHERE ref = ?`, resA.Ref).Scan(&refA)
	require.NoError(t, err, "row A should still exist after no-op prune")

	var refB string
	err = db.ContextDB.QueryRow(`SELECT ref FROM structured_memory WHERE ref = ?`, resB.Ref).Scan(&refB)
	require.NoError(t, err, "row B should still exist")

	var refC string
	err = db.ContextDB.QueryRow(`SELECT ref FROM structured_memory WHERE ref = ?`, resC.Ref).Scan(&refC)
	require.NoError(t, err, "row C should still exist")

	var refD string
	err = db.ContextDB.QueryRow(`SELECT ref FROM structured_memory WHERE ref = ?`, resD.Ref).Scan(&refD)
	require.NoError(t, err, "row D should still exist")

	// Verify PruneArchived still works for rows with reasons (365-day default)
	_, err = memory.PruneArchived(0) // 0 means default 365
	assert.NoError(t, err)
}

// TestStoreMemoryConflictResponseAlwaysIncludesFields tests MEM-12:
// store_memory for facts always returns possible_conflicts ([] when none)
// and conflict_method, even when no conflicts are found.
func TestStoreMemoryConflictResponseAlwaysIncludesFields(t *testing.T) {
	setupMemoryToolTest(t)
	out, isErr := callTool(t, "store_memory", map[string]interface{}{
		"kind":       "fact",
		"session_id": "s-conf",
		"subject":    "editor.test",
		"predicate":  "uses",
		"object":     "vim",
	})
	require.False(t, isErr, out)
	assert.Contains(t, out, "possible_conflicts", "store_memory must always return possible_conflicts")
	assert.Contains(t, out, "conflict_method", "store_memory must always return conflict_method")
	assert.IsType(t, []interface{}{}, out["possible_conflicts"], "possible_conflicts must be an array")
	assert.Contains(t, []string{"fts", "embed", "none"}, out["conflict_method"], "conflict_method must be fts, embed, or none")
}
