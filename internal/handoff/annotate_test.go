package handoff

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/trail"
)

func searchCounts(t *testing.T, sid SessionID) (searches, repeats int) {
	t.Helper()
	return count(t, `SELECT search_calls FROM handoff_children WHERE child_session_id = ?`, sid),
		count(t, `SELECT repeat_calls FROM handoff_children WHERE child_session_id = ?`, sid)
}

// AC3, AC6, OP-9, OP-10, OB-1: a fresh child's repeat of a parent search is flagged with the
// parent's hits, results the parent saw are marked, and repeats are counted.
func TestAnnotateParentMatchesAndRepeats(t *testing.T) {
	s := newTestService(t)
	now := time.Date(2026, 10, 5, 12, 0, 0, 0, time.UTC)
	setClock(t, now)
	project := indexFixture(t)
	parent := SessionID("parent-ann")
	returnSymbols(t, parent, project, "Alpha")
	trail.Record(trail.Entry{
		SessionID: string(parent), Tool: "search_semantic", Query: "Retry Backoff", HitCount: 7,
		TopHits: []string{"svc.go#Alpha@3"}, ProjectPath: project,
	})
	resp := mustCreate(t, s, CreateRequest{SessionID: parent, ProjectPath: project})
	o := mustOpen(t, s, resp.Ref, "")
	assert.Nil(t, s.Annotate("loner", SearchEvent{MatchKey: "x"}, nil), "outside a tree")

	ev := SearchEventFor(trail.Entry{
		Tool: "search_semantic", Query: "  retry   BACKOFF ", ProjectPath: project, HitCount: 2,
		CandidateHits: []string{"svc.go#Gamma@11", "svc.go#Beta@7"},
	})
	results := []map[string]any{
		{"file": "svc.go", "name": "Alpha", "start_line": 3},
		{"file": "svc.go", "name": "Beta", "start_line": float64(7)},
	}
	out := s.Annotate(o.SessionID, ev, results)
	require.NotNil(t, out)
	assert.Equal(t, map[string]any{"query": "Retry Backoff", "hit_count": 7, "zero_hit": false, "top_hits": []string{"svc.go#Alpha@3"}},
		out["parent_trail_match"])
	assert.NotContains(t, out, "sibling_trail_match")
	assert.Equal(t, true, results[0]["parent_explored"])
	assert.NotContains(t, results[1], "parent_explored")
	searches, repeats := searchCounts(t, o.SessionID)
	assert.Equal(t, []int{1, 1}, []int{searches, repeats}, "the first search is written at once; a trail match is a repeat")

	// No trail match, and half of the candidates are in the manifest: still a repeat (OB-1 b).
	other := SearchEventFor(trail.Entry{
		Tool: "search_semantic", Query: "something new", ProjectPath: project,
		CandidateHits: []string{"svc.go#Alpha@3", "svc.go#Beta@7"},
	})
	assert.Nil(t, s.Annotate(o.SessionID, other, nil))
	// A third of the candidates: not a repeat.
	third := SearchEventFor(trail.Entry{
		Tool: "search_semantic", Query: "else", ProjectPath: project,
		CandidateHits: []string{"svc.go#Alpha@3", "svc.go#Beta@7", "svc.go#Gamma@11"},
	})
	s.Annotate(o.SessionID, third, nil)
	searches, repeats = searchCounts(t, o.SessionID)
	assert.Equal(t, []int{1, 1}, []int{searches, repeats}, "coalesced within 2s")
	s.flushSearchCounters(o.SessionID)
	searches, repeats = searchCounts(t, o.SessionID)
	assert.Equal(t, []int{3, 2}, []int{searches, repeats})
	setClock(t, now.Add(3*time.Second))
	s.Annotate(o.SessionID, third, nil)
	searches, _ = searchCounts(t, o.SessionID)
	assert.Equal(t, 4, searches, "written through once the interval passed")

	// A fork child isn't marked: its dedup already skipped those results.
	fork := mustCreate(t, s, CreateRequest{SessionID: parent, ProjectPath: project, Mode: ModeFork})
	fo := mustOpen(t, s, fork.Ref, "")
	results = []map[string]any{{"file": "svc.go", "name": "Alpha", "start_line": 3}}
	out = s.Annotate(fo.SessionID, ev, results)
	assert.Contains(t, out, "parent_trail_match")
	assert.NotContains(t, results[0], "parent_explored")
}

// SP-6: a search matching another tree session's live trail entry names that session.
func TestAnnotateSiblingTrailMatch(t *testing.T) {
	s := newTestService(t)
	resp := mustCreate(t, s, CreateRequest{SessionID: "parent-sib", ProjectPath: "/p"})
	a := mustOpen(t, s, resp.Ref, "")
	b := mustOpen(t, s, resp.Ref, "")
	ev := SearchEventFor(trail.Entry{Tool: "search_semantic", Query: "retry backoff"})
	refs := `{"match_key":"` + ev.MatchKey + `","hit_count":4,"zero_hit":false,"top_hits":["c.go#Retry@9"]}`
	insert := `INSERT INTO scratchpad_entries (tree_id, author_session_id, type, text, refs_json) VALUES (?, ?, 'trail', 'search_semantic: retry backoff (4 hits)', ?)`
	exec(t, insert, resp.TreeID, a.SessionID, refs)
	exec(t, insert, resp.TreeID, b.SessionID, refs)
	exec(t, insert, "hft_othertree00000", "someone", refs)

	out := s.Annotate(b.SessionID, ev, nil)
	require.NotNil(t, out)
	m, ok := out["sibling_trail_match"].(map[string]any)
	require.True(t, ok, "%v", out)
	assert.Equal(t, string(a.SessionID), m["author"], "the other session's entry, not b's own")
	assert.Equal(t, 4, m["hit_count"])
	assert.Equal(t, []string{"c.go#Retry@9"}, m["top_hits"])
	assert.NotContains(t, out, "parent_trail_match")

	root := s.Annotate("parent-sib", ev, nil)
	assert.Contains(t, root, "sibling_trail_match", "the root session is in the tree too")
	exec(t, `UPDATE scratchpad_entries SET retracted_at = datetime('now') WHERE author_session_id = ?`, a.SessionID)
	assert.Nil(t, s.Annotate(b.SessionID, ev, nil), "retracted entries don't match")
}
