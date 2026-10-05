package handoff

import (
	"context"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	astcontext "github.com/coma-toast/ast-context-cache/internal/context"
	"github.com/coma-toast/ast-context-cache/internal/contextnotes"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/memory"
)

// AC1: the response carries a hof_ ref, a stub of at most 60 tokens, and a breakdown counting
// the manifest, trail, and pointers.
func TestCreateBreakdown(t *testing.T) {
	s := newTestService(t)
	project := indexFixture(t)
	parent := SessionID("parent-ac1")
	for i := range 40 {
		astcontext.MarkReturned(string(parent), astcontext.ReturnedSymbol{File: project + "/f" + strconv.Itoa(i) + ".go", Name: "S", StartLine: i + 1})
	}
	recordSearches(parent, project, 10, 3)
	resp := mustCreate(t, s, CreateRequest{
		SessionID: parent, ProjectPath: project, Label: "retry path",
		Pointers: []PointerInput{{Key: "svc.go|Alpha", Note: "entry point"}, {Key: "svc.go.Beta"}, {Key: "svc.go", Note: "whole file"}},
	})
	_, err := ParseHandoffRef(string(resp.Ref))
	require.NoError(t, err)
	assert.Equal(t, map[Section]int{SectionManifest: 40, SectionTrail: 10, SectionPointer: 3}, resp.Breakdown.Counts)
	assert.Positive(t, resp.Breakdown.Manifest)
	assert.Positive(t, resp.Breakdown.Trail)
	assert.Positive(t, resp.Breakdown.Pointers)
	assert.Equal(t, resp.Breakdown.Manifest+resp.Breakdown.Trail+resp.Breakdown.Pointers, resp.Breakdown.Total)
	assert.Equal(t, "["+"handoff "+string(resp.Ref)+"] retry path — call open_handoff first", resp.Stub)
	assert.LessOrEqual(t, db.EstimateTokens(resp.Stub), 60)
	assert.Equal(t, 1, resp.Depth)
	assert.True(t, s.IsTreeSession(parent))
	assert.Equal(t, 53, count(t, `SELECT COUNT(*) FROM handoff_snapshot_items WHERE handoff_ref = ?`, resp.Ref))
	assert.Equal(t, resp.Breakdown.Total, count(t, `SELECT tokens_used FROM handoff_trees WHERE tree_id = ?`, resp.TreeID))
	assert.Equal(t, 1+10+3, count(t, `SELECT entries_used FROM handoff_trees WHERE tree_id = ?`, resp.TreeID), "the manifest is one entry")

	var fqn, fp string
	var start, end int
	require.NoError(t, db.ContextDB.QueryRow(`SELECT fqn, fingerprint, start_line, end_line FROM handoff_snapshot_items
		WHERE handoff_ref = ? AND section = 'pointer' AND item_key = 'svc.go.Beta'`, resp.Ref).Scan(&fqn, &fp, &start, &end))
	beta := fixtureSymbol(t, project, "Beta")
	want, err := astcontext.SymbolFingerprint(beta.File, beta.StartLine, beta.EndLine)
	require.NoError(t, err)
	assert.Equal(t, "svc.go.Beta", fqn, "an fqn pointer resolves through the index")
	assert.Equal(t, want, fp)
	assert.Equal(t, []int{beta.StartLine, beta.EndLine}, []int{start, end})

	again := mustCreate(t, s, CreateRequest{SessionID: parent, ProjectPath: project, Brief: "second\nline two", IncludeManifest: new(bool), ExcludeAllTrail: true})
	assert.Equal(t, resp.TreeID, again.TreeID, "one tree per root session")
	assert.Equal(t, Breakdown{Counts: map[Section]int{}}, again.Breakdown)
	assert.Contains(t, again.Stub, "] second — call", "the label defaults to the brief's first line")
}

func TestCreateLongLabelStub(t *testing.T) {
	s := newTestService(t)
	resp := mustCreate(t, s, CreateRequest{SessionID: "parent-long", Label: strings.Repeat("wordy label ", 50)})
	assert.LessOrEqual(t, db.EstimateTokens(resp.Stub), 60)
	assert.True(t, strings.HasSuffix(resp.Stub, "… — call open_handoff first"))
}

func TestCreatePrunesTrail(t *testing.T) {
	s := newTestService(t)
	parent := SessionID("parent-prune")
	recordSearches(parent, "/p", 5, 1)
	resp := mustCreate(t, s, CreateRequest{SessionID: parent, ExcludeTrail: []int{0, 2}, ExcludeTrailQuery: "QUERY 1"})
	assert.Equal(t, 2, resp.Breakdown.Counts[SectionTrail], "newest (4) and third-newest (2) by index, and query 1 by substring")
	items, err := loadSnapshotItems(resp.Ref, SectionTrail)
	require.NoError(t, err)
	var labels []string
	for _, it := range items {
		labels = append(labels, it.label)
	}
	assert.Equal(t, []string{"query 3", "query 0"}, labels, "newest first")
}

func TestCreateValidation(t *testing.T) {
	s := newTestService(t)
	project := indexFixture(t)
	tests := []struct {
		name string
		req  CreateRequest
		code errs.Code
	}{
		{"no session", CreateRequest{Brief: "b"}, errs.CodeInvalidInput},
		{"no brief", CreateRequest{SessionID: "p", Brief: "  "}, errs.CodeInvalidInput},
		{"bad mode", CreateRequest{SessionID: "p", Brief: "b", Mode: "clone"}, errs.CodeInvalidInput},
		{"unknown note", CreateRequest{SessionID: "p", Brief: "b", CtxRefs: []string{"ctx_missing"}}, errs.CodeNotFound},
		{"unknown memory", CreateRequest{SessionID: "p", Brief: "b", MemRefs: []string{"mem_missing"}}, errs.CodeNotFound},
		{"unknown pointer", CreateRequest{SessionID: "p", Brief: "b", ProjectPath: project, Pointers: []PointerInput{{Key: "svc.go|Nope"}}}, errs.CodeNotFound},
		{"unknown fqn", CreateRequest{SessionID: "p", Brief: "b", ProjectPath: project, Pointers: []PointerInput{{Key: "svc.go.Nope"}}}, errs.CodeNotFound},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := s.Create(context.Background(), tt.req)
			assert.True(t, errs.HasCode(err, tt.code), "%v", err)
		})
	}
	assert.Zero(t, count(t, `SELECT COUNT(*) FROM handoffs`), "a failed create stores nothing")
	assert.Zero(t, count(t, `SELECT COUNT(*) FROM handoff_trees`))
}

// HO-7: a snapshot over the tree cap is rejected with its breakdown, and nothing is stored.
func TestCreateTreeLimit(t *testing.T) {
	s := newTestService(t)
	parent := SessionID("parent-cap")
	note, err := contextnotes.Store(string(parent), strings.Repeat("big note ", 100), "big", "", nil, "", nil, nil)
	require.NoError(t, err)
	stored, err := contextnotes.Peek(note.Ref)
	require.NoError(t, err)
	noteTokens := db.EstimateTokens(stored.Content)
	require.NoError(t, db.SetSetting(SettingTreeMaxTokens, "50"))
	_, err = s.Create(context.Background(), CreateRequest{SessionID: parent, Brief: "b", CtxRefs: []string{note.Ref}})
	require.True(t, errs.HasCode(err, CodeHandoffTreeLimitExceeded), "%v", err)
	m := ErrorMap(err)
	assert.Equal(t, string(CodeHandoffTreeLimitExceeded), m["error"])
	bd, ok := m["details"].(map[string]any)["breakdown"].(Breakdown)
	require.True(t, ok, "details carry the breakdown: %v", m["details"])
	assert.Equal(t, noteTokens, bd.Notes)
	assert.Zero(t, count(t, `SELECT COUNT(*) FROM handoffs`))

	// Within the cap alone but over it with the tree's existing usage: rejected in the tx.
	require.NoError(t, db.SetSetting(SettingTreeMaxTokens, "300"))
	mustCreate(t, s, CreateRequest{SessionID: parent, CtxRefs: []string{note.Ref}})
	_, err = s.Create(context.Background(), CreateRequest{SessionID: parent, Brief: "b", CtxRefs: []string{note.Ref}})
	require.True(t, errs.HasCode(err, CodeHandoffTreeLimitExceeded), "%v", err)
	details := ErrorMap(err)["details"].(map[string]any)
	assert.Equal(t, noteTokens, details["tokens_used"])
	assert.Equal(t, 300, details["tokens_max"])
	assert.Contains(t, details, "breakdown")
	assert.Equal(t, 1, count(t, `SELECT COUNT(*) FROM handoffs`))
}

// AC15 and HO-8: children nest handoffs in the same tree, one level deeper, up to depth 3.
func TestCreateNestedDepth(t *testing.T) {
	s := newTestService(t)
	root := mustCreate(t, s, CreateRequest{SessionID: "root-depth", ProjectPath: "/p"})
	ref, depth := root.Ref, root.Depth
	for _, want := range []int{2, 3} {
		child := mustOpen(t, s, ref, "")
		nested := mustCreate(t, s, CreateRequest{SessionID: child.SessionID})
		assert.Equal(t, root.TreeID, nested.TreeID)
		assert.Equal(t, want, nested.Depth)
		var parentChild, project string
		require.NoError(t, db.ContextDB.QueryRow(`SELECT parent_child_session_id, project_path FROM handoffs WHERE ref = ?`, nested.Ref).Scan(&parentChild, &project))
		assert.Equal(t, string(child.SessionID), parentChild)
		assert.Equal(t, "/p", project, "a nested handoff inherits the child's project")
		ref, depth = nested.Ref, nested.Depth
	}
	require.Equal(t, 3, depth)
	deepest := mustOpen(t, s, ref, "")
	_, err := s.Create(context.Background(), CreateRequest{SessionID: deepest.SessionID, Brief: "b"})
	assert.True(t, errs.HasCode(err, CodeHandoffDepthExceeded), "%v", err)
	assert.Equal(t, 3, count(t, `SELECT COUNT(*) FROM handoffs`))
}

// HO-2(d): the snapshot copies explicit memory refs and the parent's active session entries.
func TestCreateSnapshotsMemory(t *testing.T) {
	s := newTestService(t)
	parent := SessionID("parent-mem")
	global, err := memory.Store(memory.StoreInput{Kind: memory.KindProcedure, Scope: memory.ScopeGlobal, Rule: "always run make lint"})
	require.NoError(t, err)
	session, err := memory.Store(memory.StoreInput{Kind: memory.KindFact, SessionID: string(parent), Subject: "retry", Object: "exponential"})
	require.NoError(t, err)
	resp := mustCreate(t, s, CreateRequest{SessionID: parent, MemRefs: []string{global.Ref, session.Ref}})
	assert.Equal(t, 2, resp.Breakdown.Counts[SectionMemory], "deduplicated by ref")
	items, err := loadSnapshotItems(resp.Ref, SectionMemory)
	require.NoError(t, err)
	require.Len(t, items, 2)
	assert.Equal(t, global.Ref, items[0].key)
	assert.Equal(t, session.Ref, items[1].key)
}
