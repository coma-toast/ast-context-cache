package handoff

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	astcontext "github.com/coma-toast/ast-context-cache/internal/context"
	"github.com/coma-toast/ast-context-cache/internal/contextnotes"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/memory"
	"github.com/coma-toast/ast-context-cache/internal/repokey"
)

func expand(t *testing.T, s *realService, req ExpandRequest) *ExpandResponse {
	t.Helper()
	resp, err := s.Expand(context.Background(), req)
	require.NoError(t, err)
	return resp
}

func itemsByKey(items []ExpandedItem) map[string]ExpandedItem {
	out := map[string]ExpandedItem{}
	for _, it := range items {
		out[it.Key] = it
	}
	return out
}

// AC5 and OP-6: an expanded pointer is current code, and a later search doesn't resend it.
func TestExpandPointerMarksReturned(t *testing.T) {
	s := newTestService(t)
	project := indexFixture(t)
	resp := mustCreate(t, s, CreateRequest{SessionID: "parent-ac5", ProjectPath: project, Pointers: []PointerInput{{Key: "svc.go|Alpha"}}})
	o := mustOpen(t, s, resp.Ref, "")
	e := expand(t, s, ExpandRequest{Handoff: resp.Ref, SessionID: o.SessionID, Section: SectionPointer, All: true})
	require.Len(t, e.Items, 1)
	it := e.Items[0]
	assert.Equal(t, ChangeFresh, it.Change)
	assert.False(t, it.Stale)
	assert.Equal(t, "func Alpha() int {\n\treturn 1\n}", it.Content)
	assert.Equal(t, &LineRange{Start: 3, End: 5}, it.NewLines)
	assert.Nil(t, it.OldLines)
	assert.Equal(t, "svc.go.Alpha", it.FQN)
	results, savings := packedFor(t, o.SessionID, project, "Alpha")
	assert.Empty(t, results, "the expanded pointer is deduped")
	assert.Equal(t, 1, savings.DedupedCount)
	assert.Equal(t, o.TokensUsed+e.TokensUsed, e.TokensDelivered)
	assert.Equal(t, o.TokensAvailable, e.TokensAvailable)

	sk := expand(t, s, ExpandRequest{Handoff: resp.Ref, SessionID: o.SessionID, Section: SectionPointer, Items: []int64{it.ID}, Mode: "skeleton"})
	require.Len(t, sk.Items, 1)
	assert.NotContains(t, sk.Items[0].Content, "return 1", "skeleton mode drops the body")
}

// AC7 and OP-7: drift since the snapshot is reported, with the current code, and never errors.
func TestExpandStaleness(t *testing.T) {
	s := newTestService(t)
	project := indexFixture(t)
	require.NoError(t, os.WriteFile(filepath.Join(project, "notes.txt"), []byte("todo\n"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(project, "keep.txt"), []byte("keep\n"), 0o644))
	resp := mustCreate(t, s, CreateRequest{SessionID: "parent-ac7", ProjectPath: project, Pointers: []PointerInput{
		{Key: "svc.go|Alpha"}, {Key: "svc.go|Beta"}, {Key: "svc.go|Gamma"}, {Key: "notes.txt"}, {Key: "keep.txt"},
	}})
	writeAndIndex(t, project, "svc.go", "package svc\n\n// one\n// two\nfunc Alpha() int {\n\treturn 100\n}\n\nfunc Gamma() int {\n\treturn 3\n}\n")
	require.NoError(t, os.Remove(filepath.Join(project, "notes.txt")))
	o := mustOpen(t, s, resp.Ref, "")
	e := expand(t, s, ExpandRequest{Handoff: resp.Ref, SessionID: o.SessionID, Section: SectionPointer})
	got := itemsByKey(e.Items)
	require.Len(t, got, 5)

	alpha := got["svc.go|Alpha"]
	assert.Equal(t, ChangeModified, alpha.Change)
	assert.True(t, alpha.Stale)
	assert.Contains(t, alpha.Content, "return 100", "the current code")
	assert.Equal(t, &LineRange{Start: 3, End: 5}, alpha.OldLines)
	assert.Equal(t, &LineRange{Start: 5, End: 7}, alpha.NewLines)

	beta := got["svc.go|Beta"]
	assert.Equal(t, ChangeDeleted, beta.Change)
	assert.True(t, beta.Stale)
	assert.Empty(t, beta.Content)
	assert.Equal(t, &LineRange{Start: 7, End: 9}, beta.OldLines)

	gamma := got["svc.go|Gamma"]
	assert.Equal(t, ChangeMoved, gamma.Change)
	assert.True(t, gamma.Stale)
	assert.Equal(t, &LineRange{Start: 11, End: 13}, gamma.OldLines)
	assert.Equal(t, &LineRange{Start: 9, End: 11}, gamma.NewLines)

	notes := got["notes.txt"]
	assert.Equal(t, ChangeFileMissing, notes.Change)
	assert.True(t, notes.Stale)

	keep := got["keep.txt"]
	assert.Equal(t, ChangeFresh, keep.Change)
	assert.False(t, keep.Stale)
	assert.Equal(t, "keep\n", keep.Content)

	keys := astcontext.ReturnedKeys(string(o.SessionID))
	assert.Len(t, keys, 2, "only the symbols that were delivered: Alpha and Gamma")
}

// AC8 and OP-8: a child in a sibling worktree resolves pointers in its own checkout, and a fork
// child there is deduped against the parent's symbols at their worktree paths.
func TestExpandSiblingWorktree(t *testing.T) {
	s := newTestService(t)
	root := t.TempDir()
	main := filepath.Join(root, "main")
	require.NoError(t, os.MkdirAll(main, 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(main, "svc.go"), []byte(fixtureSource), 0o644))
	gitRepo(t, main)
	sibling := filepath.Join(root, "sibling")
	gitRun(t, main, "worktree", "add", "-q", "-b", "feature", sibling)
	repokey.Invalidate("")
	writeAndIndex(t, main, "svc.go", fixtureSource)
	writeAndIndex(t, sibling, "svc.go", "package svc\n\nfunc Alpha() int {\n\treturn 42\n}\n\nfunc Beta() int {\n\treturn 2\n}\n")
	parent := SessionID("parent-ac8")
	returnSymbols(t, parent, main, "Beta")
	resp := mustCreate(t, s, CreateRequest{SessionID: parent, ProjectPath: main, Mode: ModeFork, Pointers: []PointerInput{{Key: "svc.go|Alpha"}}})

	o := mustOpen(t, s, resp.Ref, sibling)
	e := expand(t, s, ExpandRequest{Handoff: resp.Ref, SessionID: o.SessionID, Section: SectionPointer})
	require.Len(t, e.Items, 1)
	assert.Contains(t, e.Items[0].Content, "return 42", "code from the child's worktree")
	assert.Equal(t, ChangeModified, e.Items[0].Change, "compared with the snapshot fingerprint")
	assert.Contains(t, astcontext.ReturnedKeys(string(o.SessionID)), filepath.Join(sibling, "svc.go")+"|Beta|7", "fork seeding maps to the worktree")
	results, _ := packedFor(t, o.SessionID, sibling, "Beta")
	assert.Empty(t, results)

	other := t.TempDir()
	o2 := mustOpen(t, s, resp.Ref, other)
	e = expand(t, s, ExpandRequest{Handoff: resp.Ref, SessionID: o2.SessionID, Section: SectionPointer})
	require.Len(t, e.Items, 1)
	assert.Contains(t, e.Items[0].Content, "return 1", "an unrelated project resolves in the snapshot's")
	assert.Equal(t, ChangeFresh, e.Items[0].Change)
}

// AC9 and HO-10: the snapshot outlives the parent's flush_context.
func TestExpandSurvivesParentFlush(t *testing.T) {
	s := newTestService(t)
	parent := SessionID("parent-ac9")
	note, err := contextnotes.Store(string(parent), "the full design", "design", "", nil, "", nil, nil)
	require.NoError(t, err)
	_, err = memory.Store(memory.StoreInput{Kind: memory.KindFact, SessionID: string(parent), Subject: "retry", Object: "capped at 5"})
	require.NoError(t, err)
	recordSearches(parent, "/p", 2, 0)
	resp := mustCreate(t, s, CreateRequest{SessionID: parent, CtxRefs: []string{note.Ref}})
	o := mustOpen(t, s, resp.Ref, "")
	_, err = contextnotes.FlushSession(string(parent))
	require.NoError(t, err)
	_, err = memory.DeleteSession(string(parent))
	require.NoError(t, err)

	e := expand(t, s, ExpandRequest{Handoff: resp.Ref, SessionID: o.SessionID, Section: SectionNote, Items: []int64{o.Notes[0].ID}})
	require.Len(t, e.Items, 1)
	assert.Equal(t, ExpandedItem{ID: o.Notes[0].ID, Section: SectionNote, Key: note.Ref, Label: "design", Content: "the full design", TokenEst: 3}, e.Items[0])
	e = expand(t, s, ExpandRequest{Handoff: resp.Ref, SessionID: o.SessionID, Section: SectionMemory, All: true})
	require.Len(t, e.Items, 1)
	assert.Contains(t, e.Items[0].Content, "capped at 5")
	e = expand(t, s, ExpandRequest{Handoff: resp.Ref, SessionID: o.SessionID, Section: SectionTrail, All: true})
	require.Len(t, e.Items, 2)
	assert.JSONEq(t, `{"tool":"search_semantic","query":"query 1","hit_count":0,"zero_hit":true,"top_hits":["svc.go#Alpha@3"]}`, e.Items[0].Content)
}

func TestExpandManifestAndBudget(t *testing.T) {
	s := newTestService(t)
	project := indexFixture(t)
	parent := SessionID("parent-budget")
	returnSymbols(t, parent, project, "Alpha", "Beta", "Gamma")
	resp := mustCreate(t, s, CreateRequest{SessionID: parent, ProjectPath: project, Pointers: []PointerInput{{Key: "svc.go|Alpha"}, {Key: "svc.go|Beta"}}})
	o := mustOpen(t, s, resp.Ref, "")
	m := expand(t, s, ExpandRequest{Handoff: resp.Ref, SessionID: o.SessionID, Section: SectionManifest})
	var keys []string
	for _, it := range m.Items {
		keys = append(keys, it.Key)
	}
	assert.Equal(t, []string{"svc.go#Alpha@3", "svc.go#Beta@7", "svc.go#Gamma@11"}, keys)

	e := expand(t, s, ExpandRequest{Handoff: resp.Ref, SessionID: o.SessionID, Section: SectionPointer, TokenBudget: 80})
	require.Len(t, e.Items, 1)
	assert.LessOrEqual(t, e.TokensUsed, 80)
	assert.True(t, e.Truncated)
	assert.Equal(t, &PageCursor{Section: SectionPointer, Offset: 1}, e.Next)
	e = expand(t, s, ExpandRequest{Handoff: resp.Ref, SessionID: o.SessionID, Section: SectionPointer, TokenBudget: 80, Next: e.Next})
	require.Len(t, e.Items, 1)
	assert.Equal(t, "svc.go|Beta", e.Items[0].Key)
	assert.False(t, e.Truncated)

	// A first item over the budget alone is cut to fit, and isn't counted as delivered.
	writeAndIndex(t, project, "big.go", "package svc\n\nfunc Big() string {\n\treturn \""+strings.Repeat("x", 2000)+"\"\n}\n")
	big := mustCreate(t, s, CreateRequest{SessionID: parent, ProjectPath: project, Pointers: []PointerInput{{Key: "big.go|Big"}}})
	o2 := mustOpen(t, s, big.Ref, "")
	cut := expand(t, s, ExpandRequest{Handoff: big.Ref, SessionID: o2.SessionID, Section: SectionPointer, TokenBudget: 200})
	require.Len(t, cut.Items, 1)
	assert.True(t, cut.Items[0].Truncated)
	assert.True(t, strings.HasPrefix(cut.Items[0].Content, "func Big() string {"))
	assert.LessOrEqual(t, cut.TokensUsed, 200)
	assert.Greater(t, cut.TokensUsed, 150, "cut to the budget, not far below it")
	assert.False(t, cut.Truncated, "nothing left to page")
	assert.Empty(t, astcontext.ReturnedKeys(string(o2.SessionID)))
}

func TestExpandValidation(t *testing.T) {
	s := newTestService(t)
	resp := mustCreate(t, s, CreateRequest{SessionID: "parent-val"})
	other := mustCreate(t, s, CreateRequest{SessionID: "parent-val"})
	o := mustOpen(t, s, resp.Ref, "")
	tests := []struct {
		name string
		req  ExpandRequest
		code errs.Code
	}{
		{"no session", ExpandRequest{Handoff: resp.Ref, Section: SectionNote}, errs.CodeInvalidInput},
		{"bad section", ExpandRequest{Handoff: resp.Ref, SessionID: o.SessionID, Section: "everything"}, errs.CodeInvalidInput},
		{"bad mode", ExpandRequest{Handoff: resp.Ref, SessionID: o.SessionID, Section: SectionPointer, Mode: "summary"}, errs.CodeInvalidInput},
		{"other handoff", ExpandRequest{Handoff: other.Ref, SessionID: o.SessionID, Section: SectionNote}, CodeHandoffNotFound},
		{"unknown handoff", ExpandRequest{Handoff: "hof_0000000000000000", SessionID: o.SessionID, Section: SectionNote}, CodeHandoffNotFound},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := s.Expand(context.Background(), tt.req)
			assert.True(t, errs.HasCode(err, tt.code), "%v", err)
		})
	}
}
