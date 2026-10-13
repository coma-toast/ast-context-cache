package handoff

import (
	"context"
	"os"
	osexec "os/exec"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/stretchr/testify/require"

	astcontext "github.com/coma-toast/ast-context-cache/internal/context"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/coma-toast/ast-context-cache/internal/search"
	"github.com/coma-toast/ast-context-cache/internal/trail"
)

// fixtureSource is the indexed file the pointer tests point into. Alpha is lines 3-5, Beta
// 7-9, and Gamma 11-13.
const fixtureSource = `package svc

func Alpha() int {
	return 1
}

func Beta() int {
	return 2
}

func Gamma() int {
	return 3
}
`

// newTestService opens a fresh database and returns a service without background loops.
func newTestService(t testing.TB) *realService {
	t.Helper()
	dbtest.Init(t)
	t.Cleanup(db.FlushWriteBuffers)
	db.WaitInitFTSRebuildForTest()
	return newService(nil)
}

// indexFixture writes fixtureSource as svc.go in a new project and indexes it.
func indexFixture(t testing.TB) string {
	t.Helper()
	project := t.TempDir()
	writeAndIndex(t, project, "svc.go", fixtureSource)
	return project
}

func writeAndIndex(t testing.TB, project, rel, src string) {
	t.Helper()
	file := filepath.Join(project, rel)
	require.NoError(t, os.WriteFile(file, []byte(src), 0o644))
	_, _, _, err := indexer.IndexFile(file, project)
	require.NoError(t, err)
}

// fixtureSymbol returns the indexed fixture symbol name in project.
func fixtureSymbol(t *testing.T, project, name string) *astcontext.SymbolRow {
	t.Helper()
	row, err := astcontext.LookupSymbol(project, "svc.go", "svc.go."+name, name)
	require.NoError(t, err)
	return row
}

// returnSymbols marks fixture symbols as returned to sid, as a search would.
func returnSymbols(t *testing.T, sid SessionID, project string, names ...string) {
	t.Helper()
	for _, n := range names {
		r := fixtureSymbol(t, project, n)
		astcontext.MarkReturned(string(sid), astcontext.ReturnedSymbol{File: r.File, Name: r.Name, ProjectPath: project, StartLine: r.StartLine})
	}
}

// recordSearches records n searches by sid, query "query <i>", each with hits hits.
func recordSearches(sid SessionID, project string, n, hits int) {
	for i := range n {
		trail.Record(trail.Entry{
			SessionID: string(sid), Tool: "search_semantic", Query: "query " + strconv.Itoa(i), ProjectPath: project,
			HitCount: hits, TopHits: []string{trail.HitRef("svc.go", "Alpha", 3)},
		})
	}
}

// mustCreate makes a handoff from parent, failing the test on error.
func mustCreate(t testing.TB, s *realService, req CreateRequest) *CreateResponse {
	t.Helper()
	if req.Brief == "" {
		req.Brief = "investigate the retry path"
	}
	resp, err := s.Create(context.Background(), req)
	require.NoError(t, err)
	return resp
}

// mustOpen opens ref as a new child, failing the test on error.
func mustOpen(t testing.TB, s *realService, ref HandoffRef, project string) *OpenResponse {
	t.Helper()
	resp, err := s.Open(context.Background(), OpenRequest{Handoff: ref, ProjectPath: project, TokenBudget: 100000})
	require.NoError(t, err)
	return resp
}

// packedFor runs the search packer for sid over one fixture symbol, as search_semantic would.
func packedFor(t *testing.T, sid SessionID, project, name string) ([]map[string]any, astcontext.SavingsMeta) {
	t.Helper()
	r := fixtureSymbol(t, project, name)
	scored := []search.ScoredResult{{Score: 1, Data: map[string]any{
		"file": r.File, "name": r.Name, "kind": r.Kind, "start_line": r.StartLine, "end_line": r.EndLine, "similarity": 1.0,
	}}}
	p, _, err := astcontext.PackScoredResults(scored, 10, project, name, "auto", string(sid), 0, astcontext.PrecisionArgs{Collapse: true})
	require.NoError(t, err)
	return p.Results, p.Savings
}

// gitRepo makes project a git repository with one commit, skipping the test without git.
func gitRepo(t *testing.T, project string) {
	t.Helper()
	// Keep fixture commits independent of the developer's git config (signing, hooks).
	t.Setenv("GIT_CONFIG_GLOBAL", os.DevNull)
	t.Setenv("GIT_CONFIG_NOSYSTEM", "1")
	for _, args := range [][]string{
		{"init", "-q"},
		{"config", "user.email", "test@example.com"},
		{"config", "user.name", "test"},
		{"add", "."},
		{"commit", "-q", "-m", "init"},
	} {
		gitRun(t, project, args...)
	}
}

func gitRun(t *testing.T, dir string, args ...string) {
	t.Helper()
	cmd := osexec.Command("git", args...)
	cmd.Dir = dir
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Skipf("git unavailable: %v: %s", err, out)
	}
}
