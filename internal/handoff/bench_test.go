package handoff

import (
	"context"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	astcontext "github.com/coma-toast/ast-context-cache/internal/context"
	"github.com/coma-toast/ast-context-cache/internal/contextnotes"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/trail"
)

// The NFR-1 fixture: a parent whose snapshot lands just under the default tree cap (64k
// tokens, 300 entries) with every section populated, which is the largest create and open the
// limits allow.
const (
	perfTrailEntries = snapshotTrailLimit
	perfManifestKeys = 600
	perfNotes        = 4
	// perfNoteBytes fills the rest of the cap (each "retry note " is 2 o200k tokens); each
	// note author holds two notes, inside the 32k-token per-session note quota.
	perfNoteBytes = 55000
	// perfChildren is how many children the scratchpad and collect fixtures open.
	perfChildren = 16
	// perfResultWords sizes a child's result (~1.2k tokens), so 16 of them fit one tree.
	perfResultWords = 120
)

// perfFixture is a service and an indexed project for the benchmarks and latency budgets.
type perfFixture struct {
	s       *realService
	project string
	ctx     context.Context
}

func newPerfFixture(tb testing.TB) *perfFixture {
	tb.Helper()
	s := newTestService(tb)
	// Open is measured hundreds of times on one handoff; lift the 16-children cap so every
	// iteration mints a new child rather than failing.
	require.NoError(tb, db.SetSetting(SettingMaxChildren, "1000000"))
	return &perfFixture{s: s, project: indexFixture(tb), ctx: context.Background()}
}

// liftTreeCaps removes the tree caps, for benchmarks whose b.N writes would outgrow them.
func liftTreeCaps(tb testing.TB) {
	tb.Helper()
	require.NoError(tb, db.SetSetting(SettingTreeMaxTokens, "1000000000"))
	require.NoError(tb, db.SetSetting(SettingTreeMaxEntries, "1000000000"))
}

// capRequest gives parent a full trail, a 600-symbol manifest, four large notes, and three
// pointers, and returns the create request that snapshots them all.
func (f *perfFixture) capRequest(tb testing.TB, parent SessionID, mode Mode) CreateRequest {
	tb.Helper()
	hits := []string{
		trail.HitRef("svc.go", "Alpha", 3), trail.HitRef("svc.go", "Beta", 7), trail.HitRef("svc.go", "Gamma", 11),
		trail.HitRef("pkg/retry.go", "Backoff", 20), trail.HitRef("pkg/retry.go", "Jitter", 40),
	}
	for i := range perfTrailEntries {
		trail.Record(trail.Entry{
			SessionID: string(parent), Tool: "get_context_capsule", Query: "query " + strconv.Itoa(i) + " about the retry backoff path",
			ProjectPath: f.project, HitCount: 7, TopHits: hits,
		})
	}
	syms := make([]astcontext.ReturnedSymbol, perfManifestKeys)
	for i := range syms {
		file := filepath.Join(f.project, "pkg", "f"+strconv.Itoa(i%40)+".go")
		syms[i] = astcontext.ReturnedSymbol{File: file, Name: "Symbol" + strconv.Itoa(i), ProjectPath: f.project, StartLine: i + 1}
	}
	astcontext.MarkReturned(string(parent), syms...)
	refs := make([]string, perfNotes)
	for i := range refs {
		author := string(parent) + "-notes-" + strconv.Itoa(i/2)
		note, err := contextnotes.Store(author, strings.Repeat("retry note ", perfNoteBytes/11), "notes "+strconv.Itoa(i), f.project, nil, "", nil, nil)
		require.NoError(tb, err)
		refs[i] = note.Ref
	}
	return CreateRequest{
		SessionID: parent, ProjectPath: f.project, Brief: "Investigate the retry path end to end", Label: "retry", Mode: mode, CtxRefs: refs,
		Pointers: []PointerInput{{Key: "svc.go|Alpha", Note: "entry point"}, {Key: "svc.go.Beta"}, {Key: "svc.go", Note: "whole file"}},
	}
}

// capHandoff creates a near-cap handoff from a new parent, checking it is near the cap.
func (f *perfFixture) capHandoff(tb testing.TB, parent SessionID, mode Mode) (CreateRequest, *CreateResponse) {
	tb.Helper()
	req := f.capRequest(tb, parent, mode)
	resp := mustCreate(tb, f.s, req)
	l := LoadLimits()
	require.Greater(tb, resp.Breakdown.Total, l.TreeMaxTokens*9/10, "the snapshot is near the token cap: %+v", resp.Breakdown)
	require.LessOrEqual(tb, resp.Breakdown.Total, l.TreeMaxTokens)
	return req, resp
}

// smallTree creates a handoff with an empty snapshot from root and opens n children.
func (f *perfFixture) smallTree(tb testing.TB, root SessionID, n int) (*CreateResponse, []SessionID) {
	tb.Helper()
	resp := mustCreate(tb, f.s, CreateRequest{SessionID: root, ProjectPath: f.project, Brief: "split the work", ExcludeAllTrail: true})
	children := make([]SessionID, n)
	for i := range children {
		children[i] = mustOpen(tb, f.s, resp.Ref, f.project).SessionID
	}
	return resp, children
}

// completedTree opens perfChildren children of a small handoff and completes each with a
// ~1.2k-token result.
func (f *perfFixture) completedTree(tb testing.TB, root SessionID) *CreateResponse {
	tb.Helper()
	resp, children := f.smallTree(tb, root, perfChildren)
	content := "FACT: the retry path is linear\n" + strings.Repeat("Detailed notes on the retry path. ", perfResultWords)
	for _, sid := range children {
		_, err := f.s.Complete(f.ctx, CompleteRequest{SessionID: sid, Content: content, Summary: "retry is linear"})
		require.NoError(tb, err)
	}
	return resp
}

// fillScratchpad posts n findings round-robin from children.
func (f *perfFixture) fillScratchpad(tb testing.TB, children []SessionID, n int) {
	tb.Helper()
	for i := range n {
		_, err := f.s.Post(f.ctx, PostRequest{SessionID: children[i%len(children)], Type: EntryTypeFinding, Text: "finding " + strconv.Itoa(i) + ": Backoff doubles", Refs: []string{"pkg/retry.go"}})
		require.NoError(tb, err)
	}
}

func BenchmarkCreate(b *testing.B) {
	f := newPerfFixture(b)
	req, _ := f.capHandoff(b, "bench-create", ModeFresh)
	flush := FlushRequest{SessionID: req.SessionID}
	_, err := f.s.Flush(f.ctx, flush)
	require.NoError(b, err)
	for b.Loop() {
		_, err := f.s.Create(f.ctx, req)
		require.NoError(b, err)
		b.StopTimer()
		_, err = f.s.Flush(f.ctx, flush)
		require.NoError(b, err)
		b.StartTimer()
	}
}

func BenchmarkOpen(b *testing.B) {
	f := newPerfFixture(b)
	_, created := f.capHandoff(b, "bench-open", ModeFresh)
	req := OpenRequest{Handoff: created.Ref, ProjectPath: f.project}
	for b.Loop() {
		_, err := f.s.Open(f.ctx, req)
		require.NoError(b, err)
	}
}

func BenchmarkExpandPointer(b *testing.B) {
	f := newPerfFixture(b)
	_, created := f.capHandoff(b, "bench-expand", ModeFresh)
	child := mustOpen(b, f.s, created.Ref, f.project).SessionID
	req := ExpandRequest{Handoff: created.Ref, SessionID: child, ProjectPath: f.project, Section: SectionPointer, All: true, Mode: "auto"}
	for b.Loop() {
		_, err := f.s.Expand(f.ctx, req)
		require.NoError(b, err)
	}
}

func BenchmarkScratchpadPost(b *testing.B) {
	f := newPerfFixture(b)
	liftTreeCaps(b)
	_, children := f.smallTree(b, "bench-post", 2)
	req := PostRequest{SessionID: children[0], Type: EntryTypeFinding, Text: "Backoff doubles without a ceiling", Refs: []string{"pkg/retry.go"}}
	for b.Loop() {
		_, err := f.s.Post(f.ctx, req)
		require.NoError(b, err)
	}
}

func BenchmarkScratchpadRead(b *testing.B) {
	f := newPerfFixture(b)
	_, children := f.smallTree(b, "bench-read", 8)
	f.fillScratchpad(b, children, 200)
	req := ReadRequest{SessionID: children[0]}
	for b.Loop() {
		_, err := f.s.Read(f.ctx, req)
		require.NoError(b, err)
	}
}

func BenchmarkClaim(b *testing.B) {
	f := newPerfFixture(b)
	liftTreeCaps(b)
	_, children := f.smallTree(b, "bench-claim", 2)
	claimReq := ClaimRequest{SessionID: children[0], Key: "pkg/retry.go", Reason: "fix backoff"}
	releaseReq := ReleaseRequest{SessionID: children[0], Key: "pkg/retry.go"}
	for b.Loop() {
		res, err := f.s.Claim(f.ctx, claimReq)
		require.NoError(b, err)
		require.Equal(b, ClaimGranted, res.Outcome)
		b.StopTimer()
		_, err = f.s.Release(f.ctx, releaseReq)
		require.NoError(b, err)
		b.StartTimer()
	}
}

func BenchmarkCollect16(b *testing.B) {
	f := newPerfFixture(b)
	created := f.completedTree(b, "bench-collect")
	req := CollectRequest{Handoff: created.Ref}
	for b.Loop() {
		res, err := f.s.Collect(f.ctx, req)
		require.NoError(b, err)
		require.Len(b, res.Children, perfChildren)
	}
}
