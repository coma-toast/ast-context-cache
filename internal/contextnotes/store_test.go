package contextnotes

import (
	"errors"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/errs"
)

const (
	testBackdateNoteQuery   = `UPDATE context_notes SET created_at = datetime('now', ?) WHERE ref = ?`
	testCountRevisionsQuery = `SELECT COUNT(*) FROM context_note_revisions WHERE ref = ?`
)

func testNotesDB(t *testing.T) {
	t.Helper()
	dbtest.Init(t)
}

func TestStoreFetchFlush(t *testing.T) {
	testNotesDB(t)
	res, err := Store("sess-1", strings.Repeat("hello ", 100), "greeting", "/tmp/proj", "tag1", "", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(res.Ref, "ctx_") {
		t.Fatalf("ref prefix: %s", res.Ref)
	}
	if res.VirtualTokensStored <= 0 {
		t.Fatalf("expected token est > 0")
	}
	fetch, err := Fetch([]string{res.Ref}, "sess-1", "")
	if err != nil {
		t.Fatal(err)
	}
	if len(fetch.Notes) != 1 {
		t.Fatalf("notes: %d", len(fetch.Notes))
	}
	if v, ok := fetch.Stats["virtual_tokens_returned"].(int); !ok || v <= 0 {
		t.Fatalf("expected access tokens, stats=%v", fetch.Stats)
	}
	list, err := List("sess-1", "", 10)
	if err != nil || list.Total != 1 {
		t.Fatalf("list total=%d err=%v", list.Total, err)
	}
	search, err := Search("hello", "sess-1", "", 5, nil)
	if err != nil {
		t.Fatal(err)
	}
	if search == nil || len(search.Notes) == 0 {
		t.Fatalf("search notes=%d", len(search.Notes))
	}
	flush, err := Flush("sess-1", nil, "", false)
	if err != nil || flush.FlushedRefs != 1 {
		t.Fatalf("flush: %+v err=%v", flush, err)
	}
	_, err = Fetch([]string{res.Ref}, "sess-1", "")
	if err != nil {
		t.Fatal(err)
	}
}

func TestStoreLimitReject(t *testing.T) {
	testNotesDB(t)
	db.SetSetting("context_max_notes_session", "1")
	db.SetSetting("context_limit_policy", "reject")
	_, err := Store("sess-limit", "first note content", "a", "", nil, "", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	_, err = Store("sess-limit", "second note should fail", "b", "", nil, "", nil, nil)
	var le *LimitError
	if !errors.As(err, &le) {
		t.Fatalf("expected LimitError, got %v", err)
	}
	if !errs.HasCode(err, errs.CodeLimitExceeded) {
		t.Fatalf("expected %s code, got %v", errs.CodeLimitExceeded, errs.CodesOf(err))
	}
	if out := LimitErrorMap(err); out["error"] != "context_limit_exceeded" || out["limit"] != "session_notes" {
		t.Fatalf("unexpected LimitErrorMap shape: %v", out)
	}
	if !strings.HasPrefix(err.Error(), "context_limit_exceeded: session_notes") {
		t.Fatalf("unexpected error text: %q", err.Error())
	}
}

func TestFetchSessionIsolation(t *testing.T) {
	testNotesDB(t)
	res, err := Store("owner", "secret", "x", "", nil, "", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	fetch, err := Fetch([]string{res.Ref}, "other-session", "")
	if err != nil {
		t.Fatal(err)
	}
	if len(fetch.Notes) != 0 {
		t.Fatalf("expected 0 notes for wrong session")
	}
}

func TestLRUEviction(t *testing.T) {
	testNotesDB(t)
	db.SetSetting("context_max_notes_session", "2")
	db.SetSetting("context_limit_policy", "lru_session")
	store := func(content, age string) string {
		res, err := Store("lru-s", content, content, "", nil, "", nil, nil)
		require.NoError(t, err)
		_, err = db.ContextDB.Exec(testBackdateNoteQuery, age, res.Ref)
		require.NoError(t, err)
		return res.Ref
	}
	oldFetched := store("one", "-2 hours")
	oldUnfetched := store("two", "-1 hours")
	_, err := Fetch([]string{oldFetched}, "lru-s", "")
	require.NoError(t, err)
	r3, err := Store("lru-s", "three", "3", "", nil, "", nil, nil)
	require.NoError(t, err)
	assert.Equal(t, []string{oldUnfetched}, r3.EvictedRefs, "a recently fetched older note must outlive a stale newer one")
	fetch, err := Fetch([]string{oldFetched, oldUnfetched, r3.Ref}, "lru-s", "")
	require.NoError(t, err)
	var got []string
	for _, n := range fetch.Notes {
		got = append(got, n.Ref)
	}
	assert.ElementsMatch(t, []string{oldFetched, r3.Ref}, got)
}

// TestFlushDeletesRevisions covers BF-6: flushing or evicting a note must not leave its
// superseded bodies behind in context_note_revisions.
func TestFlushDeletesRevisions(t *testing.T) {
	tests := []struct {
		name  string
		flush func(t *testing.T, ref string)
	}{
		{name: "refs", flush: func(t *testing.T, ref string) {
			_, err := Flush("rev-s", []string{ref}, "", false)
			require.NoError(t, err)
		}},
		{name: "session", flush: func(t *testing.T, ref string) {
			_, err := Flush("rev-s", nil, "", false)
			require.NoError(t, err)
		}},
		{name: "all", flush: func(t *testing.T, ref string) {
			_, err := Flush("", nil, "", true)
			require.NoError(t, err)
		}},
		{name: "lru eviction", flush: func(t *testing.T, ref string) {
			db.SetSetting("context_max_notes_session", "1")
			db.SetSetting("context_limit_policy", "lru_session")
			res, err := Store("rev-s", "replacement", "r", "", nil, "", nil, nil)
			require.NoError(t, err)
			require.Equal(t, []string{ref}, res.EvictedRefs)
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			testNotesDB(t)
			ref := mustStore(t, "rev-s", "v1")
			for _, c := range []string{"v2", "v3"} {
				_, err := Edit(EditInput{Action: EditAppend, Ref: ref, SessionID: "rev-s", Content: c}, nil)
				require.NoError(t, err)
			}
			countRevisions := func() int {
				var n int
				require.NoError(t, db.ContextDB.QueryRow(testCountRevisionsQuery, ref).Scan(&n))
				return n
			}
			require.Positive(t, countRevisions())
			tc.flush(t, ref)
			assert.Zero(t, countRevisions())
		})
	}
}

func TestDashboardStats(t *testing.T) {
	testNotesDB(t)
	Store("dash", "content for stats", "lbl", "", nil, "", nil, nil)
	ds := DashboardStatsFor("", 30)
	if ds.ActiveNotesCount != 1 {
		t.Fatalf("active notes: %d", ds.ActiveNotesCount)
	}
	if ds.Limits["max_notes_session"].(int) != 50 {
		t.Fatalf("limits: %+v", ds.Limits)
	}
}

func TestFlushOrphansRespectsGraceScopeAndAccess(t *testing.T) {
	testNotesDB(t)
	store := func(project, content string) string {
		t.Helper()
		res, err := Store("sess-o", content, "", project, "", "", nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		return res.Ref
	}
	backdate := func(ref string) {
		t.Helper()
		if _, err := db.ContextDB.Exec(`UPDATE context_notes SET created_at = datetime('now', '-2 hours') WHERE ref = ?`, ref); err != nil {
			t.Fatal(err)
		}
	}
	exists := func(ref string) bool {
		_, err := noteByRef(ref)
		return err == nil
	}

	oldOrphanP1 := store("/p1", "old orphan one")
	oldOrphanP2 := store("/p2", "old orphan two")
	recentOrphanP1 := store("/p1", "recent orphan")
	oldFetchedP1 := store("/p1", "old but fetched")
	for _, r := range []string{oldOrphanP1, oldOrphanP2, oldFetchedP1} {
		backdate(r)
	}
	if _, err := Fetch([]string{oldFetchedP1}, "sess-o", ""); err != nil {
		t.Fatal(err)
	}

	res, kept, err := FlushOrphans("/p1", OrphanPurgeGrace)
	if err != nil {
		t.Fatal(err)
	}
	if res.FlushedRefs != 1 || kept != 1 {
		t.Fatalf("project-scoped purge: flushed=%d kept=%d, want 1/1", res.FlushedRefs, kept)
	}
	if exists(oldOrphanP1) {
		t.Fatal("old orphan in /p1 should be purged")
	}
	if !exists(recentOrphanP1) {
		t.Fatal("recent orphan must survive the grace window")
	}
	if !exists(oldFetchedP1) {
		t.Fatal("a note that was fetched is not an orphan")
	}
	if !exists(oldOrphanP2) {
		t.Fatal("orphan in another project must survive a project-scoped purge")
	}

	if res, _, err = FlushOrphans("", OrphanPurgeGrace); err != nil || res.FlushedRefs != 1 {
		t.Fatalf("global purge: %+v err=%v", res, err)
	}
	if exists(oldOrphanP2) {
		t.Fatal("global purge should remove the remaining old orphan")
	}
}

// TestSearchLikeRespectsSession covers BF-1: the LIKE fallback used to bind its
// appended scope filters only to the content arm of the OR, so a label match
// from another session or project leaked through.
func TestSearchLikeRespectsSession(t *testing.T) {
	testNotesDB(t)
	// "zyplu" sits mid-token in each label, so the prefix FTS query misses and
	// Search must take the LIKE path.
	store := func(sid, label, proj string) string {
		res, err := Store(sid, "unrelated body text", label, proj, nil, "", nil, nil)
		require.NoError(t, err)
		return res.Ref
	}
	s1a := store("S1", "xyzzyplugh alpha", "/proj/a")
	s1b := store("S1", "xyzzyplugh beta", "/proj/b")
	store("S2", "xyzzyplugh gamma", "/proj/a")
	store("S2", "xyzzyplugh delta", "/proj/b")
	fts, err := searchNotesFTS("zyplu", "S1", "", 10)
	require.NoError(t, err)
	require.Empty(t, fts, "FTS must miss so the LIKE fallback is exercised")
	tests := []struct {
		name, sessionID, projectPath string
		want                         []string
	}{
		{name: "session only", sessionID: "S1", want: []string{s1a, s1b}},
		{name: "session and project", sessionID: "S1", projectPath: "/proj/a", want: []string{s1a}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			res, err := Search("zyplu", tc.sessionID, tc.projectPath, 10, nil)
			require.NoError(t, err)
			var got []string
			for _, n := range res.Notes {
				assert.Equal(t, tc.sessionID, n.SessionID)
				if tc.projectPath != "" {
					assert.Equal(t, tc.projectPath, n.ProjectPath)
				}
				got = append(got, n.Ref)
			}
			assert.ElementsMatch(t, tc.want, got)
		})
	}
}
