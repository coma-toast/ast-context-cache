package contextnotes

import (
	"errors"
	"strings"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
)

func mustStore(t *testing.T, sessionID, content string) string {
	t.Helper()
	res, err := Store(sessionID, content, "plan", "/tmp/proj", "", "", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	return res.Ref
}

func noteContent(t *testing.T, ref, sessionID string) string {
	t.Helper()
	res, err := Fetch([]string{ref}, sessionID, "")
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Notes) != 1 {
		t.Fatalf("expected 1 note for %s, got %d", ref, len(res.Notes))
	}
	return res.Notes[0].Content
}

// A stored note must be editable in place: that is the whole point, since the
// ctx_* stub is already written into chat and a flush+store cycle would orphan it.
func TestEditAppendKeepsRef(t *testing.T) {
	testNotesDB(t)
	ref := mustStore(t, "sess-1", "## STATE\nledger: 1")

	res, err := Edit(EditInput{Action: EditAppend, Ref: ref, SessionID: "sess-1", Content: "NEXT: ship"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !res.Changed {
		t.Fatal("expected changed=true")
	}
	if res.PreviousRevision != 1 || res.Revision != 2 {
		t.Fatalf("revisions: %d -> %d", res.PreviousRevision, res.Revision)
	}
	if got := noteContent(t, ref, "sess-1"); !strings.HasSuffix(got, "NEXT: ship") {
		t.Fatalf("append lost: %q", got)
	}
}

func TestEditReplaceByPattern(t *testing.T) {
	testNotesDB(t)
	ref := mustStore(t, "sess-1", "budget: 7/100\ndead ends: grid(0.822), hex(0.997)\nbest: 0.9992")

	res, err := Edit(EditInput{
		Action:      EditReplace,
		Ref:         ref,
		SessionID:   "sess-1",
		Pattern:     `dead ends: .*`,
		Replacement: "dead ends: pruned",
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.MatchedRegions != 1 {
		t.Fatalf("matched: %d", res.MatchedRegions)
	}
	got := noteContent(t, ref, "sess-1")
	if !strings.Contains(got, "dead ends: pruned") || !strings.Contains(got, "best: 0.9992") {
		t.Fatalf("replace wrong: %q", got)
	}
}

// Group references are what make a rewrite useful rather than blunt.
func TestEditReplaceGroupReference(t *testing.T) {
	testNotesDB(t)
	ref := mustStore(t, "sess-1", "call searched: alpha\ncall searched: beta")

	if _, err := Edit(EditInput{
		Action:      EditReplace,
		Ref:         ref,
		SessionID:   "sess-1",
		Pattern:     `searched: (\w+)`,
		Replacement: "already searched $1",
	}, nil); err != nil {
		t.Fatal(err)
	}
	got := noteContent(t, ref, "sess-1")
	if strings.Count(got, "already searched") != 2 {
		t.Fatalf("expected both matches replaced: %q", got)
	}
}

func TestEditDeleteByPattern(t *testing.T) {
	testNotesDB(t)
	ref := mustStore(t, "sess-1", "keep this\nnoise line\nkeep that")

	res, err := Edit(EditInput{Action: EditDelete, Ref: ref, SessionID: "sess-1", Pattern: `(?m)^noise line$`}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.MatchedRegions != 1 || res.TokensReclaimed <= 0 {
		t.Fatalf("matched=%d reclaimed=%d", res.MatchedRegions, res.TokensReclaimed)
	}
	if got := noteContent(t, ref, "sess-1"); strings.Contains(got, "noise line") {
		t.Fatalf("delete failed: %q", got)
	}
}

func TestEditByLineRange(t *testing.T) {
	testNotesDB(t)
	ref := mustStore(t, "sess-1", "line one\nline two\nline three\nline four")

	if _, err := Edit(EditInput{
		Action:    EditReplace,
		Ref:       ref,
		SessionID: "sess-1",
		StartLine: 2,
		EndLine:   3,
	}, nil); err != nil {
		t.Fatal(err)
	}
	got := noteContent(t, ref, "sess-1")
	if got != "line one\nline four" {
		t.Fatalf("range replace: %q", got)
	}
}

// A pattern that matches nothing is a no-op, not an error and not a revision:
// the caller needs to see the edit did not land.
func TestEditNoMatchIsNoOp(t *testing.T) {
	testNotesDB(t)
	ref := mustStore(t, "sess-1", "unchanged body")

	res, err := Edit(EditInput{Action: EditDelete, Ref: ref, SessionID: "sess-1", Pattern: `nowhere`}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Changed || res.MatchedRegions != 0 {
		t.Fatalf("expected no-op, got changed=%v matched=%d", res.Changed, res.MatchedRegions)
	}
	if res.Revision != 1 {
		t.Fatalf("no-op burned a revision: %d", res.Revision)
	}
}

// Optimistic concurrency is the guard that stops "unrestricted update" from
// becoming lost update when a handoff tree shares a ref.
func TestEditExpectRevisionConflict(t *testing.T) {
	testNotesDB(t)
	ref := mustStore(t, "sess-1", "shared note")

	if _, err := Edit(EditInput{Action: EditAppend, Ref: ref, SessionID: "sess-1", Content: "a", ExpectRevision: 1}, nil); err != nil {
		t.Fatal(err)
	}
	_, err := Edit(EditInput{Action: EditAppend, Ref: ref, SessionID: "sess-1", Content: "b", ExpectRevision: 1}, nil)
	if err == nil {
		t.Fatal("expected revision_conflict")
	}
	if errs.CodeOf(err) != errs.CodeConflict {
		t.Fatalf("expected conflict code, got %v", err)
	}
	if !strings.Contains(noteContent(t, ref, "sess-1"), "a") {
		t.Fatal("conflicting edit must not have been applied")
	}
}

func TestEditRevertRestoresPreviousBody(t *testing.T) {
	testNotesDB(t)
	original := "original body"
	ref := mustStore(t, "sess-1", original)

	if _, err := Edit(EditInput{Action: EditAppend, Ref: ref, SessionID: "sess-1", Content: "compacted away"}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := Edit(EditInput{Action: EditRewrite, Ref: ref, SessionID: "sess-1", Content: "wiped"}, nil); err != nil {
		t.Fatal(err)
	}
	res, err := Edit(EditInput{Action: EditRevert, Ref: ref, SessionID: "sess-1"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !res.Changed {
		t.Fatal("revert should report changed=true")
	}
	if got := noteContent(t, ref, "sess-1"); !strings.HasPrefix(got, "original body") {
		t.Fatalf("revert landed on the wrong revision: %q", got)
	}
	// Revision numbers only ever increase, so undo history survives the undo.
	if res.Revision <= res.PreviousRevision {
		t.Fatalf("revert did not advance the revision: %d -> %d", res.PreviousRevision, res.Revision)
	}
}

func TestEditRevertToSpecificRevision(t *testing.T) {
	testNotesDB(t)
	ref := mustStore(t, "sess-1", "v1")
	if _, err := Edit(EditInput{Action: EditAppend, Ref: ref, SessionID: "sess-1", Content: "v2"}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := Edit(EditInput{Action: EditAppend, Ref: ref, SessionID: "sess-1", Content: "v3"}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := Edit(EditInput{Action: EditRevert, Ref: ref, SessionID: "sess-1", ToRevision: 1}, nil); err != nil {
		t.Fatal(err)
	}
	if got := noteContent(t, ref, "sess-1"); !strings.HasPrefix(got, "v1") || strings.Contains(got, "v3") {
		t.Fatalf("revert to revision 1: %q", got)
	}
}

// context_notes_fts has no triggers, so an edit that skipped reindexing would
// leave search matching text the agent deliberately removed.
func TestEditReindexesSearch(t *testing.T) {
	testNotesDB(t)
	ref := mustStore(t, "sess-1", "uniqueneedletoken alpha beta")

	if _, err := Edit(EditInput{
		Action:      EditReplace,
		Ref:         ref,
		SessionID:   "sess-1",
		Pattern:     `uniqueneedletoken`,
		Replacement: "plaintext",
	}, nil); err != nil {
		t.Fatal(err)
	}
	res, err := Search("uniqueneedletoken", "sess-1", "", 5, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Notes) != 0 {
		t.Fatalf("stale FTS row still matches removed text: %d notes", len(res.Notes))
	}
	res, err = Search("plaintext", "sess-1", "", 5, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Notes) != 1 || res.Notes[0].Ref != ref {
		t.Fatalf("edited text not searchable: %+v", res.Notes)
	}
}

func TestEditGrowthRespectsQuota(t *testing.T) {
	testNotesDB(t)
	ref := mustStore(t, "sess-1", "small")

	_, err := Edit(EditInput{
		Action:    EditAppend,
		Ref:       ref,
		SessionID: "sess-1",
		Content:   strings.Repeat("padding ", 40000),
	}, nil)
	if err == nil {
		t.Fatal("expected a limit error growing a note past the session cap")
	}
	var le *LimitError
	if !errors.As(err, &le) {
		t.Fatalf("expected LimitError, got %v", err)
	}
	if got := noteContent(t, ref, "sess-1"); got != "small" {
		t.Fatalf("rejected edit still wrote: %q", got)
	}
}

func TestEditSessionOwnership(t *testing.T) {
	testNotesDB(t)
	ref := mustStore(t, "sess-1", "private")

	if _, err := Edit(EditInput{Action: EditAppend, Ref: ref, SessionID: "sess-2", Content: "x"}, nil); err == nil {
		t.Fatal("expected another session's note to be rejected")
	}
}

func TestEditRefusesToEmptyNote(t *testing.T) {
	testNotesDB(t)
	ref := mustStore(t, "sess-1", "delete me")

	if _, err := Edit(EditInput{Action: EditDelete, Ref: ref, SessionID: "sess-1", Pattern: `(?s).*`}, nil); err == nil {
		t.Fatal("expected an empty note to be refused")
	}
}

func TestEditInvalidAction(t *testing.T) {
	testNotesDB(t)
	ref := mustStore(t, "sess-1", "body")

	if _, err := Edit(EditInput{Action: "compact", Ref: ref}, nil); err == nil {
		t.Fatal("expected an unknown action to be rejected")
	}
}

func TestEditRejectsAmbiguousAddressing(t *testing.T) {
	testNotesDB(t)
	ref := mustStore(t, "sess-1", "one\ntwo\nthree")

	for _, in := range []EditInput{
		{Action: EditReplace, Ref: ref, SessionID: "sess-1"},
		{Action: EditDelete, Ref: ref, SessionID: "sess-1", Pattern: "a", StartLine: 1, EndLine: 2},
	} {
		if _, err := Edit(in, nil); err == nil {
			t.Fatalf("expected ambiguous addressing rejected: %+v", in)
		}
	}
}

func TestEditDryRunDoesNotWrite(t *testing.T) {
	testNotesDB(t)
	ref := mustStore(t, "sess-1", "keep\ndrop me")

	res, err := Edit(EditInput{
		Action:    EditDelete,
		Ref:       ref,
		SessionID: "sess-1",
		Pattern:   `drop me`,
		DryRun:    true,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Changed {
		t.Fatal("dry run reported a write")
	}
	if res.TokensReclaimed <= 0 {
		t.Fatalf("dry run should still report the token delta: %+v", res)
	}
	if got := noteContent(t, ref, "sess-1"); !strings.Contains(got, "drop me") {
		t.Fatalf("dry run wrote: %q", got)
	}
}

func TestEditRevisionRetentionIsBounded(t *testing.T) {
	testNotesDB(t)
	if err := db.SetSetting("context_max_revisions", "3"); err != nil {
		t.Fatal(err)
	}
	ref := mustStore(t, "sess-1", "v1")
	for i := 2; i <= 8; i++ {
		if _, err := Edit(EditInput{
			Action:    EditAppend,
			Ref:       ref,
			SessionID: "sess-1",
			Content:   "step" + strings.Repeat("x", i),
		}, nil); err != nil {
			t.Fatal(err)
		}
	}
	var count int
	if err := db.ContextDB.QueryRow(`SELECT COUNT(*) FROM context_note_revisions WHERE ref = ?`, ref).Scan(&count); err != nil {
		t.Fatal(err)
	}
	if count > 3 {
		t.Fatalf("revision log unbounded: %d rows", count)
	}
}

func TestEditUnknownRef(t *testing.T) {
	testNotesDB(t)
	if _, err := Edit(EditInput{Action: EditAppend, Ref: "ctx_deadbeef", Content: "x"}, nil); err == nil {
		t.Fatal("expected unknown ref to be rejected")
	}
}
