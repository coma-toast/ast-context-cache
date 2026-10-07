package contextnotes

import (
	"errors"
	"strings"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

// The paper's point is that the agent writes a function once and invokes it repeatedly
// across a trace. If defining one does not actually persist and re-invoke cleanly, the
// whole feature is a cache miss with extra steps.
func TestDefineAndApplyFnAcrossRefs(t *testing.T) {
	testNotesDB(t)
	refA := mustStore(t, "sess-1", "dead ends: grid(0.822)\ndead ends: hex(0.997)\nbest: 0.9992")
	refB := mustStore(t, "sess-1", "dead ends: ris(0.10)\nbest: 0.90")

	fn, err := DefineFn(FnDefineInput{
		Name: "compact_turns", Description: "drop dead-end candidates",
		Pattern: `dead ends: .*`, Replacement: "",
		ProjectPath: "/tmp/proj", SessionID: "sess-1",
	})
	if err != nil {
		t.Fatal(err)
	}
	if fn.Version != 1 {
		t.Fatalf("version: %d", fn.Version)
	}

	rep, err := ApplyFn(FnApplyInput{Name: "compact_turns", Refs: []string{refA, refB}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Changed != 2 {
		t.Fatalf("changed: %d of %d (errors=%v)", rep.Changed, rep.Targets, rep.Errors)
	}
	if rep.TokensReclaimed <= 0 {
		t.Fatalf("expected reclaimed tokens, got %d", rep.TokensReclaimed)
	}
	for _, ref := range []string{refA, refB} {
		if got := noteContent(t, ref, "sess-1"); strings.Contains(got, "dead ends") {
			t.Fatalf("fn did not apply to %s: %q", ref, got)
		}
	}

	stored, ok := fnByName("compact_turns")
	if !ok {
		t.Fatal("function not persisted")
	}
	if stored.CallCount != 1 || stored.NotesTouched != 2 {
		t.Fatalf("counters: calls=%d notes=%d", stored.CallCount, stored.NotesTouched)
	}
	if stored.TokensReclaimed != rep.TokensReclaimed {
		t.Fatalf("lifetime reclaimed %d != apply total %d", stored.TokensReclaimed, rep.TokensReclaimed)
	}
}

// A function applied across a whole session is the compact_turns case from the paper:
// no per-ref enumeration. Session sweep must reach every note the session owns.
func TestApplyFnAcrossSession(t *testing.T) {
	testNotesDB(t)
	refA := mustStore(t, "sess-2", "turn 1: noise\nturn 2: keep")
	refB := mustStore(t, "sess-2", "turn 1: more noise\nturn 2: also keep")
	other := mustStore(t, "sess-3", "turn 1: different session\nturn 2: theirs")

	if _, err := DefineFn(FnDefineInput{
		Name: "strip_turn1", Pattern: `turn 1: [^\n]*\n?`, Replacement: "",
	}); err != nil {
		t.Fatal(err)
	}

	rep, err := ApplyFn(FnApplyInput{Name: "strip_turn1", SessionID: "sess-2"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Targets != 2 || rep.Changed != 2 {
		t.Fatalf("targets=%d changed=%d errors=%v", rep.Targets, rep.Changed, rep.Errors)
	}
	if len(rep.Refs) != 2 {
		t.Fatalf("expected a per-ref result for each target: %+v", rep.Refs)
	}
	if got := noteContent(t, refA, "sess-2"); !strings.Contains(got, "turn 2: keep") {
		t.Fatalf("ref %s lost its kept turn: %q", refA, got)
	}
	if got := noteContent(t, refB, "sess-2"); !strings.Contains(got, "turn 2: also keep") {
		t.Fatalf("ref %s lost its kept turn: %q", refB, got)
	}
	// A session sweep must not cross into another session's notes.
	if got := noteContent(t, other, "sess-3"); !strings.Contains(got, "different session") {
		t.Fatalf("session sweep leaked into another session: %q", got)
	}
}

// Dry run is the only preview a batch edit gets. It must not write, and it must not
// credit the registry's own token-reclaimed metric — a dry run that looks like it
// saved 40k tokens makes call history a fiction.
func TestApplyFnDryRunDoesNotWriteOrCount(t *testing.T) {
	testNotesDB(t)
	ref := mustStore(t, "sess-1", "dead ends: grid(0.822)\nbest: 0.9992")
	if _, err := DefineFn(FnDefineInput{Name: "dry", Pattern: `dead ends: [^\n]*`, Replacement: ""}); err != nil {
		t.Fatal(err)
	}

	before := noteContent(t, ref, "sess-1")
	rep, err := ApplyFn(FnApplyInput{Name: "dry", Refs: []string{ref}, DryRun: true}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if rep.DryRun != true || rep.Changed != 0 {
		t.Fatalf("dry run reported a change: %+v", rep)
	}
	if rep.MatchedRegions == 0 {
		t.Fatal("dry run must still report match counts")
	}
	if got := noteContent(t, ref, "sess-1"); got != before {
		t.Fatalf("dry run wrote: %q -> %q", before, got)
	}
	stored, _ := fnByName("dry")
	if stored.CallCount != 0 || stored.TokensReclaimed != 0 {
		t.Fatalf("dry run moved counters: calls=%d tokens=%d", stored.CallCount, stored.TokensReclaimed)
	}
}

// Every apply lands in the revision log, which is what makes a bad sweep recoverable
// per note rather than a store-and-hope operation.
func TestApplyFnIsRevertiblePerNote(t *testing.T) {
	testNotesDB(t)
	ref := mustStore(t, "sess-1", "dead ends: grid(0.822)\nbest: 0.9992")
	original := noteContent(t, ref, "sess-1")
	if _, err := DefineFn(FnDefineInput{Name: "rev", Pattern: `dead ends: [^\n]*`, Replacement: ""}); err != nil {
		t.Fatal(err)
	}
	if _, err := ApplyFn(FnApplyInput{Name: "rev", Refs: []string{ref}}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := Edit(EditInput{Action: EditRevert, Ref: ref}, nil); err != nil {
		t.Fatal(err)
	}
	if got := noteContent(t, ref, "sess-1"); got != original {
		t.Fatalf("revert did not restore: %q != %q", got, original)
	}
}

// Redefining a shared name is the fn-level analogue of the lost update expect_revision
// prevents on notes: two agents define the same transform and one silently wins.
func TestDefineFnExpectVersionConflict(t *testing.T) {
	testNotesDB(t)
	if _, err := DefineFn(FnDefineInput{Name: "shared", Pattern: "a", Replacement: "b"}); err != nil {
		t.Fatal(err)
	}
	_, err := DefineFn(FnDefineInput{Name: "shared", Pattern: "c", Replacement: "d", ExpectVersion: 99})
	if err == nil {
		t.Fatal("expected revision_conflict")
	}
	if !strings.Contains(err.Error(), "revision_conflict") {
		t.Fatalf("error: %v", err)
	}
	// The right version succeeds and bumps.
	fn, err := DefineFn(FnDefineInput{Name: "shared", Pattern: "c", Replacement: "d", ExpectVersion: 1})
	if err != nil {
		t.Fatal(err)
	}
	if fn.Version != 2 {
		t.Fatalf("version after redefine: %d", fn.Version)
	}
}

// A redefine resets the counters, so call_count measures the definition in force rather
// than a lifetime total that hides how often the current version actually gets used.
func TestDefineFnResetsCountersOnRedefine(t *testing.T) {
	testNotesDB(t)
	ref := mustStore(t, "sess-1", "dead ends: x\nkeep")
	if _, err := DefineFn(FnDefineInput{Name: "reset", Pattern: `dead ends: [^\n]*`, Replacement: ""}); err != nil {
		t.Fatal(err)
	}
	if _, err := ApplyFn(FnApplyInput{Name: "reset", Refs: []string{ref}}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := DefineFn(FnDefineInput{Name: "reset", Pattern: `keep`, Replacement: "kept"}); err != nil {
		t.Fatal(err)
	}
	stored, _ := fnByName("reset")
	if stored.CallCount != 0 || stored.NotesTouched != 0 || stored.TokensReclaimed != 0 {
		t.Fatalf("counters survived redefine: %+v", stored)
	}
}

// A pattern that cannot compile is dead weight at apply time. Rejecting it at
// definition time means the agent learns before it has stored 50 copies of it.
func TestDefineFnRejectsBadInput(t *testing.T) {
	testNotesDB(t)
	if _, err := DefineFn(FnDefineInput{Name: "bad-re", Pattern: "[z-a]"}); err == nil {
		t.Fatal("expected invalid pattern rejection")
	}
	if _, err := DefineFn(FnDefineInput{Name: "no pattern"}); err == nil {
		t.Fatal("expected pattern required")
	}
	if _, err := DefineFn(FnDefineInput{Name: "bad name!", Pattern: "a"}); err == nil {
		t.Fatal("expected name charset rejection")
	}
	if _, err := DefineFn(FnDefineInput{Name: strings.Repeat("x", 65), Pattern: "a"}); err == nil {
		t.Fatal("expected name length rejection")
	}
}

// An apply must stop at the first failing note by default: a half-applied sweep is
// harder to reason about than one that stopped cleanly and reported why.
func TestApplyFnStopsOnErrorUnlessSkipping(t *testing.T) {
	testNotesDB(t)
	ref := mustStore(t, "sess-1", "dead ends: x\nkeep")
	if _, err := DefineFn(FnDefineInput{Name: "boom", Pattern: "dead ends: [^\n]*", Replacement: ""}); err != nil {
		t.Fatal(err)
	}

	rep, err := ApplyFn(FnApplyInput{Name: "boom", Refs: []string{"ctx_missing", ref}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Failed != 1 {
		t.Fatalf("failed: %d", rep.Failed)
	}
	if rep.Changed != 0 {
		t.Fatalf("apply continued past a failure without skip_errors: %d changed", rep.Changed)
	}
	if len(rep.Errors) == 0 {
		t.Fatal("expected a per-ref error entry")
	}

	rep2, err := ApplyFn(FnApplyInput{Name: "boom", Refs: []string{"ctx_missing", ref}, SkipErrors: true}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if rep2.Changed != 1 {
		t.Fatalf("skip_errors did not continue: %d changed", rep2.Changed)
	}
}

// max_replacements is the only brake on a function that matches everything, applied at
// session width.
func TestApplyFnMaxReplacements(t *testing.T) {
	testNotesDB(t)
	ref := mustStore(t, "sess-1", "x1\nx2\nx3\nx4\nx5")
	if _, err := DefineFn(FnDefineInput{Name: "cap", Pattern: "x[0-9]", Replacement: "y"}); err != nil {
		t.Fatal(err)
	}
	rep, err := ApplyFn(FnApplyInput{Name: "cap", Refs: []string{ref}, MaxReplacements: 2}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if rep.MatchedRegions != 2 {
		t.Fatalf("matched %d regions, want 2", rep.MatchedRegions)
	}
	if got := noteContent(t, ref, "sess-1"); got != "y\ny\nx3\nx4\nx5" {
		t.Fatalf("cap not honoured: %q", got)
	}
}

// Group references make a stored function genuinely reusable rather than a fixed
// string swap, which is the transform an agent actually writes to normalize notes.
func TestApplyFnGroupReferences(t *testing.T) {
	testNotesDB(t)
	ref := mustStore(t, "sess-1", "budget: 7/100")
	if _, err := DefineFn(FnDefineInput{
		Name: "renorm", Pattern: `budget: (\d+)/(\d+)`, Replacement: "budget: $1 of $2",
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := ApplyFn(FnApplyInput{Name: "renorm", Refs: []string{ref}}, nil); err != nil {
		t.Fatal(err)
	}
	if got := noteContent(t, ref, "sess-1"); got != "budget: 7 of 100" {
		t.Fatalf("groups not expanded: %q", got)
	}
}

// Retiring is a tombstone, not a delete: the name's history stays auditable and a
// retired function must refuse to run.
func TestRetireFn(t *testing.T) {
	testNotesDB(t)
	ref := mustStore(t, "sess-1", "dead ends: x\nkeep")
	if _, err := DefineFn(FnDefineInput{Name: "old", Pattern: "dead ends: [^\n]*", Replacement: ""}); err != nil {
		t.Fatal(err)
	}
	ok, err := RetireFn("old")
	if err != nil || !ok {
		t.Fatalf("retire: %v %v", ok, err)
	}
	if _, err := ApplyFn(FnApplyInput{Name: "old", Refs: []string{ref}}, nil); err == nil {
		t.Fatal("retired function still applied")
	} else if !strings.Contains(err.Error(), "function_retired") {
		t.Fatalf("error: %v", err)
	}
	if _, err := RetireFn("nope"); err == nil {
		t.Fatal("expected not-found for unknown name")
	}
}

// The registry needs a cap for the same reason the note store does: each entry is a
// transform that can be applied at scale, so an unbounded registry is unbounded
// influence over context.
func TestDefineFnRespectsMaxFns(t *testing.T) {
	testNotesDB(t)
	db.SetSetting("context_max_fns", "2")
	for i, name := range []string{"a", "b"} {
		if _, err := DefineFn(FnDefineInput{Name: name, Pattern: "x"}); err != nil {
			t.Fatalf("define %d: %v", i, err)
		}
	}
	_, err := DefineFn(FnDefineInput{Name: "c", Pattern: "x"})
	if err == nil {
		t.Fatal("expected context_fns_limit")
	}
	var le *LimitError
	if !errors.As(err, &le) {
		t.Fatalf("expected LimitError, got %T: %v", err, err)
	}
	// Retiring frees a slot; redefining an existing name must not be blocked by the cap.
	if _, err := RetireFn("a"); err != nil {
		t.Fatal(err)
	}
	if _, err := DefineFn(FnDefineInput{Name: "c", Pattern: "x"}); err != nil {
		t.Fatal(err)
	}
	if _, err := DefineFn(FnDefineInput{Name: "a", Pattern: "y"}); err != nil {
		t.Fatal(err)
	}
}

func TestListFnsMetadataOnly(t *testing.T) {
	testNotesDB(t)
	ref := mustStore(t, "sess-1", "dead ends: x\nkeep")
	if _, err := DefineFn(FnDefineInput{Name: "listed", Description: "d", Pattern: "dead ends: [^\n]*", Replacement: "", ProjectPath: "/tmp/proj"}); err != nil {
		t.Fatal(err)
	}
	if _, err := ApplyFn(FnApplyInput{Name: "listed", Refs: []string{ref}}, nil); err != nil {
		t.Fatal(err)
	}
	fns := ListFns("/tmp/proj", 10)
	if len(fns) != 1 {
		t.Fatalf("fns: %d", len(fns))
	}
	if fns[0].Pattern != "" {
		t.Fatalf("list leaked a pattern: %q", fns[0].Pattern)
	}
	if fns[0].Description != "d" || fns[0].CallCount != 1 {
		t.Fatalf("metadata: %+v", fns[0])
	}
}

// A note whose match empties it is refused by Edit, and a session sweep must survive
// that as a reported failure rather than taking the whole run down.
func TestApplyFnEmptyResultIsReportedNotFatal(t *testing.T) {
	testNotesDB(t)
	ref := mustStore(t, "sess-1", "only this")
	if _, err := DefineFn(FnDefineInput{Name: "wipe", Pattern: `[\s\S]*`, Replacement: ""}); err != nil {
		t.Fatal(err)
	}
	rep, err := ApplyFn(FnApplyInput{Name: "wipe", Refs: []string{ref}, SkipErrors: true}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Failed != 1 || rep.Changed != 0 {
		t.Fatalf("wipe should have been refused: %+v", rep)
	}
	if got := noteContent(t, ref, "sess-1"); got != "only this" {
		t.Fatalf("note was emptied anyway: %q", got)
	}
}
