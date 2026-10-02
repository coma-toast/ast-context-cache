package memory

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func storeFacts(t *testing.T, sessionID string, n int) []string {
	t.Helper()
	var refs []string
	for i := 0; i < n; i++ {
		res, err := Store(StoreInput{Kind: KindFact, Scope: ScopeSession, SessionID: sessionID,
			Subject: fmt.Sprintf("item%d", i), Predicate: "is", Object: "x"})
		if err != nil {
			t.Fatal(err)
		}
		refs = append(refs, res.Ref)
	}
	return refs
}

func activeCount(t *testing.T, refs []string) int {
	t.Helper()
	n := 0
	for _, ref := range refs {
		var vu interface{}
		if err := db.ContextDB.QueryRow(`SELECT valid_until FROM structured_memory WHERE ref = ?`, ref).Scan(&vu); err != nil {
			t.Fatal(err)
		}
		if vu == nil || vu == "" {
			n++
		}
	}
	return n
}

// Issue #8: many refs, no scope/session, one unknown ref.
func TestForgetManyRefsWithoutScope(t *testing.T) {
	testMemoryDB(t)
	refs := storeFacts(t, "sess-a", 3)
	proj, err := Store(StoreInput{Kind: KindProcedure, Scope: ScopeProject, ProjectPath: "/p", Rule: "project rule"})
	if err != nil {
		t.Fatal(err)
	}
	all := append(append([]string{}, refs...), proj.Ref, "mem_doesnotexist")
	res, err := Forget(ForgetInput{Refs: all})
	if err != nil {
		t.Fatal(err)
	}
	if res.InvalidatedRefs != 4 || len(res.Invalidated) != 4 {
		t.Fatalf("invalidated=%d %v, want 4", res.InvalidatedRefs, res.Invalidated)
	}
	if !reflect.DeepEqual(res.NotFound, []string{"mem_doesnotexist"}) {
		t.Fatalf("not_found=%v", res.NotFound)
	}
	if res.VirtualTokensFreed <= 0 {
		t.Fatalf("tokens freed=%d", res.VirtualTokensFreed)
	}
	if n := activeCount(t, append(refs, proj.Ref)); n != 0 {
		t.Fatalf("%d refs still active", n)
	}
}

func TestForgetRefsAlreadyInvalidAndIdempotent(t *testing.T) {
	testMemoryDB(t)
	refs := storeFacts(t, "sess-b", 1)
	if _, err := Forget(ForgetInput{Refs: refs}); err != nil {
		t.Fatal(err)
	}
	var before string
	db.ContextDB.QueryRow(`SELECT valid_until FROM structured_memory WHERE ref = ?`, refs[0]).Scan(&before)
	res, err := Forget(ForgetInput{Refs: refs})
	if err != nil {
		t.Fatal(err)
	}
	if res.InvalidatedRefs != 0 || !reflect.DeepEqual(res.AlreadyInvalid, refs) {
		t.Fatalf("second forget=%+v", res)
	}
	var after string
	db.ContextDB.QueryRow(`SELECT valid_until FROM structured_memory WHERE ref = ?`, refs[0]).Scan(&after)
	if before != after {
		t.Fatalf("valid_until rewritten %q -> %q", before, after)
	}
}

// An explicit scope with refs is a guard, never a widening.
func TestForgetRefsScopeGuard(t *testing.T) {
	testMemoryDB(t)
	mine := storeFacts(t, "sess-mine", 2)
	other := storeFacts(t, "sess-other", 1)
	res, err := Forget(ForgetInput{Refs: append(append([]string{}, mine...), other...), Scope: ScopeSession, SessionID: "sess-mine"})
	if err != nil {
		t.Fatal(err)
	}
	if res.InvalidatedRefs != 2 || !reflect.DeepEqual(res.ScopeMismatch, other) {
		t.Fatalf("res=%+v", res)
	}
	if activeCount(t, other) != 1 {
		t.Fatal("ref from another session was invalidated despite scope guard")
	}
	res, err = Forget(ForgetInput{Refs: other, Scope: ScopeProject})
	if err != nil || res.InvalidatedRefs != 0 || len(res.ScopeMismatch) != 1 {
		t.Fatalf("project guard on session ref: res=%+v err=%v", res, err)
	}
}

func TestForgetRejectsRefsWithAllAndBadScope(t *testing.T) {
	testMemoryDB(t)
	keep := storeFacts(t, "sess-c", 2)
	if _, err := Forget(ForgetInput{Refs: keep[:1], All: true}); err == nil {
		t.Fatal("refs + all=true should be rejected")
	}
	if _, err := Forget(ForgetInput{Subject: "item0", Scope: "bogus"}); err == nil {
		t.Fatal("invalid scope should be rejected")
	}
	if activeCount(t, keep) != 2 {
		t.Fatal("rejected calls must not invalidate anything")
	}
}

// Non-ref modes keep their existing behavior.
func TestForgetSubjectModeStaysSessionScoped(t *testing.T) {
	testMemoryDB(t)
	a := storeFacts(t, "sess-d", 1)
	b := storeFacts(t, "sess-e", 1)
	res, err := Forget(ForgetInput{Subject: "item0", SessionID: "sess-d"})
	if err != nil || res.InvalidatedRefs != 1 {
		t.Fatalf("res=%+v err=%v", res, err)
	}
	if activeCount(t, a) != 0 || activeCount(t, b) != 1 {
		t.Fatal("subject forget crossed sessions")
	}
	res, err = Forget(ForgetInput{Subject: "item0"})
	if err != nil || res.InvalidatedRefs != 0 {
		t.Fatalf("unscoped subject forget should match nothing: res=%+v err=%v", res, err)
	}
}
