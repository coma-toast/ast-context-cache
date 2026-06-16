package memory

import (
	"strings"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func testMemoryDB(t *testing.T) {
	t.Helper()
	t.Setenv("HOME", t.TempDir())
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
}

func TestExtractFromText(t *testing.T) {
	ex := ExtractFromText(`FACT: user.stack | prefers | Go errs package
RULE: Use skeleton mode when exploring unfamiliar code
user.style: compact functions`)
	if len(ex.Facts) != 2 {
		t.Fatalf("facts=%d", len(ex.Facts))
	}
	if len(ex.Procedures) != 1 {
		t.Fatalf("procedures=%d", len(ex.Procedures))
	}
	if ex.Facts[0].Subject != "user.stack" || ex.Facts[0].Predicate != "prefers" {
		t.Fatalf("fact0: %+v", ex.Facts[0])
	}
}

func TestStoreFactInvalidatesPrevious(t *testing.T) {
	testMemoryDB(t)
	r1, err := Store(StoreInput{
		Kind: KindFact, Scope: ScopeSession, SessionID: "s1",
		Subject: "user.city", Predicate: "lives_in", Object: "Mumbai",
	})
	if err != nil {
		t.Fatal(err)
	}
	r2, err := Store(StoreInput{
		Kind: KindFact, Scope: ScopeSession, SessionID: "s1",
		Subject: "user.city", Predicate: "lives_in", Object: "Bangalore",
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(r2.InvalidatedRefs) != 1 || r2.InvalidatedRefs[0] != r1.Ref {
		t.Fatalf("invalidated=%v want %s", r2.InvalidatedRefs, r1.Ref)
	}
	rec, err := Recall(RecallInput{SessionID: "s1", Query: "city", TokenBudget: 400}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(rec.Formatted, "Bangalore") {
		t.Fatalf("formatted=%q", rec.Formatted)
	}
	if strings.Contains(rec.Formatted, "Mumbai") {
		t.Fatalf("stale fact returned: %q", rec.Formatted)
	}
}

func TestStoreProcedureAndRecall(t *testing.T) {
	testMemoryDB(t)
	_, err := Store(StoreInput{
		Kind: KindProcedure, Scope: ScopeSession, SessionID: "s2",
		Rule: "Prefer get_file_context skeleton before full reads",
	})
	if err != nil {
		t.Fatal(err)
	}
	rec, err := Recall(RecallInput{SessionID: "s2", Kinds: []Kind{KindProcedure}, TokenBudget: 200}, nil)
	if err != nil || len(rec.Lines) != 1 {
		t.Fatalf("recall=%+v err=%v", rec, err)
	}
	if !strings.Contains(rec.Lines[0].Line, "skeleton") {
		t.Fatalf("line=%q", rec.Lines[0].Line)
	}
}
