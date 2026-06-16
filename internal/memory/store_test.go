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

func TestForgetByRef(t *testing.T) {
	testMemoryDB(t)
	res, err := Store(StoreInput{
		Kind: KindFact, Scope: ScopeSession, SessionID: "s3",
		Subject: "user.lang", Predicate: "prefers", Object: "Go",
	})
	if err != nil {
		t.Fatal(err)
	}
	forgot, err := Forget(ForgetInput{Refs: []string{res.Ref}})
	if err != nil || forgot.InvalidatedRefs != 1 {
		t.Fatalf("forget=%+v err=%v", forgot, err)
	}
	rec, err := Recall(RecallInput{SessionID: "s3", Query: "lang", TokenBudget: 200}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if rec.Formatted != "" {
		t.Fatalf("expected empty recall after forget, got %q", rec.Formatted)
	}
}

func TestStoreExtracted(t *testing.T) {
	testMemoryDB(t)
	ex := ExtractFromText("FACT: api.version | is | 2\nRULE: Always pass session_id on search")
	stored, err := StoreExtracted("s4", "", "", ex, ScopeSession)
	if err != nil {
		t.Fatal(err)
	}
	if len(stored) != 2 {
		t.Fatalf("stored=%d", len(stored))
	}
	rec, err := Recall(RecallInput{SessionID: "s4", TokenBudget: 400}, nil)
	if err != nil || len(rec.Lines) < 2 {
		t.Fatalf("recall lines=%d err=%v", len(rec.Lines), err)
	}
}

func TestRecallTokenBudget(t *testing.T) {
	testMemoryDB(t)
	for i := 0; i < 20; i++ {
		_, err := Store(StoreInput{
			Kind: KindFact, Scope: ScopeSession, SessionID: "s5",
			Subject: "item.count", Predicate: "equals", Object: strings.Repeat("x", 20) + string(rune('a'+i)),
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	rec, err := Recall(RecallInput{SessionID: "s5", TokenBudget: 80}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if rec.TokensUsed > 80 {
		t.Fatalf("tokens_used=%d budget=80", rec.TokensUsed)
	}
	if len(rec.Lines) >= 20 {
		t.Fatalf("expected budget to limit lines, got %d", len(rec.Lines))
	}
}
