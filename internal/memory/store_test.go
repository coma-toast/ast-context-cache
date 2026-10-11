package memory

import (
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
)

func testMemoryDB(t *testing.T) {
	t.Helper()
	dbtest.Init(t)
}

func TestExtractFromText(t *testing.T) {
	ex := ExtractFromText(`FACT: user.stack | prefers | Go errs package
RULE: Use skeleton mode when exploring unfamiliar code
user.style: compact functions`)
	if len(ex.Facts) != 1 {
		t.Fatalf("facts=%d (unmarked lines must not become facts): %+v", len(ex.Facts), ex.Facts)
	}
	if len(ex.Procedures) != 1 {
		t.Fatalf("procedures=%d", len(ex.Procedures))
	}
	if ex.Facts[0].Subject != "user.stack" || ex.Facts[0].Predicate != "prefers" {
		t.Fatalf("fact0: %+v", ex.Facts[0])
	}
}

// fieldReportNote mirrors the field-report note (issue #7): headings, prose,
// path:line text, and exactly 2 FACT: + 2 RULE: lines.
const fieldReportNote = `# bonsai model routing

## BROKEN NOW
bonsai/plugin.py:33 SERVICE_MODEL=Ternary-Bonsai is hardcoded
litellm_sync.py:120 to_litellm_params drops api_base

## WRONG BEHAVIOR
The sync job overwrites manual edits: every run.
- Status: investigating
Note: see ctx_fe05bd0e56a6 for the full trace.

` + "```" + `
FACT: inside | a | code fence
RULE: this is example code, not a rule
` + "```" + `

FACT: bonsai.service_model | is_set_at | bonsai/plugin.py:33 SERVICE_MODEL=Ternary-Bonsai
- FACT: litellm_sync.py:120 drops api_base when to_litellm_params: runs
RULE: Never hardcode SERVICE_MODEL; read it from config: models.yaml
* rule: Run sync-litellm-models --dry-run before applying
`

func TestExtractFromTextOnlyMarkedLines(t *testing.T) {
	ex := ExtractFromText(fieldReportNote)
	if len(ex.Facts) != 2 || len(ex.Procedures) != 2 {
		t.Fatalf("want 2 facts + 2 rules, got facts=%+v procedures=%+v", ex.Facts, ex.Procedures)
	}
	gotFacts := []string{
		FormatLine(Entry{Kind: KindFact, Subject: ex.Facts[0].Subject, Predicate: ex.Facts[0].Predicate, Object: ex.Facts[0].Object}),
		FormatLine(Entry{Kind: KindFact, Subject: ex.Facts[1].Subject, Predicate: ex.Facts[1].Predicate, Object: ex.Facts[1].Object}),
	}
	wantFacts := []string{
		"bonsai.service_model is_set_at bonsai/plugin.py:33 SERVICE_MODEL=Ternary-Bonsai",
		"litellm_sync.py:120 drops api_base when to_litellm_params: runs",
	}
	for i := range wantFacts {
		if gotFacts[i] != wantFacts[i] {
			t.Fatalf("fact %d = %q, want %q (text must be preserved)", i, gotFacts[i], wantFacts[i])
		}
	}
	wantRules := []string{
		"Never hardcode SERVICE_MODEL; read it from config: models.yaml",
		"Run sync-litellm-models --dry-run before applying",
	}
	for i := range wantRules {
		if ex.Procedures[i].Rule != wantRules[i] {
			t.Fatalf("rule %d = %q, want %q", i, ex.Procedures[i].Rule, wantRules[i])
		}
	}
	for _, f := range ex.Facts {
		if f.Predicate == "is" || strings.HasPrefix(f.Subject, "#") {
			t.Fatalf("mangled or heading fact: %+v", f)
		}
	}
	if len(ex.Skipped) != 0 {
		t.Fatalf("skipped=%v", ex.Skipped)
	}
}

func TestExtractFromTextReportsUnparseableMarkedLines(t *testing.T) {
	ex := ExtractFromText("FACT: bonsai/plugin.py:33 SERVICE_MODEL=Ternary-Bonsai\nRULE:\nFACT: a | b | c")
	if len(ex.Facts) != 1 || len(ex.Procedures) != 0 {
		t.Fatalf("facts=%+v procedures=%+v", ex.Facts, ex.Procedures)
	}
	if len(ex.Skipped) != 2 || ex.Skipped[0] != "FACT: bonsai/plugin.py:33 SERVICE_MODEL=Ternary-Bonsai" {
		t.Fatalf("skipped=%q", ex.Skipped)
	}
}

func TestStoreExtractedFieldReportNote(t *testing.T) {
	testMemoryDB(t)
	stored, err := StoreExtracted("s-note", "", "ctx_test", ExtractFromText(fieldReportNote), ScopeSession)
	if err != nil {
		t.Fatal(err)
	}
	if len(stored) != 4 {
		t.Fatalf("stored %d entries, want 4: %+v", len(stored), stored)
	}
	var n int
	if err := db.ContextDB.QueryRow(`SELECT COUNT(*) FROM structured_memory WHERE session_id = 's-note'`).Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n != 4 {
		t.Fatalf("structured_memory rows=%d, want 4", n)
	}
	for _, s := range stored {
		if strings.Contains(s.Line, "plugin.py is 33") || strings.Contains(s.Line, "BROKEN") {
			t.Fatalf("mangled line stored: %q", s.Line)
		}
	}
}

func TestStoreFactInvalidatesPrevious(t *testing.T) {
	testMemoryDB(t)
	r1, err := Store(StoreInput{
		Kind: KindFact, Scope: ScopeSession, SessionID: "s1",
		Subject: "user.city", Predicate: "lives_in", Object: "Mumbai", InvalidatePrevious: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	r2, err := Store(StoreInput{
		Kind: KindFact, Scope: ScopeSession, SessionID: "s1",
		Subject: "user.city", Predicate: "lives_in", Object: "Bangalore", InvalidatePrevious: true,
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

// Concurrent store_memory calls for the same subject/predicate/scope must
// still leave exactly one active fact. Without factSupersessionMu serializing
// invalidateConflicting's SELECT and the following INSERT, two goroutines can
// both see "no active fact yet" and both insert, leaving duplicates.
func TestStoreFactConcurrentSupersessionLeavesOneActiveFact(t *testing.T) {
	testMemoryDB(t)
	const n = 20
	var wg sync.WaitGroup
	errs := make(chan error, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, err := Store(StoreInput{
				Kind: KindFact, Scope: ScopeSession, SessionID: "concurrent",
				Subject: "user.city", Predicate: "lives_in", Object: fmt.Sprintf("City%d", i), InvalidatePrevious: true,
			})
			errs <- err
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	var active int
	if err := db.ContextDB.QueryRow(`SELECT COUNT(*) FROM structured_memory WHERE subject = 'user.city' AND predicate = 'lives_in' AND (valid_until IS NULL OR valid_until = '')`).Scan(&active); err != nil {
		t.Fatal(err)
	}
	if active != 1 {
		t.Fatalf("active facts for user.city/lives_in = %d, want exactly 1", active)
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
