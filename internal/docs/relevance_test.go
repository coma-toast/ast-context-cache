package docs

import (
	"math"
	"testing"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

// Field report #11 query that matched nothing relevant in the cache.
const pep562Query = "module __getattr__ PEP 562 lazy attribute"

var (
	tailscaleSection = DocEntry{
		Title:   "Tailscale API: devices",
		Content: "List devices in the tailnet. Each device object has an attribute for its hostname, addresses and tags. The API module returns JSON.",
	}
	proxmoxSection = DocEntry{
		Title:   "ProxmoxVED AGENTS.md",
		Content: "Agents should build LXC templates with the helper scripts. Each script is a bash module sourced by the installer.",
	}
	pepSection = DocEntry{
		Title:   "PEP 562 – Module __getattr__ and __dir__",
		Content: "This PEP proposes to support __getattr__ and __dir__ functions defined on modules, enabling lazy loading of submodules and deprecation warnings for module attributes.",
	}
)

// waitDocRefreshIdle waits out doc refreshes other tests left running (InstallStarterPack's
// ForceRefreshSource goroutines, TryQuietRefresh's UpdateAllSources). UpdateSource reads
// and writes through the package-global ContextDB by source id, so one that finishes
// after this test's db.Init replaces source 1's sections with whatever it fetched.
func waitDocRefreshIdle(t *testing.T) {
	t.Helper()
	deadline := time.Now().Add(60 * time.Second) // > docFetchClient's 20s timeout
	for {
		refreshMu.Lock()
		n := len(refreshing)
		refreshMu.Unlock()
		quietRefreshMu.Lock()
		busy := quietRefreshBusy
		quietRefreshMu.Unlock()
		if n == 0 && !busy {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("background doc refreshes still running (%d force, quiet=%v)", n, busy)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func seedDocs(t *testing.T, sections ...DocEntry) {
	t.Helper()
	waitDocRefreshIdle(t)
	t.Setenv("HOME", t.TempDir())
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	id, err := AddSource("test-docs", "markdown", "https://example.com/docs", "")
	if err != nil {
		t.Fatal(err)
	}
	if err := storeEntries(id, sections); err != nil {
		t.Fatal(err)
	}
	// Fresh, so a quiet refresh started later never re-fetches the fake URL over it.
	if _, err := db.ContextDB.Exec("UPDATE doc_sources SET last_updated = ? WHERE id = ?", time.Now().Format(time.RFC3339), id); err != nil {
		t.Fatal(err)
	}
}

func TestFusedScoreIsRankOnly(t *testing.T) {
	// Why the floor cannot live on the returned score: RRF gives the top hit the same
	// score whatever it is — the 0.016–0.03 band the field report saw.
	a := ScoredDoc{Entry: DocEntry{ID: 1}}
	b := ScoredDoc{Entry: DocEntry{ID: 2}}
	out := fuseDocResults([]ScoredDoc{a}, []ScoredDoc{a, b}, 10)
	if math.Abs(out[0].Score-2.0/61) > 1e-9 || math.Abs(out[1].Score-1.0/62) > 1e-9 {
		t.Fatalf("fused scores %v/%v, want 2/61 and 1/62", out[0].Score, out[1].Score)
	}
}

func TestTermCoverage(t *testing.T) {
	terms := coverageTerms(pep562Query)
	if len(terms) != 6 {
		t.Fatalf("terms=%v want 6 content terms", terms)
	}
	if c := termCoverage(terms, tailscaleSection); c >= docMinTermCoverage {
		t.Fatalf("unrelated Tailscale section coverage %.2f passes the floor", c)
	}
	if c := termCoverage(terms, proxmoxSection); c >= docMinTermCoverage {
		t.Fatalf("unrelated Proxmox section coverage %.2f passes the floor", c)
	}
	if c := termCoverage(terms, pepSection); c < 0.99 {
		t.Fatalf("PEP 562 section coverage %.2f, want 1", c)
	}
	// Stopwords do not count toward (or against) coverage.
	if got := coverageTerms("how to list devices"); len(got) != 2 {
		t.Fatalf("stopwords kept: %v", got)
	}
	if got := coverageTerms("the"); len(got) != 1 {
		t.Fatalf("all-stopword query must still have terms: %v", got)
	}
}

func TestSearchDocsNoMatchBelowFloor(t *testing.T) {
	seedDocs(t, tailscaleSection, proxmoxSection)
	r, err := SearchDocsLexical(pep562Query, 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Docs) != 0 {
		t.Fatalf("unrelated sections returned: %+v", r.Docs)
	}
	if r.BelowFloor != 2 {
		t.Fatalf("below_floor=%d want 2 (both OR-matched on module/attribute)", r.BelowFloor)
	}
	hybrid, err := SearchDocsHybridResult(pep562Query, 5, nil)
	if err != nil || len(hybrid.Docs) != 0 {
		t.Fatalf("hybrid (no embedder) returned %+v err=%v", hybrid.Docs, err)
	}
}

func TestSearchDocsRealMatchesStillReturned(t *testing.T) {
	seedDocs(t, tailscaleSection, proxmoxSection, pepSection)
	r, err := SearchDocsLexical(pep562Query, 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Docs) != 1 || r.Docs[0].Entry.Title != pepSection.Title {
		t.Fatalf("want only the PEP 562 section, got %+v", r.Docs)
	}
	for _, q := range []string{"tailscale devices", "tailscale", "LXC templates", "how to list devices"} {
		entries, err := SearchDocs(q, 5)
		if err != nil || len(entries) == 0 {
			t.Fatalf("query %q: real match dropped (err=%v)", q, err)
		}
	}
	// Dunder identifiers still match (tokenized like FTS: "__dir__" -> "dir").
	entries, err := SearchDocs("__dir__", 5)
	if err != nil || len(entries) != 1 {
		t.Fatalf("__dir__ lookup got %d entries err=%v", len(entries), err)
	}
}

func TestVectorFloor(t *testing.T) {
	t.Setenv("AST_DOCS_MIN_VECTOR_SIMILARITY", "")
	terms := coverageTerms(pep562Query)
	hits := []ScoredDoc{
		{Entry: DocEntry{ID: 1, Title: tailscaleSection.Title, Content: tailscaleSection.Content}, Score: 0.45},
		{Entry: DocEntry{ID: 2, Title: pepSection.Title, Content: pepSection.Content}, Score: 0.41},
		{Entry: DocEntry{ID: 3, Title: "Import system", Content: "Loader hooks and importlib."}, Score: 0.72},
	}
	kept := applyVectorFloor(terms, hits)
	ids := map[int]bool{}
	for _, k := range kept {
		ids[k.Entry.ID] = true
	}
	if ids[1] || !ids[2] || !ids[3] || len(kept) != 2 {
		t.Fatalf("kept %v: want low-sim unrelated dropped, lexical-supported and high-sim kept", ids)
	}
	if kept[0].Similarity != 0.41 {
		t.Fatalf("similarity not recorded: %+v", kept[0])
	}
	t.Setenv("AST_DOCS_MIN_VECTOR_SIMILARITY", "0.4")
	if got := len(applyVectorFloor(terms, hits)); got != 3 {
		t.Fatalf("env override: kept %d want 3", got)
	}
}
