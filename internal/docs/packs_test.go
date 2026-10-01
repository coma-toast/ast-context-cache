package docs

import (
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
)

func TestStarterPackShape(t *testing.T) {
	if len(StarterPack) < 3 {
		t.Fatalf("expected at least 3 starter sources, got %d", len(StarterPack))
	}
	seen := map[string]bool{}
	for _, s := range StarterPack {
		if s.Name == "" || s.URL == "" || s.Type == "" {
			t.Fatalf("incomplete entry: %+v", s)
		}
		if seen[s.Name] {
			t.Fatalf("duplicate name %q", s.Name)
		}
		seen[s.Name] = true
	}
}

// stubDocFetch serves every doc fetch a small canned page instead of the network.
func stubDocFetch(t *testing.T) {
	t.Helper()
	t.Setenv("DOC_RENDER_DISABLE", "1")
	orig := docFetchClient
	docFetchClient = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"text/html"}},
			Body:       io.NopCloser(strings.NewReader("<html><body><h1>Stub</h1><p>Stub documentation for " + r.URL.String() + "</p></body></html>")),
			Request:    r,
		}, nil
	})}
	t.Cleanup(func() { docFetchClient = orig })
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// InstallStarterPack queues a background refresh per source. They used to fetch
// the real starter-pack URLs and write the results into this test's database
// after it had returned — sometimes while t.TempDir's cleanup was removing it.
func TestInstallStarterPack(t *testing.T) {
	dbtest.Init(t)
	stubDocFetch(t)
	t.Cleanup(refreshes.Wait)

	added, results := InstallStarterPack()
	if added != len(StarterPack) {
		t.Fatalf("added=%d want %d (results=%+v)", added, len(StarterPack), results)
	}
	if len(results) != len(StarterPack) {
		t.Fatalf("results len=%d want %d", len(results), len(StarterPack))
	}
	sources, err := ListSources()
	if err != nil {
		t.Fatal(err)
	}
	if len(sources) < len(StarterPack) {
		t.Fatalf("DB sources=%d want >= %d", len(sources), len(StarterPack))
	}

	// Idempotent: second install upserts the same rows.
	added2, _ := InstallStarterPack()
	if added2 != len(StarterPack) {
		t.Fatalf("second install added=%d want %d", added2, len(StarterPack))
	}
	sources2, err := ListSources()
	if err != nil {
		t.Fatal(err)
	}
	if len(sources2) != len(sources) {
		t.Fatalf("second install grew sources %d → %d", len(sources), len(sources2))
	}

	refreshes.Wait()
	for _, r := range results {
		entries, err := ListEntriesBySource(r.ID)
		if err != nil {
			t.Fatal(err)
		}
		if len(entries) == 0 {
			t.Fatalf("source %q (id %d) has no entries after its refresh", r.Name, r.ID)
		}
	}
}
