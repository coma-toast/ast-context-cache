package docs

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"testing"
)

func TestNormalizeDocType(t *testing.T) {
	ResetRenderProbeForTest()
	os.Setenv("DOC_RENDER_DISABLE", "1")
	defer os.Unsetenv("DOC_RENDER_DISABLE")

	if got := NormalizeDocType("html", true); got != "html" {
		t.Fatalf("render_js without playwright => html, got %q", got)
	}
	if got := NormalizeDocType("webpage", false); got != "html" {
		t.Fatalf("webpage without playwright => html, got %q", got)
	}
	if got := NormalizeDocType("html", false); got != "html" {
		t.Fatalf("plain html, got %q", got)
	}
	if got := NormalizeDocType("md", false); got != "markdown" {
		t.Fatalf("md alias, got %q", got)
	}
}

func TestEntriesSparse(t *testing.T) {
	if !entriesSparse([]DocEntry{{Content: "short"}}) {
		t.Fatal("short content should be sparse")
	}
	long := make([]byte, 250)
	for i := range long {
		long[i] = 'x'
	}
	if entriesSparse([]DocEntry{{Content: string(long)}}) {
		t.Fatal("250 chars should not be sparse")
	}
}

func TestDocTypeUsesRender(t *testing.T) {
	if !docTypeUsesRender("webpage") {
		t.Fatal("webpage should use render")
	}
	if docTypeUsesRender("html") {
		t.Fatal("html should not use render by default")
	}
}

func TestRenderDisabled(t *testing.T) {
	ResetRenderProbeForTest()
	t.Setenv("DOC_RENDER_DISABLE", "1")
	if RenderEnabled() {
		t.Fatal("expected RenderEnabled false when DOC_RENDER_DISABLE=1")
	}
}

func TestFetchHTMLFallbackWithoutPlaywright(t *testing.T) {
	ResetRenderProbeForTest()
	t.Setenv("DOC_RENDER_DISABLE", "1")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		_, _ = w.Write([]byte("<html><body><h1>Hello docs</h1><p>" + strings.Repeat("x", 300) + "</p></body></html>"))
	}))
	defer srv.Close()

	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	entries, usedPW, err := fetchHTML(u, true)
	if err != nil {
		t.Fatalf("fetchHTML: %v", err)
	}
	if usedPW {
		t.Fatal("expected plain HTTP fallback, not playwright")
	}
	if len(entries) == 0 {
		t.Fatal("expected entries from plain HTML")
	}
}
