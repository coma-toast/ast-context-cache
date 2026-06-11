package docs

import "testing"

func TestNormalizeDocType(t *testing.T) {
	if got := NormalizeDocType("html", true); got != "webpage" {
		t.Fatalf("render_js html => webpage, got %q", got)
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
