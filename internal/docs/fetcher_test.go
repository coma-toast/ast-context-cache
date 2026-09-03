package docs

import (
	"net/url"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestSourceNeedsRefresh(t *testing.T) {
	if !SourceNeedsRefresh("") {
		t.Fatal("empty last_updated should refresh")
	}
	if !SourceNeedsRefresh("not-a-date") {
		t.Fatal("bad timestamp should refresh")
	}
	fresh := time.Now().Add(-24 * time.Hour).Format(time.RFC3339)
	if SourceNeedsRefresh(fresh) {
		t.Fatal("1 day old should not refresh")
	}
	stale := time.Now().Add(-8 * 24 * time.Hour).Format(time.RFC3339)
	if !SourceNeedsRefresh(stale) {
		t.Fatal("8 days old should refresh")
	}
}

// A doc source registered against a local file (a repo's own docs, a synced notes
// file) used to fail every refresh with "unsupported protocol scheme \"file\"", since
// fetchURL always called http.Get regardless of scheme.
func TestFetchURLFileScheme(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "notes.md")
	want := "# Local doc\n\nHello from disk."
	if err := os.WriteFile(path, []byte(want), 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := fetchURL("file://" + path)
	if err != nil {
		t.Fatalf("fetchURL(file://%s): %v", path, err)
	}
	if string(got) != want {
		t.Fatalf("content = %q, want %q", got, want)
	}
}

func TestFetchURLFileSchemeMissingFile(t *testing.T) {
	_, err := fetchURL("file:///no/such/path/notes.md")
	if err == nil {
		t.Fatal("expected an error for a missing local file")
	}
}

func TestFetchMarkdownFileScheme(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "notes.md")
	if err := os.WriteFile(path, []byte("# Title\n\nBody text."), 0o644); err != nil {
		t.Fatal(err)
	}

	u, err := url.Parse("file://" + path)
	if err != nil {
		t.Fatal(err)
	}
	entries, err := fetchMarkdown(u)
	if err != nil {
		t.Fatalf("fetchMarkdown: %v", err)
	}
	if len(entries) == 0 {
		t.Fatal("expected at least one chunked entry from the local markdown file")
	}
}
