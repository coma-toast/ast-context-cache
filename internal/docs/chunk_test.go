package docs

import (
	"strings"
	"testing"
)

func TestSplitMarkdownSections(t *testing.T) {
	md := "# Top\nintro\n\n## Section A\nalpha\n\n### Detail\nbeta\n\n## Section B\ngamma"
	sections := splitMarkdownSections(md)
	if len(sections) != 3 {
		t.Fatalf("sections=%d want 3", len(sections))
	}
	if sections[0].Title != "Top" || !strings.Contains(sections[0].Content, "intro") {
		t.Fatalf("top section: %+v", sections[0])
	}
	if sections[1].Title != "Section A" || !strings.Contains(sections[1].Content, "alpha") || !strings.Contains(sections[1].Content, "beta") {
		t.Fatalf("section A includes nested h3: %+v", sections[1])
	}
	if sections[2].Title != "Section B" || !strings.Contains(sections[2].Content, "gamma") {
		t.Fatalf("section B: %+v", sections[2])
	}
}

func TestChunkMarkdownSplitsLongSections(t *testing.T) {
	long := strings.Repeat("word ", 600)
	entries := chunkMarkdown("# Title\n"+long, "/doc.md")
	if len(entries) < 2 {
		t.Fatalf("expected multiple chunks, got %d", len(entries))
	}
	for _, e := range entries {
		if len(e.Content) > maxChunkChars+50 {
			t.Fatalf("chunk too large: %d", len(e.Content))
		}
	}
}

func TestChunkHTMLHeadings(t *testing.T) {
	html := `<html><head><title>Docs</title></head><body><h1>Intro</h1><p>Hello world</p><h2>Tools</h2><p>tools/list and tools/call</p></body></html>`
	entries := chunkHTML(html, "/docs/tools")
	if len(entries) == 0 {
		t.Fatal("expected html chunks")
	}
	foundTools := false
	for _, e := range entries {
		if strings.Contains(strings.ToLower(e.Title), "tools") || strings.Contains(e.Content, "tools/call") {
			foundTools = true
		}
	}
	if !foundTools {
		t.Fatalf("missing tools section: %+v", entries)
	}
}

func TestChunkJSONObject(t *testing.T) {
	raw := `{"auth":{"oauth":true},"tools":{"list":"tools/list"}}`
	entries := chunkJSON(raw, "/spec.json")
	if len(entries) < 2 {
		t.Fatalf("expected per-key chunks, got %d", len(entries))
	}
}
