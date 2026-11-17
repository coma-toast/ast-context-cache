package memory

import (
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

const maxEmbedTextBytes = 1200

func memoryVectorKey(ref string) string {
	return "mem:" + ref
}

// truncateUTF8 caps s at maxBytes without splitting a multi-byte UTF-8 rune,
// preferring a word boundary in the second half of the budget.
func truncateUTF8(s string, maxBytes int) string {
	if maxBytes <= 0 || len(s) <= maxBytes {
		return s
	}
	cut := 0
	lastSpace := -1
	for i, r := range s {
		w := utf8.RuneLen(r)
		if w < 0 {
			w = 1
		}
		if i+w > maxBytes {
			break
		}
		cut = i + w
		if unicode.IsSpace(r) {
			lastSpace = i
		}
	}
	if lastSpace > maxBytes/2 {
		return strings.TrimRightFunc(s[:lastSpace], unicode.IsSpace)
	}
	return s[:cut]
}

// EmbedText constructs a composite embedding text for e containing its kind,
// summary line/triple/rule, metadata (target, category, reason, actor, tags),
// and leading body summary, capped at 1,200 bytes on a word/rune boundary.
func EmbedText(e Entry) string {
	var parts []string
	line := FormatLine(e)
	if line != "" {
		if e.Kind != "" && !strings.HasPrefix(line, "[") {
			parts = append(parts, "["+string(e.Kind)+"] "+line)
		} else {
			parts = append(parts, line)
		}
	} else if e.Kind != "" {
		parts = append(parts, "["+string(e.Kind)+"]")
	}

	if title := strings.TrimSpace(e.Title); title != "" && !strings.Contains(line, title) {
		parts = append(parts, "title: "+title)
	}
	if target := strings.TrimSpace(e.Target); target != "" && !strings.Contains(line, target) {
		parts = append(parts, "target: "+target)
	}
	if cat := strings.TrimSpace(e.Category); cat != "" && !strings.Contains(line, cat) {
		parts = append(parts, "category: "+cat)
	}
	if reason := strings.TrimSpace(e.Reason); reason != "" && !strings.Contains(line, reason) {
		parts = append(parts, "reason: "+reason)
	}
	if actor := strings.TrimSpace(e.Actor); actor != "" {
		parts = append(parts, "actor: "+actor)
	}
	if tags := normalizeTagList(e.Tags); len(tags) > 0 {
		parts = append(parts, "tags: "+strings.Join(tags, ", "))
	}
	if body := strings.TrimSpace(e.Body); body != "" {
		parts = append(parts, body)
	}

	return truncateUTF8(strings.Join(parts, "\n"), maxEmbedTextBytes)
}

// EmbedEntryFromEntry embeds e using EmbedText(e).
func EmbedEntryFromEntry(e Entry, emb embedder.Interface) ([]float32, error) {
	return EmbedEntryWithVector(e.Ref, e.SessionID, EmbedText(e), emb)
}

// EmbedEntry embeds structured memory for semantic recall_memory.
func EmbedEntry(ref string, sessionID string, line string, emb embedder.Interface) {
	_, _ = EmbedEntryWithVector(ref, sessionID, line, emb)
}

// EmbedEntryWithVector embeds structured memory and returns the vector.
func EmbedEntryWithVector(ref string, sessionID string, line string, emb embedder.Interface) ([]float32, error) {
	if emb == nil || ref == "" || line == "" {
		return nil, nil
	}
	line = truncateUTF8(line, maxEmbedTextBytes)
	vec, err := emb.EmbedSingle(line)
	if err != nil {
		return nil, err
	}
	entry := search.VectorEntry{
		ContentHash: search.ContentHash("memory:" + ref + ":" + line),
		DocType:     "memory",
		SourceFile:  memoryVectorKey(ref),
		Name:        line,
		Kind:        "structured_memory",
		ProjectPath: sessionID,
		Vector:      vec,
	}
	_ = search.Cache.Upsert([]search.VectorEntry{entry})
	return vec, nil
}
