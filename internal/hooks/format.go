package hooks

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"
)

// The parts of the handoff tool responses the hooks read. They mirror the JSON of
// internal/handoff's response types, which the package doesn't import so a hook run stays a thin
// HTTP client; hooks_test.go decodes the real types into these to catch drift.

type createResult struct {
	Ref  string `json:"handoff"`
	Stub string `json:"stub"`
}

type openResult struct {
	Handoff   string `json:"handoff"`
	SessionID string `json:"session_id"`
	Mode      string `json:"mode"`
	Label     string `json:"label"`
	Brief     string `json:"brief"`
	Pointers  []struct {
		ID   int64  `json:"id"`
		Key  string `json:"key"`
		Note string `json:"note"`
	} `json:"pointers"`
	Notes  []itemDigest `json:"notes"`
	Memory []itemDigest `json:"memory"`
	Trail  []struct {
		Tool    string `json:"tool"`
		Query   string `json:"query"`
		Hits    int    `json:"hits"`
		ZeroHit bool   `json:"zero_hit"`
	} `json:"trail"`
	Scratchpad *struct {
		Latest   []entryHeadline `json:"latest"`
		DeadEnds []entryHeadline `json:"dead_ends"`
	} `json:"scratchpad"`
	Truncated bool `json:"truncated"`
}

type itemDigest struct {
	ID    int64  `json:"id"`
	Ref   string `json:"ref"`
	Label string `json:"label"`
}

type entryHeadline struct {
	Type     string `json:"type"`
	Headline string `json:"headline"`
}

type listResult struct {
	Handoffs []handoffSummary `json:"handoffs"`
}

type handoffSummary struct {
	Ref          string         `json:"handoff"`
	Label        string         `json:"label"`
	Children     int            `json:"children"`
	StatusCounts map[string]int `json:"status_counts"`
}

type collectResult struct {
	Children []struct {
		SessionID string `json:"session_id"`
		Status    string `json:"status"`
	} `json:"children"`
}

type completeResult struct {
	ResultRef string `json:"result_ref"`
}

// statusOrder is the order child counts are listed in.
var statusOrder = []string{"open", "done", "partial", "failed", "abandoned"}

// formatDigest renders an open digest as plain text, capped at maxDigestTokens.
func formatDigest(o *openResult) string {
	var b strings.Builder
	if o.Label != "" {
		b.WriteString("Handoff: " + o.Label + "\n")
	}
	b.WriteString("Brief: " + o.Brief + "\n")
	if len(o.Pointers) > 0 {
		b.WriteString("Pointers (id key — note):\n")
		for _, p := range o.Pointers {
			b.WriteString("- " + strconv.FormatInt(p.ID, 10) + " " + p.Key + suffix(" — ", p.Note) + "\n")
		}
	}
	writeItems(&b, "Notes", o.Notes)
	writeItems(&b, "Memory", o.Memory)
	if len(o.Trail) > 0 {
		b.WriteString("Parent searches (newest first):\n")
		for _, t := range o.Trail {
			hits := strconv.Itoa(t.Hits) + " hits"
			if t.ZeroHit {
				hits = "no hits"
			}
			b.WriteString("- " + t.Tool + " " + strconv.Quote(t.Query) + " (" + hits + ")\n")
		}
	}
	if sp := o.Scratchpad; sp != nil && len(sp.DeadEnds)+len(sp.Latest) > 0 {
		b.WriteString("Scratchpad:\n")
		for _, e := range slices.Concat(sp.DeadEnds, sp.Latest) {
			b.WriteString("- " + e.Type + ": " + e.Headline + "\n")
		}
	}
	if o.Truncated {
		b.WriteString("(digest truncated: call open_handoff action open with next to page)\n")
	}
	return truncateRunes(strings.TrimSpace(b.String()), maxDigestTokens*charsPerToken)
}

func writeItems(b *strings.Builder, title string, items []itemDigest) {
	if len(items) == 0 {
		return
	}
	b.WriteString(title + " (id ref label):\n")
	for _, it := range items {
		b.WriteString("- " + strconv.FormatInt(it.ID, 10) + " " + it.Ref + suffix(" ", it.Label) + "\n")
	}
}

// formatHandoffList lists a session's handoffs, newest first, with per-status child counts.
func formatHandoffList(sessionID string, hs []handoffSummary) string {
	var b strings.Builder
	b.WriteString("Handoffs this session created before compaction (newest first):\n")
	for i, h := range hs {
		if i == maxListedHandoffs {
			b.WriteString("- +" + strconv.Itoa(len(hs)-i) + " more (handoff action list)\n")
			break
		}
		b.WriteString("- " + h.Ref + suffix(" ", strconv.Quote(truncateRunes(h.Label, maxLabelRunes))) + ": " + childCounts(h) + "\n")
	}
	b.WriteString("Use `handoff` action `collect` with session_id=" + sessionID + " to read results.")
	return b.String()
}

func childCounts(h handoffSummary) string {
	if h.Children == 0 {
		return "no child opened yet"
	}
	var parts []string
	for _, s := range statusOrder {
		if n := h.StatusCounts[s]; n > 0 {
			parts = append(parts, strconv.Itoa(n)+" "+s)
		}
	}
	return strings.Join(parts, ", ")
}

// lastAssistantText is the text of the last assistant message in a transcript's tail, or "".
func lastAssistantText(path string) string {
	if path == "" {
		return ""
	}
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	if info, err := f.Stat(); err == nil && info.Size() > maxTranscriptScanBytes {
		if _, err := f.Seek(-maxTranscriptScanBytes, io.SeekEnd); err != nil {
			return ""
		}
	}
	tail, _ := io.ReadAll(f)
	lines := bytes.Split(tail, []byte("\n"))
	for i := len(lines) - 1; i >= 0; i-- {
		var rec struct {
			Type    string `json:"type"`
			Message struct {
				Content []struct {
					Type string `json:"type"`
					Text string `json:"text"`
				} `json:"content"`
			} `json:"message"`
		}
		if json.Unmarshal(lines[i], &rec) != nil || rec.Type != "assistant" {
			continue
		}
		var texts []string
		for _, c := range rec.Message.Content {
			if c.Type == "text" && strings.TrimSpace(c.Text) != "" {
				texts = append(texts, c.Text)
			}
		}
		if len(texts) > 0 {
			return strings.TrimSpace(strings.Join(texts, "\n"))
		}
	}
	return ""
}

// truncateRunes cuts s to at most n runes, marking a cut with "…".
func truncateRunes(s string, n int) string {
	if utf8.RuneCountInString(s) <= n {
		return s
	}
	r := []rune(s)
	return string(r[:max(0, n-1)]) + "…"
}

func suffix(sep, s string) string {
	if s == "" || s == `""` {
		return ""
	}
	return sep + s
}
