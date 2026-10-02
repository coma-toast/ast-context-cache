package memory

import (
	"strings"
)

// Extracted holds pattern-parsed facts and procedures from free text (no LLM).
type Extracted struct {
	Facts      []FactInput
	Procedures []ProcedureInput
	// Skipped lists explicit FACT:/RULE: lines that could not be turned into an
	// entry (e.g. a FACT: with fewer than three words and no pipes), so callers
	// can surface them instead of dropping them silently.
	Skipped []string
}

// FactInput is a parsed temporal fact candidate.
type FactInput struct {
	Subject   string
	Predicate string
	Object    string
}

// ProcedureInput is a parsed procedural rule candidate.
type ProcedureInput struct {
	Rule string
}

// ExtractFromText parses explicitly marked lines from store_context content.
// Only lines that start with a FACT: or RULE: marker (case-insensitive,
// optionally after a list bullet) are extracted; headings, prose, and any other
// text are ignored, as are lines inside fenced code blocks. Supported forms:
//   - FACT: subject | predicate | object
//   - FACT: subject predicate object...   (split at the first two words; the
//     text is not rewritten, so it reads back as written)
//   - RULE: free-text procedural rule
func ExtractFromText(content string) Extracted {
	var out Extracted
	inFence := false
	for _, raw := range strings.Split(content, "\n") {
		line := strings.TrimSpace(raw)
		if strings.HasPrefix(line, "```") || strings.HasPrefix(line, "~~~") {
			inFence = !inFence
			continue
		}
		if inFence || line == "" {
			continue
		}
		line = strings.TrimSpace(strings.TrimLeft(line, "-*•+ "))
		marker, body, ok := splitMarker(line)
		if !ok {
			continue
		}
		switch marker {
		case "FACT":
			if f, ok := parseFactLine(body); ok {
				out.Facts = append(out.Facts, f)
			} else {
				out.Skipped = append(out.Skipped, line)
			}
		case "RULE":
			if body != "" {
				out.Procedures = append(out.Procedures, ProcedureInput{Rule: body})
			} else {
				out.Skipped = append(out.Skipped, line)
			}
		}
	}
	return out
}

// splitMarker returns the marker (FACT or RULE) and the trimmed text after its
// colon when line starts with one, case-insensitively.
func splitMarker(line string) (string, string, bool) {
	for _, m := range []string{"FACT", "RULE"} {
		if len(line) > len(m) && strings.EqualFold(line[:len(m)], m) && line[len(m)] == ':' {
			return m, strings.TrimSpace(line[len(m)+1:]), true
		}
	}
	return "", "", false
}

// parseFactLine splits a FACT: body into subject/predicate/object without
// rewriting its text: "a | b | c" is an explicit triple; otherwise the first
// word is the subject, the second the predicate, and the verbatim remainder the
// object, so FormatLine reproduces the original line.
func parseFactLine(s string) (FactInput, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return FactInput{}, false
	}
	if parts := strings.Split(s, "|"); len(parts) == 3 {
		subj, pred, obj := strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1]), strings.TrimSpace(parts[2])
		if subj != "" && pred != "" && obj != "" {
			return FactInput{Subject: subj, Predicate: pred, Object: obj}, true
		}
	}
	fields := strings.Fields(s)
	if len(fields) < 3 {
		return FactInput{}, false
	}
	rest := strings.TrimSpace(s[len(fields[0]):])
	obj := strings.TrimSpace(rest[len(fields[1]):])
	return FactInput{Subject: fields[0], Predicate: fields[1], Object: obj}, true
}
