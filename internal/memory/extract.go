package memory

import (
	"strings"
)

// Extracted holds pattern-parsed structured memory candidates from free text (no LLM).
type Extracted struct {
	Facts         []FactInput
	Procedures    []ProcedureInput
	Preferences   []PreferenceInput
	Events        []EventInput
	ConfigChanges []ConfigChangeInput
	Discoveries   []DiscoveryInput
	// Skipped lists explicit marked lines that could not be turned into an
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

// PreferenceInput is a parsed user/project preference candidate.
type PreferenceInput struct {
	Subject   string
	Predicate string
	Object    string
}

// EventInput is a parsed event/incident candidate.
type EventInput struct {
	Title    string
	Body     string
	Category string
	Severity string
}

// ConfigChangeInput is a parsed configuration change candidate.
type ConfigChangeInput struct {
	Target      string
	BeforeValue string
	AfterValue  string
	Reason      string
}

// DiscoveryInput is a parsed discovery/finding candidate.
type DiscoveryInput struct {
	Title string
	Body  string
}

// ExtractFromText parses explicitly marked lines from store_context content.
// Only lines that start with a supported marker (FACT:, RULE:, PREFERENCE:,
// EVENT:, CONFIG_CHANGE:, DISCOVERY: — case-insensitive, optionally after a
// list bullet) are extracted; headings, prose, and any other text are ignored,
// as are lines inside fenced code blocks.
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
		case "PREFERENCE":
			if p, ok := parsePreferenceLine(body); ok {
				out.Preferences = append(out.Preferences, p)
			} else {
				out.Skipped = append(out.Skipped, line)
			}
		case "EVENT":
			if ev, ok := parseEventLine(body); ok {
				out.Events = append(out.Events, ev)
			} else {
				out.Skipped = append(out.Skipped, line)
			}
		case "CONFIG_CHANGE":
			if cc, ok := parseConfigChangeLine(body); ok {
				out.ConfigChanges = append(out.ConfigChanges, cc)
			} else {
				out.Skipped = append(out.Skipped, line)
			}
		case "DISCOVERY":
			if d, ok := parseDiscoveryLine(body); ok {
				out.Discoveries = append(out.Discoveries, d)
			} else {
				out.Skipped = append(out.Skipped, line)
			}
		}
	}
	return out
}

// splitMarker returns the marker name and the trimmed text after its colon when
// line starts with a supported marker, case-insensitively.
func splitMarker(line string) (string, string, bool) {
	for _, m := range []string{"FACT", "RULE", "PREFERENCE", "EVENT", "CONFIG_CHANGE", "DISCOVERY"} {
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

// parsePreferenceLine parses "subject | predicate | object", "subject predicate rest...",
// or a single statement mapped to subject="user", predicate="preference", object=body.
func parsePreferenceLine(s string) (PreferenceInput, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return PreferenceInput{}, false
	}
	if strings.Contains(s, "|") {
		parts := strings.Split(s, "|")
		if len(parts) == 3 {
			subj, pred, obj := strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1]), strings.TrimSpace(parts[2])
			if subj != "" && pred != "" && obj != "" {
				return PreferenceInput{Subject: subj, Predicate: pred, Object: obj}, true
			}
			return PreferenceInput{}, false
		}
	}
	fields := strings.Fields(s)
	if len(fields) >= 3 {
		rest := strings.TrimSpace(s[len(fields[0]):])
		obj := strings.TrimSpace(rest[len(fields[1]):])
		return PreferenceInput{Subject: fields[0], Predicate: fields[1], Object: obj}, true
	}
	return PreferenceInput{Subject: "user", Predicate: "preference", Object: s}, true
}

// parseEventLine parses "[severity] title | body", "title | category | body", or plain "title".
func parseEventLine(s string) (EventInput, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return EventInput{}, false
	}
	var severity string
	if strings.HasPrefix(s, "[") {
		if idx := strings.IndexByte(s, ']'); idx > 0 {
			cand := strings.ToLower(strings.TrimSpace(s[1:idx]))
			if cand != "" && IsValidSeverity(cand) {
				severity = cand
				s = strings.TrimSpace(s[idx+1:])
				if s == "" {
					return EventInput{}, false
				}
			}
		}
	}
	parts := strings.Split(s, "|")
	switch len(parts) {
	case 1:
		title := strings.TrimSpace(parts[0])
		if title == "" {
			return EventInput{}, false
		}
		return EventInput{Title: title, Severity: severity}, true
	case 2:
		title := strings.TrimSpace(parts[0])
		body := strings.TrimSpace(parts[1])
		if title == "" {
			return EventInput{}, false
		}
		return EventInput{Title: title, Body: body, Severity: severity}, true
	default:
		title := strings.TrimSpace(parts[0])
		cat := strings.TrimSpace(parts[1])
		body := strings.TrimSpace(strings.Join(parts[2:], "|"))
		if title == "" {
			return EventInput{}, false
		}
		return EventInput{Title: title, Category: cat, Body: body, Severity: severity}, true
	}
}

// parseConfigChangeLine parses "target | before -> after | reason" or "target | after | reason".
func parseConfigChangeLine(s string) (ConfigChangeInput, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return ConfigChangeInput{}, false
	}
	parts := strings.Split(s, "|")
	if len(parts) < 2 {
		return ConfigChangeInput{}, false
	}
	target := strings.TrimSpace(parts[0])
	change := strings.TrimSpace(parts[1])
	var reason string
	if len(parts) >= 3 {
		reason = strings.TrimSpace(strings.Join(parts[2:], "|"))
	}
	if target == "" || change == "" {
		return ConfigChangeInput{}, false
	}
	if idx := strings.Index(change, "->"); idx >= 0 {
		before := strings.TrimSpace(change[:idx])
		after := strings.TrimSpace(change[idx+2:])
		if after == "" {
			return ConfigChangeInput{}, false
		}
		return ConfigChangeInput{
			Target:      target,
			BeforeValue: before,
			AfterValue:  after,
			Reason:      reason,
		}, true
	}
	return ConfigChangeInput{
		Target:     target,
		AfterValue: change,
		Reason:     reason,
	}, true
}

// parseDiscoveryLine parses "title | body" or plain "title".
func parseDiscoveryLine(s string) (DiscoveryInput, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return DiscoveryInput{}, false
	}
	parts := strings.Split(s, "|")
	title := strings.TrimSpace(parts[0])
	if title == "" {
		return DiscoveryInput{}, false
	}
	var body string
	if len(parts) >= 2 {
		body = strings.TrimSpace(strings.Join(parts[1:], "|"))
	}
	return DiscoveryInput{Title: title, Body: body}, true
}
