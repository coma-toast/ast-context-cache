package memory

import (
	"fmt"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func estimateEntryTokens(e Entry) int {
	if e.TokenEst > 0 {
		return e.TokenEst
	}
	text := FormatLine(e)
	if body := strings.TrimSpace(e.Body); body != "" {
		if text != "" {
			text += "\n" + body
		} else {
			text = body
		}
	}
	return db.EstimateTokens(text)
}

func entryHeadline(e Entry) string {
	if t := strings.TrimSpace(e.Title); t != "" {
		return t
	}
	if r := strings.TrimSpace(e.Rule); r != "" {
		return r
	}
	if b := strings.TrimSpace(e.Body); b != "" {
		if nl := strings.IndexByte(b, '\n'); nl >= 0 {
			b = strings.TrimSpace(b[:nl])
		}
		if b != "" {
			return b
		}
	}
	subj := strings.TrimSpace(e.Subject)
	pred := strings.TrimSpace(e.Predicate)
	obj := strings.TrimSpace(e.Object)
	var parts []string
	if subj != "" {
		parts = append(parts, subj)
	}
	if pred != "" {
		parts = append(parts, pred)
	}
	if obj != "" {
		parts = append(parts, obj)
	}
	return strings.Join(parts, " ")
}

// FormatLine returns a compact single-line representation.
func FormatLine(e Entry) string {
	switch e.Kind {
	case KindProcedure:
		if e.Rule != "" {
			return "PROC: " + strings.TrimSpace(e.Rule)
		}
		return "PROC: (empty)"
	case KindFact:
		subj := strings.TrimSpace(e.Subject)
		pred := strings.TrimSpace(e.Predicate)
		obj := strings.TrimSpace(e.Object)
		if subj == "" && obj == "" {
			return ""
		}
		if pred == "" {
			pred = "is"
		}
		if subj == "" {
			return pred + " " + obj
		}
		if obj == "" {
			return subj + " " + pred
		}
		return subj + " " + pred + " " + obj
	case KindPreference:
		subj := strings.TrimSpace(e.Subject)
		pred := strings.TrimSpace(e.Predicate)
		obj := strings.TrimSpace(e.Object)
		if obj != "" {
			switch {
			case subj != "" && pred != "":
				return "[preference] " + subj + " " + pred + ": " + obj
			case subj != "":
				return "[preference] " + subj + ": " + obj
			case pred != "":
				return "[preference] " + pred + ": " + obj
			default:
				return "[preference] " + obj
			}
		}
		if r := strings.TrimSpace(e.Rule); r != "" {
			return "[preference] " + r
		}
		if t := strings.TrimSpace(e.Title); t != "" {
			return "[preference] " + t
		}
		return ""
	case KindEvent:
		headline := entryHeadline(e)
		target := strings.TrimSpace(e.Target)
		occ := strings.TrimSpace(e.OccurredAt)
		if headline == "" && target == "" {
			return ""
		}
		prefix := "[event]"
		if sev := strings.TrimSpace(e.Severity); sev != "" {
			prefix = "[event:" + sev + "]"
		}
		var meta []string
		if target != "" {
			meta = append(meta, target)
		}
		if occ != "" {
			meta = append(meta, occ)
		}
		out := prefix
		if headline != "" {
			out += " " + headline
		}
		if len(meta) > 0 {
			out += " (" + strings.Join(meta, ", ") + ")"
		}
		return out
	case KindConfigChange:
		target := strings.TrimSpace(e.Target)
		before := strings.TrimSpace(e.BeforeValue)
		after := strings.TrimSpace(e.AfterValue)
		reason := strings.TrimSpace(e.Reason)
		title := strings.TrimSpace(e.Title)
		var change string
		switch {
		case before != "" && after != "":
			change = before + " -> " + after
		case after != "":
			change = after
		case before != "":
			change = before + " -> (unset)"
		case title != "":
			change = title
		}
		if target == "" && change == "" {
			return ""
		}
		var out string
		switch {
		case target != "" && change != "":
			out = "[config_change] " + target + ": " + change
		case target != "":
			out = "[config_change] " + target
		default:
			out = "[config_change] " + change
		}
		if reason != "" {
			out += " (" + reason + ")"
		}
		return out
	case KindDiscovery:
		headline := entryHeadline(e)
		if headline == "" {
			return ""
		}
		if cat := strings.TrimSpace(e.Category); cat != "" {
			return "[discovery] " + headline + " (" + cat + ")"
		}
		return "[discovery] " + headline
	default:
		return ""
	}
}

// FormatCompactLine constructs a Tier-1 CompactLine with summary text and lightweight metadata.
func FormatCompactLine(e Entry, score float64) CompactLine {
	return CompactLine{
		Ref:         e.Ref,
		Kind:        e.Kind,
		Line:        FormatLine(e),
		Score:       score,
		Title:       e.Title,
		Scope:       e.Scope,
		ProjectPath: e.ProjectPath,
		OccurredAt:  e.OccurredAt,
		CreatedAt:   e.CreatedAt,
		Severity:    e.Severity,
		Category:    e.Category,
		Actor:       e.Actor,
		Target:      e.Target,
		Tags:        e.Tags,
		Verified:    e.Verified,
	}
}

func formatLines(entries []Entry) string {
	var b strings.Builder
	for i, e := range entries {
		line := FormatLine(e)
		if line == "" {
			continue
		}
		if b.Len() > 0 {
			b.WriteByte('\n')
		}
		b.WriteString("- ")
		b.WriteString(line)
		if e.ValidUntil != "" {
			b.WriteString(fmt.Sprintf(" (until %s)", e.ValidUntil))
		}
		_ = i
	}
	return b.String()
}
