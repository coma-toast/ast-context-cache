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
	return db.EstimateTokens(FormatLine(e))
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
	default:
		return ""
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
