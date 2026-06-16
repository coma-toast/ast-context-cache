package memory

import (
	"strings"
)

// Extracted holds pattern-parsed facts and procedures from free text (no LLM).
type Extracted struct {
	Facts      []FactInput
	Procedures []ProcedureInput
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

// ExtractFromText parses lightweight structured lines from store_context content.
// Supported patterns:
//   - FACT: subject | predicate | object
//   - FACT: subject predicate object
//   - subject | predicate | object
//   - subject: object  (predicate defaults to "is")
//   - RULE: / PROC: / PREFER: / PROCEDURE:
func ExtractFromText(content string) Extracted {
	var out Extracted
	for _, line := range strings.Split(content, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		line = strings.TrimLeft(line, "-*• ")
		upper := strings.ToUpper(line)
		switch {
		case strings.HasPrefix(upper, "FACT:"):
			if f, ok := parseFactLine(strings.TrimSpace(line[5:])); ok {
				out.Facts = append(out.Facts, f)
			}
		case strings.HasPrefix(upper, "RULE:"), strings.HasPrefix(upper, "PROC:"), strings.HasPrefix(upper, "PREFER:"), strings.HasPrefix(upper, "PROCEDURE:"):
			idx := strings.Index(line, ":")
			rule := strings.TrimSpace(line[idx+1:])
			if rule != "" {
				out.Procedures = append(out.Procedures, ProcedureInput{Rule: rule})
			}
		default:
			if f, ok := parseFactLine(line); ok {
				out.Facts = append(out.Facts, f)
			}
		}
	}
	return out
}

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
	if i := strings.Index(s, ":"); i > 0 && i < len(s)-1 {
		subj := strings.TrimSpace(s[:i])
		obj := strings.TrimSpace(s[i+1:])
		if subj != "" && obj != "" && !strings.Contains(subj, " ") {
			return FactInput{Subject: subj, Predicate: "is", Object: obj}, true
		}
	}
	fields := strings.Fields(s)
	if len(fields) >= 3 {
		subj := fields[0]
		obj := strings.Join(fields[2:], " ")
		return FactInput{Subject: subj, Predicate: fields[1], Object: obj}, true
	}
	return FactInput{}, false
}
