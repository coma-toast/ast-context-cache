package indexer

import (
	"strings"
)

// ExtractSkeleton extracts just the signature from source code based on language and kind.
// Returns a compact signature without implementation bodies.
func ExtractSkeleton(source, lang, kind string) string {
	lines := strings.Split(source, "\n")
	if len(lines) == 0 {
		return source
	}

	switch lang {
	case "go":
		return extractGoSkeleton(lines, kind)
	case "typescript", "tsx", "javascript":
		return extractTSSkeleton(lines, kind)
	case "python":
		return extractPythonSkeleton(lines, kind)
	case "hcl":
		return extractHCLSkeleton(lines, kind)
	case "yaml":
		return extractYAMLSkeleton(lines, kind)
	default:
		if len(lines) > 0 {
			return lines[0]
		}
		return source
	}
}

func extractGoSkeleton(lines []string, kind string) string {
	switch kind {
	case "function", "method":
		sig, _ := scanSignature(lines, goSignature)
		return sig
	case "struct":
		var result []string
		result = append(result, lines[0])
		depth := 0
		for _, line := range lines {
			trimmed := strings.TrimSpace(line)
			if strings.Contains(trimmed, "{") {
				depth++
			}
			if depth == 1 && trimmed != "" && !strings.HasPrefix(trimmed, "//") {
				if !strings.Contains(trimmed, "{") && !strings.Contains(trimmed, "}") {
					result = append(result, "\t"+trimmed)
				}
			}
			if strings.Contains(trimmed, "}") {
				depth--
				if depth == 0 {
					result = append(result, "}")
					break
				}
			}
		}
		return strings.Join(result, "\n")
	case "interface":
		var result []string
		depth := 0
		for _, line := range lines {
			trimmed := strings.TrimSpace(line)
			if strings.Contains(trimmed, "{") {
				depth++
			}
			if depth <= 1 && trimmed != "" {
				result = append(result, line)
			}
			if strings.Contains(trimmed, "}") {
				depth--
				if depth == 0 {
					break
				}
			}
		}
		return strings.Join(result, "\n")
	default:
		return lines[0]
	}
}

func extractTSSkeleton(lines []string, kind string) string {
	switch kind {
	case "function", "method":
		sig, _ := scanSignature(lines, tsSignature)
		return sig
	case "variable":
		sig := lines[0]
		if strings.Contains(sig, "{") {
			sig = strings.TrimSpace(strings.SplitN(sig, "{", 2)[0])
		}
		if strings.HasSuffix(sig, "=>") {
			sig = strings.TrimSpace(sig)
		}
		return sig
	case "class":
		result := []string{lines[0]}
		depth, skipTo, opened := 0, 0, false
		for i, line := range lines {
			trimmed := strings.TrimSpace(line)
			if depth == 1 && i >= skipTo && trimmed != "" && !strings.HasPrefix(trimmed, "//") && !strings.HasPrefix(trimmed, "}") {
				if isTSMethodLine(trimmed) {
					// Keep a wrapped parameter list whole, re-indented under the class.
					sig, end := scanSignature(lines[i:], tsSignature)
					indent := len(line) - len(strings.TrimLeft(line, " \t"))
					for _, l := range strings.Split(sig, "\n") {
						result = append(result, "  "+trimIndent(l, indent))
					}
					skipTo = i + end + 1
				} else if (strings.Contains(trimmed, ":") || strings.Contains(trimmed, "=")) && !strings.ContainsAny(trimmed, "{}") {
					result = append(result, "  "+trimmed)
				}
			}
			opened = opened || strings.Contains(line, "{")
			depth += strings.Count(line, "{") - strings.Count(line, "}")
			if opened && depth <= 0 {
				if i == 0 {
					return lines[0] // whole class on one line
				}
				result = append(result, "}")
				break
			}
		}
		return strings.Join(result, "\n")
	case "interface", "type", "enum":
		var result []string
		depth := 0
		for _, line := range lines {
			trimmed := strings.TrimSpace(line)
			if strings.Contains(trimmed, "{") {
				depth++
			}
			if depth <= 1 && trimmed != "" {
				result = append(result, line)
			}
			if strings.Contains(trimmed, "}") {
				depth--
				if depth == 0 {
					break
				}
			}
		}
		return strings.Join(result, "\n")
	default:
		return lines[0]
	}
}

// isTSMethodLine reports whether a class-body line starts a method: its "(" comes
// before any ":" or "=" (which would make it a typed or initialized property).
func isTSMethodLine(trimmed string) bool {
	paren := strings.Index(trimmed, "(")
	if paren < 0 {
		return false
	}
	other := strings.IndexAny(trimmed, ":=")
	return other < 0 || paren < other
}

func extractHCLSkeleton(lines []string, kind string) string {
	if len(lines) == 0 {
		return ""
	}
	var result []string
	result = append(result, lines[0])
	depth := 0
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.Contains(trimmed, "{") {
			depth++
		}
		if depth == 1 && trimmed != "" && !strings.HasPrefix(trimmed, "//") && !strings.HasPrefix(trimmed, "#") {
			if !strings.Contains(trimmed, "{") && !strings.Contains(trimmed, "}") {
				parts := strings.SplitN(trimmed, "=", 2)
				result = append(result, "  "+strings.TrimSpace(parts[0]))
			}
		}
		if strings.Contains(trimmed, "}") {
			depth--
			if depth == 0 {
				result = append(result, "}")
				break
			}
		}
	}
	return strings.Join(result, "\n")
}

func extractYAMLSkeleton(lines []string, kind string) string {
	if len(lines) == 0 {
		return ""
	}
	switch kind {
	case "play":
		return extractYAMLPlaySkeleton(lines)
	case "task", "handler":
		return extractYAMLTaskSkeleton(lines)
	case "key":
		return extractYAMLKeySkeleton(lines)
	default:
		return lines[0]
	}
}

func extractYAMLPlaySkeleton(lines []string) string {
	var result []string
	if len(lines) > 0 {
		result = append(result, lines[0])
	}
	baseIndent := -1
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}
		indent := len(line) - len(strings.TrimLeft(line, " "))
		if baseIndent < 0 && strings.Contains(trimmed, ":") {
			baseIndent = indent
			continue
		}
		if baseIndent >= 0 && indent == baseIndent {
			key := strings.SplitN(trimmed, ":", 2)[0]
			val := ""
			if parts := strings.SplitN(trimmed, ":", 2); len(parts) > 1 {
				val = strings.TrimSpace(parts[1])
			}
			switch key {
			case "name", "hosts", "become", "gather_facts", "connection":
				if val != "" {
					result = append(result, strings.Repeat(" ", baseIndent)+key+": "+val)
				} else {
					result = append(result, strings.Repeat(" ", baseIndent)+key+":")
				}
			case "tasks", "handlers", "pre_tasks", "post_tasks", "roles":
				result = append(result, strings.Repeat(" ", baseIndent)+key+": [...]")
			}
		}
	}
	return strings.Join(result, "\n")
}

func extractYAMLTaskSkeleton(lines []string) string {
	if len(lines) == 0 {
		return ""
	}
	var result []string
	result = append(result, lines[0])
	baseIndent := -1
	for i, line := range lines {
		if i == 0 {
			continue
		}
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}
		indent := len(line) - len(strings.TrimLeft(line, " "))
		if baseIndent < 0 {
			baseIndent = indent
		}
		if indent == baseIndent && strings.Contains(trimmed, ":") {
			key := strings.SplitN(trimmed, ":", 2)[0]
			val := ""
			if parts := strings.SplitN(trimmed, ":", 2); len(parts) > 1 {
				val = strings.TrimSpace(parts[1])
			}
			if key == "name" && val != "" {
				result = append(result, strings.Repeat(" ", indent)+key+": "+val)
			} else if val == "" || val == "|" || val == ">" {
				result = append(result, strings.Repeat(" ", indent)+key+":")
			} else {
				result = append(result, strings.Repeat(" ", indent)+key+": "+val)
			}
		}
	}
	return strings.Join(result, "\n")
}

func extractYAMLKeySkeleton(lines []string) string {
	if len(lines) == 0 {
		return ""
	}
	var result []string
	result = append(result, lines[0])
	baseIndent := -1
	for i, line := range lines {
		if i == 0 {
			continue
		}
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}
		indent := len(line) - len(strings.TrimLeft(line, " "))
		if baseIndent < 0 {
			baseIndent = indent
		}
		if indent == baseIndent && strings.Contains(trimmed, ":") {
			key := strings.SplitN(trimmed, ":", 2)[0]
			result = append(result, strings.Repeat(" ", indent)+key+":")
		}
	}
	return strings.Join(result, "\n")
}

func extractPythonSkeleton(lines []string, kind string) string {
	switch kind {
	case "function", "method":
		sig, last := scanSignature(lines, pythonSignature)
		return strings.Join(append([]string{sig}, pythonDocstring(lines, last+1)...), "\n")
	case "class":
		header, last := scanSignature(lines, pythonSignature)
		result := []string{header}
		baseIndent, haveBase := "", false
		for i := last + 1; i < len(lines); i++ {
			trimmed := strings.TrimSpace(lines[i])
			if trimmed == "" {
				continue
			}
			indent := lines[i][:len(lines[i])-len(strings.TrimLeft(lines[i], " \t"))]
			if !haveBase {
				baseIndent, haveBase = indent, true
			}
			if indent != baseIndent {
				continue
			}
			switch {
			case strings.HasPrefix(trimmed, "def "), strings.HasPrefix(trimmed, "async def "), strings.HasPrefix(trimmed, "class "):
				// Keep a wrapped parameter or base-class list whole, and skip its
				// continuation lines (a closing "):" sits at the member indent).
				sig, end := scanSignature(lines[i:], pythonSignature)
				result = append(result, sig)
				i += end
			case !strings.HasPrefix(trimmed, "#") && (strings.Contains(trimmed, "=") || strings.Contains(trimmed, ":")):
				result = append(result, baseIndent+trimmed)
			}
		}
		return strings.Join(result, "\n")
	default:
		return lines[0]
	}
}

// pythonDocstring returns the docstring starting at lines[from], if any, indented
// under its def.
func pythonDocstring(lines []string, from int) []string {
	if from >= len(lines) {
		return nil
	}
	first := strings.TrimSpace(lines[from])
	if !strings.HasPrefix(first, `"""`) && !strings.HasPrefix(first, `'''`) {
		return nil
	}
	quote := first[:3]
	if strings.Count(first, quote) >= 2 {
		return []string{"    " + first}
	}
	var out []string
	for i := from; i < len(lines); i++ {
		out = append(out, "    "+strings.TrimSpace(lines[i]))
		if i > from && strings.Contains(lines[i], quote) {
			break
		}
	}
	return out
}

// signatureSyntax describes where a language's declaration header ends.
type signatureSyntax struct {
	// terminators end the header when met outside brackets and strings.
	terminators string
	// keepTerminator keeps the terminator in the header (Python's ":").
	keepTerminator bool
	// lineComment starts a comment that runs to end of line.
	lineComment string
	// typeBraces names keywords whose following "{" opens a type literal
	// (Go's interface{} / struct{...}) rather than the body.
	typeBraces []string
	// angleBrackets counts <...> as brackets (TS/JS generics).
	angleBrackets bool
}

var (
	pythonSignature = signatureSyntax{terminators: ":", keepTerminator: true, lineComment: "#"}
	goSignature     = signatureSyntax{terminators: "{", lineComment: "//", typeBraces: []string{"interface", "struct"}}
	tsSignature     = signatureSyntax{terminators: "{;", lineComment: "//", angleBrackets: true}
)

// maxSignatureLines bounds how far a header may wrap before we give up on it.
const maxSignatureLines = 40

// scanSignature returns a declaration's header — everything before its body —
// keeping a parameter list that wraps across lines whole instead of cutting it
// at the first line. It also returns the index of the header's last line. A
// header with no terminator (an abstract or overload signature) is returned
// whole when short enough; otherwise the first line is used.
func scanSignature(lines []string, syn signatureSyntax) (string, int) {
	depth := 0
	var quote byte
	for li := 0; li < len(lines) && li < maxSignatureLines; li++ {
		line := lines[li]
		for i := 0; i < len(line); i++ {
			c := line[i]
			if quote != 0 {
				if c == '\\' {
					i++
				} else if c == quote {
					quote = 0
				}
				continue
			}
			if syn.lineComment != "" && strings.HasPrefix(line[i:], syn.lineComment) {
				break
			}
			switch {
			case c == '"' || c == '\'' || c == '`':
				quote = c
			case c == '(' || c == '[' || (syn.angleBrackets && c == '<'):
				depth++
			case c == ')' || c == ']' || (syn.angleBrackets && c == '>' && (i == 0 || line[i-1] != '=')):
				if depth > 0 {
					depth--
				}
			case c == '{' && (depth > 0 || endsWithKeyword(line[:i], syn.typeBraces)):
				depth++
			case c == '}' && depth > 0:
				depth--
			case depth == 0 && strings.IndexByte(syn.terminators, c) >= 0:
				end := i
				if syn.keepTerminator {
					end++
				}
				header := append(append([]string{}, lines[:li]...), strings.TrimRight(line[:end], " \t"))
				return strings.Join(header, "\n"), li
			}
		}
		if quote == '"' || quote == '\'' {
			quote = 0 // unterminated short string: don't let it swallow later lines
		}
	}
	if len(lines) <= maxSignatureLines {
		return strings.TrimRight(strings.Join(lines, "\n"), " \t\n"), len(lines) - 1
	}
	return lines[0], 0
}

func endsWithKeyword(s string, keywords []string) bool {
	s = strings.TrimRight(s, " \t")
	for _, k := range keywords {
		if strings.HasSuffix(s, k) {
			return true
		}
	}
	return false
}
