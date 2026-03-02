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
		sig := lines[0]
		if strings.Contains(sig, "{") {
			sig = strings.TrimSpace(strings.SplitN(sig, "{", 2)[0])
		}
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
	case "function", "variable":
		sig := lines[0]
		if strings.Contains(sig, "{") {
			sig = strings.TrimSpace(strings.SplitN(sig, "{", 2)[0])
		}
		if strings.HasSuffix(sig, "=>") {
			sig = strings.TrimSpace(sig)
		}
		return sig
	case "class":
		var result []string
		result = append(result, lines[0])
		depth := 0
		for _, line := range lines {
			trimmed := strings.TrimSpace(line)
			if strings.Contains(trimmed, "{") {
				depth++
			}
			if depth == 1 && trimmed != "" {
				isMethod := strings.Contains(trimmed, "(") && !strings.HasPrefix(trimmed, "//")
				isProperty := !strings.Contains(trimmed, "(") && (strings.Contains(trimmed, ":") || strings.Contains(trimmed, "="))
				if isMethod {
					methodSig := trimmed
					if strings.Contains(methodSig, "{") {
						methodSig = strings.TrimSpace(strings.SplitN(methodSig, "{", 2)[0])
					}
					result = append(result, "  "+methodSig)
				} else if isProperty && !strings.Contains(trimmed, "{") && !strings.Contains(trimmed, "}") {
					result = append(result, "  "+trimmed)
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

func extractPythonSkeleton(lines []string, kind string) string {
	switch kind {
	case "function":
		var result []string
		result = append(result, lines[0])
		// Include docstring if present
		if len(lines) > 1 {
			nextLine := strings.TrimSpace(lines[1])
			if strings.HasPrefix(nextLine, `"""`) || strings.HasPrefix(nextLine, `'''`) {
				quote := nextLine[:3]
				if strings.Count(nextLine, quote) >= 2 {
					result = append(result, "    "+nextLine)
				} else {
					for i := 1; i < len(lines); i++ {
						result = append(result, "    "+strings.TrimSpace(lines[i]))
						if i > 1 && strings.Contains(lines[i], quote) {
							break
						}
					}
				}
			}
		}
		return strings.Join(result, "\n")
	case "class":
		var result []string
		result = append(result, lines[0])
		baseIndent := ""
		if len(lines) > 1 {
			for _, ch := range lines[1] {
				if ch == ' ' || ch == '\t' {
					baseIndent += string(ch)
				} else {
					break
				}
			}
		}
		for i := 1; i < len(lines); i++ {
			trimmed := strings.TrimSpace(lines[i])
			if trimmed == "" {
				continue
			}
			indent := ""
			for _, ch := range lines[i] {
				if ch == ' ' || ch == '\t' {
					indent += string(ch)
				} else {
					break
				}
			}
			if indent == baseIndent {
				if strings.HasPrefix(trimmed, "def ") {
					sig := trimmed
					if strings.Contains(sig, ":") {
						sig = strings.SplitN(sig, ":", 2)[0] + ":"
					}
					result = append(result, baseIndent+sig)
				} else if strings.HasPrefix(trimmed, "class ") {
					result = append(result, baseIndent+trimmed)
				} else if !strings.HasPrefix(trimmed, "#") && (strings.Contains(trimmed, "=") || strings.Contains(trimmed, ":")) {
					result = append(result, baseIndent+trimmed)
				}
			}
		}
		return strings.Join(result, "\n")
	default:
		return lines[0]
	}
}
