package installer

import (
	"regexp"
	"strings"

	"github.com/BurntSushi/toml"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// TOML is edited as text: BurntSushi/toml only validates (it can't round-trip comments), and the
// edit replaces, appends, or removes just our [mcp_servers.ast-context-cache] table span.
const tomlManagedPrefix = "# managed by ast-context-cache"

var (
	tomlHeaderRe = regexp.MustCompile(`^\s*\[\[?\s*([^\[\]]*?)\s*\]\]?\s*(#.*)?$`)
	tomlStringRe = regexp.MustCompile(`"(?:[^"\\]|\\.)*"|'[^']*'`)
)

// tomlSpan is the byte range of our table: the managed comment line (if any) through the last
// key line, excluding trailing blank and comment lines that belong to the next table.
type tomlSpan struct {
	start, end int
}

// decodeTOML validates data and returns its decoded tables.
func decodeTOML(data string) (map[string]any, error) {
	out := map[string]any{}
	if _, err := toml.Decode(data, &out); err != nil {
		return nil, errs.WrapCodeMessage(errs.CodeInvalidInput, "failed to parse TOML", err)
	}
	return out, nil
}

// tomlTable returns the decoded table at the dotted path, e.g. mcp_servers → ast-context-cache.
func tomlTable(doc map[string]any, path ...string) (map[string]any, bool) {
	cur := doc
	for _, k := range path {
		next, ok := cur[k].(map[string]any)
		if !ok {
			return nil, false
		}
		cur = next
	}
	return cur, true
}

// findTOMLTable locates the [key] table and its sub-tables. ok is false when there is no such
// header; the caller checks the decoded document for a definition in another form.
func findTOMLTable(data, key string) (tomlSpan, bool) {
	start, end, off, lastContent, depth := -1, -1, 0, -1, 0
	inMulti := ""
	for _, ln := range splitLinesKeep(data) {
		lineStart := off
		off += len(ln)
		text := strings.TrimRight(ln, "\r\n")
		trimmed := strings.TrimSpace(text)
		if inMulti != "" || depth > 0 {
			inMulti, depth = tomlScan(text, inMulti, depth)
			if start >= 0 {
				lastContent = off
			}
			continue
		}
		hk, isHeader := tomlHeaderKey(text)
		switch {
		case isHeader && (hk == key || strings.HasPrefix(hk, key+".")):
			if start < 0 {
				start = lineStart
			}
			lastContent = off
		case isHeader && start >= 0:
			end = lastContent
		case start >= 0 && trimmed != "" && !strings.HasPrefix(trimmed, "#"):
			lastContent = off
		}
		if end >= 0 {
			break
		}
		if !isHeader {
			inMulti, depth = tomlScan(text, inMulti, depth)
		}
	}
	if start < 0 {
		return tomlSpan{}, false
	}
	if end < 0 {
		end = lastContent
	}
	// Include our managed comment directly above the header.
	if start > 0 {
		lineStart := strings.LastIndex(strings.TrimRight(data[:start], "\r\n"), "\n") + 1
		if strings.HasPrefix(strings.TrimSpace(data[lineStart:start]), tomlManagedPrefix) {
			start = lineStart
		}
	}
	return tomlSpan{start: start, end: end}, true
}

// tomlScan tracks whether the next line is still inside a multi-line string or a multi-line
// array, where a line starting with "[" is not a table header.
func tomlScan(line, inMulti string, depth int) (string, int) {
	if inMulti != "" {
		if strings.Count(line, inMulti)%2 == 1 {
			return "", depth
		}
		return inMulti, depth
	}
	for _, q := range []string{`"""`, `'''`} {
		if strings.Count(line, q)%2 == 1 {
			return q, depth
		}
	}
	code := tomlStringRe.ReplaceAllString(line, `""`)
	if i := strings.IndexByte(code, '#'); i >= 0 {
		code = code[:i]
	}
	depth += strings.Count(code, "[") - strings.Count(code, "]")
	return "", max(depth, 0)
}

// upsertTOMLTable replaces our table span with block, or appends block after a blank line.
func upsertTOMLTable(data, key, block string) string {
	if sp, ok := findTOMLTable(data, key); ok {
		return data[:sp.start] + block + data[sp.end:]
	}
	if data == "" {
		return block
	}
	nl := newlineOf(data)
	if !strings.HasSuffix(data, "\n") {
		data += nl
	}
	return data + nl + block
}

// removeTOMLTable deletes our table span, and the blank separator line before it when the table
// was last in the file.
func removeTOMLTable(data, key string) (string, bool) {
	sp, ok := findTOMLTable(data, key)
	if !ok {
		return data, false
	}
	before, after := data[:sp.start], data[sp.end:]
	if before == "" || strings.HasSuffix(before, "\n\n") {
		after = strings.TrimPrefix(strings.TrimPrefix(after, "\r"), "\n")
	}
	if strings.TrimSpace(after) == "" {
		after = ""
		switch {
		case strings.HasSuffix(before, "\r\n\r\n"):
			before = before[:len(before)-2]
		case strings.HasSuffix(before, "\n\n"):
			before = before[:len(before)-1]
		}
	}
	return before + after, true
}

// tomlHeaderKey returns the normalized dotted key of a [table] or [[array]] header line.
func tomlHeaderKey(line string) (string, bool) {
	m := tomlHeaderRe.FindStringSubmatch(line)
	if m == nil {
		return "", false
	}
	parts := strings.Split(m[1], ".")
	for i, p := range parts {
		parts[i] = strings.Trim(strings.TrimSpace(p), `"'`)
	}
	return strings.Join(parts, "."), true
}

// tomlString quotes s as a TOML basic string.
func tomlString(s string) string {
	return `"` + strings.NewReplacer(`\`, `\\`, `"`, `\"`).Replace(s) + `"`
}

func splitLinesKeep(s string) []string {
	if s == "" {
		return nil
	}
	return strings.SplitAfter(s, "\n")
}
