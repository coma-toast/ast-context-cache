// Package render turns code-tool results into compact text: a summary line, then a
// heading and fenced block per symbol (Text), or one line per hit (Locations).
package render

import (
	"fmt"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/tokens"
)

// Response is one code tool's packed result, ready to render.
type Response struct {
	Tool, Query string
	// Root, when set, is the project path that result files are shown relative to.
	Root string
	// Results are the packed result maps (file, start_line, end_line, kind, name,
	// qualified_name, source|skeleton|summary, mode, score|similarity).
	Results []map[string]any
	// Total is the candidate count before packing; it is carried for callers and not printed.
	Total, Withheld, WithheldTokens int
	Collapsed                       []Collapse
	NoMatch                         *NoMatch
	Notes                           []string
}

// Collapse is a group of distractor or duplicate hits folded under one result.
type Collapse struct {
	Into  string   `json:"into"`
	Paths []string `json:"paths"`
	Count int      `json:"count"`
}

// NoMatch reports that the best hit was too weak to count as a match.
type NoMatch struct {
	BestScore float64 `json:"best_score"`
	Hint      string  `json:"hint"`
}

// bodyKeys are the result keys holding a symbol body, with the mode each implies.
var bodyKeys = []struct{ key, mode string }{{"source", "full"}, {"skeleton", "skeleton"}, {"summary", "summary"}}

// Text renders r as a summary line followed by a heading and fenced block per result.
func Text(r Response) string {
	var body strings.Builder
	for _, res := range r.Results {
		writeResult(&body, r.Root, res)
	}
	for _, c := range r.Collapsed {
		fmt.Fprintf(&body, "collapsed into %s: %s\n", c.Into, strings.Join(c.Paths, ", "))
	}
	return header(r, tokens.Count(body.String())) + body.String()
}

// Locations renders r as a summary line followed by one `file:start-end kind name score` line per hit.
func Locations(r Response) string {
	var body strings.Builder
	for _, res := range r.Results {
		line := location(r.Root, res) + " " + str(res, "kind") + " " + displayName(res)
		if s, ok := score(res); ok {
			line += " " + strconv.FormatFloat(s, 'g', 3, 64)
		}
		body.WriteString(strings.TrimRight(line, " ") + "\n")
	}
	return header(r, tokens.Count(body.String())) + body.String()
}

// LangFromExt maps a file's extension to its fenced-code language, or "" when unknown.
func LangFromExt(file string) string {
	switch strings.ToLower(filepath.Ext(file)) {
	case ".go":
		return "go"
	case ".py":
		return "python"
	case ".ts", ".tsx":
		return "typescript"
	case ".js", ".jsx":
		return "javascript"
	case ".rs":
		return "rust"
	case ".rb":
		return "ruby"
	case ".java":
		return "java"
	case ".sh":
		return "bash"
	case ".fish":
		return "fish"
	case ".yaml", ".yml":
		return "yaml"
	default:
		return ""
	}
}

// header is the summary line plus any hint and notes, each newline-terminated.
func header(r Response, tok int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s %q · %d results · %d tok", r.Tool, r.Query, len(r.Results), tok)
	if r.Withheld > 0 {
		fmt.Fprintf(&b, " · withheld %d (%d tok)", r.Withheld, r.WithheldTokens)
	}
	if n := collapsedCount(r.Collapsed); n > 0 {
		fmt.Fprintf(&b, " · also %d similar in tests", n)
	}
	if r.NoMatch != nil {
		fmt.Fprintf(&b, " · no match (best %.2f)", r.NoMatch.BestScore)
	}
	b.WriteString("\n")
	if r.NoMatch != nil && r.NoMatch.Hint != "" {
		b.WriteString(r.NoMatch.Hint + "\n")
	}
	for _, n := range r.Notes {
		b.WriteString(n + "\n")
	}
	return b.String()
}

func writeResult(b *strings.Builder, root string, res map[string]any) {
	text, mode := resultBody(res)
	heading := "### " + location(root, res) + " " + str(res, "kind") + " " + displayName(res)
	if mode != "" {
		heading += " (" + mode + ")"
	}
	b.WriteString(strings.TrimRight(heading, " ") + "\n")
	if text == "" {
		return
	}
	fence := "```"
	for strings.Contains(text, fence) {
		fence += "`"
	}
	b.WriteString(fence + LangFromExt(str(res, "file")) + "\n" + strings.TrimRight(text, "\n") + "\n" + fence + "\n")
}

// resultBody returns the first body present and its mode; an explicit mode key wins.
func resultBody(res map[string]any) (string, string) {
	var text, mode string
	for _, k := range bodyKeys {
		if s := str(res, k.key); s != "" {
			text, mode = s, k.mode
			break
		}
	}
	if m := str(res, "mode"); m != "" {
		mode = m
	}
	return text, mode
}

func location(root string, res map[string]any) string {
	file := relPath(root, str(res, "file"))
	start, end := intField(res, "start_line"), intField(res, "end_line")
	if start <= 0 {
		return file
	}
	if end < start {
		end = start
	}
	return file + ":" + strconv.Itoa(start) + "-" + strconv.Itoa(end)
}

func relPath(root, file string) string {
	if root == "" || !filepath.IsAbs(file) {
		return filepath.ToSlash(file)
	}
	rel, err := filepath.Rel(root, file)
	if err != nil || strings.HasPrefix(rel, "..") {
		return filepath.ToSlash(file)
	}
	return filepath.ToSlash(rel)
}

func displayName(res map[string]any) string {
	if q := str(res, "qualified_name"); q != "" {
		return q
	}
	return str(res, "name")
}

func score(res map[string]any) (float64, bool) {
	for _, k := range []string{"score", "similarity"} {
		if f, ok := floatField(res, k); ok {
			return f, true
		}
	}
	return 0, false
}

func collapsedCount(cs []Collapse) int {
	n := 0
	for _, c := range cs {
		n += c.Count
	}
	return n
}

func str(m map[string]any, k string) string {
	s, _ := m[k].(string)
	return s
}

func intField(m map[string]any, k string) int {
	f, _ := floatField(m, k)
	return int(f)
}

func floatField(m map[string]any, k string) (float64, bool) {
	switch v := m[k].(type) {
	case float64:
		return v, true
	case float32:
		return float64(v), true
	case int:
		return float64(v), true
	case int64:
		return float64(v), true
	case int32:
		return float64(v), true
	default:
		return 0, false
	}
}
