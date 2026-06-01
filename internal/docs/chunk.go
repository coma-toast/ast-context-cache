package docs

import (
	"encoding/json"
	"regexp"
	"strconv"
	"strings"
)

const maxChunkChars = 2000

var (
	htmlScriptRe = regexp.MustCompile(`(?is)<script[^>]*>.*?</script>`)
	htmlStyleRe  = regexp.MustCompile(`(?is)<style[^>]*>.*?</style>`)
	htmlHeadingRe = regexp.MustCompile(`(?is)<h([1-4])[^>]*>(.*?)</h([1-4])>`)
	htmlTagRe    = regexp.MustCompile(`<[^>]+>`)
	htmlWSRe     = regexp.MustCompile(`\s+`)
)

func chunkMarkdown(content, path string) []DocEntry {
	sections := splitMarkdownSections(content)
	var out []DocEntry
	for _, s := range sections {
		out = append(out, splitLongSection(s.Title, s.Content, path)...)
	}
	if len(out) == 0 && strings.TrimSpace(content) != "" {
		out = append(out, DocEntry{Title: path, Content: strings.TrimSpace(content), Path: path})
	}
	return out
}

func chunkHTML(html, path string) []DocEntry {
	html = htmlScriptRe.ReplaceAllString(html, " ")
	html = htmlStyleRe.ReplaceAllString(html, " ")
	title := path
	if m := regexp.MustCompile(`(?is)<title[^>]*>(.*?)</title>`).FindStringSubmatch(html); len(m) > 1 {
		title = cleanHTMLText(m[1])
	}
	sections := extractHTMLSections(html)
	if len(sections) == 0 {
		text := cleanHTMLText(html)
		if text == "" {
			return nil
		}
		return splitLongSection(title, text, path)
	}
	var out []DocEntry
	for _, s := range sections {
		if s.Title == "" {
			s.Title = title
		}
		out = append(out, splitLongSection(s.Title, s.Content, path)...)
	}
	return out
}

func chunkJSON(raw, path string) []DocEntry {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	var parsed interface{}
	if json.Unmarshal([]byte(raw), &parsed) != nil {
		return []DocEntry{{Title: path, Content: raw, Path: path}}
	}
	switch v := parsed.(type) {
	case map[string]interface{}:
		var out []DocEntry
		for key, val := range v {
			b, _ := json.MarshalIndent(val, "", "  ")
			content := string(b)
			out = append(out, splitLongSection(key, content, path)...)
		}
		if len(out) > 0 {
			return out
		}
	case []interface{}:
		var b strings.Builder
		for i, item := range v {
			if i > 0 {
				b.WriteByte('\n')
			}
			line, _ := json.Marshal(item)
			b.Write(line)
		}
		return splitLongSection(path, b.String(), path)
	}
	return []DocEntry{{Title: path, Content: raw, Path: path}}
}

func splitLongSection(title, content, path string) []DocEntry {
	content = strings.TrimSpace(content)
	if content == "" {
		return nil
	}
	if len(content) <= maxChunkChars {
		return []DocEntry{{Title: strings.TrimSpace(title), Content: content, Path: path}}
	}
	paras := splitParagraphs(content)
	var out []DocEntry
	var buf strings.Builder
	part := 1
	flush := func() {
		if buf.Len() == 0 {
			return
		}
		chunkTitle := strings.TrimSpace(title)
		if part > 1 {
			chunkTitle = chunkTitle + " (part " + strconv.Itoa(part) + ")"
		}
		out = append(out, DocEntry{Title: chunkTitle, Content: strings.TrimSpace(buf.String()), Path: path})
		buf.Reset()
		part++
	}
	for _, p := range paras {
		if len(p) > maxChunkChars {
			flush()
			for i := 0; i < len(p); i += maxChunkChars {
				end := i + maxChunkChars
				if end > len(p) {
					end = len(p)
				}
				chunkTitle := strings.TrimSpace(title)
				if part > 1 || end < len(p) {
					chunkTitle = chunkTitle + " (part " + strconv.Itoa(part) + ")"
				}
				out = append(out, DocEntry{Title: chunkTitle, Content: p[i:end], Path: path})
				part++
			}
			continue
		}
		if buf.Len()+len(p)+2 > maxChunkChars {
			flush()
		}
		if buf.Len() > 0 {
			buf.WriteString("\n\n")
		}
		buf.WriteString(p)
	}
	flush()
	return out
}

func splitParagraphs(content string) []string {
	parts := regexp.MustCompile(`\n\s*\n`).Split(content, -1)
	var out []string
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	if len(out) == 0 {
		return []string{content}
	}
	return out
}

func extractHTMLSections(html string) []markdownSection {
	loc := htmlHeadingRe.FindAllStringSubmatchIndex(html, -1)
	if len(loc) == 0 {
		return nil
	}
	var sections []markdownSection
	for i, m := range loc {
		openLevel := html[m[2]:m[3]]
		closeLevel := html[m[6]:m[7]]
		if openLevel != closeLevel {
			continue
		}
		title := cleanHTMLText(html[m[4]:m[5]])
		start := m[1]
		end := len(html)
		if i+1 < len(loc) {
			end = loc[i+1][0]
		}
		body := cleanHTMLText(html[start:end])
		if title != "" && body != "" {
			sections = append(sections, markdownSection{Title: title + " (h" + openLevel + ")", Content: body})
		}
	}
	return sections
}

func cleanHTMLText(s string) string {
	s = htmlTagRe.ReplaceAllString(s, " ")
	s = htmlWSRe.ReplaceAllString(s, " ")
	return strings.TrimSpace(s)
}
