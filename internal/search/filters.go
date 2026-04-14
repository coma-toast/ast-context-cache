package search

import (
	"path/filepath"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

// SearchFilters narrows hybrid / vector / BM25 recall by path, symbol kind, or language (file extension).
// Nil or Empty() means no filtering.
type SearchFilters struct {
	PathPrefix string
	Kinds      []string
	Language   string
}

// ParseSearchFilters builds filters from MCP tool arguments (optional).
func ParseSearchFilters(args map[string]interface{}) *SearchFilters {
	f := &SearchFilters{}
	if p, ok := args["path_prefix"].(string); ok {
		f.PathPrefix = strings.TrimSpace(p)
	}
	if lang, ok := args["language"].(string); ok {
		f.Language = strings.ToLower(strings.TrimSpace(lang))
	}
	switch v := args["kinds"].(type) {
	case string:
		for _, part := range strings.Split(v, ",") {
			part = strings.TrimSpace(part)
			if part != "" {
				f.Kinds = append(f.Kinds, part)
			}
		}
	case []interface{}:
		for _, item := range v {
			if s, ok := item.(string); ok {
				s = strings.TrimSpace(s)
				if s != "" {
					f.Kinds = append(f.Kinds, s)
				}
			}
		}
	}
	if k, ok := args["kind"].(string); ok {
		k = strings.TrimSpace(k)
		if k != "" {
			f.Kinds = append(f.Kinds, k)
		}
	}
	f.Kinds = dedupeStringsPreserveOrder(f.Kinds)
	if f.Empty() {
		return nil
	}
	return f
}

func dedupeStringsPreserveOrder(in []string) []string {
	if len(in) < 2 {
		return in
	}
	seen := make(map[string]struct{}, len(in))
	out := in[:0]
	for _, s := range in {
		if _, ok := seen[s]; ok {
			continue
		}
		seen[s] = struct{}{}
		out = append(out, s)
	}
	return out
}

// symbolFilterSQL returns an SQL fragment (no leading AND) using alias `s` for the symbols table,
// plus positional args. Pushes down kind, language (file suffix), and path_prefix when possible
// so FTS / fallback read fewer rows before Go-side filterScoredResults.
func symbolFilterSQL(f *SearchFilters, projectPath string) (string, []interface{}) {
	if f == nil || f.Empty() {
		return "", nil
	}
	var parts []string
	var args []interface{}
	if len(f.Kinds) > 0 {
		ph := strings.Repeat("?,", len(f.Kinds))
		ph = strings.TrimSuffix(ph, ",")
		parts = append(parts, "s.kind IN ("+ph+")")
		for _, k := range f.Kinds {
			args = append(args, k)
		}
	}
	if f.Language != "" {
		exts := languageExtensions(f.Language)
		if len(exts) > 0 {
			var ors []string
			for _, ext := range exts {
				e := strings.TrimPrefix(strings.ToLower(ext), ".")
				ors = append(ors, "LOWER(s.file) LIKE ?")
				args = append(args, "%."+e)
			}
			parts = append(parts, "("+strings.Join(ors, " OR ")+")")
		}
	}
	if f.PathPrefix != "" {
		clause, a := pathPrefixSQLClause(projectPath, f.PathPrefix)
		if clause != "" {
			parts = append(parts, clause)
			args = append(args, a...)
		}
	}
	if len(parts) == 0 {
		return "", nil
	}
	return strings.Join(parts, " AND "), args
}

func pathPrefixSQLClause(projectPath, prefix string) (string, []interface{}) {
	prefix = strings.TrimSpace(prefix)
	if prefix == "" {
		return "", nil
	}
	p := filepath.ToSlash(prefix)
	p = strings.TrimPrefix(p, "./")
	var abs string
	if filepath.IsAbs(p) || strings.HasPrefix(p, "/") {
		abs = p
	} else {
		abs = filepath.ToSlash(filepath.Join(projectPath, strings.Trim(p, "/")))
	}
	// Match fileHasPathPrefix: exact path or anything under that directory.
	return "(s.file = ? OR s.file LIKE ?)", []interface{}{abs, abs + "/%"}
}

// CacheKey returns a stable string for deduplication caches (empty if no filters).
func (f *SearchFilters) CacheKey() string {
	if f == nil || f.Empty() {
		return ""
	}
	var b strings.Builder
	b.WriteString("p:")
	b.WriteString(f.PathPrefix)
	b.WriteString("|k:")
	b.WriteString(strings.Join(f.Kinds, ","))
	b.WriteString("|l:")
	b.WriteString(f.Language)
	return b.String()
}

// Empty reports whether any filter is active.
func (f *SearchFilters) Empty() bool {
	return f == nil || (f.PathPrefix == "" && len(f.Kinds) == 0 && f.Language == "")
}

// MatchesSymbol returns whether a symbol row matches all active constraints.
func (f *SearchFilters) MatchesSymbol(file, kind, projectPath string) bool {
	if f == nil || f.Empty() {
		return true
	}
	if len(f.Kinds) > 0 {
		ok := false
		for _, k := range f.Kinds {
			if kind == k {
				ok = true
				break
			}
		}
		if !ok {
			return false
		}
	}
	if f.Language != "" {
		exts := languageExtensions(f.Language)
		if len(exts) == 0 || !fileHasAnyExtension(file, exts) {
			return false
		}
	}
	if f.PathPrefix != "" {
		return fileHasPathPrefix(file, projectPath, f.PathPrefix)
	}
	return true
}

func fileHasAnyExtension(file string, exts []string) bool {
	lower := strings.ToLower(file)
	for _, ext := range exts {
		ext = strings.ToLower(ext)
		if !strings.HasPrefix(ext, ".") {
			ext = "." + ext
		}
		if strings.HasSuffix(lower, ext) {
			return true
		}
	}
	return false
}

// fileHasPathPrefix matches path_prefix against project-relative paths and common absolute layouts.
func fileHasPathPrefix(file, projectPath, prefix string) bool {
	prefix = filepath.ToSlash(strings.TrimSpace(prefix))
	if prefix == "" {
		return true
	}
	prefix = strings.TrimPrefix(prefix, "./")
	fileSlash := filepath.ToSlash(file)
	// Absolute prefix: compare to full file path
	if filepath.IsAbs(prefix) {
		p := filepath.ToSlash(prefix)
		return strings.HasPrefix(fileSlash, p) || strings.HasPrefix(fileSlash, strings.TrimSuffix(p, "/")+"/")
	}
	rel := filepath.ToSlash(db.RelPath(file, projectPath))
	p := strings.Trim(prefix, "/")
	return rel == p || strings.HasPrefix(rel, p+"/") || strings.HasPrefix(rel, p)
}

// languageExtensions maps a coarse language name to file suffixes used in the repo.
func languageExtensions(lang string) []string {
	switch strings.ToLower(lang) {
	case "go", "golang":
		return []string{".go"}
	case "python", "py":
		return []string{".py"}
	case "typescript", "ts":
		return []string{".ts", ".tsx"}
	case "javascript", "js":
		return []string{".js", ".jsx", ".mjs", ".cjs"}
	case "rust", "rs":
		return []string{".rs"}
	case "ruby", "rb":
		return []string{".rb"}
	case "java":
		return []string{".java"}
	case "bash", "sh":
		return []string{".sh"}
	case "fish":
		return []string{".fish"}
	case "yaml", "yml":
		return []string{".yaml", ".yml"}
	default:
		return nil
	}
}

func filterScoredResults(scored []ScoredResult, projectPath string, filters *SearchFilters) []ScoredResult {
	if filters == nil || filters.Empty() {
		return scored
	}
	var out []ScoredResult
	for _, s := range scored {
		file, _ := s.Data["file"].(string)
		kind, _ := s.Data["kind"].(string)
		if filters.MatchesSymbol(file, kind, projectPath) {
			out = append(out, s)
		}
	}
	return out
}
