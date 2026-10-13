package context

import (
	"crypto/sha256"
	"path"
	"path/filepath"
	"regexp"
	"slices"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/render"
)

const (
	settingCollapseGlobs = "collapse_globs"
	// testsGroup collects distractor hits with no non-distractor result of the same name.
	testsGroup = "tests"
)

// builtinDistractorGlobs are test, mock, vendored and generated paths. A pattern ending in
// "/" matches a directory segment anywhere in the path; one containing "/" otherwise
// matches the whole path; the rest match the base name.
var builtinDistractorGlobs = []string{
	"*_test.go", "test_*.py", "*_test.py", "*.spec.ts", "*.test.ts", "*.spec.js", "*.test.js",
	"__mocks__/", "mocks/", "mock_*.go", "vendor/", "node_modules/", "testdata/", "*.pb.go", "*_gen.go",
}

// distractorQueryRe marks queries that are about tests or mocks; collapse is skipped for them.
var distractorQueryRe = regexp.MustCompile(`(?i)\b(test|spec|mock)`)

// IsDistractorPath reports whether rel is a test, mock, vendored or generated file under the
// built-in patterns or the extra globs (same pattern rules).
func IsDistractorPath(rel string, extra []string) bool {
	rel = filepath.ToSlash(rel)
	for _, g := range builtinDistractorGlobs {
		if matchDistractorGlob(rel, g) {
			return true
		}
	}
	for _, g := range extra {
		if matchDistractorGlob(rel, g) {
			return true
		}
	}
	return false
}

// CollapseDistractors folds distractor hits and same-signature duplicates (same name, kind
// and skeleton, or source when no skeleton) out of results. Each folds under the first
// non-distractor result with the same name, or into the generic "tests" group when there is
// none. Nothing is collapsed when !enabled, the query is about tests/specs/mocks, or no
// result is a non-distractor.
func CollapseDistractors(results []map[string]any, query string, enabled bool) (kept []map[string]any, collapsed []render.Collapse) {
	if !enabled || distractorQueryRe.MatchString(query) {
		return results, nil
	}
	extra := collapseGlobs()
	distractor := make([]bool, len(results))
	primary := map[string]bool{}
	for i, r := range results {
		distractor[i] = IsDistractorPath(mapStr(r, "file"), extra)
		if !distractor[i] {
			primary[mapStr(r, "name")] = true
		}
	}
	if len(primary) == 0 {
		return results, nil
	}
	groups := map[string]int{}
	seenSig := map[string]bool{}
	add := func(into string, r map[string]any) {
		i, ok := groups[into]
		if !ok {
			i = len(collapsed)
			groups[into] = i
			collapsed = append(collapsed, render.Collapse{Into: into})
		}
		c := &collapsed[i]
		c.Count++
		if f := mapStr(r, "file"); f != "" && !slices.Contains(c.Paths, f) {
			c.Paths = append(c.Paths, f)
		}
	}
	for i, r := range results {
		name, sig := mapStr(r, "name"), signature(r)
		switch {
		case distractor[i] && primary[name]:
			add(name, r)
		case distractor[i]:
			add(testsGroup, r)
		case sig != "" && seenSig[sig]:
			add(name, r)
		default:
			if sig != "" {
				seenSig[sig] = true
			}
			kept = append(kept, r)
		}
	}
	return kept, collapsed
}

func matchDistractorGlob(rel, g string) bool {
	g = strings.TrimSpace(filepath.ToSlash(g))
	switch {
	case g == "":
		return false
	case strings.HasSuffix(g, "/"):
		dir := strings.TrimSuffix(g, "/")
		return strings.HasPrefix(rel, dir+"/") || strings.Contains(rel, "/"+dir+"/")
	case strings.Contains(g, "/"):
		ok, _ := path.Match(g, rel)
		return ok
	default:
		ok, _ := path.Match(g, path.Base(rel))
		return ok
	}
}

// collapseGlobs parses the comma-separated collapse_globs setting.
func collapseGlobs() []string {
	var out []string
	for _, g := range strings.Split(db.GetSetting(settingCollapseGlobs, ""), ",") {
		if g = strings.TrimSpace(g); g != "" {
			out = append(out, g)
		}
	}
	return out
}

// signature identifies a symbol's shape across files: name, kind and a hash of its
// skeleton (or source when no skeleton is attached). Empty when there is no body.
func signature(r map[string]any) string {
	body := mapStr(r, "skeleton")
	if body == "" {
		body = mapStr(r, "source")
	}
	if body == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(body))
	return mapStr(r, "name") + "|" + mapStr(r, "kind") + "|" + string(sum[:])
}

func mapStr(m map[string]any, k string) string {
	s, _ := m[k].(string)
	return s
}
