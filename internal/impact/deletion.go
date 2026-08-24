package impact

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
)

// HandleCheckDeletionSafety reports whether the symbols a change removes from a
// file are still referenced anywhere. It parses the base revision and the working
// tree copy of the file with the same tree-sitter extraction the indexer uses, then
// runs the impact graph over every name that disappeared, sibling worktrees
// included.
//
// Symbols are compared by name, so a rename reads as one removal plus one
// addition and a symbol merely moved within the file is not reported as removed.
// That is deliberate: the question this answers is "does this name still have
// callers", which a rename must also answer.
func HandleCheckDeletionSafety(args map[string]interface{}, projectPath string) string {
	if projectPath == "" {
		return `{"error": "project_path required"}`
	}
	file := strArg(args, "file")
	if file == "" {
		return `{"error": "file required"}`
	}
	baseRef := strArg(args, "base_ref")
	if baseRef == "" {
		baseRef = "origin/main"
	}

	rel, err := relativeToProject(file, projectPath)
	if err != nil {
		return errJSON(err)
	}
	lang := indexer.GetLanguage(rel)
	if lang == "" {
		return errJSON(fmt.Errorf("unsupported file type: %s", rel))
	}

	baseContent, err := runGit(projectPath, "show", baseRef+":./"+filepath.ToSlash(rel))
	if err != nil {
		return errJSON(err)
	}
	// A file deleted outright has no working-tree copy; every base symbol is gone.
	currentContent, _ := os.ReadFile(filepath.Join(projectPath, rel))

	removed := removedNames(
		indexer.ParseSymbols([]byte(baseContent), lang),
		indexer.ParseSymbols(currentContent, lang),
	)

	stillReferenced := map[string][]string{}
	safe := []string{}
	for _, name := range removed {
		res, err := Graph(name, projectPath, true)
		if err != nil {
			continue
		}
		refs := confirmedReferences(dependents(res, []string{rel}), name)
		if len(refs) > 0 {
			stillReferenced[name] = refs
		} else {
			safe = append(safe, name)
		}
	}

	data, _ := json.Marshal(map[string]interface{}{
		"file":                       rel,
		"base_ref":                   baseRef,
		"removed_symbols":            removed,
		"still_referenced_elsewhere": stillReferenced,
		"safe_to_delete":             safe,
		"checked_scope":              projectlinks.ResolveScopeWithRepoSiblings(projectPath, true),
	})
	return string(data)
}

// maxConfirmBytes caps how much of a dependent file is read back when confirming
// a reference; anything larger is reported without confirmation.
const maxConfirmBytes = 2 << 20

// confirmedReferences narrows the impact graph's file-level import edges down to
// the files whose text actually mentions the name. Edges record module imports,
// not per-symbol calls, so without this every symbol in an imported file would
// look unsafe to delete. Files that cannot be read are kept, erring toward unsafe.
func confirmedReferences(entries []Entry, name string) []string {
	var out []string
	for _, e := range entries {
		if mentionsSymbol(e.Abs, name) {
			out = append(out, e.File)
		}
	}
	return out
}

func mentionsSymbol(absFile, name string) bool {
	if absFile == "" || name == "" {
		return true
	}
	info, err := os.Stat(absFile)
	if err != nil || info.Size() > maxConfirmBytes {
		return true
	}
	content, err := os.ReadFile(absFile)
	if err != nil {
		return true
	}
	text := string(content)
	for i := 0; ; {
		idx := strings.Index(text[i:], name)
		if idx < 0 {
			return false
		}
		start := i + idx
		end := start + len(name)
		if !isIdentChar(byteAt(text, start-1)) && !isIdentChar(byteAt(text, end)) {
			return true
		}
		i = end
	}
}

func byteAt(s string, i int) byte {
	if i < 0 || i >= len(s) {
		return ' '
	}
	return s[i]
}

func isIdentChar(b byte) bool {
	return b == '_' || b == '$' ||
		(b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || (b >= '0' && b <= '9')
}

// removedNames returns names present in base but absent from current.
func removedNames(base, current []indexer.SymbolDef) []string {
	kept := make(map[string]bool, len(current))
	for _, s := range current {
		kept[s.Name] = true
	}
	seen := map[string]bool{}
	removed := []string{}
	for _, s := range base {
		if kept[s.Name] || seen[s.Name] {
			continue
		}
		seen[s.Name] = true
		removed = append(removed, s.Name)
	}
	sort.Strings(removed)
	return removed
}

// relativeToProject accepts an absolute or project-relative file and returns it
// relative to the project root.
func relativeToProject(file, projectPath string) (string, error) {
	projectPath = projectlinks.NormalizePath(projectPath)
	if !filepath.IsAbs(file) {
		file = filepath.Join(projectPath, file)
	}
	rel, err := filepath.Rel(projectPath, filepath.Clean(file))
	if err != nil || rel == "." || strings.HasPrefix(rel, "..") {
		return "", fmt.Errorf("file must live inside project_path")
	}
	return rel, nil
}
