package context

import (
	"regexp"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
)

const (
	// selectCalleeSymbolsPrefix is completed with one placeholder per callee name and
	// selectCalleeSymbolsSuffix; same-file symbols sort first.
	selectCalleeSymbolsPrefix = "SELECT name, kind, file, start_line, end_line, COALESCE(fqn,'') FROM symbols WHERE project_path = ? AND kind IN ('function','method') AND name IN ("
	selectCalleeSymbolsSuffix = ") ORDER BY file = ? DESC, file, start_line"
	maxEditCallees            = 10

	// Per-language words that look like calls (`if (`, `len(`) but never name a project symbol.
	commonCallKeywords = "if for while switch return catch function sizeof typeof"
	goCallKeywords     = "func go defer select case range make new len cap append copy delete panic recover print println close complex real imag min max clear string int int8 int16 int32 int64 uint uint8 uint16 uint32 uint64 uintptr byte rune float32 float64 bool error any map chan struct interface"
	pythonCallKeywords = "def class elif except with lambda not and or in is assert del yield await print len range str int float bool list dict set tuple isinstance issubclass super type getattr setattr hasattr enumerate zip map filter sorted min max sum any all open repr iter next"
	jsCallKeywords     = "else do new delete void await async import export require super constructor get set catch finally throw class this console parseInt parseFloat String Number Boolean Array Object Promise Symbol Map Set"
	rustCallKeywords   = "fn match loop impl where move unsafe let mut Some Ok Err Box Vec String Rc Arc println print format vec panic assert assert_eq"
	shellCallKeywords  = "then elif fi do done case esac in end function"
)

// calleeCallRe captures identifiers followed by an opening parenthesis.
var calleeCallRe = regexp.MustCompile(`\b([A-Za-z_][A-Za-z0-9_]*)\s*\(`)

// callKeywords maps an indexer language to the words dropped from callee candidates.
var callKeywords = map[string]map[string]bool{
	"":           wordSet(commonCallKeywords),
	"go":         wordSet(commonCallKeywords, goCallKeywords),
	"python":     wordSet(commonCallKeywords, pythonCallKeywords),
	"javascript": wordSet(commonCallKeywords, jsCallKeywords),
	"typescript": wordSet(commonCallKeywords, jsCallKeywords),
	"tsx":        wordSet(commonCallKeywords, jsCallKeywords),
	"rust":       wordSet(commonCallKeywords, rustCallKeywords),
	"bash":       wordSet(commonCallKeywords, shellCallKeywords),
	"fish":       wordSet(commonCallKeywords, shellCallKeywords),
}

// EditView builds the edit-mode view of target: the target with its exact source span
// (mode "edit"), plus up to 10 skeletons of the project functions and methods its body
// calls, same-file callees first.
func EditView(projectPath string, target map[string]any, fileCache map[string][]string) (map[string]any, []map[string]any, error) {
	file, name := mapStr(target, "file"), mapStr(target, "name")
	start, end := coerceInt(target["start_line"]), coerceInt(target["end_line"])
	if file == "" || start <= 0 || end < start {
		return nil, nil, errs.NewCode(errs.CodeInvalidInput, "edit target needs a file and line span", "file", file, "start_line", start, "end_line", end)
	}
	src := indexer.ReadSourceRange(file, start, end, fileCache)
	if src == "" {
		return nil, nil, errs.NewCode(errs.CodeNotFound, "unable to read edit target source", "file", file, "start_line", start, "end_line", end)
	}
	out := make(map[string]any, len(target)+2)
	for k, v := range target {
		out[k] = v
	}
	delete(out, "skeleton")
	delete(out, "summary")
	out["source"], out["mode"] = src, "edit"
	callees, err := resolveCallees(projectPath, file, calleeNames(src, file, name, mapStr(target, "qualified_name")), fileCache)
	if err != nil {
		return nil, nil, err
	}
	return out, callees, nil
}

// calleeNames returns the distinct called identifiers in src, in order of first use,
// minus the target's own name and the language's keywords.
func calleeNames(src, file, name, qualified string) []string {
	kw, ok := callKeywords[indexer.GetLanguage(file)]
	if !ok {
		kw = callKeywords[""]
	}
	own := map[string]bool{name: true}
	if i := strings.LastIndex(qualified, "."); i >= 0 {
		own[qualified[i+1:]] = true
	}
	seen := map[string]bool{}
	var out []string
	for _, m := range calleeCallRe.FindAllStringSubmatch(src, -1) {
		id := m[1]
		if own[id] || kw[id] || seen[id] {
			continue
		}
		seen[id] = true
		out = append(out, id)
	}
	return out
}

// resolveCallees looks the names up as project functions and methods, keeps the first
// match per name (same file first), caps at maxEditCallees and renders each as a skeleton.
func resolveCallees(projectPath, file string, names []string, fileCache map[string][]string) ([]map[string]any, error) {
	if len(names) == 0 {
		return nil, nil
	}
	conn, err := db.IndexReader()
	if err != nil {
		return nil, errs.WrapMessage("unable to open index for edit callees", err)
	}
	args := make([]any, 0, len(names)+2)
	args = append(args, projectPath)
	for _, n := range names {
		args = append(args, n)
	}
	args = append(args, file)
	q := selectCalleeSymbolsPrefix + strings.TrimSuffix(strings.Repeat("?,", len(names)), ",") + selectCalleeSymbolsSuffix
	rows, err := conn.Query(q, args...)
	if err != nil {
		return nil, errs.WrapMessage("unable to query edit callees", err, "file", file)
	}
	defer rows.Close()
	seen := map[string]bool{}
	var out []map[string]any
	for rows.Next() && len(out) < maxEditCallees {
		var name, kind, cFile, fqn string
		var start, end int
		if err := rows.Scan(&name, &kind, &cFile, &start, &end, &fqn); err != nil {
			return nil, errs.WrapMessage("unable to scan edit callee", err, "file", file)
		}
		if seen[name] {
			continue
		}
		seen[name] = true
		data := map[string]any{"name": name, "kind": kind, "file": cFile, "start_line": start, "end_line": end, "mode": "skeleton"}
		if qn := db.QualifiedName(fqn, cFile, name); qn != name {
			data["qualified_name"] = qn
		}
		ApplyMode(data, "skeleton", cFile, name, projectPath, start, end, fileCache)
		out = append(out, data)
	}
	if err := rows.Err(); err != nil {
		return nil, errs.WrapMessage("unable to read edit callees", err, "file", file)
	}
	return out, nil
}

func wordSet(lists ...string) map[string]bool {
	set := map[string]bool{}
	for _, l := range lists {
		for _, w := range strings.Fields(l) {
			set[w] = true
		}
	}
	return set
}
