package impact

import (
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"sort"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
)

// Entry is one file that depends on the analyzed symbol.
type Entry struct {
	File   string `json:"file"`
	Target string `json:"target"`
	Kind   string `json:"kind"`
	// Callers are the functions/methods in File whose own body imports the
	// symbol (function-local imports), e.g. "LlamaCppClient.load_model". Empty
	// when File only imports it at module level.
	Callers []string `json:"callers,omitempty"`
	// Abs is the indexed absolute path behind File, for callers that need to
	// read the file back; it stays out of the tool payload.
	Abs string `json:"-"`
}

// Result is the blast radius of a single symbol, with paths relative to the
// project (or sibling checkout) that owns them.
type Result struct {
	Symbol     string   `json:"symbol"`
	DefinedIn  []string `json:"defined_in"`
	ImpactedBy []Entry  `json:"impacted_by"`
	TotalFiles int      `json:"total_files"`
	Scope      []string `json:"checked_scope"`
}

// Graph computes the impact graph for symbol. When includeSiblings is set the
// query also covers other indexed checkouts of the same repo, so a change made in
// one worktree shows the callers living in another branch's worktree.
//
// A file depends on the symbol when one of its imports (module level or inside a
// function) loads the defining file and can reach the symbol, explicitly imports
// the symbol's name from the defining file or its package, or names the symbol
// as one of its own module path segments. Every comparison is on whole names and
// path segments (see resolveModule); nothing matches by substring.
func Graph(symbol, projectPath string, includeSiblings bool) (*Result, error) {
	if projectPath == "" {
		return nil, errors.New("project_path required")
	}
	if symbol == "" {
		return nil, errors.New("symbol required")
	}

	scopeFrag, scopeArgs, scope := projectlinks.ScopeSQLWithRepoSiblings("", projectPath, includeSiblings)

	conn, err := db.IndexReader()
	if err != nil {
		return nil, err
	}

	defs, codeDefs, err := definitionFiles(conn, scopeFrag, scopeArgs, symbol)
	if err != nil {
		return nil, err
	}
	sources, err := candidateSources(conn, scopeFrag, scopeArgs, symbol, defs)
	if err != nil {
		return nil, err
	}
	edgesByFile, err := importEdges(conn, scopeFrag, scopeArgs, sources)
	if err != nil {
		return nil, err
	}

	var impacts []Entry
	for _, src := range sources {
		if e, ok := matchSource(src, edgesByFile[src], symbol, defs, codeDefs); ok {
			impacts = append(impacts, e)
		}
	}

	defined := make([]string, 0, len(defs))
	for _, k := range defs {
		defined = append(defined, RelPathInScope(k, projectPath, scope))
	}

	relImpacts := make([]Entry, len(impacts))
	for i, imp := range impacts {
		imp.Abs = imp.File
		imp.File = RelPathInScope(imp.File, projectPath, scope)
		relImpacts[i] = imp
	}

	return &Result{
		Symbol:     symbol,
		DefinedIn:  defined,
		ImpactedBy: relImpacts,
		TotalFiles: len(relImpacts),
		Scope:      scope,
	}, nil
}

// definitionFiles returns the files defining symbol, matched case-sensitively
// since every indexed language is case-sensitive (folding made the YAML key
// `hosts:` a definition of the Python constant HOSTS). Code definitions win: a
// YAML key, Markdown heading or log line sharing the name is data, not a
// declaration a code change can break, so non-code files are reported only when
// nothing in code defines the name (e.g. asking about an Ansible role or key).
func definitionFiles(conn *sql.DB, scopeFrag string, scopeArgs []interface{}, symbol string) (defs []string, codeDefs bool, err error) {
	nameFrag, nameArgs := exactNameMatchSQL("", symbol)
	rows, err := conn.Query("SELECT DISTINCT file FROM symbols WHERE "+scopeFrag+" AND "+nameFrag+" ORDER BY file",
		append(append([]interface{}{}, scopeArgs...), nameArgs...)...)
	if err != nil {
		return nil, false, err
	}
	defer rows.Close()
	var code, other []string
	for rows.Next() {
		var f string
		if rows.Scan(&f) != nil {
			continue
		}
		if isCodeDefinitionFile(f) {
			code = append(code, f)
		} else {
			other = append(other, f)
		}
	}
	if err := rows.Err(); err != nil {
		return nil, false, err
	}
	if len(code) > 0 {
		return code, true, nil
	}
	return other, false, nil
}

func isCodeDefinitionFile(file string) bool {
	switch indexer.GetLanguage(file) {
	case "", "yaml", "markdown", "plaintext":
		return false
	}
	return true
}

// maxPrefilterNeedles bounds the LIKE prefilter; past it every import edge in
// scope is examined instead.
const maxPrefilterNeedles = 200

// candidateSources narrows the scope's import edges to files that could match,
// with a case-insensitive substring prefilter that matchSource then checks
// exactly. Targets without any name characters ("." / "..") are always kept
// since they resolve relative to the importing file.
func candidateSources(conn *sql.DB, scopeFrag string, scopeArgs []interface{}, symbol string, defs []string) ([]string, error) {
	needles := map[string]bool{symbol: true}
	for _, d := range defs {
		base := filepath.Base(d)
		needles[strings.TrimSuffix(base, filepath.Ext(base))] = true
		dir := filepath.Dir(d)
		needles[filepath.Base(dir)] = true
		if !isCodeDefinitionFile(d) {
			needles[filepath.Base(filepath.Dir(dir))] = true
		}
	}
	q := "SELECT DISTINCT source_file FROM edges WHERE " + scopeFrag + " AND kind IN ('import', 'import_names')"
	args := append([]interface{}{}, scopeArgs...)
	if len(needles) <= maxPrefilterNeedles {
		var ors []string
		for n := range needles {
			if n == "" || n == "." || n == string(filepath.Separator) {
				continue
			}
			ors = append(ors, "target LIKE ?")
			args = append(args, "%"+n+"%")
		}
		ors = append(ors, "target NOT GLOB '*[A-Za-z0-9_$]*'")
		q += " AND (" + strings.Join(ors, " OR ") + ")"
	}
	rows, err := conn.Query(q+" ORDER BY source_file", args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var f string
		if rows.Scan(&f) == nil {
			out = append(out, f)
		}
	}
	return out, rows.Err()
}

type importEdge struct {
	scope, target, kind string
}

// importEdges loads every import edge of the candidate files, so a file's
// statements for one module can be judged together.
func importEdges(conn *sql.DB, scopeFrag string, scopeArgs []interface{}, files []string) (map[string][]importEdge, error) {
	out := make(map[string][]importEdge, len(files))
	const batch = 500
	for i := 0; i < len(files); i += batch {
		chunk := files[i:min(i+batch, len(files))]
		args := append([]interface{}{}, scopeArgs...)
		for _, f := range chunk {
			args = append(args, f)
		}
		rows, err := conn.Query("SELECT source_file, COALESCE(source_symbol, ''), target, kind FROM edges WHERE "+scopeFrag+
			" AND kind IN ('import', 'import_names') AND source_file IN (?"+strings.Repeat(",?", len(chunk)-1)+") ORDER BY id", args...)
		if err != nil {
			return nil, err
		}
		for rows.Next() {
			var f string
			var e importEdge
			if rows.Scan(&f, &e.scope, &e.target, &e.kind) == nil {
				out[f] = append(out[f], e)
			}
		}
		rows.Close()
		if err := rows.Err(); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// importGroup is everything one scope of a file imports from one module.
type importGroup struct {
	scope, module string
	statements    int        // 'import' edges: one per statement
	named         [][]string // names bound by the statements that list them
}

// matchSource decides whether src depends on symbol. Its import statements are
// grouped per (enclosing function, module): a statement that lists names
// (`from m import a`) can only reach those names, while one that lists none
// (`import m`, require, or an index built before names were recorded) reaches
// the whole module.
func matchSource(src string, edges []importEdge, symbol string, defs []string, codeDefs bool) (Entry, bool) {
	var order []string
	groups := map[string]*importGroup{}
	for _, e := range edges {
		module, names := e.target, []string(nil)
		if e.kind == "import_names" {
			i := strings.LastIndex(e.target, indexer.ImportNamesSep)
			if i < 0 {
				continue
			}
			module, names = e.target[:i], strings.Split(e.target[i+len(indexer.ImportNamesSep):], ",")
		}
		key := e.scope + "\x00" + module
		g := groups[key]
		if g == nil {
			g = &importGroup{scope: e.scope, module: module}
			groups[key] = g
			order = append(order, key)
		}
		if e.kind == "import_names" {
			g.named = append(g.named, names)
		} else {
			g.statements++
		}
	}

	entry := Entry{File: src}
	matched := false
	callers := map[string]bool{}
	for _, key := range order {
		g := groups[key]
		kind, ok := matchGroup(g, src, symbol, defs, codeDefs)
		if !ok {
			continue
		}
		if !matched || (kind == "import_names" && entry.Kind != "import_names") {
			entry.Target, entry.Kind = g.module, kind
		}
		matched = true
		if g.scope != "" {
			callers[g.scope] = true
		}
	}
	if !matched {
		return Entry{}, false
	}
	for c := range callers {
		entry.Callers = append(entry.Callers, c)
	}
	sort.Strings(entry.Callers)
	return entry, true
}

func matchGroup(g *importGroup, src, symbol string, defs []string, codeDefs bool) (string, bool) {
	best := matchNone
	for _, d := range defs {
		if m := resolveModule(g.module, src, d); m > best {
			best = m
		}
	}
	namesSymbol, wildcard := false, false
	for _, names := range g.named {
		for _, n := range names {
			switch n {
			case symbol:
				namesSymbol = true
			case "*":
				wildcard = true
			}
		}
	}
	// An explicit import of the name: from the defining file or its package, or
	// from anywhere when no code declares it (PEP 562 module __getattr__,
	// re-exports of third-party names).
	if namesSymbol && (!codeDefs || best >= matchPackage) {
		return "import_names", true
	}
	if best == matchFile && (g.statements > len(g.named) || wildcard) {
		return "import", true
	}
	// `from pkg import mod` where mod is the defining file.
	for _, names := range g.named {
		for _, n := range names {
			if n == "*" {
				continue
			}
			sub := subModule(g.module, n, src)
			for _, d := range defs {
				if resolveModule(sub, src, d) == matchFile {
					return "import_names", true
				}
			}
		}
	}
	if moduleNamesSymbol(g.module, src, symbol) {
		return "import", true
	}
	return "", false
}

// subModule is the specifier of name imported as a submodule of module.
func subModule(module, name, src string) string {
	if langFamily(src) == "python" {
		if strings.HasSuffix(module, ".") {
			return module + name
		}
		return module + "." + name
	}
	return strings.TrimSuffix(module, "/") + "/" + name
}

// RelPathInScope shortens file against whichever scoped project owns it, so a hit
// from a sibling checkout stays readable instead of showing as a bare absolute path.
func RelPathInScope(file, projectPath string, scope []string) string {
	best := ""
	for _, p := range scope {
		if projectlinks.IsUnderPath(file, p) && len(p) > len(best) {
			best = p
		}
	}
	if best == "" {
		best = projectPath
	}
	return db.RelPath(file, best)
}

func HandleImpactGraph(args map[string]interface{}, projectPath string) string {
	symbol, _ := args["symbol"].(string)
	res, err := Graph(symbol, projectPath, false)
	if err != nil {
		return fmt.Sprintf(`{"error": "%s"}`, err.Error())
	}
	data, _ := json.Marshal(map[string]interface{}{
		"symbol":      res.Symbol,
		"defined_in":  res.DefinedIn,
		"impacted_by": res.ImpactedBy,
		"total_files": res.TotalFiles,
	})
	return string(data)
}
