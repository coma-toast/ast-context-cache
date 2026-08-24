package impact

import (
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
)

// Entry is one file that depends on the analyzed symbol.
type Entry struct {
	File   string `json:"file"`
	Target string `json:"target"`
	Kind   string `json:"kind"`
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
func Graph(symbol, projectPath string, includeSiblings bool) (*Result, error) {
	if projectPath == "" {
		return nil, errors.New("project_path required")
	}
	if symbol == "" {
		return nil, errors.New("symbol required")
	}

	symbolLower := strings.ToLower(symbol)
	scopeFrag, scopeArgs, scope := projectlinks.ScopeSQLWithRepoSiblings("", projectPath, includeSiblings)

	conn, err := db.IndexReader()
	if err != nil {
		return nil, err
	}

	symbolRows, err := conn.Query(
		"SELECT DISTINCT file FROM symbols WHERE "+scopeFrag+" AND LOWER(name) = ?",
		append(scopeArgs, symbolLower)...)
	if err != nil {
		return nil, err
	}
	defer symbolRows.Close()

	symbolFiles := map[string]bool{}
	for symbolRows.Next() {
		var f string
		symbolRows.Scan(&f)
		symbolFiles[f] = true
	}

	var impacts []Entry

	edgeRows, err := conn.Query(
		"SELECT source_file, target, kind FROM edges WHERE "+scopeFrag+" AND (LOWER(target) LIKE ? OR LOWER(target) LIKE ?)",
		append(scopeArgs, "%"+symbolLower+"%", "%/"+symbolLower)...)
	if err != nil {
		return nil, err
	}
	defer edgeRows.Close()

	seen := map[string]bool{}
	for edgeRows.Next() {
		var srcFile, target, kind string
		edgeRows.Scan(&srcFile, &target, &kind)
		if !seen[srcFile] {
			seen[srcFile] = true
			impacts = append(impacts, Entry{File: srcFile, Target: target, Kind: kind})
		}
	}

	for f := range symbolFiles {
		base := strings.ToLower(filepath.Base(f))
		// Import specifiers usually drop the extension ("./page" for page.ts), so
		// match the bare stem as a path suffix as well as the full basename.
		stem := strings.TrimSuffix(base, filepath.Ext(base))
		depRows, _ := conn.Query(
			"SELECT source_file, target, kind FROM edges WHERE "+scopeFrag+
				" AND (LOWER(target) LIKE ? OR LOWER(target) LIKE ? OR LOWER(target) = ?)",
			append(scopeArgs, "%"+base+"%", "%/"+stem, stem)...)
		if depRows != nil {
			for depRows.Next() {
				var srcFile, target, kind string
				depRows.Scan(&srcFile, &target, &kind)
				if !seen[srcFile] {
					seen[srcFile] = true
					impacts = append(impacts, Entry{File: srcFile, Target: target, Kind: kind})
				}
			}
			depRows.Close()
		}
	}

	defined := make([]string, 0, len(symbolFiles))
	for k := range symbolFiles {
		defined = append(defined, RelPathInScope(k, projectPath, scope))
	}

	relImpacts := make([]Entry, len(impacts))
	for i, imp := range impacts {
		relImpacts[i] = Entry{
			File:   RelPathInScope(imp.File, projectPath, scope),
			Target: imp.Target,
			Kind:   imp.Kind,
			Abs:    imp.File,
		}
	}

	return &Result{
		Symbol:     symbol,
		DefinedIn:  defined,
		ImpactedBy: relImpacts,
		TotalFiles: len(seen),
		Scope:      scope,
	}, nil
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
