package impact

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
)

// HandleDiffImpact reports the blast radius of a whole branch: every file the
// diff touches, the indexed symbols those files define, and which files still
// depend on each of them. Sibling checkouts of the same repo are included, so a
// change made on one branch's worktree surfaces callers living in another.
func HandleDiffImpact(args map[string]interface{}, projectPath string) string {
	if projectPath == "" {
		return `{"error": "project_path required"}`
	}
	baseRef := strArg(args, "base_ref")
	if baseRef == "" {
		baseRef = "origin/main"
	}
	headRef := strArg(args, "head_ref")
	if headRef == "" {
		headRef = "HEAD"
	}

	changed, err := changedFiles(projectPath, baseRef, headRef)
	if err != nil {
		return errJSON(err)
	}

	scope := projectlinks.ResolveScopeWithRepoSiblings(projectPath, true)
	symbolNames := symbolsInFiles(changed, scope)

	impacted := map[string][]string{}
	for _, name := range symbolNames {
		res, err := Graph(name, projectPath, true)
		if err != nil {
			continue
		}
		files := dependentFiles(res, changed)
		if len(files) > 0 {
			impacted[name] = files
		}
	}

	data, _ := json.Marshal(map[string]interface{}{
		"base_ref":        baseRef,
		"head_ref":        headRef,
		"changed_files":   changed,
		"symbols_changed": symbolNames,
		"impacted_by":     impacted,
		"checked_scope":   scope,
	})
	return string(data)
}

// dependents returns the impacted entries that are not themselves part of the
// diff — those are the places a change can break without being reviewed.
func dependents(res *Result, changed []string) []Entry {
	inDiff := map[string]bool{}
	for _, f := range changed {
		inDiff[filepath.Clean(f)] = true
	}
	seen := map[string]bool{}
	var out []Entry
	for _, e := range res.ImpactedBy {
		f := filepath.Clean(e.File)
		if inDiff[f] || seen[f] {
			continue
		}
		seen[f] = true
		out = append(out, e)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].File < out[j].File })
	return out
}

// dependentFiles is dependents reduced to display paths.
func dependentFiles(res *Result, changed []string) []string {
	entries := dependents(res, changed)
	out := make([]string, 0, len(entries))
	for _, e := range entries {
		out = append(out, e.File)
	}
	return out
}

// changedFiles lists repo-relative paths changed between two refs.
func changedFiles(projectPath, baseRef, headRef string) ([]string, error) {
	out, err := runGit(projectPath, "diff", "--name-only", "--relative", baseRef+"..."+headRef)
	if err != nil {
		return nil, err
	}
	var files []string
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			files = append(files, line)
		}
	}
	return files, nil
}

// symbolsInFiles returns the distinct indexed symbol names defined in any of the
// given repo-relative files, across every project path in scope. Everything the
// indexer stores is already top-level, so no extra filtering is needed.
func symbolsInFiles(relFiles, scope []string) []string {
	if len(relFiles) == 0 || len(scope) == 0 {
		return nil
	}
	conn, err := db.IndexReader()
	if err != nil {
		return nil
	}
	seen := map[string]bool{}
	var names []string
	for _, rel := range relFiles {
		var abs []interface{}
		for _, root := range scope {
			abs = append(abs, filepath.Join(root, rel))
		}
		ph := strings.TrimSuffix(strings.Repeat("?,", len(abs)), ",")
		rows, err := conn.Query("SELECT DISTINCT name FROM symbols WHERE file IN ("+ph+")", abs...)
		if err != nil {
			continue
		}
		for rows.Next() {
			var n string
			if rows.Scan(&n) != nil || n == "" || seen[n] {
				continue
			}
			seen[n] = true
			names = append(names, n)
		}
		rows.Close()
	}
	sort.Strings(names)
	return names
}

func runGit(projectPath string, args ...string) (string, error) {
	cmd := exec.Command("git", append([]string{"-C", projectPath}, args...)...)
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("git %s failed: %v", strings.Join(args, " "), err)
	}
	return string(out), nil
}

func strArg(args map[string]interface{}, key string) string {
	v, _ := args[key].(string)
	return strings.TrimSpace(v)
}

func errJSON(err error) string {
	data, _ := json.Marshal(map[string]string{"error": err.Error()})
	return string(data)
}
