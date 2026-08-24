package impact

import (
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
)

// HandleDiffImpact reports the blast radius of a whole branch: every file the
// diff touches, the indexed symbols those files define, and which files still
// depend on each of them. Sibling checkouts of the same repo are included, so a
// change made on one branch's worktree surfaces callers living in another.
//
// With a "pr" number instead of refs it answers the same question about a GitHub
// pull request that is not checked out anywhere: the file list comes from gh over
// the network and is matched against whichever local checkout is already indexed.
func HandleDiffImpact(args map[string]interface{}, projectPath string) string {
	if projectPath == "" {
		return `{"error": "project_path required"}`
	}
	if pr := intArg(args, "pr"); pr > 0 {
		return diffImpactForPR(args, projectPath, pr)
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
	return diffImpactResult(projectPath, changed, map[string]interface{}{
		"base_ref": baseRef,
		"head_ref": headRef,
	})
}

// diffImpactForPR resolves a pull request's changed files through gh (no local
// checkout of that branch needed) and runs the same analysis over them.
func diffImpactForPR(args map[string]interface{}, projectPath string, pr int) string {
	repo := strArg(args, "repo")
	if repo == "" {
		repo = originRepo(projectPath)
	}
	info, err := fetchPullRequest(projectPath, repo, pr)
	if err != nil {
		return errJSON(err)
	}
	prefix := repoPrefix(projectPath)
	changed := make([]string, 0, len(info.Files))
	for _, f := range info.Files {
		if rel, ok := stripPrefix(f.Path, prefix); ok {
			changed = append(changed, rel)
		}
	}
	return diffImpactResult(projectPath, changed, map[string]interface{}{
		"pr":       info.Number,
		"pr_title": info.Title,
		"repo":     repo,
	})
}

// A sweeping branch or PR can touch hundreds of symbols with thousands of
// dependents; these caps keep one call's payload usable for an agent.
const (
	maxAnalyzedSymbols     = 200
	maxDependentsPerSymbol = 50
)

// diffImpactResult runs the symbol lookup and impact graph over a set of
// project-relative changed files and renders the shared payload.
func diffImpactResult(projectPath string, changed []string, extra map[string]interface{}) string {
	scope := projectlinks.ResolveScopeWithRepoSiblings(projectPath, true)
	symbolNames := symbolsInFiles(changed, scope)

	analyzed := symbolNames
	truncated := false
	if len(analyzed) > maxAnalyzedSymbols {
		analyzed = analyzed[:maxAnalyzedSymbols]
		truncated = true
	}

	impacted := map[string][]string{}
	for _, name := range analyzed {
		res, err := Graph(name, projectPath, true)
		if err != nil {
			continue
		}
		files := dependentFiles(res, changed)
		if len(files) == 0 {
			continue
		}
		if len(files) > maxDependentsPerSymbol {
			files = files[:maxDependentsPerSymbol]
			truncated = true
		}
		impacted[name] = files
	}

	out := map[string]interface{}{
		"changed_files":    changed,
		"symbols_changed":  symbolNames,
		"symbols_analyzed": len(analyzed),
		"impacted_by":      impacted,
		"checked_scope":    scope,
		"truncated":        truncated,
	}
	for k, v := range extra {
		out[k] = v
	}
	data, _ := json.Marshal(out)
	return string(data)
}

// pullRequest is the slice of gh's PR JSON this tool needs.
type pullRequest struct {
	Number int    `json:"number"`
	Title  string `json:"title"`
	Files  []struct {
		Path string `json:"path"`
	} `json:"files"`
}

func fetchPullRequest(projectPath, repo string, pr int) (*pullRequest, error) {
	ghArgs := []string{"pr", "view", strconv.Itoa(pr), "--json", "number,title,files"}
	if repo != "" {
		ghArgs = append(ghArgs, "--repo", repo)
	}
	cmd := exec.Command("gh", ghArgs...)
	cmd.Dir = projectPath
	out, err := cmd.Output()
	if err != nil {
		detail := ""
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			detail = ": " + strings.TrimSpace(string(exitErr.Stderr))
		}
		return nil, fmt.Errorf("gh pr view %d failed: %v%s", pr, err, detail)
	}
	var info pullRequest
	if err := json.Unmarshal(out, &info); err != nil {
		return nil, fmt.Errorf("parse gh output: %v", err)
	}
	return &info, nil
}

// originRepo returns "owner/name" for the project's origin remote, or "" when it
// is not a recognisable GitHub remote (gh then falls back to its own detection).
func originRepo(projectPath string) string {
	out, err := runGit(projectPath, "remote", "get-url", "origin")
	if err != nil {
		return ""
	}
	url := strings.TrimSuffix(strings.TrimSpace(out), ".git")
	i := strings.Index(url, "github.com")
	if i < 0 {
		return ""
	}
	return strings.TrimLeft(url[i+len("github.com"):], ":/")
}

// repoPrefix is projectPath's path within its repository ("" at the repo root).
// PR file lists are repo-root relative, indexed files are project relative.
func repoPrefix(projectPath string) string {
	out, err := runGit(projectPath, "rev-parse", "--show-prefix")
	if err != nil {
		return ""
	}
	return strings.TrimSpace(out)
}

func stripPrefix(path, prefix string) (string, bool) {
	if prefix == "" {
		return path, true
	}
	if !strings.HasPrefix(path, prefix) {
		return "", false
	}
	return strings.TrimPrefix(path, prefix), true
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

// intArg reads a numeric argument, tolerating the float64 JSON decodes into and
// the string some MCP clients send.
func intArg(args map[string]interface{}, key string) int {
	switch v := args[key].(type) {
	case float64:
		return int(v)
	case int:
		return v
	case string:
		n, err := strconv.Atoi(strings.TrimSpace(v))
		if err != nil {
			return 0
		}
		return n
	}
	return 0
}

func errJSON(err error) string {
	data, _ := json.Marshal(map[string]string{"error": err.Error()})
	return string(data)
}
