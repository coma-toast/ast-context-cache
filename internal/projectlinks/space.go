package projectlinks

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// spaceMarkers are files wtg writes at a space root. Any one of them, or a child
// checkout that is a git worktree (.git is a file), marks a directory of
// checkouts as a space rather than a plain folder that happens to hold clones.
var spaceMarkers = []string{"go.work", "SPACE.md"}

// SpaceCheckouts returns the git checkouts directly inside dir when dir is a wtg
// space root: dir is not a checkout itself, holds at least one child directory
// with a .git entry, and carries a space marker (go.work, SPACE.md) or a child
// worktree. It returns nil for anything else, so plain folders (a notes vault, a
// single repo) keep being indexed as one project.
func SpaceCheckouts(dir string) []string {
	dir = NormalizePath(dir)
	if dir == "" || hasGitEntry(dir) {
		return nil
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	marked := false
	for _, m := range spaceMarkers {
		if st, err := os.Stat(filepath.Join(dir, m)); err == nil && !st.IsDir() {
			marked = true
			break
		}
	}
	var out []string
	for _, e := range entries {
		if !e.IsDir() || strings.HasPrefix(e.Name(), ".") {
			continue
		}
		child := filepath.Join(dir, e.Name())
		st, err := os.Stat(filepath.Join(child, ".git"))
		if err != nil {
			continue
		}
		if !st.IsDir() {
			marked = true // a worktree's .git is a file pointing at the main clone
		}
		out = append(out, child)
	}
	if !marked || len(out) == 0 {
		return nil
	}
	sort.Strings(out)
	return out
}

// IsSpaceRoot reports whether dir is a wtg space root (see SpaceCheckouts).
func IsSpaceRoot(dir string) bool {
	return len(SpaceCheckouts(dir)) > 0
}

// IndexTargets returns the projects that indexing or watching projectPath really
// means: each child checkout of a space root, or projectPath itself.
func IndexTargets(projectPath string) []string {
	if kids := SpaceCheckouts(projectPath); len(kids) > 0 {
		return kids
	}
	if p := NormalizePath(projectPath); p != "" {
		return []string{p}
	}
	return nil
}

// LinkSpaceCheckouts links each of kids under the space root so that tools called
// with the root as project_path search every checkout (ResolveScope), and returns
// the children it newly linked. Already-linked children cost nothing beyond one
// Links query, so callers may run it on every tool call.
func LinkSpaceCheckouts(root string, kids []string) []string {
	root = NormalizePath(root)
	existing, _ := Links(root)
	have := make(map[string]bool, len(existing))
	for _, e := range existing {
		have[e] = true
	}
	var linked []string
	for _, k := range kids {
		k = NormalizePath(k)
		if k == "" || have[k] {
			continue
		}
		if err := CreateLink(root, k, true); err != nil {
			logger.Warn("Failed to link space checkout", "space", root, "checkout", k, "error", err)
			continue
		}
		linked = append(linked, k)
	}
	if len(linked) > 0 {
		logger.Info("Linked space checkouts", "space", root, "checkouts", linked)
	}
	return linked
}

// OwningCheckout returns the child checkout of space root that contains path, or
// "" when path is outside every checkout (e.g. SPACE.md at the root).
func OwningCheckout(path, root string) string {
	for _, k := range SpaceCheckouts(root) {
		if IsUnderPath(path, k) {
			return k
		}
	}
	return ""
}

func hasGitEntry(dir string) bool {
	_, err := os.Stat(filepath.Join(dir, ".git"))
	return err == nil
}
