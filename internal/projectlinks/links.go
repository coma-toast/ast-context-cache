package projectlinks

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/repokey"
)

// ProjectLink records a parent container project referencing an indexed child project.
type ProjectLink struct {
	Parent    string
	Child     string
	Auto      bool
	CreatedAt string
}

// NormalizePath returns a canonical absolute project path.
func NormalizePath(p string) string {
	p = strings.TrimSpace(p)
	if p == "" {
		return ""
	}
	if abs, err := filepath.Abs(p); err == nil {
		p = abs
	}
	return filepath.Clean(p)
}

// IsStrictSubpath reports whether child is a strict subdirectory of parent.
func IsStrictSubpath(child, parent string) bool {
	child = NormalizePath(child)
	parent = NormalizePath(parent)
	if child == "" || parent == "" || child == parent {
		return false
	}
	sep := string(os.PathSeparator)
	prefix := parent
	if !strings.HasSuffix(prefix, sep) {
		prefix += sep
	}
	return strings.HasPrefix(child, prefix)
}

// IsUnderPath reports whether absPath is under root (inclusive of root itself).
func IsUnderPath(absPath, root string) bool {
	absPath = NormalizePath(absPath)
	root = NormalizePath(root)
	if absPath == "" || root == "" {
		return false
	}
	if absPath == root {
		return true
	}
	sep := string(os.PathSeparator)
	prefix := root
	if !strings.HasSuffix(prefix, sep) {
		prefix += sep
	}
	return strings.HasPrefix(absPath, prefix)
}

// IndexedProjectPaths returns all distinct indexed project paths.
func IndexedProjectPaths() []string {
	conn, err := db.IndexReader()
	if err != nil {
		return nil
	}
	rows, err := conn.Query(`SELECT DISTINCT project_path FROM symbols WHERE project_path IS NOT NULL AND project_path != '' AND project_path != '.'`)
	if err != nil {
		return nil
	}
	defer rows.Close()
	var out []string
	seen := map[string]bool{}
	for rows.Next() {
		var pp string
		if rows.Scan(&pp) != nil {
			continue
		}
		pp = NormalizePath(pp)
		if pp == "" || seen[pp] {
			continue
		}
		seen[pp] = true
		out = append(out, pp)
	}
	return out
}

// RepoSiblings returns already-indexed project paths that are other checkouts of
// the same git repository as projectPath (WTG worktrees on different branches).
func RepoSiblings(projectPath string) []string {
	projectPath = NormalizePath(projectPath)
	if projectPath == "" {
		return nil
	}
	key := repokey.Key(projectPath)
	if key == "" {
		return nil
	}
	var out []string
	for _, p := range IndexedProjectPaths() {
		if p == projectPath {
			continue
		}
		if repokey.Key(p) == key {
			out = append(out, p)
		}
	}
	return out
}

// DetectAutoLinks returns indexed project paths that are strict subdirs of parent.
func DetectAutoLinks(parent string) []string {
	parent = NormalizePath(parent)
	if parent == "" {
		return nil
	}
	var out []string
	for _, p := range IndexedProjectPaths() {
		if IsStrictSubpath(p, parent) {
			out = append(out, p)
		}
	}
	return out
}

// Links returns child paths linked under parent.
func Links(parent string) ([]string, error) {
	parent = NormalizePath(parent)
	if parent == "" || db.DB == nil {
		return nil, nil
	}
	rows, err := db.DB.Query(`SELECT child_path FROM project_links WHERE parent_path = ? ORDER BY child_path`, parent)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var c string
		if rows.Scan(&c) == nil {
			c = NormalizePath(c)
			if c != "" {
				out = append(out, c)
			}
		}
	}
	return out, nil
}

// Parents returns parent paths that link child.
func Parents(child string) ([]string, error) {
	child = NormalizePath(child)
	if child == "" || db.DB == nil {
		return nil, nil
	}
	rows, err := db.DB.Query(`SELECT parent_path FROM project_links WHERE child_path = ? ORDER BY parent_path`, child)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var p string
		if rows.Scan(&p) == nil {
			p = NormalizePath(p)
			if p != "" {
				out = append(out, p)
			}
		}
	}
	return out, nil
}

// ResolveScope returns project_path values to include when querying parent (parent + linked children).
func ResolveScope(projectPath string) []string {
	projectPath = NormalizePath(projectPath)
	if projectPath == "" {
		return nil
	}
	linked, err := Links(projectPath)
	if err != nil || len(linked) == 0 {
		return []string{projectPath}
	}
	scope := make([]string, 0, 1+len(linked))
	scope = append(scope, projectPath)
	scope = append(scope, linked...)
	return scope
}

// ResolveScopeWithRepoSiblings returns the normal resolved scope and, when
// includeSiblings is set, every already-indexed checkout of the same repo (and
// their linked children). Sibling inclusion is opt-in: tools that predate it keep
// querying a single worktree, while cross-branch tools see the whole repo.
func ResolveScopeWithRepoSiblings(projectPath string, includeSiblings bool) []string {
	scope := ResolveScope(projectPath)
	if !includeSiblings || len(scope) == 0 {
		return scope
	}
	seen := make(map[string]bool, len(scope))
	for _, p := range scope {
		seen[p] = true
	}
	for _, sib := range RepoSiblings(projectPath) {
		for _, p := range ResolveScope(sib) {
			if p == "" || seen[p] {
				continue
			}
			seen[p] = true
			scope = append(scope, p)
		}
	}
	return scope
}

// ScopeContains reports whether projectPath is in the resolved scope of root.
func ScopeContains(root, projectPath string) bool {
	root = NormalizePath(root)
	projectPath = NormalizePath(projectPath)
	for _, p := range ResolveScope(root) {
		if p == projectPath {
			return true
		}
	}
	return false
}

// IsUnderLinkedChild reports whether absPath falls under a linked child of parent.
func IsUnderLinkedChild(absPath, parent string) bool {
	parent = NormalizePath(parent)
	absPath = NormalizePath(absPath)
	if parent == "" || absPath == "" {
		return false
	}
	linked, err := Links(parent)
	if err != nil {
		return false
	}
	for _, child := range linked {
		if IsUnderPath(absPath, child) {
			return true
		}
	}
	return false
}

// LinkedChildRoot returns the linked child root if absPath is under one, else "".
func LinkedChildRoot(absPath, parent string) string {
	parent = NormalizePath(parent)
	absPath = NormalizePath(absPath)
	linked, err := Links(parent)
	if err != nil {
		return ""
	}
	var best string
	bestLen := 0
	for _, child := range linked {
		if !IsUnderPath(absPath, child) {
			continue
		}
		if l := len(child); l > bestLen {
			best = child
			bestLen = l
		}
	}
	return best
}

// OwningProject returns the project_path that owns symbols for filePath when queried from parent scope.
func OwningProject(filePath, parent string) string {
	parent = NormalizePath(parent)
	filePath = NormalizePath(filePath)
	if child := LinkedChildRoot(filePath, parent); child != "" {
		return child
	}
	return parent
}

// ScopeSQL returns "alias.project_path IN (?,?,?)" and args for the resolved scope.
func ScopeSQL(alias, projectPath string) (string, []interface{}) {
	return scopeSQLFor(alias, projectPath, ResolveScope(projectPath))
}

// ScopeSQLWithRepoSiblings is ScopeSQL over ResolveScopeWithRepoSiblings; it also
// returns the scope it built so tools can report which project paths they searched.
func ScopeSQLWithRepoSiblings(alias, projectPath string, includeSiblings bool) (string, []interface{}, []string) {
	scope := ResolveScopeWithRepoSiblings(projectPath, includeSiblings)
	frag, args := scopeSQLFor(alias, projectPath, scope)
	if len(scope) == 0 {
		scope = []string{NormalizePath(projectPath)}
	}
	return frag, args, scope
}

func scopeSQLFor(alias, projectPath string, scope []string) (string, []interface{}) {
	col := "project_path"
	if alias != "" {
		col = alias + ".project_path"
	}
	if len(scope) == 0 {
		return col + " = ?", []interface{}{projectPath}
	}
	if len(scope) == 1 {
		return col + " = ?", []interface{}{scope[0]}
	}
	ph := strings.Repeat("?,", len(scope))
	ph = strings.TrimSuffix(ph, ",")
	args := make([]interface{}, len(scope))
	for i, p := range scope {
		args[i] = p
	}
	return col + " IN (" + ph + ")", args
}

func validateLink(parent, child string) error {
	parent = NormalizePath(parent)
	child = NormalizePath(child)
	if parent == "" || child == "" {
		return fmt.Errorf("parent_path and child_path required")
	}
	if parent == child {
		return fmt.Errorf("parent and child must differ")
	}
	if !IsStrictSubpath(child, parent) {
		return fmt.Errorf("child must be a subdirectory of parent")
	}
	if parents, _ := Parents(parent); len(parents) > 0 {
		return fmt.Errorf("parent is already linked under another container")
	}
	if st, err := os.Stat(parent); err != nil || !st.IsDir() {
		return fmt.Errorf("parent path not found")
	}
	if st, err := os.Stat(child); err != nil || !st.IsDir() {
		return fmt.Errorf("child path not found")
	}
	return nil
}

// CreateLink creates a parent→child link and cleans up parent duplicate rows.
func CreateLink(parent, child string, auto bool) error {
	if err := validateLink(parent, child); err != nil {
		return err
	}
	parent = NormalizePath(parent)
	child = NormalizePath(child)
	autoInt := 0
	if auto {
		autoInt = 1
	}
	if db.DB == nil {
		return fmt.Errorf("database not initialized")
	}
	if _, err := db.DB.Exec(`INSERT OR IGNORE INTO project_links (parent_path, child_path, auto_linked) VALUES (?, ?, ?)`, parent, child, autoInt); err != nil {
		return err
	}
	return CleanupParentDuplicates(parent, child)
}

// Unlink removes a parent→child link.
func Unlink(parent, child string) error {
	parent = NormalizePath(parent)
	child = NormalizePath(child)
	if parent == "" || child == "" || db.DB == nil {
		return fmt.Errorf("parent_path and child_path required")
	}
	_, err := db.DB.Exec(`DELETE FROM project_links WHERE parent_path = ? AND child_path = ?`, parent, child)
	return err
}

// RemoveLinksForPath deletes all links where path is parent or child.
func RemoveLinksForPath(path string) error {
	path = NormalizePath(path)
	if path == "" || db.DB == nil {
		return nil
	}
	_, err := db.DB.Exec(`DELETE FROM project_links WHERE parent_path = ? OR child_path = ?`, path, path)
	return err
}

// SyncAutoLinks detects and links indexed child projects under parent; returns newly linked paths.
func SyncAutoLinks(parent string) ([]string, error) {
	parent = NormalizePath(parent)
	if parent == "" {
		return nil, nil
	}
	candidates := DetectAutoLinks(parent)
	var linked []string
	for _, child := range candidates {
		existing, _ := Links(parent)
		already := false
		for _, e := range existing {
			if e == child {
				already = true
				break
			}
		}
		if already {
			continue
		}
		if err := CreateLink(parent, child, true); err != nil {
			continue
		}
		linked = append(linked, child)
	}
	return linked, nil
}

// ShouldSkipDirDuringWalk reports whether dirPath should be skipped during parent indexing/watching.
func ShouldSkipDirDuringWalk(dirPath, parent string) bool {
	dirPath = NormalizePath(dirPath)
	parent = NormalizePath(parent)
	if dirPath == parent {
		return false
	}
	linked, err := Links(parent)
	if err != nil {
		return false
	}
	for _, child := range linked {
		if dirPath == child {
			return true
		}
	}
	return false
}
