package purge

import (
	"sort"

	"github.com/coma-toast/ast-context-cache/internal/cache"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedqueue"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
	"github.com/coma-toast/ast-context-cache/internal/search"
	"github.com/coma-toast/ast-context-cache/internal/watcher"
)

const (
	selectIndexedFileProjectsQuery = `SELECT DISTINCT project_path FROM indexed_files WHERE project_path IS NOT NULL AND project_path != ''`
	countProjectIndexedFilesQuery  = `SELECT COUNT(*) FROM indexed_files WHERE project_path = ?`
)

// SpaceRoot is an indexed project whose path is a wtg space root: a directory
// of checkouts that used to be indexed whole, duplicating each checkout's index.
type SpaceRoot struct {
	Path    string `json:"path"`
	Files   int    `json:"files"`
	Symbols int    `json:"symbols"`
	// Checkouts are the git checkouts directly inside the space.
	Checkouts []string `json:"checkouts"`
	// Unindexed are checkouts with no index of their own yet. A root is covered,
	// and safe to prune, only when this is empty.
	Unindexed []string `json:"unindexed,omitempty"`
	Covered   bool     `json:"covered"`
	Pruned    bool     `json:"pruned,omitempty"`
	Error     string   `json:"error,omitempty"`
}

// SpaceRoots lists every indexed project that is a wtg space root. With apply,
// each covered root (every checkout indexed as its own project) is pruned: its
// checkouts are linked under it so searches with the root as project_path still
// reach them, and the root's own index rows (symbols, edges, vectors, indexed
// files, summaries, pending embeds) are deleted. Notes, memory, handoffs and query
// history recorded under the root are kept. Without apply nothing is changed.
func SpaceRoots(apply bool) []SpaceRoot {
	var out []SpaceRoot
	for _, p := range indexedProjects() {
		kids := projectlinks.SpaceCheckouts(p)
		if len(kids) == 0 {
			continue
		}
		r := SpaceRoot{Path: p, Checkouts: kids}
		r.Symbols, r.Files = projectlinks.LinkStats(p)
		for _, k := range kids {
			if indexedFileCount(k) == 0 {
				r.Unindexed = append(r.Unindexed, k)
			}
		}
		r.Covered = len(r.Unindexed) == 0
		if apply && r.Covered {
			if err := pruneSpaceRoot(p, kids); err != nil {
				r.Error = err.Error()
			} else {
				r.Pruned = true
			}
		}
		out = append(out, r)
	}
	return out
}

func pruneSpaceRoot(root string, kids []string) error {
	watcher.DeleteWatcher(root)
	embedqueue.RemoveProject(root)
	projectlinks.LinkSpaceCheckouts(root, kids)
	if err := deleteIndexData(root, contextRefs{}); err != nil {
		return err
	}
	// RemoveProject marks the root cancelled for embeds; nothing embeds under it
	// any more, but clear the mark so a later explicit index_files is not ignored.
	embedqueue.UnmarkProjectCancelled(root)
	cache.Candidates.ClearProject(root)
	search.Cache.DeleteByProject(root)
	logger.Info("Pruned space root index", "space", root, "checkouts", len(kids))
	return nil
}

// indexedProjects is every project_path with symbols or indexed files.
func indexedProjects() []string {
	seen := map[string]bool{}
	for _, p := range projectlinks.IndexedProjectPaths() {
		seen[p] = true
	}
	if conn, err := db.IndexReader(); err == nil {
		if rows, err := conn.Query(selectIndexedFileProjectsQuery); err == nil {
			for rows.Next() {
				var p string
				if rows.Scan(&p) == nil {
					if p = projectlinks.NormalizePath(p); p != "" {
						seen[p] = true
					}
				}
			}
			rows.Close()
		}
	}
	out := make([]string, 0, len(seen))
	for p := range seen {
		out = append(out, p)
	}
	sort.Strings(out)
	return out
}

func indexedFileCount(projectPath string) int {
	conn, err := db.IndexReader()
	if err != nil {
		return 0
	}
	var n int
	conn.QueryRow(countProjectIndexedFilesQuery, projectPath).Scan(&n)
	return n
}
