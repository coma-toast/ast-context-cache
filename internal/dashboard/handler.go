package dashboard

import (
	"context"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/projectmeta"
)

var serverStartTime = time.Now()

var (
	projectsCacheMu sync.Mutex
	projectsCache   []components.Project
	projectsCacheAt time.Time
)

const projectsCacheTTL = 15 * time.Second

func loadProjects(pid string) []components.Project {
	_ = pid
	projectsCacheMu.Lock()
	if len(projectsCache) > 0 && time.Since(projectsCacheAt) < projectsCacheTTL {
		out := make([]components.Project, len(projectsCache))
		copy(out, projectsCache)
		projectsCacheMu.Unlock()
		return out
	}
	projectsCacheMu.Unlock()
	ps := loadProjectsFresh()
	projectsCacheMu.Lock()
	projectsCache = ps
	projectsCacheAt = time.Now()
	projectsCacheMu.Unlock()
	return ps
}

func loadProjectsForPage() ([]components.Project, bool) {
	projectsCacheMu.Lock()
	if len(projectsCache) > 0 {
		out := make([]components.Project, len(projectsCache))
		copy(out, projectsCache)
		stale := time.Since(projectsCacheAt) >= projectsCacheTTL
		projectsCacheMu.Unlock()
		if stale {
			go func() { loadProjects("") }()
		}
		return out, false
	}
	projectsCacheMu.Unlock()
	go func() { loadProjects("") }()
	return nil, true
}

func loadProjectsFresh() []components.Project {
	if db.IndexDB == nil || db.DB == nil {
		return nil
	}
	type symCount struct {
		symbols int
		files   int
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	symCounts := map[string]symCount{}
	symRows, err := db.IndexDB.QueryContext(ctx, "SELECT project_path, COUNT(*), COUNT(DISTINCT file) FROM symbols WHERE project_path IS NOT NULL GROUP BY project_path")
	if err == nil {
		defer symRows.Close()
		for symRows.Next() {
			var pp string
			var sc symCount
			symRows.Scan(&pp, &sc.symbols, &sc.files)
			symCounts[pp] = sc
		}
	}
	queryCounts := map[string]int{}
	rows, err := db.DB.QueryContext(ctx, "SELECT DISTINCT project_path, COUNT(*) FROM queries WHERE project_path IS NOT NULL AND project_path != '' AND project_path != '.' GROUP BY project_path")
	if err == nil {
		defer rows.Close()
		for rows.Next() {
			var p string
			var c int
			rows.Scan(&p, &c)
			queryCounts[p] = c
		}
	}
	allPaths := map[string]bool{}
	for pp := range symCounts {
		allPaths[pp] = true
	}
	for pp := range queryCounts {
		allPaths[pp] = true
	}
	for _, pp := range projectmeta.DiscoverPaths() {
		allPaths[pp] = true
	}
	pinned := map[string]bool{}
	for _, p := range db.GetPinnedProjects() {
		pinned[p] = true
	}
	var ps []components.Project
	for pp := range allPaths {
		if projectmeta.IsExcluded(pp) {
			continue
		}
		meta := projectmeta.Enrich(pp)
		sc := symCounts[pp]
		label := meta.Label
		if label == "" {
			label = filepath.Base(pp)
		}
		ps = append(ps, components.Project{
			Path:        pp,
			Name:        meta.RepoName,
			Label:       label,
			Workspace:   meta.Workspace,
			Branch:      meta.Branch,
			RepoKey:     meta.RepoKey,
			QueryCount:  queryCounts[pp],
			SymbolCount: sc.symbols,
			FileCount:   sc.files,
			Pinned:      pinned[pp],
		})
	}
	sort.Slice(ps, func(i, j int) bool {
		li, lj := strings.ToLower(ps[i].Label), strings.ToLower(ps[j].Label)
		if li != lj {
			return li < lj
		}
		return ps[i].Path < ps[j].Path
	})
	return ps
}
