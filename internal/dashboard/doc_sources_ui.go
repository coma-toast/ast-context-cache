package dashboard

import (
	"net/http"
	"strconv"

	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/docs"
)

const DefaultDocSourcesPerPage = 10

func parseDocSourcesPageQuery(r *http.Request) int {
	page := 1
	if p := r.URL.Query().Get("doc_sources_page"); p != "" {
		if n, err := strconv.Atoi(p); err == nil && n > 0 {
			page = n
		}
	}
	return page
}

func appendMemoryDocSources(m *components.MemoryData, page int) {
	sources, total, page, err := docs.ListSourcesPaged(page, DefaultDocSourcesPerPage)
	if err != nil {
		return
	}
	m.DocSourcesTotal = total
	m.DocSourcesPage = page
	m.DocSourcesPerPage = DefaultDocSourcesPerPage
	for _, s := range sources {
		age, stale := components.FormatDocSourceAge(s.LastUpdated, docs.DocSourceMaxAge)
		m.DocSources = append(m.DocSources, components.IndexDocSource{
			ID:         s.ID,
			Name:       s.Name,
			Type:       s.Type,
			URL:        s.URL,
			Age:        age,
			Stale:      stale,
			Refreshing: docs.IsRefreshing(s.ID),
		})
	}
}
