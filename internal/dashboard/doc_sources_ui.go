package dashboard

import (
	"net/http"
	"strconv"
	"time"

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

func appendIndexDocSources(h *components.IndexHealth, page int) {
	sources, total, page, err := docs.ListSourcesPaged(page, DefaultDocSourcesPerPage)
	if err != nil {
		return
	}
	h.DocSourcesTotal = total
	h.DocSourcesPage = page
	h.DocSourcesPerPage = DefaultDocSourcesPerPage
	for _, s := range sources {
		age, stale := components.FormatDocSourceAge(s.LastUpdated, docs.DocSourceMaxAge)
		h.DocSources = append(h.DocSources, components.IndexDocSource{
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

func loadSettingsDocSources(page int) ([]components.DocSource, int, int) {
	sources, total, page, err := docs.ListSourcesPaged(page, DefaultDocSourcesPerPage)
	if err != nil {
		return nil, 0, page
	}
	var out []components.DocSource
	for _, s := range sources {
		updated := "Never"
		if s.LastUpdated != "" {
			if t, err := time.Parse("2006-01-02T15:04:05Z07:00", s.LastUpdated); err == nil {
				updated = t.Format("Jan 2, 2006 15:04")
			} else if t, err := time.Parse("2006-01-02 15:04:05", s.LastUpdated); err == nil {
				updated = t.Format("Jan 2, 2006 15:04")
			} else {
				updated = s.LastUpdated
			}
		}
		out = append(out, components.DocSource{
			ID:          s.ID,
			Name:        s.Name,
			Type:        s.Type,
			URL:         s.URL,
			Version:     s.Version,
			LastUpdated: updated,
			Refreshing:  docs.IsRefreshing(s.ID),
		})
	}
	return out, total, page
}
