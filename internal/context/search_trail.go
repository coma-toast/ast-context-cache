package context

import (
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/search"
	"github.com/coma-toast/ast-context-cache/internal/trail"
)

// SearchTrailEntry describes one ranked search for the search trail. HitCount is every ranked
// candidate, before session dedup; the hit refs cover the first window candidates, which are
// the ones the call considers delivering. scored must not have been packed yet, since packing
// rewrites each hit's file to its relative form.
func SearchTrailEntry(tool string, q CandidateQuery, mode string, scored []search.ScoredResult, window int) trail.Entry {
	hits := trailHits(scored, window, q.ProjectPath)
	return trail.Entry{
		Tool: tool, Query: q.Query, FiltersKey: q.Filters.NormalizedKey(q.ProjectPath), Mode: mode, DocType: q.DocType,
		ProjectPath: q.ProjectPath, HitCount: len(scored), ZeroHit: len(scored) == 0,
		TopHits: hits[:min(len(hits), trail.MaxTopHits)], CandidateHits: hits,
	}
}

// trailHits formats the first window candidates as trail hit refs, resolving any missing start
// lines the way packing will.
func trailHits(scored []search.ScoredResult, window int, projectPath string) []string {
	window = min(window, len(scored))
	hits := make([]string, 0, window)
	for _, r := range scored[:window] {
		h := hitFromScored(r, projectPath)
		file, _ := h.Data["file"].(string)
		name, _ := h.Data["name"].(string)
		hits = append(hits, trail.HitRef(db.RelPath(file, projectPath), name, h.StartLine))
	}
	return hits
}
