package mcp

import (
	"github.com/coma-toast/ast-context-cache/internal/docs"
)

const docNoMatchHint = "no cached documentation matched this query above the relevance floor; fetch the page with fetch_doc (it is then cached for future search_docs calls)"

// handleSearchDocs runs the floored doc search. Every response carries no_match (true
// when nothing passed the floor) and below_floor (candidate sections discarded).
func handleSearchDocs(query string, limit int) map[string]interface{} {
	if emb == nil {
		r, err := docs.SearchDocsLexical(query, limit)
		if err != nil {
			return map[string]interface{}{"error": err.Error()}
		}
		entries := make([]docs.DocEntry, len(r.Docs))
		for i, s := range r.Docs {
			entries[i] = s.Entry
		}
		return withDocNoMatch(map[string]interface{}{
			"query":   query,
			"results": entries,
			"total":   len(entries),
		}, r.BelowFloor)
	}
	r, err := docs.SearchDocsHybridResult(query, limit, emb)
	if err != nil {
		return map[string]interface{}{"error": err.Error()}
	}
	results := make([]map[string]interface{}, 0, len(r.Docs))
	for _, s := range r.Docs {
		row := map[string]interface{}{
			"id":            s.Entry.ID,
			"source_id":     s.Entry.SourceID,
			"title":         s.Entry.Title,
			"content":       s.Entry.Content,
			"path":          s.Entry.Path,
			"content_hash":  s.Entry.ContentHash,
			"updated_at":    s.Entry.UpdatedAt,
			"score":         s.Score,
			"term_coverage": s.TermCoverage,
		}
		if s.Similarity > 0 {
			row["vector_similarity"] = s.Similarity
		}
		results = append(results, row)
	}
	return withDocNoMatch(map[string]interface{}{
		"query":   query,
		"results": results,
		"total":   len(results),
		"hybrid":  true,
	}, r.BelowFloor)
}

func withDocNoMatch(out map[string]interface{}, belowFloor int) map[string]interface{} {
	total, _ := out["total"].(int)
	out["no_match"] = total == 0
	out["below_floor"] = belowFloor
	if total == 0 {
		out["hint"] = docNoMatchHint
	}
	return out
}
