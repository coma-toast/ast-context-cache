package docs

import (
	"sort"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

const docRRFK = 60

// ScoredDoc pairs a cached section with a relevance score.
type ScoredDoc struct {
	Entry DocEntry
	Score float64
}

// SearchDocs runs FTS over cached doc sections with LIKE fallback.
func SearchDocs(query string, limit int) ([]DocEntry, error) {
	scored, err := searchDocsFTS(query, limit)
	if err != nil {
		return nil, err
	}
	if len(scored) == 0 {
		scored, err = searchDocsLike(query, limit)
		if err != nil {
			return nil, err
		}
	}
	out := make([]DocEntry, len(scored))
	for i, s := range scored {
		out[i] = s.Entry
	}
	return out, nil
}

// SearchDocsHybrid merges FTS and vector recall for doc sections.
func SearchDocsHybrid(query string, limit int, emb embedder.Interface) ([]ScoredDoc, error) {
	if limit <= 0 {
		limit = 10
	}
	fts, err := searchDocsFTS(query, limit*2)
	if err != nil {
		return nil, err
	}
	if len(fts) == 0 {
		fts, err = searchDocsLike(query, limit*2)
		if err != nil {
			return nil, err
		}
	}
	var vector []ScoredDoc
	if emb != nil {
		vector, _ = searchDocsVector(query, limit*2, emb)
	}
	if len(vector) == 0 {
		if len(fts) > limit {
			fts = fts[:limit]
		}
		return fts, nil
	}
	return fuseDocResults(fts, vector, limit), nil
}

func searchDocsFTS(query string, limit int) ([]ScoredDoc, error) {
	if limit <= 0 {
		limit = 10
	}
	ftsQuery := search.BuildFTSQuery(splitTerms(query))
	if ftsQuery == "" {
		return nil, nil
	}
	rows, err := db.ContextDB.Query(`
		SELECT dc.id, dc.source_id, dc.title, dc.content, COALESCE(dc.path,''), COALESCE(dc.content_hash,''), dc.updated_at, f.rank
		FROM docs_fts f
		JOIN doc_content dc ON f.rowid = dc.id
		WHERE docs_fts MATCH ?
		ORDER BY f.rank
		LIMIT ?`, ftsQuery, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return scanScoredDocs(rows)
}

func searchDocsLike(query string, limit int) ([]ScoredDoc, error) {
	rows, err := db.ContextDB.Query(`
		SELECT dc.id, dc.source_id, dc.title, dc.content, COALESCE(dc.path,''), COALESCE(dc.content_hash,''), dc.updated_at
		FROM doc_content dc
		WHERE dc.title LIKE ? OR dc.content LIKE ?
		ORDER BY dc.updated_at DESC
		LIMIT ?`, "%"+query+"%", "%"+query+"%", limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []ScoredDoc
	for rows.Next() {
		var e DocEntry
		err := rows.Scan(&e.ID, &e.SourceID, &e.Title, &e.Content, &e.Path, &e.ContentHash, &e.UpdatedAt)
		if err != nil {
			continue
		}
		out = append(out, ScoredDoc{Entry: e, Score: 0.5})
	}
	return out, nil
}

func searchDocsVector(query string, limit int, emb embedder.Interface) ([]ScoredDoc, error) {
	queryVec, err := emb.EmbedSingle(query)
	if err != nil {
		return nil, err
	}
	scored := search.Cache.SearchDoc(queryVec, limit)
	out := make([]ScoredDoc, 0, len(scored))
	for _, s := range scored {
		entry, ok := entryFromVectorHit(s)
		if !ok {
			continue
		}
		out = append(out, ScoredDoc{Entry: entry, Score: s.Score})
	}
	return out, nil
}

func entryFromVectorHit(s search.ScoredResult) (DocEntry, bool) {
	id, _ := s.Data["doc_id"].(int)
	if id == 0 {
		return DocEntry{}, false
	}
	var e DocEntry
	err := db.ContextDB.QueryRow(`
		SELECT id, source_id, title, content, COALESCE(path,''), COALESCE(content_hash,''), updated_at
		FROM doc_content WHERE id = ?`, id).Scan(&e.ID, &e.SourceID, &e.Title, &e.Content, &e.Path, &e.ContentHash, &e.UpdatedAt)
	return e, err == nil
}

func fuseDocResults(fts, vector []ScoredDoc, limit int) []ScoredDoc {
	type fused struct {
		entry DocEntry
		score float64
	}
	seen := map[int]*fused{}
	add := func(rank int, s ScoredDoc) {
		if s.Entry.ID == 0 {
			return
		}
		bump := 1.0 / float64(docRRFK+rank+1)
		if e, ok := seen[s.Entry.ID]; ok {
			e.score += bump
		} else {
			seen[s.Entry.ID] = &fused{entry: s.Entry, score: bump}
		}
	}
	for i, s := range fts {
		add(i, s)
	}
	for i, s := range vector {
		add(i, s)
	}
	out := make([]ScoredDoc, 0, len(seen))
	for _, e := range seen {
		out = append(out, ScoredDoc{Entry: e.entry, Score: e.score})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	if len(out) > limit {
		out = out[:limit]
	}
	return out
}

func splitTerms(query string) []string {
	return search.QueryTerms(query)
}

func scanScoredDocs(rows interface {
	Next() bool
	Scan(...any) error
}) ([]ScoredDoc, error) {
	var out []ScoredDoc
	for rows.Next() {
		e, rank, err := scanDocEntryRank(rows)
		if err != nil {
			continue
		}
		out = append(out, ScoredDoc{Entry: e, Score: -rank})
	}
	return out, nil
}

func scanDocEntryRank(rows interface {
	Scan(...any) error
}) (DocEntry, float64, error) {
	var e DocEntry
	var rank float64
	err := rows.Scan(&e.ID, &e.SourceID, &e.Title, &e.Content, &e.Path, &e.ContentHash, &e.UpdatedAt, &rank)
	return e, rank, err
}
