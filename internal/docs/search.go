package docs

import (
	"sort"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

const docRRFK = 60

const (
	searchDocsFTSQuery = `
		SELECT dc.id, dc.source_id, dc.title, dc.content, COALESCE(dc.path,''), COALESCE(dc.content_hash,''), dc.updated_at, f.rank
		FROM docs_fts f
		JOIN doc_content dc ON f.rowid = dc.id
		WHERE docs_fts MATCH ?
		ORDER BY f.rank
		LIMIT ?`
	searchDocsLikeQuery = `
		SELECT dc.id, dc.source_id, dc.title, dc.content, COALESCE(dc.path,''), COALESCE(dc.content_hash,''), dc.updated_at
		FROM doc_content dc
		WHERE dc.title LIKE ? OR dc.content LIKE ?
		ORDER BY dc.updated_at DESC
		LIMIT ?`
	selectDocEntryQuery = `
		SELECT id, source_id, title, content, COALESCE(path,''), COALESCE(content_hash,''), updated_at
		FROM doc_content WHERE id = ?`
)

// ScoredDoc pairs a cached section with a relevance score.
type ScoredDoc struct {
	Entry DocEntry
	Score float64
	// TermCoverage is the fraction of the query's terms found in the section (0..1).
	TermCoverage float64
	// Similarity is cosine similarity to the query when vector recall found the section.
	Similarity float64
}

// DocSearchResult is a doc search after the relevance floor (relevance.go): the
// sections that passed, and how many distinct candidate sections fell below it.
type DocSearchResult struct {
	Docs       []ScoredDoc
	BelowFloor int
}

// SearchDocs runs FTS over cached doc sections with LIKE fallback.
func SearchDocs(query string, limit int) ([]DocEntry, error) {
	r, err := SearchDocsLexical(query, limit)
	if err != nil {
		return nil, err
	}
	out := make([]DocEntry, len(r.Docs))
	for i, s := range r.Docs {
		out[i] = s.Entry
	}
	return out, nil
}

// SearchDocsLexical is SearchDocs with scores and the relevance-floor count.
func SearchDocsLexical(query string, limit int) (DocSearchResult, error) {
	if limit <= 0 {
		limit = 10
	}
	terms := coverageTerms(query)
	raw, kept, err := lexicalDocs(query, terms, limit)
	if err != nil {
		return DocSearchResult{}, err
	}
	if len(kept) > limit {
		kept = kept[:limit]
	}
	return DocSearchResult{Docs: kept, BelowFloor: countBelowFloor([][]ScoredDoc{raw}, [][]ScoredDoc{kept})}, nil
}

// lexicalDocs returns FTS candidates and those passing the lexical floor, falling back to
// a whole-query LIKE match when no FTS hit survives the floor.
func lexicalDocs(query string, terms [][]string, limit int) (raw, kept []ScoredDoc, err error) {
	raw, err = searchDocsFTS(query, limit)
	if err != nil {
		return nil, nil, err
	}
	kept = applyLexicalFloor(terms, raw)
	if len(kept) > 0 {
		return raw, kept, nil
	}
	like, err := searchDocsLike(query, limit)
	if err != nil {
		return nil, nil, err
	}
	return append(raw, like...), applyLexicalFloor(terms, like), nil
}

// SearchDocsHybrid merges FTS and vector recall for doc sections.
func SearchDocsHybrid(query string, limit int, emb embedder.Interface) ([]ScoredDoc, error) {
	r, err := SearchDocsHybridResult(query, limit, emb)
	return r.Docs, err
}

// SearchDocsHybridResult is SearchDocsHybrid plus the relevance-floor count. Floors
// apply per signal before fusion because the fused RRF score is rank-only.
func SearchDocsHybridResult(query string, limit int, emb embedder.Interface) (DocSearchResult, error) {
	if limit <= 0 {
		limit = 10
	}
	terms := coverageTerms(query)
	lexRaw, lex, err := lexicalDocs(query, terms, limit*2)
	if err != nil {
		return DocSearchResult{}, err
	}
	var vecRaw, vector []ScoredDoc
	if emb != nil {
		vecRaw, _ = searchDocsVector(query, limit*2, emb)
		vector = applyVectorFloor(terms, vecRaw)
	}
	below := countBelowFloor([][]ScoredDoc{lexRaw, vecRaw}, [][]ScoredDoc{lex, vector})
	if len(vector) == 0 {
		if len(lex) > limit {
			lex = lex[:limit]
		}
		return DocSearchResult{Docs: lex, BelowFloor: below}, nil
	}
	return DocSearchResult{Docs: fuseDocResults(lex, vector, limit), BelowFloor: below}, nil
}

func searchDocsFTS(query string, limit int) ([]ScoredDoc, error) {
	if limit <= 0 {
		limit = 10
	}
	ftsQuery := search.BuildFTSQuery(splitTerms(query))
	if ftsQuery == "" {
		return nil, nil
	}
	rows, err := db.ContextDB.Query(searchDocsFTSQuery, ftsQuery, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return scanScoredDocs(rows)
}

func searchDocsLike(query string, limit int) ([]ScoredDoc, error) {
	rows, err := db.ContextDB.Query(searchDocsLikeQuery, "%"+query+"%", "%"+query+"%", limit)
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
	// VectorCache.SearchDoc only orders results when it has more than limit candidates;
	// RRF needs true rank order.
	sort.SliceStable(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	return out, nil
}

func entryFromVectorHit(s search.ScoredResult) (DocEntry, bool) {
	id, _ := s.Data["doc_id"].(int)
	if id == 0 {
		return DocEntry{}, false
	}
	var e DocEntry
	err := db.ContextDB.QueryRow(selectDocEntryQuery, id).Scan(&e.ID, &e.SourceID, &e.Title, &e.Content, &e.Path, &e.ContentHash, &e.UpdatedAt)
	return e, err == nil
}

func fuseDocResults(fts, vector []ScoredDoc, limit int) []ScoredDoc {
	seen := map[int]*ScoredDoc{}
	add := func(rank int, s ScoredDoc) {
		if s.Entry.ID == 0 {
			return
		}
		bump := 1.0 / float64(docRRFK+rank+1)
		e, ok := seen[s.Entry.ID]
		if !ok {
			e = &ScoredDoc{Entry: s.Entry}
			seen[s.Entry.ID] = e
		}
		e.Score += bump
		e.TermCoverage = max(e.TermCoverage, s.TermCoverage)
		e.Similarity = max(e.Similarity, s.Similarity)
	}
	for i, s := range fts {
		add(i, s)
	}
	for i, s := range vector {
		add(i, s)
	}
	ids := make([]int, 0, len(seen))
	for id := range seen {
		ids = append(ids, id)
	}
	sort.Ints(ids)
	out := make([]ScoredDoc, len(ids))
	for i, id := range ids {
		out[i] = *seen[id]
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		return out[i].Entry.ID < out[j].Entry.ID
	})
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
},
) ([]ScoredDoc, error) {
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
},
) (DocEntry, float64, error) {
	var e DocEntry
	var rank float64
	err := rows.Scan(&e.ID, &e.SourceID, &e.Title, &e.Content, &e.Path, &e.ContentHash, &e.UpdatedAt, &rank)
	return e, rank, err
}
