package memory

import (
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

// Conflict represents a potential supersession conflict candidate.
type Conflict struct {
	Ref        string  `json:"ref"`
	Line       string  `json:"line"`
	Similarity float64 `json:"similarity"`
}

type embedResult struct {
	conflicts []Conflict
	err       error
}

// ConflictCandidates finds active memory entries that might conflict with the given entry (MEM-12–14).
// It embeds FormatLine(e) in a goroutine with a timeout (default 300ms, configured via AST_CONFLICT_TIMEOUT_MS),
// searches vector cache, and falls back to FTS if it times out or embedder is nil.
func ConflictCandidates(e Entry, emb embedder.Interface, timeout time.Duration) ([]Conflict, string) {
	if timeout <= 0 {
		timeoutMs := db.SettingInt("conflict_timeout_ms", "AST_CONFLICT_TIMEOUT_MS", 300)
		timeout = time.Duration(timeoutMs) * time.Millisecond
	}
	if emb == nil {
		return ftsConflictCandidates(e), "fts"
	}

	resCh := make(chan embedResult, 1)
	go func() {
		line := FormatLine(e)
		vec, err := emb.EmbedSingle(line)
		if err != nil {
			resCh <- embedResult{err: err}
			return
		}
		// Search memory cache with scope/session filter
		sessionID := e.SessionID
		includeGlobal := e.Scope != ScopeSession
		scored := search.Cache.SearchMemory(vec, sessionID, includeGlobal, 10)
		var conflicts []Conflict
		for _, s := range scored {
			ref, _ := s.Data["ref"].(string)
			if ref == "" || ref == e.Ref {
				continue
			}
			cand, err := entryByRef(ref)
			if err != nil || cand.ValidUntil != "" {
				continue
			}
			conflicts = append(conflicts, Conflict{
				Ref:        ref,
				Line:       FormatLine(cand),
				Similarity: float64(s.Score),
			})
		}
		resCh <- embedResult{conflicts: conflicts}
	}()

	select {
	case r := <-resCh:
		if r.err != nil {
			return ftsConflictCandidates(e), "fts"
		}
		return r.conflicts, "embed"
	case <-time.After(timeout):
		return ftsConflictCandidates(e), "fts"
	}
}

func ftsConflictCandidates(e Entry) []Conflict {
	// Fallback FTS search over subject/object terms; limit to active facts only.
	terms := search.QueryTerms(e.Subject + " " + e.Object)
	if len(terms) == 0 {
		return nil
	}
	ftsQuery := search.BuildFTSQuery(terms)
	if ftsQuery == "" {
		return nil
	}
	kinds := []Kind{KindFact}
	clause, args := filterClauses(RecallInput{
		SessionID:           e.SessionID,
		ProjectPath:         e.ProjectPath,
		Scope:               e.Scope,
		Kinds:               kinds,
		IncludeRepoSiblings: e.Scope == ScopeProject,
	}, "sm.", smEntryValidity)
	q := `SELECT sm.ref, sm.kind, sm.scope, sm.session_id, sm.project_path, sm.subject, sm.predicate, sm.object, sm.rule,
		sm.valid_from, sm.valid_until, sm.superseded_by, sm.source_ref, sm.token_est, sm.access_count, sm.last_accessed_at, sm.created_at,
		sm.invalidated_reason, sm.world_valid_at, sm.world_invalid_at, sm.expires_at, sm.quarantined
		FROM structured_memory_fts
		JOIN structured_memory sm ON sm.ref = structured_memory_fts.ref
		WHERE structured_memory_fts MATCH ?` + clause + ` ORDER BY bm25(structured_memory_fts) LIMIT 10`

	rows, err := db.ContextDB.Query(q, append([]any{ftsContentColumnsFilterStart + ftsQuery + ")"}, args...)...)
	if err != nil {
		return nil
	}
	defer rows.Close()

	var conflicts []Conflict
	for rows.Next() {
		cand, err := scanEntry(rows)
		if err != nil || cand.Ref == e.Ref {
			continue
		}
		conflicts = append(conflicts, Conflict{
			Ref:        cand.Ref,
			Line:       FormatLine(cand),
			Similarity: 0.5,
		})
	}
	return conflicts
}
