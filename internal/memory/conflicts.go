package memory

import (
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
	"github.com/coma-toast/ast-context-cache/internal/repokey"
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
	targetKind := e.Kind
	if targetKind == "" {
		targetKind = KindFact
	}
	if targetKind != KindFact && targetKind != KindDiscovery {
		return nil, "fts"
	}
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
			if err != nil || cand.ValidUntil != "" || cand.Quarantined != 0 {
				continue
			}
			// Filter to matching kind, same scope, and project/repo siblings
			if cand.Kind != targetKind {
				continue
			}
			if cand.Scope != e.Scope {
				continue
			}
			if e.Scope == ScopeProject && cand.ProjectPath != e.ProjectPath {
				// Check repo siblings for project scope
				if !isRepoSibling(e.ProjectPath, cand.ProjectPath) {
					continue
				}
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
	targetKind := e.Kind
	if targetKind == "" {
		targetKind = KindFact
	}
	if targetKind != KindFact && targetKind != KindDiscovery {
		return nil
	}
	// Fallback FTS search over title/body/subject/predicate/object terms; limit to matching active kind.
	terms := search.QueryTerms(e.Title + " " + e.Body + " " + e.Subject + " " + e.Predicate + " " + e.Object)
	if len(terms) == 0 {
		return nil
	}
	ftsQuery := search.BuildFTSQuery(terms)
	if ftsQuery == "" {
		return nil
	}
	kinds := []Kind{targetKind}
	clause, args := filterClauses(RecallInput{
		SessionID:           e.SessionID,
		ProjectPath:         e.ProjectPath,
		Scope:               e.Scope,
		Kinds:               kinds,
		IncludeRepoSiblings: e.Scope == ScopeProject,
	}, "sm.", smEntryValidity)
	q := `SELECT ` + entrySelectColsAliased("sm", false) + `
		FROM structured_memory_kb_fts
		JOIN structured_memory sm ON sm.ref = structured_memory_kb_fts.ref
		WHERE structured_memory_kb_fts MATCH ?` + clause + ` ORDER BY bm25(structured_memory_kb_fts) LIMIT 10`

	rows, err := db.ContextDB.Query(q, append([]any{ftsKBContentColumnsFilterStart + ftsQuery + ")"}, args...)...)
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

// isRepoSibling reports whether projA and projB are in the same repo
// (checked via the project_links index).
func isRepoSibling(projA, projB string) bool {
	// Same project path
	if projA == projB {
		return true
	}
	keyA := repokey.Key(projA)
	keyB := repokey.Key(projB)
	if keyA == "" || keyB == "" {
		return false
	}
	if keyA == keyB {
		return true
	}
	// Check if projB is a repo sibling of projA
	for _, sib := range projectlinks.RepoSiblings(projA) {
		if sib == projB {
			return true
		}
	}
	return false
}
