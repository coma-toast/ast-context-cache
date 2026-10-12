package memory

import (
	"math"
	"sort"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/flags"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

const (
	searchEntriesLikeQuery = `SELECT ref, kind, scope, session_id, project_path, subject, predicate, object, rule,
		valid_from, valid_until, superseded_by, source_ref, token_est, access_count, last_accessed_at, created_at,
		invalidated_reason, world_valid_at, world_invalid_at, expires_at, quarantined
		FROM structured_memory WHERE (subject LIKE ? OR predicate LIKE ? OR object LIKE ? OR rule LIKE ?)`
	// The FTS table has no alias so bm25() and MATCH can name it.
	searchEntriesFTSQuery = `SELECT sm.ref, sm.kind, sm.scope, sm.session_id, sm.project_path, sm.subject, sm.predicate, sm.object, sm.rule,
		sm.valid_from, sm.valid_until, sm.superseded_by, sm.source_ref, sm.token_est, sm.access_count, sm.last_accessed_at, sm.created_at,
		sm.invalidated_reason, sm.world_valid_at, sm.world_invalid_at, sm.expires_at, sm.quarantined
		FROM structured_memory_fts
		JOIN structured_memory sm ON sm.ref = structured_memory_fts.ref
		WHERE structured_memory_fts MATCH ?`
	orderByBM25RefLimitClause    = ` ORDER BY bm25(structured_memory_fts), sm.ref LIMIT ?`
	orderByAccessRefLimitClause  = ` ORDER BY access_count DESC, ref LIMIT ?`
	ftsContentColumnsFilterStart = `{subject predicate object rule} : (`
)

// searchEntries runs the FTS, LIKE and (with an embedder) vector searches and fuses their
// rankings with RRF. It returns the fused entries best first and each ref's fused score.
func searchEntries(in RecallInput, emb embedder.Interface) ([]Entry, map[string]float64, error) {
	fts, err := searchFTS(in)
	if err != nil {
		logger.Warn("Memory FTS search failed, using the other rankings", "error", err)
	}
	like, err := searchLike(in)
	if err != nil {
		return nil, nil, err
	}
	var vec []Entry
	if emb != nil {
		if vec, err = vectorSearch(in, emb); err != nil {
			logger.Warn("Memory vector search failed, using the lexical rankings", "error", err)
		}
	}
	entries, scores := fuseEntries(fts, like, vec)
	entries = applyDecay(entries, scores, in)
	entries = applyTTLExpiry(entries, in)
	return entries, scores, nil
}

// decayFactor returns the recency multiplier for an entry, clamped to 0.3–1.5x (MEM-6).
// Procedures are exempt by default (MEM-7). Half-lives are settings: session scope
// defaults to 24 hours, project scope to 30 days, and global scope has no decay.
func decayFactor(e Entry, in RecallInput) float64 {
	if e.Kind == KindProcedure {
		if !db.SettingBool("memory_decay_procedures", "AST_FEATURE_MEMORY_DECAY_PROCEDURES", false) {
			return 1.0
		}
	}
	if e.Scope == ScopeGlobal {
		return 1.0
	}
	halfLifeDays := 1.0
	if e.Scope == ScopeProject {
		halfLifeDays = 30.0
	}
	// MEM-6: half-lives are settings. The session half-life is in hours, the
	// project half-life in days; both fall back to the defaults above.
	if e.Scope == ScopeSession {
		if h := db.SettingInt("memory_halflife_session_hours", "AST_MEMORY_HALFLIFE_SESSION_HOURS", 24); h > 0 {
			halfLifeDays = float64(h) / 24.0
		}
	} else if e.Scope == ScopeProject {
		if d := db.SettingInt("memory_halflife_project_days", "AST_MEMORY_HALFLIFE_PROJECT_DAYS", 30); d > 0 {
			halfLifeDays = float64(d)
		}
	}
	ageDays := 0.0
	if e.LastAccessedAt != "" {
		if t, err := time.Parse("2006-01-02 15:04:05", e.LastAccessedAt); err == nil {
			ageDays = time.Since(t).Hours() / 24.0
		}
	}
	decay := 0.5
	if e.AccessCount > 1 {
		decay = 0.5 * (1.0 + 1.0/float64(e.AccessCount))
	}
	factor := math.Pow(decay, ageDays/halfLifeDays)
	if factor < 0.3 {
		factor = 0.3
	}
	if factor > 1.5 {
		factor = 1.5
	}
	return factor
}

// applyDecay multiplies each fused score by its recency factor and re-sorts.
func applyDecay(entries []Entry, scores map[string]float64, in RecallInput) []Entry {
	if !flags.Enabled(flags.KeyMemoryV2) {
		return entries
	}
	for i := range entries {
		if f := decayFactor(entries[i], in); f != 1.0 {
			scores[entries[i].Ref] *= f
		}
	}
	sort.Slice(entries, func(i, j int) bool {
		if si, sj := scores[entries[i].Ref], scores[entries[j].Ref]; si != sj {
			return si > sj
		}
		return entries[i].Ref < entries[j].Ref
	})
	return entries
}

// applyTTLExpiry invalidates entries past their TTL (MEM-8) and removes them from output.
func applyTTLExpiry(entries []Entry, in RecallInput) []Entry {
	if !flags.Enabled(flags.KeyMemoryV2) {
		return entries
	}
	now := time.Now().UTC()
	out := entries[:0]
	for _, e := range entries {
		if e.ExpiresAt != "" {
			if t, err := time.Parse("2006-01-02 15:04:05", e.ExpiresAt); err == nil && t.Before(now) {
				memoryInvalidateTTL(e.Ref)
				continue
			}
		}
		out = append(out, e)
	}
	return out
}

func memoryInvalidateTTL(ref string) {
	db.ContextDB.Exec(`UPDATE structured_memory SET valid_until = datetime('now'), invalidated_reason = ? WHERE ref = ?`, ReasonTTL, ref)
	deleteFTS(ref)
}

// fuseEntries merges ranked entry lists with RRF, ordered by fused score descending then ref.
func fuseEntries(lists ...[]Entry) ([]Entry, map[string]float64) {
	byRef := map[string]Entry{}
	refLists := make([][]string, len(lists))
	for i, list := range lists {
		for _, e := range list {
			refLists[i] = append(refLists[i], e.Ref)
			byRef[e.Ref] = e
		}
	}
	scores := search.RRF(refLists...)
	out := make([]Entry, 0, len(byRef))
	for _, e := range byRef {
		out = append(out, e)
	}
	sort.Slice(out, func(i, j int) bool {
		if si, sj := scores[out[i].Ref], scores[out[j].Ref]; si != sj {
			return si > sj
		}
		return out[i].Ref < out[j].Ref
	})
	return out, scores
}

// searchFTS matches the query against the content columns only (never ref), best bm25 first.
func searchFTS(in RecallInput) ([]Entry, error) {
	ftsQuery := search.BuildFTSQuery(search.QueryTerms(in.Query))
	if ftsQuery == "" {
		return nil, nil
	}
	clause, args := filterClauses(in, "sm.", smEntryValidity)
	args = append([]any{ftsContentColumnsFilterStart + ftsQuery + ")"}, args...)
	return queryEntries(searchEntriesFTSQuery+clause+orderByBM25RefLimitClause, append(args, in.Limit*2)...)
}

func searchLike(in RecallInput) ([]Entry, error) {
	likeQ := `%` + in.Query + `%`
	clause, args := filterClauses(in, "", entryValidity)
	args = append([]any{likeQ, likeQ, likeQ, likeQ}, args...)
	return queryEntries(searchEntriesLikeQuery+clause+orderByAccessRefLimitClause, append(args, in.Limit*2)...)
}

func vectorSearch(in RecallInput, emb embedder.Interface) ([]Entry, error) {
	vec, err := emb.EmbedSingle(in.Query)
	if err != nil {
		return nil, err
	}
	// Session-less vectors only hold project or global entries, which a session-scoped
	// recall can never return. Any other scope gets every memory vector's candidates.
	scored := search.Cache.SearchMemory(vec, in.SessionID, in.Scope != ScopeSession, in.Limit*2)
	var refs []string
	for _, s := range scored {
		if ref, _ := s.Data["ref"].(string); ref != "" {
			refs = append(refs, ref)
		}
	}
	if len(refs) == 0 {
		return nil, nil
	}
	// Vectors only know the storing session, so the re-select applies the same validity,
	// scope and kind filters as the FTS and LIKE paths.
	args := make([]any, 0, len(refs))
	for _, r := range refs {
		args = append(args, r)
	}
	clause, filterArgs := filterClauses(in, "", entryValidity)
	q := selectEntriesByRefsQuery + strings.TrimSuffix(strings.Repeat("?,", len(refs)), ",") + ")" + clause
	entries, err := queryEntries(q, append(args, filterArgs...)...)
	if err != nil {
		return nil, err
	}
	entries = orderByRefs(entries, refs)
	if len(entries) > in.Limit*2 {
		entries = entries[:in.Limit*2]
	}
	return entries, nil
}

// orderByRefs returns entries in refs order (the vector similarity rank),
// dropping duplicate refs.
func orderByRefs(entries []Entry, refs []string) []Entry {
	byRef := make(map[string]Entry, len(entries))
	for _, e := range entries {
		byRef[e.Ref] = e
	}
	out := make([]Entry, 0, len(entries))
	for _, r := range refs {
		if e, ok := byRef[r]; ok {
			out = append(out, e)
			delete(byRef, r)
		}
	}
	return out
}
