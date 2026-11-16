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
	searchEntriesLikeWhere = ` FROM structured_memory WHERE (subject LIKE ? OR predicate LIKE ? OR object LIKE ? OR rule LIKE ? OR title LIKE ? OR body LIKE ? OR target LIKE ? OR tags LIKE ? OR reason LIKE ? OR category LIKE ? OR actor LIKE ?)`
	// The FTS table has no alias so bm25() and MATCH can name it.
	searchEntriesKBFTSFrom = ` FROM structured_memory_kb_fts
		JOIN structured_memory sm ON sm.ref = structured_memory_kb_fts.ref
		WHERE structured_memory_kb_fts MATCH ?`
	orderByKBBM25RefLimitClause    = ` ORDER BY bm25(structured_memory_kb_fts), sm.ref LIMIT ?`
	orderByAccessRefLimitClause    = ` ORDER BY access_count DESC, ref LIMIT ?`
	ftsKBContentColumnsFilterStart = `{title subject predicate object rule body tags target before_value after_value reason category actor} : (`
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

// decayFactor returns the recency/verified multiplier for an entry.
// Base time-decay is clamped to 0.3–1.5x (MEM-6); Verified entries receive a
// 1.25x boost clamped up to 1.75x max.
// Procedures (unless memory_decay_procedures), preferences, events, config_change,
// discoveries, and global entries are exempt from time-decay (base factor 1.0).
func decayFactor(e Entry, in RecallInput) float64 {
	factor := 1.0
	exempt := false
	switch e.Kind {
	case KindPreference, KindEvent, KindConfigChange, KindDiscovery:
		exempt = true
	case KindProcedure:
		if !db.SettingBool("memory_decay_procedures", "AST_FEATURE_MEMORY_DECAY_PROCEDURES", false) {
			exempt = true
		}
	}
	if e.Scope == ScopeGlobal {
		exempt = true
	}

	if !exempt {
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
		// recency: half-life since the last access.
		recency := 1.0
		if e.LastAccessedAt != "" {
			if t, err := time.Parse("2006-01-02 15:04:05", e.LastAccessedAt); err == nil {
				sinceLast := time.Since(t).Hours() / 24.0
				if sinceLast < 0 {
					sinceLast = 0
				}
				recency = math.Pow(0.5, sinceLast/halfLifeDays)
			}
		}
		// frequency: more accesses raise the factor, never lower it. k=4 tops out near 1.5.
		frequency := 1.0
		if e.AccessCount > 0 {
			frequency = 1.0 + math.Log2(1.0+float64(e.AccessCount))/4.0
		}
		// age: mild decay for entries created long ago, independent of access recency.
		age := 1.0
		if e.CreatedAt != "" {
			if t, err := time.Parse("2006-01-02 15:04:05", e.CreatedAt); err == nil {
				ageDays := time.Since(t).Hours() / 24.0
				if ageDays < 0 {
					ageDays = 0
				}
				age = math.Pow(0.5, ageDays/(halfLifeDays*4.0))
			}
		}
		factor = recency * frequency * age
		if factor < 0.3 {
			factor = 0.3
		}
		if factor > 1.5 {
			factor = 1.5
		}
	}

	if e.Verified {
		factor *= 1.25
		if factor > 1.75 {
			factor = 1.75
		}
	}
	return factor
}

// applyDecay multiplies each fused score by its recency/verified factor and re-sorts.
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

// searchFTS matches the query against the KB FTS content columns only (never ref), best bm25 first.
func searchFTS(in RecallInput) ([]Entry, error) {
	ftsQuery := search.BuildFTSQuery(search.QueryTerms(in.Query))
	if ftsQuery == "" {
		return nil, nil
	}
	fullTier2 := in.Expand || len(in.Refs) > 0
	clause, args := filterClauses(in, "sm.", smEntryValidity)
	args = append([]any{ftsKBContentColumnsFilterStart + ftsQuery + ")"}, args...)
	q := "SELECT " + entrySelectColsAliased("sm", fullTier2) + searchEntriesKBFTSFrom + clause + orderByKBBM25RefLimitClause
	return queryEntries(q, append(args, in.Limit*2)...)
}

func searchLike(in RecallInput) ([]Entry, error) {
	fullTier2 := in.Expand || len(in.Refs) > 0
	likeQ := `%` + in.Query + `%`
	clause, args := filterClauses(in, "", entryValidity)
	likeArgs := make([]any, 0, 11+len(args)+1)
	for i := 0; i < 11; i++ {
		likeArgs = append(likeArgs, likeQ)
	}
	likeArgs = append(likeArgs, args...)
	likeArgs = append(likeArgs, in.Limit*2)
	q := "SELECT " + entrySelectCols(fullTier2) + searchEntriesLikeWhere + clause + orderByAccessRefLimitClause
	return queryEntries(q, likeArgs...)
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
	fullTier2 := in.Expand || len(in.Refs) > 0
	clause, filterArgs := filterClauses(in, "", entryValidity)
	q := "SELECT " + entrySelectCols(fullTier2) + " FROM structured_memory WHERE ref IN (" + strings.TrimSuffix(strings.Repeat("?,", len(refs)), ",") + ")" + clause
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
