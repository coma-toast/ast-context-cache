package contextnotes

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

// Note is a stored virtual context entry.
type Note struct {
	Ref           string `json:"ref"`
	SessionID     string `json:"session_id"`
	ProjectPath   string `json:"project_path,omitempty"`
	Label         string `json:"label,omitempty"`
	Content       string `json:"content,omitempty"`
	Tags          string `json:"tags,omitempty"`
	TokenEst      int    `json:"virtual_tokens,omitempty"`
	AccessCount   int    `json:"access_count,omitempty"`
	CreatedAt     string `json:"created_at,omitempty"`
	LastAccessed  string `json:"last_accessed_at,omitempty"`
}

// LimitError is returned when storage caps are exceeded.
type LimitError struct {
	Limit    string `json:"limit"`
	Current  int    `json:"current"`
	Max      int    `json:"max"`
	WouldAdd int    `json:"would_add"`
}

func (e *LimitError) Error() string {
	return fmt.Sprintf("context_limit_exceeded: %s (%d/%d, would add %d)", e.Limit, e.Current, e.Max, e.WouldAdd)
}

// StoreResult is returned from Store.
type StoreResult struct {
	Ref                 string                 `json:"ref"`
	Label               string                 `json:"label,omitempty"`
	VirtualTokensStored int                    `json:"virtual_tokens_stored"`
	SessionID           string                 `json:"session_id"`
	EvictedRefs         []string               `json:"evicted_refs,omitempty"`
	Stats               map[string]interface{} `json:"stats"`
}

func newRef() (string, error) {
	b := make([]byte, 6)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return "ctx_" + hex.EncodeToString(b), nil
}

func normalizeTags(raw interface{}) string {
	if raw == nil {
		return ""
	}
	switch v := raw.(type) {
	case string:
		return strings.TrimSpace(v)
	case []interface{}:
		parts := make([]string, 0, len(v))
		for _, item := range v {
			if s, ok := item.(string); ok && strings.TrimSpace(s) != "" {
				parts = append(parts, strings.TrimSpace(s))
			}
		}
		return strings.Join(parts, ",")
	default:
		b, _ := json.Marshal(v)
		return string(b)
	}
}

// Store saves virtual context and returns a stable ref.
func Store(sessionID, content, label, projectPath string, tags interface{}, emb embedder.Interface) (*StoreResult, error) {
	sessionID = strings.TrimSpace(sessionID)
	content = strings.TrimSpace(content)
	if sessionID == "" || content == "" {
		return nil, errors.New("session_id and content required")
	}
	tokenEst := db.EstimateTokens(content)
	lim := LoadLimits()
	if tokenEst > lim.MaxTokensSession {
		return nil, &LimitError{Limit: "single_note_tokens", Current: tokenEst, Max: lim.MaxTokensSession, WouldAdd: tokenEst}
	}
	tagStr := normalizeTags(tags)
	var evicted []string
	var err error
	if lim.Policy == "lru_session" {
		evicted, err = evictSessionLRU(sessionID, tokenEst, lim)
		if err != nil {
			return nil, err
		}
	} else if err = checkLimits(sessionID, tokenEst, lim); err != nil {
		return nil, err
	}
	ref, err := newRef()
	if err != nil {
		return nil, err
	}
	hash := search.ContentHash(content)
	_, err = db.DB.Exec(`INSERT INTO context_notes (ref, session_id, project_path, label, content, content_hash, tags, token_est)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		ref, sessionID, projectPath, label, content, hash, tagStr, tokenEst)
	if err != nil {
		return nil, err
	}
	indexNoteFTS(ref, sessionID, label, content)
	bumpSessionStore(sessionID, projectPath, tokenEst)
	if emb != nil {
		go EmbedNote(ref, sessionID, label, content, emb)
	}
	stats := BuildStatsBlock(sessionID, lim)
	return &StoreResult{
		Ref:                 ref,
		Label:               label,
		VirtualTokensStored: tokenEst,
		SessionID:           sessionID,
		EvictedRefs:         evicted,
		Stats:               stats,
	}, nil
}

func checkLimits(sessionID string, addTokens int, lim Limits) error {
	var sessionNotes, sessionTokens int
	db.DB.QueryRow(`SELECT COUNT(*), COALESCE(SUM(token_est),0) FROM context_notes WHERE session_id = ?`, sessionID).
		Scan(&sessionNotes, &sessionTokens)
	globalNotes, globalTokens := GlobalRollup()
	if sessionNotes+1 > lim.MaxNotesSession {
		return &LimitError{Limit: "session_notes", Current: sessionNotes, Max: lim.MaxNotesSession, WouldAdd: 1}
	}
	if sessionTokens+addTokens > lim.MaxTokensSession {
		return &LimitError{Limit: "session_tokens", Current: sessionTokens, Max: lim.MaxTokensSession, WouldAdd: addTokens}
	}
	if globalNotes+1 > lim.MaxNotesGlobal {
		return &LimitError{Limit: "global_notes", Current: globalNotes, Max: lim.MaxNotesGlobal, WouldAdd: 1}
	}
	if globalTokens+addTokens > lim.MaxTokensGlobal {
		return &LimitError{Limit: "global_tokens", Current: globalTokens, Max: lim.MaxTokensGlobal, WouldAdd: addTokens}
	}
	return nil
}

func evictSessionLRU(sessionID string, needTokens int, lim Limits) ([]string, error) {
	var evicted []string
	for {
		limitErr := checkLimits(sessionID, needTokens, lim)
		if limitErr == nil {
			return evicted, nil
		} else if _, ok := limitErr.(*LimitError); !ok {
			return evicted, limitErr
		} else if le, ok := limitErr.(*LimitError); ok && (le.Limit == "global_notes" || le.Limit == "global_tokens") {
			return evicted, le
		}
		ref, _, ok := oldestSessionNote(sessionID)
		if !ok {
			return evicted, limitErr
		}
		deleteRefs([]string{ref}, "")
		evicted = append(evicted, ref)
	}
}

func oldestSessionNote(sessionID string) (ref string, tokens int, ok bool) {
	err := db.DB.QueryRow(`SELECT ref, token_est FROM context_notes WHERE session_id = ?
		ORDER BY created_at ASC, access_count ASC LIMIT 1`, sessionID).Scan(&ref, &tokens)
	return ref, tokens, err == nil
}

func scanNote(row interface{ Scan(...any) error }) (Note, error) {
	var n Note
	err := row.Scan(&n.Ref, &n.SessionID, &n.ProjectPath, &n.Label, &n.Content, &n.Tags, &n.TokenEst, &n.AccessCount, &n.CreatedAt, &n.LastAccessed)
	if err != nil {
		return n, err
	}
	if n.TokenEst == 0 && n.Content != "" {
		n.TokenEst = db.EstimateTokens(n.Content)
	}
	return n, nil
}

func noteByRef(ref string) (Note, error) {
	row := db.DB.QueryRow(`SELECT ref, session_id, COALESCE(project_path,''), COALESCE(label,''), content,
		COALESCE(tags,''), token_est, access_count, created_at, COALESCE(last_accessed_at,'')
		FROM context_notes WHERE ref = ?`, ref)
	return scanNote(row)
}

func deleteRefs(refs []string, sessionID string) (tokensFreed int, count int, sessions map[string]int) {
	sessions = map[string]int{}
	for _, ref := range refs {
		ref = strings.TrimSpace(ref)
		if ref == "" {
			continue
		}
		var sid string
		var tok int
		err := db.DB.QueryRow(`SELECT session_id, token_est FROM context_notes WHERE ref = ?`, ref).Scan(&sid, &tok)
		if err != nil {
			continue
		}
		if sessionID != "" && sid != sessionID {
			continue
		}
		db.DB.Exec(`DELETE FROM context_notes WHERE ref = ?`, ref)
		deleteNoteFTS(ref)
		deleteNoteVector(ref)
		tokensFreed += tok
		count++
		sessions[sid] += tok
		adjustSessionStore(sid, -1, -tok)
	}
	return tokensFreed, count, sessions
}

func deleteBySession(sessionID, projectPath string) (tokensFreed int, count int) {
	q := `SELECT ref, token_est FROM context_notes WHERE session_id = ?`
	args := []any{sessionID}
	if projectPath != "" {
		q += ` AND project_path = ?`
		args = append(args, projectPath)
	}
	rows, err := db.DB.Query(q, args...)
	if err != nil {
		return 0, 0
	}
	defer rows.Close()
	var refs []string
	for rows.Next() {
		var ref string
		var tok int
		rows.Scan(&ref, &tok)
		refs = append(refs, ref)
	}
	tokensFreed, count, _ = deleteRefs(refs, sessionID)
	return tokensFreed, count
}

func deleteAll() (tokensFreed int, count int) {
	rows, err := db.DB.Query(`SELECT ref FROM context_notes`)
	if err != nil {
		return 0, 0
	}
	defer rows.Close()
	var refs []string
	for rows.Next() {
		var ref string
		rows.Scan(&ref)
		refs = append(refs, ref)
	}
	tokensFreed, count, _ = deleteRefs(refs, "")
	db.DB.Exec(`DELETE FROM context_session_stats`)
	return tokensFreed, count
}

func parseRefList(raw interface{}) []string {
	if raw == nil {
		return nil
	}
	switch v := raw.(type) {
	case string:
		if strings.TrimSpace(v) != "" {
			return []string{strings.TrimSpace(v)}
		}
	case []interface{}:
		out := make([]string, 0, len(v))
		for _, item := range v {
			if s, ok := item.(string); ok && strings.TrimSpace(s) != "" {
				out = append(out, strings.TrimSpace(s))
			}
		}
		return out
	case []string:
		out := make([]string, 0, len(v))
		for _, s := range v {
			if strings.TrimSpace(s) != "" {
				out = append(out, strings.TrimSpace(s))
			}
		}
		return out
	}
	return nil
}

// FetchResult holds notes retrieved by ref.
type FetchResult struct {
	Notes []Note                 `json:"notes"`
	Stats map[string]interface{} `json:"stats"`
}

// Fetch returns notes by ref, optionally scoped to session_id.
func Fetch(refsRaw interface{}, sessionID string) (*FetchResult, error) {
	refs := parseRefList(refsRaw)
	if len(refs) == 0 {
		return nil, errors.New("refs required")
	}
	notes := make([]Note, 0, len(refs))
	totalReturned := 0
	for _, ref := range refs {
		n, err := noteByRef(ref)
		if err != nil {
			continue
		}
		if sessionID != "" && n.SessionID != sessionID {
			continue
		}
		tok := n.TokenEst
		if tok == 0 {
			tok = db.EstimateTokens(n.Content)
		}
		RecordAccess(ref, n.SessionID, n.ProjectPath, "fetch_context", tok)
		totalReturned += tok
		notes = append(notes, n)
	}
	stats := map[string]interface{}{
		"virtual_tokens_returned": totalReturned,
		"refs_accessed":           len(notes),
	}
	if sessionID != "" {
		r := SessionRollupFor(sessionID)
		stats["session_virtual_accessed_total"] = r.VirtualTokensAccessed
	}
	return &FetchResult{Notes: notes, Stats: stats}, nil
}

// ListResult is metadata-only listing.
type ListResult struct {
	Notes []Note `json:"notes"`
	Total int    `json:"total"`
}

// List returns note metadata for a session.
func List(sessionID, projectPath string, limit int) (*ListResult, error) {
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return nil, errors.New("session_id required")
	}
	if limit <= 0 {
		limit = 20
	}
	var total int
	countQ := `SELECT COUNT(*) FROM context_notes WHERE session_id = ?`
	countArgs := []any{sessionID}
	q := `SELECT ref, session_id, COALESCE(project_path,''), COALESCE(label,''), '',
		COALESCE(tags,''), token_est, access_count, created_at, COALESCE(last_accessed_at,'')
		FROM context_notes WHERE session_id = ?`
	args := []any{sessionID}
	if projectPath != "" {
		q += ` AND project_path = ?`
		countQ += ` AND project_path = ?`
		args = append(args, projectPath)
		countArgs = append(countArgs, projectPath)
	}
	db.DB.QueryRow(countQ, countArgs...).Scan(&total)
	q += ` ORDER BY created_at DESC LIMIT ?`
	args = append(args, limit)
	rows, err := db.DB.Query(q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var notes []Note
	for rows.Next() {
		n, err := scanNote(rows)
		if err != nil {
			continue
		}
		notes = append(notes, n)
	}
	return &ListResult{Notes: notes, Total: total}, nil
}

// FlushResult reports deleted virtual context.
type FlushResult struct {
	FlushedRefs        int                    `json:"flushed_refs"`
	VirtualTokensFreed int                    `json:"virtual_tokens_freed"`
	Scope              string                 `json:"scope"`
	Stats              map[string]interface{} `json:"stats"`
}

// Flush deletes stored context by scope.
func Flush(sessionID string, refsRaw interface{}, projectPath string, all bool) (*FlushResult, error) {
	refs := parseRefList(refsRaw)
	var tokensFreed, count int
	scope := ""
	switch {
	case all:
		tokensFreed, count = deleteAll()
		scope = "all"
	case len(refs) > 0:
		tokensFreed, count, _ = deleteRefs(refs, sessionID)
		scope = "refs"
	case sessionID != "":
		tokensFreed, count = deleteBySession(sessionID, projectPath)
		scope = "session"
	default:
		return nil, errors.New("scope required: session_id, refs, or all=true")
	}
	inv := LiveInventory("")
	stats := map[string]interface{}{
		"active_inventory_tokens": inv.ActiveInventoryTok,
		"active_notes_count":      inv.ActiveNotesCount,
	}
	return &FlushResult{
		FlushedRefs:        count,
		VirtualTokensFreed: tokensFreed,
		Scope:              scope,
		Stats:              stats,
	}, nil
}

// ScoredNote pairs a note with search score.
type ScoredNote struct {
	Note  Note    `json:"note"`
	Score float64 `json:"score"`
}

// SearchResult holds search matches.
type SearchResult struct {
	Notes []Note                 `json:"notes"`
	Stats map[string]interface{} `json:"stats"`
}

// Search finds notes by FTS (and optional vector hybrid).
func Search(query, sessionID, projectPath string, limit int, emb embedder.Interface) (*SearchResult, error) {
	query = strings.TrimSpace(query)
	if query == "" {
		return nil, errors.New("query required")
	}
	if limit <= 0 {
		limit = 5
	}
	scored, err := searchNotesHybrid(query, sessionID, projectPath, limit, emb)
	if err != nil {
		return nil, err
	}
	notes := make([]Note, 0, len(scored))
	totalReturned := 0
	for _, s := range scored {
		n := s.Note
		tok := n.TokenEst
		if tok == 0 && n.Content != "" {
			tok = db.EstimateTokens(n.Content)
		}
		RecordAccess(n.Ref, n.SessionID, n.ProjectPath, "search_context", tok)
		totalReturned += tok
		notes = append(notes, n)
	}
	stats := map[string]interface{}{
		"virtual_tokens_returned": totalReturned,
		"refs_accessed":           len(notes),
	}
	if sessionID != "" {
		r := SessionRollupFor(sessionID)
		stats["session_virtual_accessed_total"] = r.VirtualTokensAccessed
	}
	return &SearchResult{Notes: notes, Stats: stats}, nil
}

func searchNotesFTS(query, sessionID, projectPath string, limit int) ([]ScoredNote, error) {
	ftsQuery := search.BuildFTSQuery(search.QueryTerms(query))
	if ftsQuery == "" {
		return nil, nil
	}
	q := `SELECT cn.ref, cn.session_id, COALESCE(cn.project_path,''), COALESCE(cn.label,''), cn.content,
		COALESCE(cn.tags,''), cn.token_est, cn.access_count, cn.created_at, COALESCE(cn.last_accessed_at,'')
		FROM context_notes_fts f
		JOIN context_notes cn ON cn.ref = f.ref
		WHERE context_notes_fts MATCH ?`
	args := []any{ftsQuery}
	if sessionID != "" {
		q += ` AND cn.session_id = ?`
		args = append(args, sessionID)
	}
	if projectPath != "" {
		q += ` AND cn.project_path = ?`
		args = append(args, projectPath)
	}
	q += ` ORDER BY cn.created_at DESC LIMIT ?`
	args = append(args, limit)
	rows, err := db.DB.Query(q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []ScoredNote
	for rows.Next() {
		var n Note
		if err := rows.Scan(&n.Ref, &n.SessionID, &n.ProjectPath, &n.Label, &n.Content, &n.Tags, &n.TokenEst, &n.AccessCount, &n.CreatedAt, &n.LastAccessed); err != nil {
			continue
		}
		out = append(out, ScoredNote{Note: n, Score: 1.0})
	}
	return out, nil
}

func searchNotesLike(query, sessionID, projectPath string, limit int) ([]ScoredNote, error) {
	q := `SELECT ref, session_id, COALESCE(project_path,''), COALESCE(label,''), content,
		COALESCE(tags,''), token_est, access_count, created_at, COALESCE(last_accessed_at,'')
		FROM context_notes WHERE label LIKE ? OR content LIKE ?`
	args := []any{"%" + query + "%", "%" + query + "%"}
	if sessionID != "" {
		q += ` AND session_id = ?`
		args = append(args, sessionID)
	}
	if projectPath != "" {
		q += ` AND project_path = ?`
		args = append(args, projectPath)
	}
	q += ` ORDER BY created_at DESC LIMIT ?`
	args = append(args, limit)
	rows, err := db.DB.Query(q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []ScoredNote
	for rows.Next() {
		n, err := scanNote(rows)
		if err != nil {
			continue
		}
		out = append(out, ScoredNote{Note: n, Score: 0.5})
	}
	return out, nil
}

func searchNotesHybrid(query, sessionID, projectPath string, limit int, emb embedder.Interface) ([]ScoredNote, error) {
	fts, err := searchNotesFTS(query, sessionID, projectPath, limit*2)
	if err != nil {
		fts = nil
	}
	if len(fts) == 0 {
		fts, err = searchNotesLike(query, sessionID, projectPath, limit*2)
		if err != nil {
			return nil, err
		}
	}
	var vector []ScoredNote
	if emb != nil {
		vector, _ = searchNotesVector(query, sessionID, limit*2, emb)
	}
	if len(vector) == 0 {
		if len(fts) > limit {
			fts = fts[:limit]
		}
		return fts, nil
	}
	return fuseNoteResults(fts, vector, limit), nil
}

func searchNotesVector(query, sessionID string, limit int, emb embedder.Interface) ([]ScoredNote, error) {
	queryVec, err := emb.EmbedSingle(query)
	if err != nil {
		return nil, err
	}
	scored := search.Cache.SearchNote(queryVec, sessionID, limit)
	out := make([]ScoredNote, 0, len(scored))
	for _, s := range scored {
		ref, _ := s.Data["ref"].(string)
		if ref == "" {
			continue
		}
		n, err := noteByRef(ref)
		if err != nil {
			continue
		}
		out = append(out, ScoredNote{Note: n, Score: s.Score})
	}
	return out, nil
}

func fuseNoteResults(fts, vector []ScoredNote, limit int) []ScoredNote {
	const k = 60
	type fused struct {
		n     Note
		score float64
	}
	seen := map[string]*fused{}
	add := func(rank int, s ScoredNote) {
		if s.Note.Ref == "" {
			return
		}
		bump := 1.0 / float64(k+rank+1)
		if e, ok := seen[s.Note.Ref]; ok {
			e.score += bump
		} else {
			seen[s.Note.Ref] = &fused{n: s.Note, score: bump}
		}
	}
	for i, s := range fts {
		add(i, s)
	}
	for i, s := range vector {
		add(i, s)
	}
	out := make([]ScoredNote, 0, len(seen))
	for _, e := range seen {
		out = append(out, ScoredNote{Note: e.n, Score: e.score})
	}
	// sort by score desc
	for i := 0; i < len(out); i++ {
		maxIdx := i
		for j := i + 1; j < len(out); j++ {
			if out[j].Score > out[maxIdx].Score {
				maxIdx = j
			}
		}
		out[i], out[maxIdx] = out[maxIdx], out[i]
	}
	if len(out) > limit {
		out = out[:limit]
	}
	return out
}

// LimitErrorMap converts LimitError to JSON-friendly map.
func LimitErrorMap(err error) map[string]interface{} {
	var le *LimitError
	if errors.As(err, &le) {
		return map[string]interface{}{
			"error":      "context_limit_exceeded",
			"limit":      le.Limit,
			"current":    le.Current,
			"max":        le.Max,
			"would_add":  le.WouldAdd,
			"suggestion": "call flush_context(session_id=...) or raise limits in dashboard settings",
		}
	}
	return map[string]interface{}{"error": err.Error()}
}
