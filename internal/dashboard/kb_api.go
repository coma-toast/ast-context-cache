package dashboard

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/coma-toast/ast-context-cache/internal/contextnotes"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/docs"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/handoff"
	"github.com/coma-toast/ast-context-cache/internal/mcp"
	"github.com/coma-toast/ast-context-cache/internal/memory"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

const (
	kbDefaultPerPage = 50
	kbMaxPerPage     = 200
	snippetMaxRunes  = 200
)

func registerKBAPI(mux *http.ServeMux) {
	mux.HandleFunc("/api/dashboard/kb", handleKBListJSON)
	mux.HandleFunc("/api/dashboard/kb/entry", handleKBEntryJSON)
	mux.HandleFunc("/api/dashboard/kb/entry/action", handleKBEntryActionJSON)
	mux.HandleFunc("/api/dashboard/kb/export", handleKBExport)
	mux.HandleFunc("/api/dashboard/context-notes", handleContextNotesJSON)
	mux.HandleFunc("/api/dashboard/context-notes/detail", handleContextNoteDetailJSON)
	mux.HandleFunc("/api/dashboard/doc-entries", handleDocEntriesJSON)
}

func kbErrorStatus(err error) int {
	switch {
	case errs.HasCode(err, errs.CodeInvalidInput):
		return http.StatusBadRequest
	case errs.HasCode(err, errs.CodeNotFound):
		return http.StatusNotFound
	case errs.HasCode(err, errs.CodeConflict):
		return http.StatusConflict
	default:
		return http.StatusInternalServerError
	}
}

func writeKBError(w http.ResponseWriter, err error, logMsg string) {
	status := kbErrorStatus(err)
	if status == http.StatusInternalServerError {
		logger.Warn(logMsg, "error", err)
	}
	writeJSONStatusError(w, status, err.Error())
}

func writeJSONStatusError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": msg})
}

func requireContextDB(w http.ResponseWriter) bool {
	if db.ContextDB == nil {
		writeJSONStatusError(w, http.StatusServiceUnavailable, "context database unavailable")
		return false
	}
	return true
}

func parsePageParams(r *http.Request, defaultPerPage, maxPerPage int) (page, perPage int, err error) {
	q := r.URL.Query()
	page = 1
	if v := strings.TrimSpace(q.Get("page")); v != "" {
		n, convErr := strconv.Atoi(v)
		if convErr != nil || n < 1 {
			return 0, 0, errs.NewCode(errs.CodeInvalidInput, "page must be a positive integer")
		}
		page = n
	}
	perPage = defaultPerPage
	if v := strings.TrimSpace(q.Get("per_page")); v != "" {
		n, convErr := strconv.Atoi(v)
		if convErr != nil || n < 1 {
			return 0, 0, errs.NewCode(errs.CodeInvalidInput, "per_page must be a positive integer")
		}
		perPage = n
	}
	if maxPerPage > 0 && perPage > maxPerPage {
		perPage = maxPerPage
	}
	return page, perPage, nil
}

func parseCSVParam(raw string) []string {
	if strings.TrimSpace(raw) == "" {
		return nil
	}
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if trimmed := strings.TrimSpace(p); trimmed != "" {
			out = append(out, trimmed)
		}
	}
	return out
}

func kbQueryInputFromRequest(r *http.Request) (memory.KBQueryInput, error) {
	q := r.URL.Query()
	page, perPage, err := parsePageParams(r, kbDefaultPerPage, kbMaxPerPage)
	if err != nil {
		return memory.KBQueryInput{}, err
	}

	var kinds []memory.Kind
	for _, raw := range parseCSVParam(q.Get("kinds")) {
		k := memory.Kind(strings.ToLower(raw))
		if !memory.IsValidKind(k) {
			return memory.KBQueryInput{}, errs.NewCode(errs.CodeInvalidInput, "invalid kind: %s", raw)
		}
		kinds = append(kinds, k)
	}

	scope := memory.Scope(strings.ToLower(strings.TrimSpace(q.Get("scope"))))
	projectID := strings.TrimSpace(q.Get("project_id"))
	verifiedRaw := strings.ToLower(strings.TrimSpace(q.Get("verified_only")))
	verifiedOnly := verifiedRaw == "true" || verifiedRaw == "1"

	return memory.KBQueryInput{
		Query:                    strings.TrimSpace(q.Get("q")),
		Kinds:                    kinds,
		Scope:                    scope,
		ProjectPath:              projectID,
		IncludeGlobalWithProject: projectID != "" && scope == "",
		SessionID:                strings.TrimSpace(q.Get("session_id")),
		Status:                   strings.TrimSpace(q.Get("status")),
		Tags:                     parseCSVParam(q.Get("tags")),
		Actor:                    strings.TrimSpace(q.Get("actor")),
		Since:                    strings.TrimSpace(q.Get("since")),
		Until:                    strings.TrimSpace(q.Get("until")),
		VerifiedOnly:             verifiedOnly,
		Sort:                     strings.TrimSpace(q.Get("sort")),
		Page:                     page,
		PerPage:                  perPage,
		Embedder:                 mcp.GetEmbedder(),
	}, nil
}

func handleKBListJSON(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSONStatusError(w, http.StatusMethodNotAllowed, "GET required")
		return
	}
	if !requireContextDB(w) {
		return
	}
	in, err := kbQueryInputFromRequest(r)
	if err != nil {
		writeKBError(w, err, "Invalid KB query parameters")
		return
	}
	res, err := memory.QueryKB(in)
	if err != nil {
		writeKBError(w, err, "Failed to query KB entries")
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(res)
}

type kbCreateRequest struct {
	Kind               memory.Kind  `json:"kind"`
	Scope              memory.Scope `json:"scope"`
	SessionID          string       `json:"session_id"`
	ProjectPath        string       `json:"project_path"`
	ProjectID          string       `json:"project_id"`
	Subject            string       `json:"subject"`
	Predicate          string       `json:"predicate"`
	Object             string       `json:"object"`
	Rule               string       `json:"rule"`
	SourceRef          string       `json:"source_ref"`
	InvalidatePrevious *bool        `json:"invalidate_previous"`
	WorldValidAt       string       `json:"world_valid_at"`
	WorldInvalidAt     string       `json:"world_invalid_at"`
	TTL                int          `json:"ttl"`
	TTLHours           int          `json:"ttl_hours"`
	TTLMinutes         int          `json:"ttl_minutes"`
	ExpiresAt          string       `json:"expires_at"`
	Title              string       `json:"title"`
	Body               string       `json:"body"`
	Tags               []string     `json:"tags"`
	Actor              string       `json:"actor"`
	Category           string       `json:"category"`
	Severity           string       `json:"severity"`
	Target             string       `json:"target"`
	BeforeValue        string       `json:"before_value"`
	AfterValue         string       `json:"after_value"`
	Reason             string       `json:"reason"`
	OccurredAt         string       `json:"occurred_at"`
	Verified           bool         `json:"verified"`
}

func handleKBEntryJSON(w http.ResponseWriter, r *http.Request) {
	if !requireContextDB(w) {
		return
	}
	switch r.Method {
	case http.MethodGet:
		getKBEntryJSON(w, r)
	case http.MethodPost:
		createKBEntryJSON(w, r)
	case http.MethodPut:
		updateKBEntryJSON(w, r)
	default:
		writeJSONStatusError(w, http.StatusMethodNotAllowed, "GET, POST, or PUT required")
	}
}

func getKBEntryJSON(w http.ResponseWriter, r *http.Request) {
	ref := strings.TrimSpace(r.URL.Query().Get("ref"))
	if ref == "" {
		writeJSONStatusError(w, http.StatusBadRequest, "ref required")
		return
	}
	entry, err := memory.Peek(ref)
	if err != nil {
		writeKBError(w, err, "Failed to load KB entry")
		return
	}

	history := []memory.HistoryEntry{}
	histRes, err := memory.History(memory.HistoryInput{Ref: ref, IncludeWorld: true})
	if err != nil {
		if !errs.HasCode(err, errs.CodeNotFound) {
			logger.Warn("Failed to load KB entry history", "ref", ref, "error", err)
		}
	} else if histRes != nil && len(histRes.Entries) > 0 {
		history = histRes.Entries
	}

	conflicts := []memory.Conflict{}
	if entry.ValidUntil == "" && entry.Quarantined == 0 && (entry.Kind == memory.KindFact || entry.Kind == memory.KindDiscovery) {
		if cands, _ := memory.ConflictCandidates(*entry, mcp.GetEmbedder(), 0); len(cands) > 0 {
			conflicts = cands
		}
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"entry":     entry,
		"history":   history,
		"conflicts": conflicts,
	})
}

func createKBEntryJSON(w http.ResponseWriter, r *http.Request) {
	var req kbCreateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSONStatusError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}

	projectPath := strings.TrimSpace(req.ProjectPath)
	if projectPath == "" {
		projectPath = strings.TrimSpace(req.ProjectID)
	}
	actor := strings.TrimSpace(req.Actor)
	if actor == "" {
		actor = "human"
	}
	invalidatePrev := true
	if req.InvalidatePrevious != nil {
		invalidatePrev = *req.InvalidatePrevious
	}

	res, err := memory.Store(memory.StoreInput{
		Kind:               memory.Kind(strings.ToLower(strings.TrimSpace(string(req.Kind)))),
		Scope:              memory.Scope(strings.ToLower(strings.TrimSpace(string(req.Scope)))),
		SessionID:          req.SessionID,
		ProjectPath:        projectPath,
		Subject:            req.Subject,
		Predicate:          req.Predicate,
		Object:             req.Object,
		Rule:               req.Rule,
		SourceRef:          req.SourceRef,
		InvalidatePrevious: invalidatePrev,
		WorldValidAt:       req.WorldValidAt,
		WorldInvalidAt:     req.WorldInvalidAt,
		TTL:                req.TTL,
		TTLHours:           req.TTLHours,
		TTLMinutes:         req.TTLMinutes,
		ExpiresAt:          req.ExpiresAt,
		Title:              req.Title,
		Body:               req.Body,
		Tags:               req.Tags,
		Actor:              actor,
		Category:           req.Category,
		Severity:           req.Severity,
		Target:             req.Target,
		BeforeValue:        req.BeforeValue,
		AfterValue:         req.AfterValue,
		Reason:             req.Reason,
		OccurredAt:         req.OccurredAt,
		Verified:           req.Verified,
	})
	if err != nil {
		writeKBError(w, err, "Failed to create KB entry")
		return
	}

	entry, err := memory.Peek(res.Ref)
	if err != nil {
		writeKBError(w, err, "Failed to reload created KB entry")
		return
	}

	emb := mcp.GetEmbedder()
	if emb != nil {
		item := *entry
		go func() {
			_, _ = memory.EmbedEntryFromEntry(item, emb)
		}()
	}

	conflicts := []memory.Conflict{}
	if entry.Kind == memory.KindFact || entry.Kind == memory.KindDiscovery {
		if cands, _ := memory.ConflictCandidates(*entry, emb, 0); len(cands) > 0 {
			conflicts = cands
		}
	}

	realtime.Notify(realtime.KnowledgeBase)

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"entry":            entry,
		"warnings":         nonNilStrings(res.Warnings),
		"redacted_fields":  nonNilStrings(res.RedactedFields),
		"invalidated_refs": nonNilStrings(res.InvalidatedRefs),
		"conflicts":        conflicts,
	})
}

func updateKBEntryJSON(w http.ResponseWriter, r *http.Request) {
	var in memory.UpdateInput
	if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
		writeJSONStatusError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}

	entry, warnings, err := memory.UpdateEntry(in, mcp.GetEmbedder())
	if err != nil {
		writeKBError(w, err, "Failed to update KB entry")
		return
	}

	realtime.Notify(realtime.KnowledgeBase)

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"entry":           entry,
		"warnings":        nonNilStrings(warnings),
		"redacted_fields": nonNilStrings(entry.RedactedFields),
	})
}

type kbActionRequest struct {
	Ref      string `json:"ref"`
	Action   string `json:"action"`
	Verified *bool  `json:"verified,omitempty"`
}

func handleKBEntryActionJSON(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSONStatusError(w, http.StatusMethodNotAllowed, "POST required")
		return
	}
	if !requireContextDB(w) {
		return
	}

	var req kbActionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSONStatusError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}

	ref := strings.TrimSpace(req.Ref)
	if ref == "" {
		writeJSONStatusError(w, http.StatusBadRequest, "ref required")
		return
	}
	action := strings.ToLower(strings.TrimSpace(req.Action))
	emb := mcp.GetEmbedder()

	var result any
	switch action {
	case "forget":
		res, err := memory.Forget(memory.ForgetInput{Refs: []string{ref}})
		if err != nil {
			writeKBError(w, err, "Failed to forget KB entry")
			return
		}
		if len(res.NotFound) > 0 {
			writeJSONStatusError(w, http.StatusNotFound, errs.NewCode(errs.CodeNotFound, "memory entry %q not found", ref).Error())
			return
		}
		result = res
	case "restore":
		res, err := memory.Restore(memory.RestoreInput{Refs: []string{ref}, Embedder: emb})
		if err != nil {
			writeKBError(w, err, "Failed to restore KB entry")
			return
		}
		if len(res.Conflicts) > 0 {
			msg := strings.Join(res.Conflicts, "; ")
			if strings.HasSuffix(res.Conflicts[0], ": not found") {
				writeJSONStatusError(w, http.StatusNotFound, msg)
			} else {
				writeJSONStatusError(w, http.StatusConflict, msg)
			}
			return
		}
		result = res
	case "verify":
		verified := true
		if req.Verified != nil {
			verified = *req.Verified
		}
		if err := memory.SetVerified(ref, verified); err != nil {
			writeKBError(w, err, "Failed to verify KB entry")
			return
		}
	case "approve_quarantine":
		if err := memory.ApproveQuarantined(ref, emb); err != nil {
			writeKBError(w, err, "Failed to approve quarantined KB entry")
			return
		}
	case "reject_quarantine":
		if err := memory.RejectQuarantined(ref); err != nil {
			writeKBError(w, err, "Failed to reject quarantined KB entry")
			return
		}
	default:
		writeJSONStatusError(w, http.StatusBadRequest, "invalid action: must be forget, restore, verify, approve_quarantine, or reject_quarantine")
		return
	}

	realtime.Notify(realtime.KnowledgeBase)

	entry, err := memory.Peek(ref)
	if err != nil && !errs.HasCode(err, errs.CodeNotFound) {
		logger.Warn("Failed to reload KB entry after action", "ref", ref, "action", action, "error", err)
	}

	out := map[string]any{
		"status": "ok",
		"action": action,
		"ref":    ref,
		"entry":  entry,
	}
	if result != nil {
		out["result"] = result
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(out)
}

func handleKBExport(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSONStatusError(w, http.StatusMethodNotAllowed, "GET required")
		return
	}
	if !requireContextDB(w) {
		return
	}

	in, err := kbQueryInputFromRequest(r)
	if err != nil {
		writeKBError(w, err, "Invalid KB export parameters")
		return
	}
	format := strings.ToLower(strings.TrimSpace(r.URL.Query().Get("format")))
	if format == "" {
		format = "json"
	}

	data, err := memory.ExportEntries(in, format)
	if err != nil {
		writeKBError(w, err, "Failed to export KB entries")
		return
	}

	if format == "markdown" || format == "md" {
		w.Header().Set("Content-Type", "text/markdown; charset=utf-8")
		w.Header().Set("Content-Disposition", `attachment; filename="knowledge-base.md"`)
	} else {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Content-Disposition", `attachment; filename="knowledge-base.json"`)
	}
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(data)
}

type contextNoteListItem struct {
	Ref               string `json:"ref"`
	SessionID         string `json:"session_id"`
	ProjectPath       string `json:"project_path,omitempty"`
	Label             string `json:"label,omitempty"`
	Snippet           string `json:"snippet"`
	Tags              string `json:"tags,omitempty"`
	Kind              string `json:"kind,omitempty"`
	MetadataJSON      string `json:"metadata_json,omitempty"`
	TokenEst          int    `json:"virtual_tokens"`
	AccessCount       int    `json:"access_count"`
	Revision          int    `json:"revision"`
	CreatedAt         string `json:"created_at,omitempty"`
	LastAccessedAt    string `json:"last_accessed_at,omitempty"`
	HandoffTreeActive bool   `json:"handoff_tree_active"`
}

type contextNoteDetail struct {
	*contextnotes.Note
	Revision int `json:"revision"`
}

type contextNoteRevisionRow struct {
	Revision  int    `json:"revision"`
	Content   string `json:"content"`
	TokenEst  int    `json:"token_est"`
	Op        string `json:"op"`
	CreatedAt string `json:"created_at"`
}

func handleContextNotesJSON(w http.ResponseWriter, r *http.Request) {
	if !requireContextDB(w) {
		return
	}
	switch r.Method {
	case http.MethodGet:
		listContextNotesJSON(w, r)
	case http.MethodDelete:
		deleteContextNoteJSON(w, r)
	default:
		writeJSONStatusError(w, http.StatusMethodNotAllowed, "GET or DELETE required")
	}
}

func listContextNotesJSON(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	page, perPage, err := parsePageParams(r, kbDefaultPerPage, kbMaxPerPage)
	if err != nil {
		writeKBError(w, err, "Invalid context-notes pagination parameters")
		return
	}

	var where []string
	var args []any

	if sid := strings.TrimSpace(q.Get("session_id")); sid != "" {
		where = append(where, "cn.session_id = ?")
		args = append(args, sid)
	}
	if pid := strings.TrimSpace(q.Get("project_id")); pid != "" {
		where = append(where, "cn.project_path = ?")
		args = append(args, pid)
	}
	if kind := strings.TrimSpace(q.Get("kind")); kind != "" {
		where = append(where, "COALESCE(cn.kind, '') = ?")
		args = append(args, kind)
	}
	if tags := parseCSVParam(q.Get("tags")); len(tags) > 0 {
		var tagParts []string
		for _, t := range tags {
			lt := strings.ToLower(t)
			jsonPat := `%"` + escapeSQLLike(lt) + `"%`
			csvPat := "%," + escapeSQLLike(lt) + ",%"
			tagParts = append(tagParts, `(lower(COALESCE(cn.tags,'')) LIKE ? ESCAPE '\' OR ',' || REPLACE(lower(COALESCE(cn.tags,'')), ' ', '') || ',' LIKE ? ESCAPE '\')`)
			args = append(args, jsonPat, csvPat)
		}
		where = append(where, "("+strings.Join(tagParts, " OR ")+")")
	}
	if searchQ := strings.TrimSpace(q.Get("q")); searchQ != "" {
		for _, term := range strings.Fields(searchQ) {
			pat := "%" + escapeSQLLike(term) + "%"
			where = append(where, `(cn.ref LIKE ? ESCAPE '\' OR COALESCE(cn.label,'') LIKE ? ESCAPE '\' OR cn.content LIKE ? ESCAPE '\' OR COALESCE(cn.tags,'') LIKE ? ESCAPE '\' OR cn.session_id LIKE ? ESCAPE '\')`)
			args = append(args, pat, pat, pat, pat, pat)
		}
	}

	whereSQL := ""
	if len(where) > 0 {
		whereSQL = " WHERE " + strings.Join(where, " AND ")
	}

	var total int
	countQ := "SELECT COUNT(*) FROM context_notes cn" + whereSQL
	if err := db.ContextDB.QueryRow(countQ, args...).Scan(&total); err != nil {
		logger.Warn("Failed to count context notes", "error", err)
		writeJSONStatusError(w, http.StatusInternalServerError, "failed to count context notes")
		return
	}

	offset := (page - 1) * perPage
	listQ := `SELECT cn.ref, cn.session_id, COALESCE(cn.project_path,''), COALESCE(cn.label,''),
		SUBSTR(cn.content, 1, 800), COALESCE(cn.tags,''), COALESCE(cn.kind,''), COALESCE(cn.metadata_json,''),
		COALESCE(cn.token_est, 0), COALESCE(cn.access_count, 0), COALESCE(cn.revision, 1),
		cn.created_at, COALESCE(cn.last_accessed_at,'')
		FROM context_notes cn` + whereSQL + ` ORDER BY cn.created_at DESC, cn.ref DESC LIMIT ? OFFSET ?`
	listArgs := append(append([]any{}, args...), perPage, offset)

	rows, err := db.ContextDB.Query(listQ, listArgs...)
	if err != nil {
		logger.Warn("Failed to query context notes", "error", err)
		writeJSONStatusError(w, http.StatusInternalServerError, "failed to list context notes")
		return
	}
	defer rows.Close()

	items := []contextNoteListItem{}
	for rows.Next() {
		var item contextNoteListItem
		var preview string
		if scanErr := rows.Scan(
			&item.Ref, &item.SessionID, &item.ProjectPath, &item.Label,
			&preview, &item.Tags, &item.Kind, &item.MetadataJSON,
			&item.TokenEst, &item.AccessCount, &item.Revision,
			&item.CreatedAt, &item.LastAccessedAt,
		); scanErr != nil {
			logger.Warn("Failed to scan context note row", "error", scanErr)
			continue
		}
		item.Snippet = makeSnippet(preview, snippetMaxRunes)
		if item.TokenEst == 0 && preview != "" {
			item.TokenEst = db.EstimateTokens(preview)
		}
		if item.Kind == contextnotes.KindHandoffResult {
			item.HandoffTreeActive = handoffTreeActiveForNote(item.Ref, item.SessionID)
		}
		items = append(items, item)
	}
	if err := rows.Err(); err != nil {
		logger.Warn("Failed to iterate context notes", "error", err)
		writeJSONStatusError(w, http.StatusInternalServerError, "failed to iterate context notes")
		return
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"items":       items,
		"total":       total,
		"page":        page,
		"per_page":    perPage,
		"total_pages": totalPagesCount(total, perPage),
	})
}

func deleteContextNoteJSON(w http.ResponseWriter, r *http.Request) {
	ref := strings.TrimSpace(r.URL.Query().Get("ref"))
	if ref == "" {
		writeJSONStatusError(w, http.StatusBadRequest, "ref required")
		return
	}
	if _, err := contextnotes.Peek(ref); err != nil {
		writeKBError(w, err, "Failed to look up context note")
		return
	}
	res, err := contextnotes.Flush("", []string{ref}, "", false)
	if err != nil {
		writeKBError(w, err, "Failed to delete context note")
		return
	}
	realtime.Notify(realtime.KnowledgeBase | realtime.Stats)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"status":               "deleted",
		"ref":                  ref,
		"flushed_refs":         res.FlushedRefs,
		"virtual_tokens_freed": res.VirtualTokensFreed,
		"scope":                res.Scope,
		"stats":                res.Stats,
	})
}

func handleContextNoteDetailJSON(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSONStatusError(w, http.StatusMethodNotAllowed, "GET required")
		return
	}
	if !requireContextDB(w) {
		return
	}

	ref := strings.TrimSpace(r.URL.Query().Get("ref"))
	if ref == "" {
		writeJSONStatusError(w, http.StatusBadRequest, "ref required")
		return
	}
	note, err := contextnotes.Peek(ref)
	if err != nil {
		writeKBError(w, err, "Failed to load context note detail")
		return
	}

	rev := 1
	_ = db.ContextDB.QueryRow(`SELECT COALESCE(revision, 1) FROM context_notes WHERE ref = ?`, ref).Scan(&rev)

	revisions := []contextNoteRevisionRow{}
	rows, err := db.ContextDB.Query(
		`SELECT revision, content, COALESCE(token_est, 0), COALESCE(op, ''), created_at
		FROM context_note_revisions WHERE ref = ? ORDER BY revision DESC`,
		ref,
	)
	if err != nil {
		logger.Warn("Failed to query context note revisions", "ref", ref, "error", err)
		writeJSONStatusError(w, http.StatusInternalServerError, "failed to query context note revisions")
		return
	}
	defer rows.Close()
	for rows.Next() {
		var rr contextNoteRevisionRow
		if scanErr := rows.Scan(&rr.Revision, &rr.Content, &rr.TokenEst, &rr.Op, &rr.CreatedAt); scanErr == nil {
			revisions = append(revisions, rr)
		}
	}
	if err := rows.Err(); err != nil {
		logger.Warn("Failed to iterate context note revisions", "ref", ref, "error", err)
		writeJSONStatusError(w, http.StatusInternalServerError, "failed to iterate context note revisions")
		return
	}

	activeTree := note.Kind == contextnotes.KindHandoffResult && handoffTreeActiveForNote(note.Ref, note.SessionID)

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"note":                contextNoteDetail{Note: note, Revision: rev},
		"revisions":           revisions,
		"active_handoff_tree": activeTree,
	})
}

func handleDocEntriesJSON(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSONStatusError(w, http.StatusMethodNotAllowed, "GET required")
		return
	}
	if !requireContextDB(w) {
		return
	}

	q := r.URL.Query()
	sourceIDStr := strings.TrimSpace(q.Get("source_id"))
	if sourceIDStr == "" {
		writeJSONStatusError(w, http.StatusBadRequest, "source_id required")
		return
	}
	sourceID, err := strconv.Atoi(sourceIDStr)
	if err != nil || sourceID <= 0 {
		writeJSONStatusError(w, http.StatusBadRequest, "source_id must be a positive integer")
		return
	}
	page, perPage, err := parsePageParams(r, kbDefaultPerPage, kbMaxPerPage)
	if err != nil {
		writeKBError(w, err, "Invalid doc-entries pagination parameters")
		return
	}

	entries, err := docs.ListEntriesBySource(sourceID)
	if err != nil {
		logger.Warn("Failed to list doc entries", "source_id", sourceID, "error", err)
		writeJSONStatusError(w, http.StatusInternalServerError, "failed to list doc entries")
		return
	}
	if len(entries) == 0 {
		var exists int
		if checkErr := db.ContextDB.QueryRow(`SELECT 1 FROM doc_sources WHERE id = ?`, sourceID).Scan(&exists); checkErr != nil {
			writeJSONStatusError(w, http.StatusNotFound, "doc source not found")
			return
		}
	}

	filterQ := strings.ToLower(strings.TrimSpace(q.Get("q")))
	filtered := entries
	if filterQ != "" {
		filtered = make([]docs.DocEntry, 0, len(entries))
		for _, e := range entries {
			if strings.Contains(strings.ToLower(e.Title), filterQ) ||
				strings.Contains(strings.ToLower(e.Content), filterQ) ||
				strings.Contains(strings.ToLower(e.Path), filterQ) {
				filtered = append(filtered, e)
			}
		}
	}

	total := len(filtered)
	offset := (page - 1) * perPage
	pageEntries := []docs.DocEntry{}
	if offset < total {
		end := offset + perPage
		if end > total {
			end = total
		}
		pageEntries = filtered[offset:end]
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"source_id":   sourceID,
		"entries":     pageEntries,
		"total":       total,
		"page":        page,
		"per_page":    perPage,
		"total_pages": totalPagesCount(total, perPage),
	})
}

func handoffTreeActiveForNote(ref, sessionID string) bool {
	if db.ContextDB == nil {
		return false
	}
	cutoff := db.SQLTime(time.Now().UTC().Add(-handoff.LoadLimits().TTL()))
	var active int
	err := db.ContextDB.QueryRow(`
		SELECT 1
		FROM handoff_trees ht
		JOIN handoff_children hc ON hc.tree_id = ht.tree_id
		WHERE ht.last_access_at >= ?
		  AND (
			hc.child_session_id = ?
			OR hc.result_ref = ?
			OR EXISTS (
				SELECT 1 FROM handoff_results hr
				WHERE hr.result_ref = ? AND hr.child_session_id = hc.child_session_id
			)
		  )
		LIMIT 1`,
		cutoff, sessionID, ref, ref,
	).Scan(&active)
	return err == nil && active == 1
}

func makeSnippet(content string, maxRunes int) string {
	s := strings.TrimSpace(strings.Join(strings.Fields(content), " "))
	if maxRunes <= 0 || utf8.RuneCountInString(s) <= maxRunes {
		return s
	}
	var b strings.Builder
	count := 0
	for _, r := range s {
		if count >= maxRunes {
			break
		}
		b.WriteRune(r)
		count++
	}
	return strings.TrimSpace(b.String()) + "…"
}

func escapeSQLLike(s string) string {
	s = strings.ReplaceAll(s, `\`, `\\`)
	s = strings.ReplaceAll(s, `%`, `\%`)
	s = strings.ReplaceAll(s, `_`, `\_`)
	return s
}

func totalPagesCount(total, perPage int) int {
	if total <= 0 || perPage <= 0 {
		return 0
	}
	return (total + perPage - 1) / perPage
}

func nonNilStrings(s []string) []string {
	if s == nil {
		return []string{}
	}
	return s
}
