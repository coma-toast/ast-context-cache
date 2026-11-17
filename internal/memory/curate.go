package memory

import (
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

const (
	updateEntryInPlaceQuery = `UPDATE structured_memory SET
		title = ?, body = ?, subject = ?, predicate = ?, object = ?, rule = ?,
		tags = ?, actor = ?, category = ?, severity = ?, target = ?,
		before_value = ?, after_value = ?, reason = ?, occurred_at = ?,
		verified = ?, verified_at = ?, redacted_fields = ?, token_est = ?
		WHERE ref = ?`
	setVerifiedTrueQuery  = `UPDATE structured_memory SET verified = 1, verified_at = ? WHERE ref = ?`
	setVerifiedFalseQuery = `UPDATE structured_memory SET verified = 0, verified_at = NULL WHERE ref = ?`
	kbSearchCandidateCap  = 500
	kbExportMaxRows       = 10000
)

// UpdateInput configures a partial update or versioned supersession of a memory entry.
type UpdateInput struct {
	Ref         string    `json:"ref"`
	Title       *string   `json:"title,omitempty"`
	Body        *string   `json:"body,omitempty"`
	Subject     *string   `json:"subject,omitempty"`
	Predicate   *string   `json:"predicate,omitempty"`
	Object      *string   `json:"object,omitempty"`
	Rule        *string   `json:"rule,omitempty"`
	Tags        *[]string `json:"tags,omitempty"`
	Actor       *string   `json:"actor,omitempty"`
	Category    *string   `json:"category,omitempty"`
	Severity    *string   `json:"severity,omitempty"`
	Target      *string   `json:"target,omitempty"`
	BeforeValue *string   `json:"before_value,omitempty"`
	AfterValue  *string   `json:"after_value,omitempty"`
	Reason      *string   `json:"reason,omitempty"`
	OccurredAt  *string   `json:"occurred_at,omitempty"`
	Verified    *bool     `json:"verified,omitempty"`
	NewVersion  bool      `json:"new_version,omitempty"`
}

// UpdateEntry edits an existing active memory entry in place or creates a superseding
// revision when NewVersion is true or substantive fields change on a preference or
// config_change. Returns the resulting entry and any secret-redaction warnings.
func UpdateEntry(in UpdateInput, emb embedder.Interface) (*Entry, []string, error) {
	ref := strings.TrimSpace(in.Ref)
	if ref == "" {
		return nil, nil, errs.NewCode(errs.CodeInvalidInput, "ref required")
	}

	existing, err := entryByRef(ref)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, nil, errs.NewCode(errs.CodeNotFound, "memory entry %q not found", ref)
		}
		return nil, nil, errs.WrapCodeMessage(errs.CodeInternal, "failed to load memory entry", err, "ref", ref)
	}
	if existing.ValidUntil != "" {
		return nil, nil, errs.NewCode(errs.CodeInvalidInput, "cannot update invalidated memory entry %q", ref)
	}

	next := existing
	if in.Title != nil {
		next.Title = strings.TrimSpace(*in.Title)
	}
	if in.Body != nil {
		body := strings.TrimSpace(*in.Body)
		if len(body) > maxBodyBytes {
			return nil, nil, errs.NewCode(errs.CodeInvalidInput, "body exceeds 16 KB limit; use store_context for large content and pass ctx_* ref in source_ref")
		}
		next.Body = body
	}
	if in.Subject != nil {
		next.Subject = strings.TrimSpace(*in.Subject)
	}
	if in.Predicate != nil {
		next.Predicate = strings.TrimSpace(*in.Predicate)
	}
	if in.Object != nil {
		next.Object = strings.TrimSpace(*in.Object)
	}
	if in.Rule != nil {
		next.Rule = strings.TrimSpace(*in.Rule)
	}
	if in.Tags != nil {
		next.Tags = normalizeTags(*in.Tags)
	}
	if in.Actor != nil {
		next.Actor = strings.TrimSpace(*in.Actor)
	}
	if in.Category != nil {
		next.Category = strings.TrimSpace(*in.Category)
	}
	if in.Severity != nil {
		next.Severity = strings.ToLower(strings.TrimSpace(*in.Severity))
	}
	if in.Target != nil {
		next.Target = strings.TrimSpace(*in.Target)
	}
	if in.BeforeValue != nil {
		next.BeforeValue = strings.TrimSpace(*in.BeforeValue)
	}
	if in.AfterValue != nil {
		next.AfterValue = strings.TrimSpace(*in.AfterValue)
	}
	if in.Reason != nil {
		next.Reason = strings.TrimSpace(*in.Reason)
	}
	if in.OccurredAt != nil {
		raw := strings.TrimSpace(*in.OccurredAt)
		if raw == "" {
			next.OccurredAt = ""
		} else {
			parsed, err := db.NormalizeSQLTime(raw)
			if err != nil {
				return nil, nil, err
			}
			next.OccurredAt = parsed
		}
	}
	if in.Verified != nil {
		next.Verified = *in.Verified
		if *in.Verified {
			if !existing.Verified || next.VerifiedAt == "" {
				next.VerifiedAt = db.SQLTime(time.Now().UTC())
			}
		} else {
			next.VerifiedAt = ""
		}
	}

	if !IsValidSeverity(next.Severity) {
		return nil, nil, errs.NewCode(errs.CodeInvalidInput, fmt.Sprintf("invalid severity: %s", next.Severity), "severity", next.Severity)
	}
	if next.Kind == KindEvent && next.Severity == "" {
		next.Severity = SeverityInfo
	}

	if err := validateEntryKindFields(&next); err != nil {
		return nil, nil, err
	}

	_, warnings := RedactEntry(&next)
	next.TokenEst = 0
	next.TokenEst = estimateEntryTokens(next)

	substantiveChange := false
	switch next.Kind {
	case KindPreference:
		substantiveChange = next.Subject != existing.Subject ||
			next.Predicate != existing.Predicate ||
			next.Object != existing.Object
	case KindConfigChange:
		substantiveChange = next.Target != existing.Target ||
			next.BeforeValue != existing.BeforeValue ||
			next.AfterValue != existing.AfterValue
	}

	tagsEnc := EncodeTags(next.Tags)
	redactedEnc := EncodeTags(next.RedactedFields)

	if in.NewVersion || substantiveChange {
		newRefStr, err := newRef("mem_")
		if err != nil {
			return nil, nil, err
		}
		next.Ref = newRefStr
		next.ValidUntil = ""
		next.SupersededBy = ""
		next.InvalidatedReason = ""
		next.AccessCount = 0
		next.LastAccessedAt = ""

		factSupersessionMu.Lock()
		tx, err := db.ContextDB.Begin()
		if err != nil {
			factSupersessionMu.Unlock()
			return nil, nil, errs.WrapCodeMessage(errs.CodeInternal, "failed to begin memory version transaction", err, "ref", ref)
		}
		_, err = tx.Exec(insertEntryQuery,
			newRefStr, string(next.Kind), string(next.Scope), nullIfEmpty(next.SessionID), nullIfEmpty(next.ProjectPath),
			nullIfEmpty(next.Subject), nullIfEmpty(next.Predicate), nullIfEmpty(next.Object), nullIfEmpty(next.Rule),
			nullIfEmpty(next.SourceRef), next.TokenEst,
			nullIfEmpty(next.WorldValidAt),
			nullIfEmpty(next.WorldInvalidAt),
			nullIfEmpty(next.ExpiresAt),
			next.Quarantined,
			nullIfEmpty(next.Title),
			nullIfEmpty(next.Body),
			nullIfEmpty(tagsEnc),
			nullIfEmpty(next.Actor),
			nullIfEmpty(next.Category),
			nullIfEmpty(next.Severity),
			nullIfEmpty(next.Target),
			nullIfEmpty(next.BeforeValue),
			nullIfEmpty(next.AfterValue),
			nullIfEmpty(next.Reason),
			nullIfEmpty(next.OccurredAt),
			boolToInt(next.Verified),
			nullIfEmpty(next.VerifiedAt),
			nullIfEmpty(redactedEnc),
		)
		if err != nil {
			tx.Rollback()
			factSupersessionMu.Unlock()
			return nil, nil, errs.WrapCodeMessage(errs.CodeInternal, "failed to insert new memory version", err, "ref", ref)
		}
		if _, err := tx.Exec(supersedeEntryQuery, newRefStr, ReasonSuperseded, existing.Ref); err != nil {
			tx.Rollback()
			factSupersessionMu.Unlock()
			return nil, nil, errs.WrapCodeMessage(errs.CodeInternal, "failed to supersede prior memory version", err, "ref", ref)
		}
		if err := tx.Commit(); err != nil {
			factSupersessionMu.Unlock()
			return nil, nil, errs.WrapCodeMessage(errs.CodeInternal, "failed to commit memory version", err, "ref", ref)
		}
		factSupersessionMu.Unlock()

		deleteFTS(existing.Ref)
		_ = search.Cache.DeleteRefs("memory", []string{memoryVectorKey(existing.Ref)})
		if next.Quarantined == 0 {
			if err := indexEntryFTS(next); err != nil {
				logger.Warn("Failed to index updated memory FTS", "ref", newRefStr, "error", err)
			}
			if emb != nil {
				item := next
				go func() {
					_, _ = EmbedEntryFromEntry(item, emb)
				}()
			}
		}

		created, err := entryByRef(newRefStr)
		if err != nil {
			return &next, warnings, nil
		}
		return &created, warnings, nil
	}

	_, err = db.ContextDB.Exec(updateEntryInPlaceQuery,
		nullIfEmpty(next.Title),
		nullIfEmpty(next.Body),
		nullIfEmpty(next.Subject),
		nullIfEmpty(next.Predicate),
		nullIfEmpty(next.Object),
		nullIfEmpty(next.Rule),
		nullIfEmpty(tagsEnc),
		nullIfEmpty(next.Actor),
		nullIfEmpty(next.Category),
		nullIfEmpty(next.Severity),
		nullIfEmpty(next.Target),
		nullIfEmpty(next.BeforeValue),
		nullIfEmpty(next.AfterValue),
		nullIfEmpty(next.Reason),
		nullIfEmpty(next.OccurredAt),
		boolToInt(next.Verified),
		nullIfEmpty(next.VerifiedAt),
		nullIfEmpty(redactedEnc),
		next.TokenEst,
		existing.Ref,
	)
	if err != nil {
		return nil, nil, errs.WrapCodeMessage(errs.CodeInternal, "failed to update memory entry", err, "ref", existing.Ref)
	}

	if next.Quarantined == 0 {
		if err := indexEntryFTS(next); err != nil {
			logger.Warn("Failed to reindex updated memory FTS", "ref", existing.Ref, "error", err)
		}
		if emb != nil {
			item := next
			go func() {
				_, _ = EmbedEntryFromEntry(item, emb)
			}()
		}
	}

	updated, err := entryByRef(existing.Ref)
	if err != nil {
		return &next, warnings, nil
	}
	return &updated, warnings, nil
}

func validateEntryKindFields(e *Entry) error {
	switch e.Kind {
	case KindFact:
		if e.Predicate == "" {
			e.Predicate = "is"
		}
		if e.Subject == "" || e.Object == "" {
			return errs.NewCode(errs.CodeInvalidInput, "subject and object required for fact")
		}
	case KindProcedure:
		if e.Rule == "" {
			return errs.NewCode(errs.CodeInvalidInput, "rule required for procedure")
		}
	case KindPreference:
		if e.Object == "" && e.Rule != "" {
			e.Object = e.Rule
		}
		if e.Subject == "" {
			e.Subject = "user"
		}
		if e.Predicate == "" {
			e.Predicate = "prefers"
		}
		if e.Subject == "" || e.Object == "" {
			return errs.NewCode(errs.CodeInvalidInput, "subject and object required for preference")
		}
	case KindEvent:
		if e.Title == "" && (e.Subject == "" || e.Object == "") {
			return errs.NewCode(errs.CodeInvalidInput, "title (or subject and object) required for event")
		}
	case KindDiscovery:
		if e.Title == "" && (e.Subject == "" || e.Object == "") {
			return errs.NewCode(errs.CodeInvalidInput, "title (or subject and object) required for discovery")
		}
	case KindConfigChange:
		if e.Target == "" {
			return errs.NewCode(errs.CodeInvalidInput, "target required for config_change")
		}
		if e.AfterValue == "" && e.Body == "" {
			return errs.NewCode(errs.CodeInvalidInput, "after_value or body required for config_change")
		}
	default:
		return errs.NewCode(errs.CodeInvalidInput, "kind must be fact, procedure, preference, event, config_change, or discovery")
	}
	return nil
}

// SetVerified toggles the verified flag and verified_at timestamp on a memory entry.
func SetVerified(ref string, verified bool) error {
	ref = strings.TrimSpace(ref)
	if ref == "" {
		return errs.NewCode(errs.CodeInvalidInput, "ref required")
	}
	var res sql.Result
	var err error
	if verified {
		res, err = db.ContextDB.Exec(setVerifiedTrueQuery, db.SQLTime(time.Now().UTC()), ref)
	} else {
		res, err = db.ContextDB.Exec(setVerifiedFalseQuery, ref)
	}
	if err != nil {
		return errs.WrapCodeMessage(errs.CodeInternal, "failed to update verified status", err, "ref", ref)
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return errs.NewCode(errs.CodeNotFound, "memory entry %q not found", ref)
	}
	return nil
}

// KBQueryInput configures paginated knowledge base browsing, search, and timeline queries.
type KBQueryInput struct {
	Query                    string             `json:"query,omitempty"`
	Kinds                    []Kind             `json:"kinds,omitempty"`
	Scope                    Scope              `json:"scope,omitempty"`
	ProjectPath              string             `json:"project_path,omitempty"`
	IncludeGlobalWithProject bool               `json:"include_global_with_project,omitempty"`
	SessionID                string             `json:"session_id,omitempty"`
	Status                   string             `json:"status,omitempty"` // active (default), archived, quarantined, all
	Tags                     []string           `json:"tags,omitempty"`
	Actor                    string             `json:"actor,omitempty"`
	Since                    string             `json:"since,omitempty"`
	Until                    string             `json:"until,omitempty"`
	VerifiedOnly             bool               `json:"verified_only,omitempty"`
	Sort                     string             `json:"sort,omitempty"` // occurred_desc, created_desc, access_desc
	Page                     int                `json:"page,omitempty"`
	PerPage                  int                `json:"per_page,omitempty"`
	Embedder                 embedder.Interface `json:"-"`
}

// KBCounts reports active entry counts by kind plus active quarantined count.
type KBCounts struct {
	Fact         int `json:"fact"`
	Procedure    int `json:"procedure"`
	Preference   int `json:"preference"`
	Event        int `json:"event"`
	ConfigChange int `json:"config_change"`
	Discovery    int `json:"discovery"`
	Total        int `json:"total"`
	Quarantined  int `json:"quarantined"`
}

// KBQueryResult is the paginated response returned by QueryKB.
type KBQueryResult struct {
	Entries    []Entry  `json:"entries"`
	Total      int      `json:"total"`
	Page       int      `json:"page"`
	PerPage    int      `json:"per_page"`
	TotalPages int      `json:"total_pages"`
	Counts     KBCounts `json:"counts"`
}

// QueryKB searches and lists memory entries for the dashboard Explorer, Timeline, and Preferences views.
func QueryKB(in KBQueryInput) (*KBQueryResult, error) {
	in.Query = strings.TrimSpace(in.Query)
	in.ProjectPath = strings.TrimSpace(in.ProjectPath)
	in.SessionID = strings.TrimSpace(in.SessionID)
	in.Actor = strings.TrimSpace(in.Actor)
	in.Status = strings.ToLower(strings.TrimSpace(in.Status))
	in.Sort = strings.ToLower(strings.TrimSpace(in.Sort))

	if in.Since = strings.TrimSpace(in.Since); in.Since != "" {
		norm, err := db.NormalizeSQLTime(in.Since)
		if err != nil {
			return nil, err
		}
		in.Since = norm
	}
	if in.Until = strings.TrimSpace(in.Until); in.Until != "" {
		norm, err := db.NormalizeSQLTime(in.Until)
		if err != nil {
			return nil, err
		}
		in.Until = norm
	}

	if in.Page <= 0 {
		in.Page = 1
	}
	if in.PerPage <= 0 {
		in.PerPage = 50
	}
	if in.PerPage > 200 {
		in.PerPage = 200
	}

	if err := validateKBStatus(in.Status); err != nil {
		return nil, err
	}
	if err := validateKBSort(in.Sort); err != nil {
		return nil, err
	}

	baseScope, baseScopeArgs, err := kbScopeClause(in, "")
	if err != nil {
		return nil, err
	}

	counts, err := queryKBCounts(baseScope, baseScopeArgs)
	if err != nil {
		return nil, err
	}

	filterSQL, filterArgs, err := kbFilterClauses(in, "")
	if err != nil {
		return nil, err
	}

	if in.Query != "" {
		matched, err := queryKBSearch(in)
		if err != nil {
			return nil, err
		}
		if in.Sort != "" {
			sortKBEntries(matched, in.Sort)
		}
		total := len(matched)
		totalPages := calcTotalPages(total, in.PerPage)
		offset := (in.Page - 1) * in.PerPage
		var pageEntries []Entry
		if offset < total {
			end := offset + in.PerPage
			if end > total {
				end = total
			}
			pageEntries = matched[offset:end]
		} else {
			pageEntries = []Entry{}
		}
		return &KBQueryResult{
			Entries:    pageEntries,
			Total:      total,
			Page:       in.Page,
			PerPage:    in.PerPage,
			TotalPages: totalPages,
			Counts:     counts,
		}, nil
	}

	var total int
	countQ := "SELECT COUNT(*) FROM structured_memory WHERE 1=1" + filterSQL
	if err := db.ContextDB.QueryRow(countQ, filterArgs...).Scan(&total); err != nil {
		return nil, errs.WrapCodeMessage(errs.CodeInternal, "failed to count KB entries", err)
	}

	orderSQL := kbOrderBySQL(in.Sort)
	offset := (in.Page - 1) * in.PerPage
	listQ := "SELECT " + entrySelectCols(true) + " FROM structured_memory WHERE 1=1" + filterSQL + orderSQL + " LIMIT ? OFFSET ?"
	listArgs := append(append([]any{}, filterArgs...), in.PerPage, offset)
	entries, err := queryEntries(listQ, listArgs...)
	if err != nil {
		return nil, errs.WrapCodeMessage(errs.CodeInternal, "failed to query KB entries", err)
	}
	if entries == nil {
		entries = []Entry{}
	}

	return &KBQueryResult{
		Entries:    entries,
		Total:      total,
		Page:       in.Page,
		PerPage:    in.PerPage,
		TotalPages: calcTotalPages(total, in.PerPage),
		Counts:     counts,
	}, nil
}

func validateKBStatus(status string) error {
	switch status {
	case "", "active", "archived", "quarantined", "all":
		return nil
	default:
		return errs.NewCode(errs.CodeInvalidInput, "invalid status: %s", status)
	}
}

func validateKBSort(sortBy string) error {
	switch sortBy {
	case "", "occurred_desc", "created_desc", "access_desc":
		return nil
	default:
		return errs.NewCode(errs.CodeInvalidInput, "invalid sort: %s", sortBy)
	}
}

func calcTotalPages(total, perPage int) int {
	if total <= 0 || perPage <= 0 {
		return 0
	}
	return (total + perPage - 1) / perPage
}

func kbOrderBySQL(sortBy string) string {
	switch sortBy {
	case "occurred_desc":
		return ` ORDER BY COALESCE(NULLIF(occurred_at,''), created_at) DESC, created_at DESC, ref DESC`
	case "access_desc":
		return ` ORDER BY access_count DESC, created_at DESC, ref DESC`
	default:
		return ` ORDER BY created_at DESC, ref DESC`
	}
}

func sortKBEntries(entries []Entry, sortBy string) {
	switch sortBy {
	case "occurred_desc":
		sort.SliceStable(entries, func(i, j int) bool {
			ti := entries[i].OccurredAt
			if ti == "" {
				ti = entries[i].CreatedAt
			}
			tj := entries[j].OccurredAt
			if tj == "" {
				tj = entries[j].CreatedAt
			}
			if ti != tj {
				return ti > tj
			}
			if entries[i].CreatedAt != entries[j].CreatedAt {
				return entries[i].CreatedAt > entries[j].CreatedAt
			}
			return entries[i].Ref > entries[j].Ref
		})
	case "access_desc":
		sort.SliceStable(entries, func(i, j int) bool {
			if entries[i].AccessCount != entries[j].AccessCount {
				return entries[i].AccessCount > entries[j].AccessCount
			}
			if entries[i].CreatedAt != entries[j].CreatedAt {
				return entries[i].CreatedAt > entries[j].CreatedAt
			}
			return entries[i].Ref > entries[j].Ref
		})
	case "created_desc":
		sort.SliceStable(entries, func(i, j int) bool {
			if entries[i].CreatedAt != entries[j].CreatedAt {
				return entries[i].CreatedAt > entries[j].CreatedAt
			}
			return entries[i].Ref > entries[j].Ref
		})
	}
}

func resolveKBProjectPaths(projectPath string) []string {
	if projectPath == "" {
		return nil
	}
	paths := projectlinks.ResolveScopeWithRepoSiblings(projectPath, true)
	if len(paths) == 0 {
		return []string{projectPath}
	}
	return paths
}

func kbScopeClause(in KBQueryInput, prefix string) (string, []any, error) {
	scopeCol := prefix + "scope"
	sessionCol := prefix + "session_id"
	projectCol := prefix + "project_path"
	projFrag, projArgs := projectMatch(projectCol, resolveKBProjectPaths(in.ProjectPath))

	if in.Scope != "" {
		switch in.Scope {
		case ScopeSession:
			if in.SessionID != "" {
				return andFragment + scopeCol + scopeSessionFragment + sessionCol + eqParamFragment, []any{in.SessionID}, nil
			}
			return andFragment + scopeCol + ` = 'session'`, nil, nil
		case ScopeProject:
			if projFrag != "" {
				return andFragment + scopeCol + scopeProjectFragment + andFragment + projFrag, projArgs, nil
			}
			return andFragment + scopeCol + scopeProjectFragment, nil, nil
		case ScopeGlobal:
			return andFragment + scopeCol + scopeGlobalFragment, nil, nil
		default:
			return "", nil, errs.NewCode(errs.CodeInvalidInput, "invalid scope: %s", in.Scope)
		}
	}

	var clause string
	var args []any
	if projFrag != "" {
		if in.IncludeGlobalWithProject {
			clause += andFragment + "(" + projFrag + orFragment + scopeCol + scopeGlobalFragment + ")"
		} else {
			clause += andFragment + projFrag
		}
		args = append(args, projArgs...)
	}
	if in.SessionID != "" {
		clause += andFragment + sessionCol + eqParamFragment
		args = append(args, in.SessionID)
	}
	return clause, args, nil
}

func kbStatusClause(status, prefix string) string {
	switch status {
	case "", "active":
		return andFragment + "(" + prefix + "valid_until IS NULL OR " + prefix + "valid_until = '') AND COALESCE(" + prefix + "quarantined, 0) = 0"
	case "archived":
		return andFragment + prefix + "valid_until IS NOT NULL AND " + prefix + "valid_until != ''"
	case "quarantined":
		return andFragment + "(" + prefix + "valid_until IS NULL OR " + prefix + "valid_until = '') AND COALESCE(" + prefix + "quarantined, 0) = 1"
	case "all":
		return ""
	default:
		return ""
	}
}

func kbFilterClauses(in KBQueryInput, prefix string) (string, []any, error) {
	scopeSQL, args, err := kbScopeClause(in, prefix)
	if err != nil {
		return "", nil, err
	}
	statusSQL := kbStatusClause(in.Status, prefix)
	kindSQL, kindArgs := kindClause(in.Kinds, prefix)
	tagsSQL, tagsArgs := tagsClause(in.Tags, prefix)
	actorSQL, actorArgs := actorClause(in.Actor, prefix)
	trSQL, trArgs := timeRangeClause(in.Since, in.Until, prefix)
	verSQL := verifiedClause(in.VerifiedOnly, prefix)

	args = append(args, kindArgs...)
	args = append(args, tagsArgs...)
	args = append(args, actorArgs...)
	args = append(args, trArgs...)
	return statusSQL + scopeSQL + kindSQL + tagsSQL + actorSQL + trSQL + verSQL, args, nil
}

func queryKBCounts(baseScope string, baseScopeArgs []any) (KBCounts, error) {
	var counts KBCounts
	q := `SELECT kind, COALESCE(quarantined, 0), COUNT(*)
		FROM structured_memory
		WHERE (valid_until IS NULL OR valid_until = '')` + baseScope + `
		GROUP BY kind, COALESCE(quarantined, 0)`
	rows, err := db.ContextDB.Query(q, baseScopeArgs...)
	if err != nil {
		return counts, errs.WrapCodeMessage(errs.CodeInternal, "failed to count KB entries by kind", err)
	}
	defer rows.Close()

	for rows.Next() {
		var kind string
		var quarantined, n int
		if err := rows.Scan(&kind, &quarantined, &n); err != nil {
			return counts, errs.WrapCodeMessage(errs.CodeInternal, "failed to scan KB counts", err)
		}
		if quarantined != 0 {
			counts.Quarantined += n
			continue
		}
		counts.Total += n
		switch Kind(kind) {
		case KindFact:
			counts.Fact += n
		case KindProcedure:
			counts.Procedure += n
		case KindPreference:
			counts.Preference += n
		case KindEvent:
			counts.Event += n
		case KindConfigChange:
			counts.ConfigChange += n
		case KindDiscovery:
			counts.Discovery += n
		}
	}
	return counts, rows.Err()
}

func queryKBSearch(in KBQueryInput) ([]Entry, error) {
	var ftsEntries []Entry
	if ftsQuery := search.BuildFTSQuery(search.QueryTerms(in.Query)); ftsQuery != "" {
		clause, args, err := kbFilterClauses(in, "sm.")
		if err != nil {
			return nil, err
		}
		args = append([]any{ftsKBContentColumnsFilterStart + ftsQuery + ")"}, args...)
		args = append(args, kbSearchCandidateCap)
		q := "SELECT " + entrySelectColsAliased("sm", true) + searchEntriesKBFTSFrom + clause + orderByKBBM25RefLimitClause
		ftsEntries, err = queryEntries(q, args...)
		if err != nil {
			logger.Warn("KB FTS search failed, falling back to LIKE/vector", "error", err)
		}
	}

	clause, filterArgs, err := kbFilterClauses(in, "")
	if err != nil {
		return nil, err
	}
	likeQ := "%" + in.Query + "%"
	likeArgs := make([]any, 0, 11+len(filterArgs)+1)
	for i := 0; i < 11; i++ {
		likeArgs = append(likeArgs, likeQ)
	}
	likeArgs = append(likeArgs, filterArgs...)
	likeArgs = append(likeArgs, kbSearchCandidateCap)
	likeQuery := "SELECT " + entrySelectCols(true) + searchEntriesLikeWhere + clause + kbOrderBySQL(in.Sort) + " LIMIT ?"
	likeEntries, err := queryEntries(likeQuery, likeArgs...)
	if err != nil {
		return nil, errs.WrapCodeMessage(errs.CodeInternal, "failed to run KB LIKE search", err)
	}

	var vecEntries []Entry
	if in.Embedder != nil {
		if vec, err := in.Embedder.EmbedSingle(in.Query); err == nil && len(vec) > 0 {
			scored := search.Cache.SearchMemory(vec, in.SessionID, in.Scope != ScopeSession, kbSearchCandidateCap)
			var refs []string
			for _, s := range scored {
				if ref, _ := s.Data["ref"].(string); ref != "" {
					refs = append(refs, ref)
				}
			}
			if len(refs) > 0 {
				vecArgs := make([]any, 0, len(refs)+len(filterArgs))
				for _, r := range refs {
					vecArgs = append(vecArgs, r)
				}
				vecArgs = append(vecArgs, filterArgs...)
				q := "SELECT " + entrySelectCols(true) + " FROM structured_memory WHERE ref IN (" + strings.TrimSuffix(strings.Repeat("?,", len(refs)), ",") + ")" + clause
				if rows, err := queryEntries(q, vecArgs...); err == nil {
					vecEntries = orderByRefs(rows, refs)
				}
			}
		} else if err != nil {
			logger.Warn("KB vector search failed, using lexical rankings", "error", err)
		}
	}

	fused, _ := fuseEntries(ftsEntries, likeEntries, vecEntries)
	return fused, nil
}

// ExportEntries renders matching knowledge base entries in "markdown" or "json" format.
func ExportEntries(in KBQueryInput, format string) ([]byte, error) {
	format = strings.ToLower(strings.TrimSpace(format))
	if format == "" {
		format = "json"
	}
	switch format {
	case "json", "markdown", "md":
	default:
		return nil, errs.NewCode(errs.CodeInvalidInput, "export format must be 'markdown' or 'json'")
	}

	in.Page = 1
	in.PerPage = 200
	all := make([]Entry, 0)
	for {
		res, err := QueryKB(in)
		if err != nil {
			return nil, err
		}
		all = append(all, res.Entries...)
		if len(all) >= res.Total || len(all) >= kbExportMaxRows || len(res.Entries) == 0 {
			break
		}
		in.Page++
	}

	if format == "json" {
		return json.MarshalIndent(all, "", "  ")
	}
	return []byte(renderEntriesMarkdown(all)), nil
}

func renderEntriesMarkdown(entries []Entry) string {
	var b strings.Builder
	b.WriteString("# Knowledge Base Export\n\n")
	fmt.Fprintf(&b, "Exported %d entries.\n", len(entries))
	for _, e := range entries {
		headline := entryHeadline(e)
		if headline == "" {
			headline = FormatLine(e)
		}
		if headline == "" {
			headline = e.Ref
		}
		fmt.Fprintf(&b, "\n## [%s] %s (`%s`)\n\n", e.Kind, headline, e.Ref)
		var meta []string
		meta = append(meta, "scope: "+string(e.Scope))
		if e.ProjectPath != "" {
			meta = append(meta, "project: `"+e.ProjectPath+"`")
		}
		if e.SessionID != "" {
			meta = append(meta, "session: `"+e.SessionID+"`")
		}
		if e.Verified {
			meta = append(meta, "verified: true")
		}
		if e.Quarantined != 0 {
			meta = append(meta, "quarantined: true")
		}
		if e.Severity != "" {
			meta = append(meta, "severity: "+e.Severity)
		}
		if e.Category != "" {
			meta = append(meta, "category: "+e.Category)
		}
		if e.Actor != "" {
			meta = append(meta, "actor: "+e.Actor)
		}
		if e.Target != "" {
			meta = append(meta, "target: `"+e.Target+"`")
		}
		if len(e.Tags) > 0 {
			meta = append(meta, "tags: "+strings.Join(e.Tags, ", "))
		}
		if e.OccurredAt != "" {
			meta = append(meta, "occurred_at: "+e.OccurredAt)
		}
		if e.CreatedAt != "" {
			meta = append(meta, "created_at: "+e.CreatedAt)
		}
		if e.ValidUntil != "" {
			meta = append(meta, "valid_until: "+e.ValidUntil)
		}
		if e.InvalidatedReason != "" {
			meta = append(meta, "invalidated_reason: "+e.InvalidatedReason)
		}
		if e.SupersededBy != "" {
			meta = append(meta, "superseded_by: `"+e.SupersededBy+"`")
		}
		b.WriteString("- " + strings.Join(meta, " | ") + "\n")
		if e.Subject != "" || e.Predicate != "" || e.Object != "" {
			fmt.Fprintf(&b, "- triple: %s %s %s\n", e.Subject, e.Predicate, e.Object)
		}
		if e.Rule != "" {
			fmt.Fprintf(&b, "- rule: %s\n", e.Rule)
		}
		if e.BeforeValue != "" || e.AfterValue != "" {
			fmt.Fprintf(&b, "- change: `%s` -> `%s`\n", e.BeforeValue, e.AfterValue)
		}
		if e.Reason != "" {
			fmt.Fprintf(&b, "- reason: %s\n", e.Reason)
		}
		if e.Body != "" {
			b.WriteString("\n" + e.Body + "\n")
		}
	}
	return b.String()
}
