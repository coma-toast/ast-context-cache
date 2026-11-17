package memory

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
)

const (
	maxBodyBytes = 16 * 1024

	insertEntryQuery = `INSERT INTO structured_memory
			(ref, kind, scope, session_id, project_path, subject, predicate, object, rule, source_ref, token_est,
			 world_valid_at, world_invalid_at, expires_at, quarantined,
			 title, body, tags, actor, category, severity, target, before_value, after_value, reason, occurred_at,
			 verified, verified_at, redacted_fields)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
	selectActiveFactRefsQuery   = `SELECT ref FROM structured_memory WHERE kind = ? AND subject = ? AND predicate = ? AND (valid_until IS NULL OR valid_until = '')`
	selectActiveTargetRefsQuery = `SELECT ref FROM structured_memory WHERE kind = 'config_change' AND target = ? AND (valid_until IS NULL OR valid_until = '')`
	andSessionScopeClause       = ` AND scope = 'session' AND session_id = ?`
	andProjectScopeClause       = ` AND scope = 'project' AND project_path = ?`
	andGlobalScopeClause        = ` AND scope = 'global'`
	// MEM-1: all invalidation routes record a reason (contradiction, user_forget,
	// ttl, stale_anchor, quarantine, restored, superseded) — rows are never deleted on invalidate.
	supersedeEntryQuery = `UPDATE structured_memory SET valid_until = datetime('now'), superseded_by = NULLIF(?, ''), invalidated_reason = ? WHERE ref = ?`
)

// entrySelectCols returns the comma-separated column list for scanning an Entry.
// When fullTier2 is false, body/before_value/after_value/reason are selected as empty
// strings so Tier-1 recall never reads large Tier-2 payloads from SQLite.
func entrySelectCols(fullTier2 bool) string {
	return entrySelectColsAliased("", fullTier2)
}

// entrySelectColsAliased returns the comma-separated column list qualified with alias
// (e.g. "sm") for scanning an Entry in joined queries.
func entrySelectColsAliased(alias string, fullTier2 bool) string {
	p := ""
	if alias != "" {
		p = strings.TrimSuffix(alias, ".") + "."
	}
	bodyCol := "''"
	beforeCol := "''"
	afterCol := "''"
	reasonCol := "''"
	if fullTier2 {
		bodyCol = p + "body"
		beforeCol = p + "before_value"
		afterCol = p + "after_value"
		reasonCol = p + "reason"
	}
	return fmt.Sprintf(
		"%sref, %skind, %sscope, %ssession_id, %sproject_path, %ssubject, %spredicate, %sobject, %srule, "+
			"%svalid_from, %svalid_until, %ssuperseded_by, %ssource_ref, %stoken_est, %saccess_count, %slast_accessed_at, %screated_at, "+
			"%sinvalidated_reason, %sworld_valid_at, %sworld_invalid_at, %sexpires_at, %squarantined, "+
			"%stitle, %s, %stags, %sactor, %scategory, %sseverity, %starget, %s, %s, %s, "+
			"%soccurred_at, %sverified, %sverified_at, %sredacted_fields",
		p, p, p, p, p, p, p, p, p,
		p, p, p, p, p, p, p, p,
		p, p, p, p, p,
		p, bodyCol, p, p, p, p, p, beforeCol, afterCol, reasonCol,
		p, p, p, p,
	)
}

// Invalidation reasons (MEM-1). They are stored in invalidated_reason so the
// history chain can explain why any row stopped being active.
const (
	ReasonContradiction = "contradiction"
	ReasonUserForget    = "user_forget"
	ReasonTTL           = "ttl"
	ReasonStaleAnchor   = "stale_anchor"
	ReasonQuarantine    = "quarantine"
	ReasonRestored      = "restored"
)

// factSupersessionMu serializes the read-then-write fact-supersession sequence
// in Store: without it, two concurrent store_memory calls for the same
// subject/predicate/scope can both see the same "no active fact yet" state and
// both insert, leaving duplicate active facts (invalidateConflicting's own
// SELECT wouldn't see the other call's not-yet-committed INSERT).
var factSupersessionMu sync.Mutex

// normalizeTime parses and normalizes a client value; an empty string means "use
// system time" for that bound.
func normalizeTime(v string) string {
	if v == "" {
		return ""
	}
	if t, err := time.Parse(time.RFC3339, v); err == nil {
		return db.SQLTime(t.UTC())
	}
	// Fall back to naive parsing if RFC3339 failed.
	if t, err := time.ParseInLocation("2006-01-02", v, time.Local); err == nil {
		return db.SQLTime(t)
	}
	// Any other parseable datetime form via SQLite's datetime() compatibility.
	if t, err := time.Parse(time.RFC3339Nano, v); err == nil {
		return db.SQLTime(t.UTC())
	}
	return ""
}

// StoreInput is the payload for store_memory.
// MEM-5/MEM-8: optional world-time and TTL fields. valid_at/invalid_at are world-time
// anchors (when the fact was/is true in the world); ttl is an absolute TTL.
type StoreInput struct {
	Kind               Kind
	Scope              Scope
	SessionID          string
	ProjectPath        string
	Subject            string
	Predicate          string
	Object             string
	Rule               string
	SourceRef          string
	InvalidatePrevious bool
	WorldValidAt       string // MEM-5: world_valid_at; normalized, default system time
	WorldInvalidAt     string // MEM-5: world_invalid_at; normalized, default system time
	TTL                int    // MEM-8: TTL in days (Nd); >0 writes expires_at
	TTLHours           int    // MEM-8: TTL hours (Nh); preserved with minute precision
	TTLMinutes         int    // MEM-8: TTL minutes (Nm); preserved with minute precision
	ExpiresAt          string // MEM-8: absolute expires_at; if set overrides TTL
	Title              string
	Body               string
	Tags               []string
	Actor              string
	Category           string
	Severity           string
	Target             string
	BeforeValue        string
	AfterValue         string
	Reason             string
	OccurredAt         string
	Verified           bool
}

// StoreResult is returned from Store.
type StoreResult struct {
	Ref                 string   `json:"ref"`
	Kind                Kind     `json:"kind"`
	VirtualTokensStored int      `json:"virtual_tokens_stored"`
	InvalidatedRefs     []string `json:"invalidated_refs,omitempty"`
	Line                string   `json:"line"`
	Warnings            []string `json:"warnings,omitempty"`
	RedactedFields      []string `json:"redacted_fields,omitempty"`
}

func newRef(prefix string) (string, error) {
	b := make([]byte, 6)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return prefix + hex.EncodeToString(b), nil
}

func defaultScopeForKind(k Kind, sessionID, projectPath string) Scope {
	switch k {
	case KindPreference:
		if projectPath != "" {
			return ScopeProject
		}
		return ScopeGlobal
	case KindEvent, KindDiscovery, KindConfigChange:
		if projectPath != "" {
			return ScopeProject
		}
		if sessionID == "" {
			return ScopeGlobal
		}
		return ScopeSession
	default:
		return ScopeSession
	}
}

func normalizeScope(s Scope, sessionID, projectPath string) (Scope, error) {
	switch s {
	case "", ScopeSession:
		if strings.TrimSpace(sessionID) == "" {
			return "", errs.NewCode(errs.CodeInvalidInput, "session_id required for session scope")
		}
		return ScopeSession, nil
	case ScopeProject:
		if strings.TrimSpace(projectPath) == "" {
			return "", errs.NewCode(errs.CodeInvalidInput, "project_path required for project scope")
		}
		return ScopeProject, nil
	case ScopeGlobal:
		return ScopeGlobal, nil
	default:
		return "", errs.NewCode(errs.CodeInvalidInput, fmt.Sprintf("invalid scope: %s", s), "scope", s)
	}
}

func normalizeTags(tags []string) []string {
	if len(tags) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(tags))
	var out []string
	for _, raw := range tags {
		t := strings.ToLower(strings.TrimSpace(raw))
		if t == "" {
			continue
		}
		if _, ok := seen[t]; ok {
			continue
		}
		seen[t] = struct{}{}
		out = append(out, t)
	}
	return out
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

// Store saves a structured memory entry. With InvalidatePrevious, facts and preferences
// supersede active entries sharing kind, subject, predicate, and scope. Config changes
// supersede active entries sharing target and scope.
func Store(in StoreInput) (*StoreResult, error) {
	if len(in.Body) > maxBodyBytes {
		return nil, errs.NewCode(errs.CodeInvalidInput, "body exceeds 16 KB limit; use store_context for large content and pass ctx_* ref in source_ref")
	}

	in.SessionID = strings.TrimSpace(in.SessionID)
	in.ProjectPath = strings.TrimSpace(in.ProjectPath)
	in.SourceRef = strings.TrimSpace(in.SourceRef)
	in.Subject = strings.TrimSpace(in.Subject)
	in.Predicate = strings.TrimSpace(in.Predicate)
	in.Object = strings.TrimSpace(in.Object)
	in.Rule = strings.TrimSpace(in.Rule)
	in.Title = strings.TrimSpace(in.Title)
	in.Body = strings.TrimSpace(in.Body)
	in.Actor = strings.TrimSpace(in.Actor)
	in.Category = strings.TrimSpace(in.Category)
	in.Severity = strings.ToLower(strings.TrimSpace(in.Severity))
	in.Target = strings.TrimSpace(in.Target)
	in.BeforeValue = strings.TrimSpace(in.BeforeValue)
	in.AfterValue = strings.TrimSpace(in.AfterValue)
	in.Reason = strings.TrimSpace(in.Reason)
	in.OccurredAt = strings.TrimSpace(in.OccurredAt)

	if in.Kind == KindPinned {
		return nil, errs.NewCode(errs.CodeInvalidInput, "pinned notes must be stored via store_context, not store_memory")
	}
	if !IsValidKind(in.Kind) {
		return nil, errs.NewCode(errs.CodeInvalidInput, "kind must be fact, procedure, preference, event, config_change, or discovery")
	}

	if in.Scope == "" {
		in.Scope = defaultScopeForKind(in.Kind, in.SessionID, in.ProjectPath)
	}
	scope, err := normalizeScope(in.Scope, in.SessionID, in.ProjectPath)
	if err != nil {
		return nil, err
	}

	if !IsValidSeverity(in.Severity) {
		return nil, errs.NewCode(errs.CodeInvalidInput, fmt.Sprintf("invalid severity: %s", in.Severity), "severity", in.Severity)
	}
	if in.Kind == KindEvent && in.Severity == "" {
		in.Severity = SeverityInfo
	}

	occurredAt := ""
	if in.OccurredAt != "" {
		parsed, err := db.NormalizeSQLTime(in.OccurredAt)
		if err != nil {
			return nil, err
		}
		occurredAt = parsed
	} else if in.Kind == KindEvent || in.Kind == KindConfigChange {
		occurredAt = db.SQLTime(time.Now().UTC())
	}

	switch in.Kind {
	case KindFact:
		if in.Predicate == "" {
			in.Predicate = "is"
		}
		if in.Subject == "" || in.Object == "" {
			return nil, errs.NewCode(errs.CodeInvalidInput, "subject and object required for fact")
		}
	case KindProcedure:
		if in.Rule == "" {
			return nil, errs.NewCode(errs.CodeInvalidInput, "rule required for procedure")
		}
	case KindPreference:
		if in.Object == "" && in.Rule != "" {
			in.Object = in.Rule
		}
		if in.Subject == "" {
			in.Subject = "user"
		}
		if in.Predicate == "" {
			in.Predicate = "prefers"
		}
		if in.Subject == "" || in.Object == "" {
			return nil, errs.NewCode(errs.CodeInvalidInput, "subject and object required for preference")
		}
	case KindEvent:
		if in.Title == "" && (in.Subject == "" || in.Object == "") {
			return nil, errs.NewCode(errs.CodeInvalidInput, "title (or subject and object) required for event")
		}
	case KindDiscovery:
		if in.Title == "" && (in.Subject == "" || in.Object == "") {
			return nil, errs.NewCode(errs.CodeInvalidInput, "title (or subject and object) required for discovery")
		}
	case KindConfigChange:
		if in.Target == "" {
			return nil, errs.NewCode(errs.CodeInvalidInput, "target required for config_change")
		}
		if in.AfterValue == "" && in.Body == "" {
			return nil, errs.NewCode(errs.CodeInvalidInput, "after_value or body required for config_change")
		}
	}

	ref, err := newRef("mem_")
	if err != nil {
		return nil, err
	}

	var verifiedAt string
	if in.Verified {
		verifiedAt = db.SQLTime(time.Now().UTC())
	}

	entry := Entry{
		Ref:         ref,
		Kind:        in.Kind,
		Scope:       scope,
		SessionID:   in.SessionID,
		ProjectPath: in.ProjectPath,
		Subject:     in.Subject,
		Predicate:   in.Predicate,
		Object:      in.Object,
		Rule:        in.Rule,
		SourceRef:   in.SourceRef,
		Title:       in.Title,
		Body:        in.Body,
		Tags:        normalizeTags(in.Tags),
		Actor:       in.Actor,
		Category:    in.Category,
		Severity:    in.Severity,
		Target:      in.Target,
		BeforeValue: in.BeforeValue,
		AfterValue:  in.AfterValue,
		Reason:      in.Reason,
		OccurredAt:  occurredAt,
		Verified:    in.Verified,
		VerifiedAt:  verifiedAt,
	}

	// Redact secrets across all text fields BEFORE token estimation, SQLite insert, and FTS indexing.
	_, warnings := RedactEntry(&entry)
	entry.TokenEst = estimateEntryTokens(entry)

	// MEM-5/MEM-8: normalize world time and compute expires_at from TTL.
	worldValidAt := normalizeTime(in.WorldValidAt)
	worldInvalidAt := normalizeTime(in.WorldInvalidAt)
	var expiresAt string
	if in.ExpiresAt != "" {
		parsed, err := db.NormalizeSQLTime(in.ExpiresAt)
		if err == nil {
			expiresAt = parsed
		} else {
			// ignore bad expires_at; fall through to TTL below
			expiresAt = ""
		}
	}
	if expiresAt == "" && in.TTL > 0 {
		d := time.Duration(in.TTL) * 24 * time.Hour
		d += time.Duration(in.TTLHours) * time.Hour
		d += time.Duration(in.TTLMinutes) * time.Minute
		expiresAt = db.SQLTime(time.Now().UTC().Add(d))
	}

	tagsEnc := EncodeTags(entry.Tags)
	redactedEnc := EncodeTags(entry.RedactedFields)

	insert := func() error {
		_, err := db.ContextDB.Exec(insertEntryQuery,
			ref, string(entry.Kind), string(scope), nullIfEmpty(entry.SessionID), nullIfEmpty(entry.ProjectPath),
			nullIfEmpty(entry.Subject), nullIfEmpty(entry.Predicate), nullIfEmpty(entry.Object), nullIfEmpty(entry.Rule),
			nullIfEmpty(entry.SourceRef), entry.TokenEst,
			nullIfEmpty(worldValidAt),
			nullIfEmpty(worldInvalidAt),
			nullIfEmpty(expiresAt),
			0, // quarantined = 0 (not quarantined) by default
			nullIfEmpty(entry.Title),
			nullIfEmpty(entry.Body),
			nullIfEmpty(tagsEnc),
			nullIfEmpty(entry.Actor),
			nullIfEmpty(entry.Category),
			nullIfEmpty(entry.Severity),
			nullIfEmpty(entry.Target),
			nullIfEmpty(entry.BeforeValue),
			nullIfEmpty(entry.AfterValue),
			nullIfEmpty(entry.Reason),
			nullIfEmpty(entry.OccurredAt),
			boolToInt(entry.Verified),
			nullIfEmpty(entry.VerifiedAt),
			nullIfEmpty(redactedEnc),
		)
		return err
	}

	var invalidated []string
	switch entry.Kind {
	case KindFact, KindPreference:
		if in.InvalidatePrevious {
			factSupersessionMu.Lock()
			invalidated, _ = invalidateConflicting(entry.Kind, scope, entry.SessionID, entry.ProjectPath, entry.Subject, entry.Predicate, ref)
			err = insert()
			factSupersessionMu.Unlock()
		} else {
			err = insert()
		}
	case KindConfigChange:
		factSupersessionMu.Lock()
		invalidated, _ = invalidateTargetChain(scope, entry.SessionID, entry.ProjectPath, entry.Target, ref)
		err = insert()
		factSupersessionMu.Unlock()
	default:
		err = insert()
	}
	if err != nil {
		return nil, err
	}

	if err := indexEntryFTS(entry); err != nil {
		logger.Warn("Failed to index memory FTS", "ref", ref, "error", err)
	}
	return &StoreResult{
		Ref:                 ref,
		Kind:                entry.Kind,
		VirtualTokensStored: entry.TokenEst,
		InvalidatedRefs:     invalidated,
		Line:                FormatLine(entry),
		Warnings:            warnings,
		RedactedFields:      entry.RedactedFields,
	}, nil
}

func nullIfEmpty(s string) interface{} {
	if s == "" {
		return nil
	}
	return s
}

// StoreExtracted saves all extracted memory entries from text.
func StoreExtracted(sessionID, projectPath, sourceRef string, ex Extracted, scope Scope) ([]StoreResult, error) {
	var results []StoreResult
	for _, f := range ex.Facts {
		res, err := Store(StoreInput{
			Kind:               KindFact,
			Scope:              scope,
			SessionID:          sessionID,
			ProjectPath:        projectPath,
			Subject:            f.Subject,
			Predicate:          f.Predicate,
			Object:             f.Object,
			SourceRef:          sourceRef,
			InvalidatePrevious: true,
		})
		if err != nil {
			continue
		}
		results = append(results, *res)
	}
	for _, p := range ex.Procedures {
		res, err := Store(StoreInput{
			Kind:        KindProcedure,
			Scope:       scope,
			SessionID:   sessionID,
			ProjectPath: projectPath,
			Rule:        p.Rule,
			SourceRef:   sourceRef,
		})
		if err != nil {
			continue
		}
		results = append(results, *res)
	}
	for _, pref := range ex.Preferences {
		res, err := Store(StoreInput{
			Kind:               KindPreference,
			Scope:              scope,
			SessionID:          sessionID,
			ProjectPath:        projectPath,
			Subject:            pref.Subject,
			Predicate:          pref.Predicate,
			Object:             pref.Object,
			SourceRef:          sourceRef,
			InvalidatePrevious: true,
		})
		if err != nil {
			continue
		}
		results = append(results, *res)
	}
	for _, ev := range ex.Events {
		res, err := Store(StoreInput{
			Kind:        KindEvent,
			Scope:       scope,
			SessionID:   sessionID,
			ProjectPath: projectPath,
			Title:       ev.Title,
			Body:        ev.Body,
			Category:    ev.Category,
			Severity:    ev.Severity,
			SourceRef:   sourceRef,
		})
		if err != nil {
			continue
		}
		results = append(results, *res)
	}
	for _, cc := range ex.ConfigChanges {
		res, err := Store(StoreInput{
			Kind:        KindConfigChange,
			Scope:       scope,
			SessionID:   sessionID,
			ProjectPath: projectPath,
			Target:      cc.Target,
			BeforeValue: cc.BeforeValue,
			AfterValue:  cc.AfterValue,
			Reason:      cc.Reason,
			SourceRef:   sourceRef,
		})
		if err != nil {
			continue
		}
		results = append(results, *res)
	}
	for _, d := range ex.Discoveries {
		res, err := Store(StoreInput{
			Kind:        KindDiscovery,
			Scope:       scope,
			SessionID:   sessionID,
			ProjectPath: projectPath,
			Title:       d.Title,
			Body:        d.Body,
			SourceRef:   sourceRef,
		})
		if err != nil {
			continue
		}
		results = append(results, *res)
	}
	return results, nil
}

func scanEntry(row interface{ Scan(...any) error }) (Entry, error) {
	var e Entry
	var kind, scope string
	var sessionID, projectPath, subject, predicate, object, rule, sourceRef, validFrom, validUntil, supersededBy, lastAccessed, createdAt interface{}
	var invalidatedReason, worldValidAt, worldInvalidAt, expiresAt interface{}
	var quarantined interface{}
	var title, body, tags, actor, category, severity, target, beforeValue, afterValue, reason, occurredAt interface{}
	var verified, verifiedAt, redactedFields interface{}
	err := row.Scan(
		&e.Ref, &kind, &scope, &sessionID, &projectPath, &subject, &predicate, &object, &rule,
		&validFrom, &validUntil, &supersededBy, &sourceRef, &e.TokenEst, &e.AccessCount, &lastAccessed, &createdAt,
		&invalidatedReason, &worldValidAt, &worldInvalidAt, &expiresAt, &quarantined,
		&title, &body, &tags, &actor, &category, &severity, &target, &beforeValue, &afterValue, &reason,
		&occurredAt, &verified, &verifiedAt, &redactedFields,
	)
	if err != nil {
		return e, err
	}
	e.Kind = Kind(kind)
	e.Scope = Scope(scope)
	e.SessionID = strVal(sessionID)
	e.ProjectPath = strVal(projectPath)
	e.Subject = strVal(subject)
	e.Predicate = strVal(predicate)
	e.Object = strVal(object)
	e.Rule = strVal(rule)
	e.ValidFrom = strVal(validFrom)
	e.ValidUntil = strVal(validUntil)
	e.SupersededBy = strVal(supersededBy)
	e.SourceRef = strVal(sourceRef)
	e.LastAccessedAt = strVal(lastAccessed)
	e.CreatedAt = strVal(createdAt)
	e.InvalidatedReason = strVal(invalidatedReason)
	e.WorldValidAt = strVal(worldValidAt)
	e.WorldInvalidAt = strVal(worldInvalidAt)
	e.ExpiresAt = strVal(expiresAt)
	e.Quarantined = intVal(quarantined)
	e.Title = strVal(title)
	e.Body = strVal(body)
	e.Tags = DecodeTags(strVal(tags))
	e.Actor = strVal(actor)
	e.Category = strVal(category)
	e.Severity = strVal(severity)
	e.Target = strVal(target)
	e.BeforeValue = strVal(beforeValue)
	e.AfterValue = strVal(afterValue)
	e.Reason = strVal(reason)
	e.OccurredAt = strVal(occurredAt)
	e.Verified = intVal(verified) != 0
	e.VerifiedAt = strVal(verifiedAt)
	e.RedactedFields = DecodeTags(strVal(redactedFields))
	return e, nil
}

func strVal(v interface{}) string {
	if v == nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	return fmt.Sprint(v)
}

func intVal(v interface{}) int {
	if v == nil {
		return 0
	}
	switch n := v.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		return int(n)
	default:
		return 0
	}
}

func entryByRef(ref string) (Entry, error) {
	q := "SELECT " + entrySelectCols(true) + " FROM structured_memory WHERE ref = ?"
	row := db.ContextDB.QueryRow(q, ref)
	return scanEntry(row)
}
