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
	insertEntryQuery = `INSERT INTO structured_memory
			(ref, kind, scope, session_id, project_path, subject, predicate, object, rule, source_ref, token_est,
			 world_valid_at, world_invalid_at, expires_at)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
	selectActiveFactRefsQuery = `SELECT ref FROM structured_memory WHERE kind = 'fact' AND subject = ? AND predicate = ? AND (valid_until IS NULL OR valid_until = '')`
	andSessionScopeClause     = ` AND scope = 'session' AND session_id = ?`
	andProjectScopeClause     = ` AND scope = 'project' AND project_path = ?`
	andGlobalScopeClause      = ` AND scope = 'global'`
	// MEM-1: all invalidation routes record a reason (contradiction, user_forget,
	// ttl, stale_anchor, quarantine, restored) — rows are never deleted on invalidate.
	supersedeEntryQuery = `UPDATE structured_memory SET valid_until = datetime('now'), superseded_by = ?, invalidated_reason = ? WHERE ref = ?`
	// Entry queries now select the v2 columns too. Old rows read as zero values.
	selectEntryByRefQuery = `SELECT ref, kind, scope, session_id, project_path, subject, predicate, object, rule,
		valid_from, valid_until, superseded_by, source_ref, token_est, access_count, last_accessed_at, created_at,
		invalidated_reason, world_valid_at, world_invalid_at, expires_at
		FROM structured_memory WHERE ref = ?`
)

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

// StoreInput is the payload for store_memory.
// MEM-5/MEM-8: optional world-time and TTL fields. valid_at/invalid_at are world-time
// anchors (when the fact was/is true in the world); ttl is an absolute TTL.
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
	ExpiresAt          string // MEM-8: absolute expires_at; if set overrides TTL
}

// StoreResult is returned from Store.
type StoreResult struct {
	Ref                 string   `json:"ref"`
	Kind                Kind     `json:"kind"`
	VirtualTokensStored int      `json:"virtual_tokens_stored"`
	InvalidatedRefs     []string `json:"invalidated_refs,omitempty"`
	Line                string   `json:"line"`
}

func newRef(prefix string) (string, error) {
	b := make([]byte, 6)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return prefix + hex.EncodeToString(b), nil
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

// Store saves a fact or procedure. With InvalidatePrevious a fact supersedes the active facts
// sharing its subject, predicate and scope.
func Store(in StoreInput) (*StoreResult, error) {
	in.SessionID = strings.TrimSpace(in.SessionID)
	in.ProjectPath = strings.TrimSpace(in.ProjectPath)
	scope, err := normalizeScope(in.Scope, in.SessionID, in.ProjectPath)
	if err != nil {
		return nil, err
	}
	var entry Entry
	switch in.Kind {
	case KindFact:
		in.Subject = strings.TrimSpace(in.Subject)
		in.Predicate = strings.TrimSpace(in.Predicate)
		in.Object = strings.TrimSpace(in.Object)
		if in.Predicate == "" {
			in.Predicate = "is"
		}
		if in.Subject == "" || in.Object == "" {
			return nil, errs.NewCode(errs.CodeInvalidInput, "subject and object required for fact")
		}
		entry = Entry{
			Kind:        KindFact,
			Scope:       scope,
			SessionID:   in.SessionID,
			ProjectPath: in.ProjectPath,
			Subject:     in.Subject,
			Predicate:   in.Predicate,
			Object:      in.Object,
			SourceRef:   in.SourceRef,
		}
	case KindProcedure:
		in.Rule = strings.TrimSpace(in.Rule)
		if in.Rule == "" {
			return nil, errs.NewCode(errs.CodeInvalidInput, "rule required for procedure")
		}
		entry = Entry{
			Kind:        KindProcedure,
			Scope:       scope,
			SessionID:   in.SessionID,
			ProjectPath: in.ProjectPath,
			Rule:        in.Rule,
			SourceRef:   in.SourceRef,
		}
	default:
		return nil, errs.NewCode(errs.CodeInvalidInput, "kind must be fact or procedure")
	}
	ref, err := newRef("mem_")
	if err != nil {
		return nil, err
	}
	entry.Ref = ref
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
		expiresAt = db.SQLTime(time.Now().UTC().AddDate(0, 0, in.TTL))
	}

	insert := func() error {
		_, err := db.ContextDB.Exec(insertEntryQuery,
			ref, string(entry.Kind), string(scope), nullIfEmpty(in.SessionID), nullIfEmpty(in.ProjectPath),
			nullIfEmpty(entry.Subject), nullIfEmpty(entry.Predicate), nullIfEmpty(entry.Object), nullIfEmpty(entry.Rule),
			nullIfEmpty(entry.SourceRef), entry.TokenEst,
			nullIfEmpty(in.InvalidatedReason),
			nullIfEmpty(worldValidAt),
			nullIfEmpty(worldInvalidAt),
			nullIfEmpty(expiresAt))
		return err
	}
	var invalidated []string
	if in.Kind == KindFact && in.InvalidatePrevious {
		factSupersessionMu.Lock()
		invalidated, _ = invalidateConflicting(scope, in.SessionID, in.ProjectPath, in.Subject, in.Predicate, ref)
		err = insert()
		factSupersessionMu.Unlock()
	} else {
		err = insert()
	}
	if err != nil {
		return nil, err
	}
	indexFTS(ref, entry.Subject, entry.Predicate, entry.Object, entry.Rule)
	return &StoreResult{
		Ref:                 ref,
		Kind:                entry.Kind,
		VirtualTokensStored: entry.TokenEst,
		InvalidatedRefs:     invalidated,
		Line:                FormatLine(entry),
	}, nil
}

func nullIfEmpty(s string) interface{} {
	if s == "" {
		return nil
	}
	return s
}

// StoreExtracted saves all extracted facts/procedures from text.
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
	return results, nil
}

func scanEntry(row interface{ Scan(...any) error }) (Entry, error) {
	var e Entry
	var kind, scope string
	var sessionID, projectPath, subject, predicate, object, rule, sourceRef, validFrom, validUntil, supersededBy, lastAccessed, createdAt interface{}
	var invalidatedReason, worldValidAt, worldInvalidAt, expiresAt interface{}
	err := row.Scan(&e.Ref, &kind, &scope, &sessionID, &projectPath, &subject, &predicate, &object, &rule, &validFrom, &validUntil, &supersededBy, &sourceRef, &e.TokenEst, &e.AccessCount, &lastAccessed, &createdAt, &invalidatedReason, &worldValidAt, &worldInvalidAt, &expiresAt)
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

func entryByRef(ref string) (Entry, error) {
	row := db.ContextDB.QueryRow(selectEntryByRefQuery, ref)
	return scanEntry(row)
}
