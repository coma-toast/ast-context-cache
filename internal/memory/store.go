package memory

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

// StoreInput is the payload for store_memory.
type StoreInput struct {
	Kind                Kind
	Scope               Scope
	SessionID           string
	ProjectPath         string
	Subject             string
	Predicate           string
	Object              string
	Rule                string
	SourceRef           string
	InvalidatePrevious  bool
}

// StoreResult is returned from Store.
type StoreResult struct {
	Ref                 string `json:"ref"`
	Kind                Kind   `json:"kind"`
	VirtualTokensStored int    `json:"virtual_tokens_stored"`
	InvalidatedRefs     []string `json:"invalidated_refs,omitempty"`
	Line                string `json:"line"`
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
			return "", errors.New("session_id required for session scope")
		}
		return ScopeSession, nil
	case ScopeProject:
		if strings.TrimSpace(projectPath) == "" {
			return "", errors.New("project_path required for project scope")
		}
		return ScopeProject, nil
	case ScopeGlobal:
		return ScopeGlobal, nil
	default:
		return "", fmt.Errorf("invalid scope: %s", s)
	}
}

// Store saves a fact or procedure with optional temporal invalidation of prior facts.
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
			return nil, errors.New("subject and object required for fact")
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
			return nil, errors.New("rule required for procedure")
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
		return nil, errors.New("kind must be fact or procedure")
	}
	ref, err := newRef("mem_")
	if err != nil {
		return nil, err
	}
	entry.Ref = ref
	entry.TokenEst = estimateEntryTokens(entry)
	var invalidated []string
	if in.Kind == KindFact {
		invalidated, _ = invalidateConflicting(scope, in.SessionID, in.ProjectPath, in.Subject, in.Predicate, ref)
	}
	_, err = db.DB.Exec(`INSERT INTO structured_memory
		(ref, kind, scope, session_id, project_path, subject, predicate, object, rule, source_ref, token_est)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		ref, string(entry.Kind), string(scope), nullIfEmpty(in.SessionID), nullIfEmpty(in.ProjectPath),
		nullIfEmpty(entry.Subject), nullIfEmpty(entry.Predicate), nullIfEmpty(entry.Object), nullIfEmpty(entry.Rule),
		nullIfEmpty(entry.SourceRef), entry.TokenEst)
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

func invalidateConflicting(scope Scope, sessionID, projectPath, subject, predicate, newRef string) ([]string, error) {
	q := `SELECT ref FROM structured_memory WHERE kind = 'fact' AND subject = ? AND predicate = ? AND (valid_until IS NULL OR valid_until = '')`
	args := []any{subject, predicate}
	switch scope {
	case ScopeSession:
		q += ` AND scope = 'session' AND session_id = ?`
		args = append(args, sessionID)
	case ScopeProject:
		q += ` AND scope = 'project' AND project_path = ?`
		args = append(args, projectPath)
	case ScopeGlobal:
		q += ` AND scope = 'global'`
	}
	rows, err := db.DB.Query(q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var refs []string
	for rows.Next() {
		var ref string
		if rows.Scan(&ref) == nil && ref != "" && ref != newRef {
			refs = append(refs, ref)
		}
	}
	for _, ref := range refs {
		db.DB.Exec(`UPDATE structured_memory SET valid_until = datetime('now'), superseded_by = ? WHERE ref = ?`, newRef, ref)
	}
	return refs, nil
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
	err := row.Scan(&e.Ref, &kind, &scope, &sessionID, &projectPath, &subject, &predicate, &object, &rule, &validFrom, &validUntil, &supersededBy, &sourceRef, &e.TokenEst, &e.AccessCount, &lastAccessed, &createdAt)
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
	row := db.DB.QueryRow(`SELECT ref, kind, scope, session_id, project_path, subject, predicate, object, rule,
		valid_from, valid_until, superseded_by, source_ref, token_est, access_count, last_accessed_at, created_at
		FROM structured_memory WHERE ref = ?`, ref)
	return scanEntry(row)
}
