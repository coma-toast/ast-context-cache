package contextnotes

import (
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

const (
	KindPinned      = "pinned"
	pinnedMaxTokens = 500
)

// PinnedNote is a project-scoped pinned note.
type PinnedNote struct {
	Ref         string `json:"ref"`
	ProjectPath string `json:"project_path"`
	Content     string `json:"content"`
	TokenEst    int    `json:"token_est"`
	CreatedAt   string `json:"created_at"`
	Kind        string `json:"kind"`
}

// StorePinned stores or updates a pinned note for a project. It enforces a 500-token
// cap (AC19) and keeps one note per project (PIN-1). Each update is applied as a
// revision write (PIN-1) so the superseded body lands in context_note_revisions
// and the ref stays stable.
func StorePinned(projectPath, content, origin string) (*PinnedNote, error) {
	projectPath = strings.TrimSpace(projectPath)
	if projectPath == "" {
		return nil, errs.NewCode(errs.CodeInvalidInput, "project_path required for pinned note")
	}
	tokenEst := db.EstimateTokens(content)
	if tokenEst > pinnedMaxTokens {
		return nil, errs.NewCode(errs.CodeInvalidInput, fmt.Sprintf("pinned note exceeds max tokens: %d (max %d)", tokenEst, pinnedMaxTokens))
	}

	// Check if a pinned note exists for this project
	var existingRef string
	err := db.ContextDB.QueryRow(
		`SELECT ref FROM context_notes WHERE kind = ? AND project_path = ?`,
		KindPinned, projectPath,
	).Scan(&existingRef)

	if errors.Is(err, sql.ErrNoRows) {
		// New pinned note - insert
		ref, err := newRef()
		if err != nil {
			return nil, err
		}
		hash := search.ContentHash(content)

		_, err = db.ContextDB.Exec(
			`INSERT INTO context_notes (ref, session_id, project_path, label, content, content_hash, tags, kind, metadata_json, token_est, revision, last_accessed_at)
			 VALUES (?, 'pinned', ?, ?, ?, ?, ?, ?, ?, ?, 1, datetime('now'))`,
			ref, projectPath, "pinned", content, hash, "", KindPinned, "{}", tokenEst,
		)
		if err != nil {
			return nil, err
		}
		indexNoteFTS(ref, "pinned", "pinned", content)
		return &PinnedNote{
			Ref:         ref,
			ProjectPath: projectPath,
			Content:     content,
			TokenEst:    tokenEst,
			CreatedAt:   db.SQLTime(time.Now().UTC()),
			Kind:        KindPinned,
		}, nil
	}

	// Update existing pinned note using revision logic (PIN-1: keep ref + revisions)
	var note Note
	var currentRevision int
	err = db.ContextDB.QueryRow(
		`SELECT ref, session_id, COALESCE(project_path,''), COALESCE(label,''), content,
			COALESCE(tags,''), COALESCE(kind,''), COALESCE(metadata_json,''), token_est, access_count, created_at, COALESCE(last_accessed_at,''),
			COALESCE(revision, 1)
			FROM context_notes WHERE kind = ? AND project_path = ?`,
		KindPinned, projectPath,
	).Scan(&note.Ref, &note.SessionID, &note.ProjectPath, &note.Label, &note.Content,
		&note.Tags, &note.Kind, &note.MetadataJSON, &note.TokenEst, &note.AccessCount,
		&note.CreatedAt, &note.LastAccessed, &currentRevision)
	if err != nil {
		return nil, err
	}

	// Write the superseded body as a revision before overwriting (PIN-1).
	if err := writeRevision(note.Ref, currentRevision, note.Content, "pinned-update", note.TokenEst); err != nil {
		return nil, errs.WrapMessage("failed to record pinned note revision", err, "ref", note.Ref)
	}
	hash := search.ContentHash(content)
	if _, err := db.ContextDB.Exec(updateNoteContentQuery, content, hash, tokenEst, currentRevision+1, note.Ref); err != nil {
		return nil, errs.WrapMessage("failed to update pinned note content", err, "ref", note.Ref)
	}
	reindexNoteFTS(note.Ref, note.SessionID, note.Label, content)
	pruneRevisions(note.Ref)

	return &PinnedNote{
		Ref:         note.Ref,
		ProjectPath: projectPath,
		Content:     content,
		TokenEst:    tokenEst,
		CreatedAt:   note.CreatedAt,
		Kind:        note.Kind,
	}, nil
}

// Pinned returns the pinned note for a project (PIN-1).
func Pinned(projectPath string) (*PinnedNote, error) {
	projectPath = strings.TrimSpace(projectPath)
	if projectPath == "" {
		return nil, errs.NewCode(errs.CodeInvalidInput, "project_path required")
	}

	var note PinnedNote
	err := db.ContextDB.QueryRow(
		`SELECT ref, project_path, COALESCE(content,''), token_est, created_at
		 FROM context_notes WHERE kind = ? AND project_path = ?`,
		KindPinned, projectPath,
	).Scan(&note.Ref, &note.ProjectPath, &note.Content, &note.TokenEst, &note.CreatedAt)
	if err != nil {
		return nil, err
	}
	note.Kind = KindPinned
	return &note, nil
}

// routePinnedKind routes store_context with kind=pinned to StorePinned.
// Called from contextnotes.Store when kind="pinned".
func routePinnedKind(projectPath, content string) (*PinnedNote, error) {
	return StorePinned(projectPath, content, "store_context")
}
