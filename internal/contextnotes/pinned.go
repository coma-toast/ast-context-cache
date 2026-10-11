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

const KindPinned = "pinned"
const pinnedMaxTokens = 500

// PinnedNote is a project-scoped pinned note.
type PinnedNote struct {
	Ref       string `json:"ref"`
	ProjectPath string `json:"project_path"`
	Content   string `json:"content"`
	TokenEst  int    `json:"token_est"`
	CreatedAt string `json:"created_at"`
}

// StorePinned stores or updates a pinned note for a project. It enforces a 500-token
// cap (AC19) and keeps one note per project (PIN-1). The change is applied as
// EditRewrite (PIN-1) so the ref and revisions are kept.
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
			`INSERT INTO context_notes (ref, session_id, project_path, label, content, content_hash, tags, kind, metadata_json, token_est, last_accessed_at)
			 VALUES (?, 'pinned', ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))`,
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
		}, nil
	}

	// Update existing pinned note using edit logic (PIN-1: keep ref/revisions)
	var note Note
	err = db.ContextDB.QueryRow(
		`SELECT ref, session_id, COALESCE(project_path,''), COALESCE(label,''), content,
			COALESCE(tags,''), COALESCE(kind,''), COALESCE(metadata_json,''), token_est, access_count, created_at, COALESCE(last_accessed_at,'')
			FROM context_notes WHERE kind = ? AND project_path = ?`,
		KindPinned, projectPath,
	).Scan(&note.Ref, &note.SessionID, &note.ProjectPath, &note.Label, &note.Content,
		&note.Tags, &note.Kind, &note.MetadataJSON, &note.TokenEst, &note.AccessCount,
		&note.CreatedAt, &note.LastAccessed)
	if err != nil {
		return nil, err
	}

	// Apply EditRewrite logic - replace content while keeping ref
	_, err = db.ContextDB.Exec(
		`UPDATE context_notes SET content = ?, content_hash = ?, token_est = ?,
			last_accessed_at = datetime('now') WHERE ref = ?`,
		content, search.ContentHash(content), tokenEst, note.Ref,
	)
	if err != nil {
		return nil, err
	}

	// Update FTS index (PIN-1)
	deleteNoteFTS(note.Ref)
	indexNoteFTS(note.Ref, note.SessionID, note.Label, content)

	return &PinnedNote{
		Ref:         note.Ref,
		ProjectPath: projectPath,
		Content:     content,
		TokenEst:    tokenEst,
		CreatedAt:   note.CreatedAt,
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
	return &note, nil
}

// routePinnedKind routes store_context with kind=pinned to StorePinned.
// Called from contextnotes.Store when kind="pinned".
func routePinnedKind(projectPath, content string) (*PinnedNote, error) {
	return StorePinned(projectPath, content, "store_context")
}
