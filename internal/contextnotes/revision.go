package contextnotes

import "github.com/coma-toast/ast-context-cache/internal/db"

// Revisions exist because context-as-a-file editing makes stored context mutable:
// every edit writes the body it replaced into context_note_revisions before
// overwriting the live copy, so a bad in-place compaction is one revert away
// instead of a lost note and a dangling ctx_* stub.

const (
	insertRevisionQuery = `INSERT OR REPLACE INTO context_note_revisions (ref, revision, content, token_est, op)
		VALUES (?, ?, ?, ?, ?)`
	selectRevisionQuery = `SELECT content, token_est FROM context_note_revisions WHERE ref = ? AND revision = ?`
	// Revert walks backwards through the newest bodies, so revision numbers only ever
	// increase and a revert never overwrites a history row.
	selectLatestRevisionQuery = `SELECT revision, content, token_est FROM context_note_revisions
		WHERE ref = ? ORDER BY revision DESC LIMIT 1`
	listRevisionsQuery = `SELECT revision, token_est, COALESCE(op,''), created_at
		FROM context_note_revisions WHERE ref = ? ORDER BY revision DESC LIMIT ?`
	pruneRevisionsQuery = `DELETE FROM context_note_revisions WHERE ref = ? AND revision NOT IN (
		SELECT revision FROM context_note_revisions WHERE ref = ? ORDER BY revision DESC LIMIT ?)`
	deleteRevisionsQuery = `DELETE FROM context_note_revisions WHERE ref = ?`
)

const defaultMaxRevisions = 10

// LoadMaxRevisions is how many superseded bodies to retain per note. Revisions are
// recovery, not an audit log: unbounded retention would let an agent that edits one
// note in a loop grow context.db without limit, bypassing the store quotas entirely.
func LoadMaxRevisions() int {
	n := db.SettingInt("context_max_revisions", "AST_CONTEXT_MAX_REVISIONS", defaultMaxRevisions)
	if n < 0 {
		n = 0
	}
	return n
}

type revisionRow struct {
	Revision  int
	Content   string
	TokenEst  int
	Op        string
	CreatedAt string
}

func writeRevision(ref string, revision int, content, op string, tokenEst int) error {
	_, err := db.ContextDB.Exec(insertRevisionQuery, ref, revision, content, tokenEst, op)
	return err
}

// latestRevision returns the most recently superseded body for a note, which is
// what an unqualified revert restores.
func latestRevision(ref string) (revisionRow, bool) {
	var r revisionRow
	err := db.ContextDB.QueryRow(selectLatestRevisionQuery, ref).
		Scan(&r.Revision, &r.Content, &r.TokenEst)
	return r, err == nil
}

func revisionAt(ref string, revision int) (string, int, bool) {
	var content string
	var tokenEst int
	err := db.ContextDB.QueryRow(selectRevisionQuery, ref, revision).Scan(&content, &tokenEst)
	return content, tokenEst, err == nil
}

// ListRevisions returns revision metadata (never content) newest-first.
func ListRevisions(ref string, limit int) []revisionRow {
	if limit <= 0 {
		limit = defaultMaxRevisions
	}
	rows, err := db.ContextDB.Query(listRevisionsQuery, ref, limit)
	if err != nil {
		return nil
	}
	defer rows.Close()
	var out []revisionRow
	for rows.Next() {
		var r revisionRow
		if err := rows.Scan(&r.Revision, &r.TokenEst, &r.Op, &r.CreatedAt); err == nil {
			out = append(out, r)
		}
	}
	return out
}

func pruneRevisions(ref string) {
	max := LoadMaxRevisions()
	if max <= 0 {
		db.ContextDB.Exec(deleteRevisionsQuery, ref)
		return
	}
	db.ContextDB.Exec(pruneRevisionsQuery, ref, ref, max)
}
