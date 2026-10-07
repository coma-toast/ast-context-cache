// Package purge removes every trace of a project from the local caches and
// sweeps for projects whose directory has been deleted from disk.
//
// WTG spaces are created and thrown away constantly, so without this the index
// accumulates rows for checkouts that no longer exist.
package purge

import (
	"database/sql"

	"github.com/coma-toast/ast-context-cache/internal/cache"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedqueue"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
	"github.com/coma-toast/ast-context-cache/internal/projectmeta"
	"github.com/coma-toast/ast-context-cache/internal/search"
	"github.com/coma-toast/ast-context-cache/internal/watcher"
)

const (
	deleteProjectQueriesQuery      = "DELETE FROM queries WHERE project_path = ?"
	deleteProjectMemoryAccessQuery = "DELETE FROM memory_access WHERE project_path = ?"
	deleteSessionsUnderPathQuery   = "DELETE FROM sessions WHERE file_path LIKE ?"
	selectSymbolCountsQuery        = `SELECT (SELECT count(*) FROM symbols WHERE project_path = ?), (SELECT count(*) FROM symbols)`
	deleteProjectSymbolsQuery      = "DELETE FROM symbols WHERE project_path = ?"
	selectNoteRefsQuery            = `SELECT ref FROM context_notes WHERE project_path = ? AND ref != ''`
	selectMemoryRefsQuery          = `SELECT ref FROM structured_memory WHERE project_path = ? AND ref != ''`
	deleteRefVectorQuery           = `DELETE FROM vectors WHERE doc_type = ? AND source_file = ?`
	deleteNoteFTSQuery             = `DELETE FROM context_notes_fts WHERE ref = ?`
	deleteProjectNotesQuery        = `DELETE FROM context_notes WHERE project_path = ?`
	// Revision bodies have no project_path of their own, so they are keyed by the
	// note refs collected for the purge — the same order that clears the FTS mirrors.
	deleteNoteRevisionsQuery         = `DELETE FROM context_note_revisions WHERE ref = ?`
	deleteMemoryFTSQuery             = `DELETE FROM structured_memory_fts WHERE ref = ?`
	deleteProjectMemoryQuery         = `DELETE FROM structured_memory WHERE project_path = ?`
	deleteProjectKVRepairEventsQuery = `DELETE FROM kv_repair_events WHERE project_path = ?`
	deleteFromQueryPrefix            = "DELETE FROM "
	whereProjectPathQuerySuffix      = " WHERE project_path = ?"
	// Handoff rows go whole-tree for trees rooted in the project, plus any handoff or child row
	// that names the project itself. Dependents are deleted before the rows the subqueries read.
	projectTreeIDsSubquery           = `(SELECT tree_id FROM handoff_trees WHERE project_path = ?)`
	whereProjectTreeQuerySuffix      = ` WHERE tree_id IN ` + projectTreeIDsSubquery
	whereProjectOrTreeQuerySuffix    = ` WHERE project_path = ? OR tree_id IN ` + projectTreeIDsSubquery
	deleteProjectSnapshotItemsQuery  = `DELETE FROM handoff_snapshot_items WHERE handoff_ref IN (SELECT ref FROM handoffs` + whereProjectOrTreeQuerySuffix + `)`
	deleteProjectHandoffResultsQuery = `DELETE FROM handoff_results WHERE child_session_id IN (SELECT child_session_id FROM handoff_children` + whereProjectOrTreeQuerySuffix + `)`
	deleteProjectHandoffTreesQuery   = `DELETE FROM handoff_trees WHERE project_path = ?`
)

var (
	handoffTreeKeyedTables = []string{"scratchpad_entries", "handoff_claims", "handoff_claim_queue", "handoff_claim_grants"}
	handoffProjectTables   = []string{"handoff_children", "handoffs"}
)

// ProjectData deletes all indexed and remembered data for projectPath: symbols,
// edges, vectors, indexed files, summaries, query history, context sessions, stored
// notes and structured memory.
//
// This is the permanent-deletion purge. The dashboard's "reset" action reuses it
// because a reset re-indexes from scratch anyway; nothing here is recoverable
// from the project directory alone, so callers must be sure the project is going
// away or being rebuilt.
func ProjectData(projectPath string) error {
	projectPath = watcher.NormalizeProjectPath(projectPath)
	if projectPath == "" {
		return errs.NewCode(errs.CodeInvalidInput, "project_path required")
	}

	embedqueue.RemoveProject(projectPath)

	// Stop any active watcher immediately — otherwise the next file-save event under
	// this directory silently repopulates the index we're about to wipe below, with
	// no explicit tool call involved. Un-pin too, since a still-pinned deleted project
	// gets auto-watched (and thus re-indexed) again the next time ast-mcp starts.
	watcher.DeleteWatcher(projectPath)
	db.TogglePinnedProject(projectPath, false)
	// Tombstone the path so passive filesystem discovery (DiscoverPaths, run at every
	// ast-mcp startup) doesn't re-surface it in the dashboard's project list either.
	// Only an explicit tool call against this exact path should bring it back.
	projectmeta.MarkDeleted(projectPath)
	// Drop parent/child monorepo-container links involving this path — otherwise a
	// stale row survives the delete and, since validateLink refuses to link a parent
	// that's "already linked under another container", can block a legitimate new
	// link from ever being created at this path again.
	projectlinks.RemoveLinksForPath(projectPath)

	refs, err := collectContextRefs(projectPath)
	if err != nil {
		return errs.WrapMessage("failed to list notes and memory", err, "project_path", projectPath)
	}
	if err := deleteIndexData(projectPath, refs); err != nil {
		return errs.WrapMessage("failed to purge index data", err, "project_path", projectPath)
	}
	refs.dropCachedVectors()

	if db.DB != nil {
		db.DB.Exec(deleteProjectQueriesQuery, projectPath)
		db.DB.Exec(deleteProjectMemoryAccessQuery, projectPath)
		// sessions has no project_path column (it tracks get_context_capsule dedup by
		// session_id, not by project), so scope by the file_path prefix instead.
		db.DB.Exec(deleteSessionsUnderPathQuery, projectPath+"/%")
	}
	purgeContextData(projectPath, refs)

	cache.Candidates.ClearProject(projectPath)
	search.Cache.DeleteByProject(projectPath)
	logger.Info("Deleted all indexed data and memory for project", "project_path", projectPath)
	return nil
}

// afterSymbolDelete, when set by tests, runs inside deleteIndexData's transaction
// right after the project's symbols are deleted; a non-nil error aborts the purge.
var afterSymbolDelete func() error

// deleteIndexData removes projectPath's rows from index.db in one write
// transaction on the index writer. The symbol FTS indexes are therefore either
// fully updated or untouched: a failed statement rolls everything back, and a WAL
// quiesce can't land part-way through, because it flushes the index writer (and
// so waits for this transaction) before closing the pool.
//
// The project's note and memory vectors go in the same transaction, because they
// are keyed by ref rather than project_path and the notes themselves are deleted
// from context.db afterwards, when index writes may already be quiesced. If the
// transaction fails, ProjectData stops before touching context.db and the
// project's symbols are still indexed, so the deleted-project sweep retries it.
func deleteIndexData(projectPath string, refs contextRefs) error {
	return db.IndexWrite(func(tx *sql.Tx) error {
		// Clear the trigger-free tables first. That takes the write lock before the
		// symbol counts below are read, so nothing can change them before the delete.
		for _, table := range []string{"edges", "vectors", "indexed_files", "summaries", "embed_pending"} {
			if _, err := tx.Exec(deleteFromQueryPrefix+table+whereProjectPathQuerySuffix, projectPath); err != nil {
				return errs.WrapMessage("failed to delete table rows", err, "table", table)
			}
		}
		if err := refs.deleteVectors(tx); err != nil {
			return err
		}
		var projectSymbols, totalSymbols int
		if err := tx.QueryRow(selectSymbolCountsQuery, projectPath).Scan(&projectSymbols, &totalSymbols); err != nil {
			return errs.WrapMessage("failed to count symbols", err)
		}
		deleteSymbols := func() error {
			if _, err := tx.Exec(deleteProjectSymbolsQuery, projectPath); err != nil {
				return errs.WrapMessage("failed to delete symbols", err)
			}
			if afterSymbolDelete != nil {
				return afterSymbolDelete()
			}
			return nil
		}
		// A full FTS rebuild re-tokenizes every remaining symbol, while the delete
		// triggers only touch this project's rows, so the rebuild only pays off when
		// this project is most of the index. A deleted worktree is usually a few
		// thousand symbols out of hundreds of thousands.
		if projectSymbols*2 > totalSymbols {
			return db.WithoutFTSTriggers(tx, deleteSymbols)
		}
		return deleteSymbols()
	})
}

// contextRefs are the refs of a project's stored notes and structured memory.
// Their vectors live in index.db under doc_type note/memory with source_file
// "note:<ref>" / "mem:<ref>", and project_path set to the session ID.
type contextRefs struct {
	notes, memories []string
}

func collectContextRefs(projectPath string) (contextRefs, error) {
	var refs contextRefs
	if db.ContextDB == nil {
		return refs, nil
	}
	var err error
	if refs.notes, err = queryRefs(selectNoteRefsQuery, projectPath); err != nil {
		return refs, errs.WrapMessage("failed to query context notes", err)
	}
	if refs.memories, err = queryRefs(selectMemoryRefsQuery, projectPath); err != nil {
		return refs, errs.WrapMessage("failed to query structured memory", err)
	}
	return refs, nil
}

func (r contextRefs) noteKeys() []string   { return prefixed("note:", r.notes) }
func (r contextRefs) memoryKeys() []string { return prefixed("mem:", r.memories) }

func (r contextRefs) deleteVectors(tx *sql.Tx) error {
	del := func(docType string, keys []string) error {
		for _, key := range keys {
			if _, err := tx.Exec(deleteRefVectorQuery, docType, key); err != nil {
				return errs.WrapMessage("failed to delete vector", err, "doc_type", docType, "source_file", key)
			}
		}
		return nil
	}
	if err := del("note", r.noteKeys()); err != nil {
		return err
	}
	return del("memory", r.memoryKeys())
}

func (r contextRefs) dropCachedVectors() {
	search.Cache.DeleteBySourceFiles("note", r.noteKeys())
	search.Cache.DeleteBySourceFiles("memory", r.memoryKeys())
}

// purgeContextData removes stored notes, structured memory, and handoff trees for the project.
// Both carry standalone FTS mirrors keyed by ref rather than by project_path, so
// they are cleaned up by the refs collected at the start of the purge. Their
// vectors were already deleted with the rest of the index data.
func purgeContextData(projectPath string, refs contextRefs) {
	if db.ContextDB == nil {
		return
	}
	for _, ref := range refs.notes {
		db.ContextDB.Exec(deleteNoteFTSQuery, ref)
		db.ContextDB.Exec(deleteNoteRevisionsQuery, ref)
	}
	db.ContextDB.Exec(deleteProjectNotesQuery, projectPath)

	for _, ref := range refs.memories {
		db.ContextDB.Exec(deleteMemoryFTSQuery, ref)
	}
	db.ContextDB.Exec(deleteProjectMemoryQuery, projectPath)

	db.ContextDB.Exec(deleteProjectKVRepairEventsQuery, projectPath)

	if err := db.HandoffTx(func(tx *sql.Tx) error { return deleteHandoffData(tx, projectPath) }); err != nil {
		logger.Warn("Failed to purge handoff trees for project", "project_path", projectPath, "error", err)
	}
}

// deleteHandoffData removes the project's handoff trees in one handoff write transaction, so a
// tree is either gone or intact. The child sessions' notes and memory carry project_path and
// were deleted above with the rest of the project's notes.
func deleteHandoffData(tx *sql.Tx, projectPath string) error {
	for _, q := range []string{deleteProjectSnapshotItemsQuery, deleteProjectHandoffResultsQuery} {
		if _, err := tx.Exec(q, projectPath, projectPath); err != nil {
			return errs.WrapMessage("failed to delete handoff rows", err)
		}
	}
	for _, table := range handoffTreeKeyedTables {
		if _, err := tx.Exec(deleteFromQueryPrefix+table+whereProjectTreeQuerySuffix, projectPath); err != nil {
			return errs.WrapMessage("failed to delete handoff rows", err, "table", table)
		}
	}
	for _, table := range handoffProjectTables {
		if _, err := tx.Exec(deleteFromQueryPrefix+table+whereProjectOrTreeQuerySuffix, projectPath, projectPath); err != nil {
			return errs.WrapMessage("failed to delete handoff rows", err, "table", table)
		}
	}
	if _, err := tx.Exec(deleteProjectHandoffTreesQuery, projectPath); err != nil {
		return errs.WrapMessage("failed to delete handoff trees", err)
	}
	return nil
}

func queryRefs(query, projectPath string) ([]string, error) {
	rows, err := db.ContextDB.Query(query, projectPath)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var ref string
		if err := rows.Scan(&ref); err != nil {
			return nil, err
		}
		out = append(out, ref)
	}
	return out, rows.Err()
}

func prefixed(prefix string, refs []string) []string {
	out := make([]string, len(refs))
	for i, ref := range refs {
		out[i] = prefix + ref
	}
	return out
}
