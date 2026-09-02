// Package purge removes every trace of a project from the local caches and
// sweeps for projects whose directory has been deleted from disk.
//
// WTG spaces are created and thrown away constantly, so without this the index
// accumulates rows for checkouts that no longer exist.
package purge

import (
	"fmt"

	"github.com/coma-toast/ast-context-cache/internal/cache"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedqueue"
	"github.com/coma-toast/ast-context-cache/internal/search"
	"github.com/coma-toast/ast-context-cache/internal/watcher"
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
		return fmt.Errorf("project_path required")
	}

	embedqueue.RemoveProject(projectPath)

	conn, err := db.IndexReader()
	if err != nil {
		return err
	}
	// FTS triggers are dropped around the bulk delete and the index rebuilt after,
	// which is far cheaper than firing a trigger per removed symbol.
	conn.Exec("DROP TRIGGER IF EXISTS symbols_fts_ins")
	conn.Exec("DROP TRIGGER IF EXISTS symbols_fts_del")
	_, delErr := conn.Exec("DELETE FROM symbols WHERE project_path = ?", projectPath)
	if delErr == nil {
		conn.Exec("DELETE FROM edges WHERE project_path = ?", projectPath)
		conn.Exec("DELETE FROM vectors WHERE project_path = ?", projectPath)
		conn.Exec("DELETE FROM indexed_files WHERE project_path = ?", projectPath)
		conn.Exec("DELETE FROM summaries WHERE project_path = ?", projectPath)
		conn.Exec("DELETE FROM embed_pending WHERE project_path = ?", projectPath)
		conn.Exec(`INSERT INTO symbols_fts(symbols_fts) VALUES('rebuild')`)
	}
	db.EnsureFTSTriggers()
	if delErr != nil {
		return delErr
	}

	if db.DB != nil {
		db.DB.Exec("DELETE FROM queries WHERE project_path = ?", projectPath)
		db.DB.Exec("DELETE FROM memory_access WHERE project_path = ?", projectPath)
		// sessions has no project_path column (it tracks get_context_capsule dedup by
		// session_id, not by project), so scope by the file_path prefix instead.
		db.DB.Exec("DELETE FROM sessions WHERE file_path LIKE ?", projectPath+"/%")
	}
	purgeContextData(projectPath)

	cache.GlobalCache.ClearProject(projectPath)
	search.Cache.DeleteByProject(projectPath)
	return nil
}

// purgeContextData removes stored notes and structured memory for the project.
// Both carry standalone FTS mirrors and vectors keyed by ref rather than by
// project_path, so rows are collected first and cleaned up by ref.
func purgeContextData(projectPath string) {
	if db.ContextDB == nil {
		return
	}
	noteRefs := contextRefs(`SELECT ref FROM context_notes WHERE project_path = ?`, projectPath)
	for _, ref := range noteRefs {
		db.ContextDB.Exec(`DELETE FROM context_notes_fts WHERE ref = ?`, ref)
		deleteRefVector("note", "note:"+ref)
	}
	db.ContextDB.Exec(`DELETE FROM context_notes WHERE project_path = ?`, projectPath)

	memRefs := contextRefs(`SELECT ref FROM structured_memory WHERE project_path = ?`, projectPath)
	for _, ref := range memRefs {
		db.ContextDB.Exec(`DELETE FROM structured_memory_fts WHERE ref = ?`, ref)
		deleteRefVector("memory", "mem:"+ref)
	}
	db.ContextDB.Exec(`DELETE FROM structured_memory WHERE project_path = ?`, projectPath)

	db.ContextDB.Exec(`DELETE FROM kv_repair_events WHERE project_path = ?`, projectPath)
}

func contextRefs(query, projectPath string) []string {
	rows, err := db.ContextDB.Query(query, projectPath)
	if err != nil {
		return nil
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var ref string
		if rows.Scan(&ref) == nil && ref != "" {
			out = append(out, ref)
		}
	}
	return out
}

func deleteRefVector(docType, key string) {
	if conn, err := db.IndexReader(); err == nil {
		conn.Exec(`DELETE FROM vectors WHERE doc_type = ? AND source_file = ?`, docType, key)
	}
	search.Cache.DeleteNoteByRef(key)
}
