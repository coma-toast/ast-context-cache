package indexer

import (
	"database/sql"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

const (
	deleteFileSummariesQuery     = "DELETE FROM summaries WHERE file_path = ? AND project_path = ?"
	deleteIndexedFileQuery       = "DELETE FROM indexed_files WHERE file = ? AND project_path = ?"
	deleteFileEmbedPendingQuery  = "DELETE FROM embed_pending WHERE file = ? AND project_path = ?"
	selectProjectIndexFilesQuery = `
		SELECT file FROM indexed_files WHERE project_path = ?1
		UNION SELECT DISTINCT file FROM symbols WHERE project_path = ?1
		UNION SELECT DISTINCT source_file FROM edges WHERE project_path = ?1
		UNION SELECT DISTINCT file FROM embed_pending WHERE project_path = ?1
		UNION SELECT DISTINCT source_file FROM vectors WHERE project_path = ?1 AND COALESCE(doc_type, 'code') = 'code'`
)

// ErrSymlinkAlias is returned by IndexFile for a path that is not indexed under
// its own name: a symlink to another file inside the project (the target is
// indexed under its real path instead, so its symbols appear once), or a
// dangling / looping / non-regular symlink.
var ErrSymlinkAlias = errs.New("symlink not indexed")

// purgeFileQueries delete every per-file index row. Only code vectors go:
// doc, note and memory vectors are keyed by their own sources, never by an
// indexed file's path.
var purgeFileQueries = []string{
	deleteFileSymbolsQuery,
	deleteFileEdgesQuery,
	deleteFileCodeVectorsQuery,
	deleteFileSummariesQuery,
	deleteIndexedFileQuery,
	deleteFileEmbedPendingQuery,
}

// OnFilePurged, when set (main wires embedqueue.ForgetFile), runs after
// PurgeFile so in-memory state outside this package (the embed queue's pending
// retry set) forgets the file too.
var OnFilePurged func(filePath, projectPath string)

// PurgeFile removes every index row for one file (symbols, edges, vectors,
// summaries, indexed_files, embed_pending) plus its in-memory vectors and the
// project's cached query results, then calls OnFilePurged. Used when a file is
// gone from disk or must not be indexed under this path (symlink alias).
func PurgeFile(filePath, projectPath string) error {
	err := db.IndexWrite(func(tx *sql.Tx) error {
		for _, q := range purgeFileQueries {
			if _, err := tx.Exec(q, filePath, projectPath); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		return err
	}
	search.Cache.DeleteByFile(filePath, projectPath)
	if OnFilePurged != nil {
		OnFilePurged(filePath, projectPath)
	}
	notifyIndexCommitted(projectPath)
	return nil
}

// FileGone reports whether path definitely no longer exists (Lstat says
// ENOENT). Permission or I/O errors are not treated as gone, so a transient
// failure never destroys index data.
func FileGone(path string) bool {
	_, err := os.Lstat(path)
	return errors.Is(err, fs.ErrNotExist)
}

// SymlinkAlias reports whether filePath must not be indexed under its own name
// because resolving symlinks shows it is:
//   - an alias of another in-project file that the walk indexes under its real
//     path (target is returned), so a file and a link to it yield one set of
//     symbols; or
//   - a dangling or looping symlink, or one whose target is not a regular file
//     (a FIFO or device would block or misbehave on read); target is "".
//
// A symlink resolving outside the project is indexed under the link path (it is
// the project's only copy of that content), provided the target is a regular
// file. A path whose only symlink is the project root itself (e.g. macOS
// /var -> /private/var) is not an alias.
func SymlinkAlias(filePath, projectPath string) (target string, alias bool) {
	real, err := filepath.EvalSymlinks(filePath)
	if err != nil {
		// Missing files are not aliases (callers handle ENOENT); a symlink that
		// cannot be resolved is dangling or a loop.
		if fi, lerr := os.Lstat(filePath); lerr == nil && fi.Mode()&os.ModeSymlink != 0 {
			return "", true
		}
		return "", false
	}
	if real == filePath {
		return "", false
	}
	root := filepath.Clean(projectPath)
	if r, err := filepath.EvalSymlinks(root); err == nil {
		root = r
	}
	rel, err := filepath.Rel(root, real)
	inProject := err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
	canonical := filepath.Join(projectPath, rel)
	// Checked before the regular-file test: when the only symlink is the project
	// root, the file is treated exactly as if the root weren't a symlink.
	if inProject && canonical == filePath {
		return "", false
	}
	if fi, err := os.Stat(real); err != nil || !fi.Mode().IsRegular() {
		return "", true
	}
	if !inProject {
		return "", false
	}
	if !walkIndexes(rel, canonical) {
		// The target would not be indexed on its own (skipped dir or not a code
		// file), so the link is the only way this content gets indexed.
		return "", false
	}
	return canonical, true
}

// walkIndexes approximates whether a project walk visits the project-relative
// file rel (it is a code file and no ancestor directory is skipped).
func walkIndexes(rel, abs string) bool {
	if !IsCodeFile(abs) {
		return false
	}
	for dir := filepath.Dir(rel); dir != "." && dir != string(filepath.Separator); dir = filepath.Dir(dir) {
		if ShouldSkipDir(filepath.Base(dir)) {
			return false
		}
	}
	return true
}

// SkipSymlinkAlias is the cheap walk-time check: info comes from filepath.Walk
// (Lstat), so only entries that are themselves symlinks pay for resolution.
// Rows previously indexed under an alias path are purged. Returns true when the
// walk should skip path.
func SkipSymlinkAlias(path, projectPath string, info os.FileInfo) bool {
	if info == nil || info.Mode()&os.ModeSymlink == 0 {
		return false
	}
	if _, alias := SymlinkAlias(path, projectPath); !alias {
		return false
	}
	_ = PurgeFile(path, projectPath)
	return true
}

// ProjectFilesInIndex returns every file path that has any per-file row for
// projectPath, not just those in indexed_files: rows written before
// indexed_files existed, or left behind by a partial delete, would otherwise
// never be reconciled against the filesystem.
func ProjectFilesInIndex(projectPath string) []string {
	conn, err := db.IndexReader()
	if err != nil {
		return nil
	}
	rows, err := conn.Query(selectProjectIndexFilesQuery, projectPath)
	if err != nil {
		logger.Warn("Failed to list indexed files", "project", projectPath, "error", err)
		return nil
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var f string
		if rows.Scan(&f) == nil && f != "" {
			out = append(out, f)
		}
	}
	return out
}

// PruneMissingFiles purges index rows for files under dirPath (owned by
// projectPath) that no longer exist on disk, e.g. files deleted while the server
// was down so no watcher saw the delete. Nothing is pruned when dirPath itself is
// missing (an unmounted volume is not a mass deletion; the deleted-project sweep
// owns that case). Returns the number of files purged.
func PruneMissingFiles(dirPath, projectPath string) int {
	dirPath = filepath.Clean(dirPath)
	if fi, err := os.Stat(dirPath); err != nil || !fi.IsDir() {
		return 0
	}
	prefix := dirPath + string(filepath.Separator)
	n := 0
	for _, f := range ProjectFilesInIndex(projectPath) {
		if f != dirPath && !strings.HasPrefix(f, prefix) {
			continue
		}
		if !FileGone(f) {
			continue
		}
		if err := PurgeFile(f, projectPath); err != nil {
			logger.Warn("Failed to purge missing file", "file", f, "error", err)
			continue
		}
		n++
	}
	if n > 0 {
		logger.Info("Purged files no longer on disk", "files", n, "dir", dirPath)
	}
	return n
}

// dropStaleEmbedJob reports whether an embed job for filePath should be dropped
// instead of run: the file is gone from disk, or it is a symlink alias. Its rows
// are purged so later catch-up / pending retries do not re-queue it (embedding a
// missing file would otherwise "succeed" on empty source and keep its symbols
// searchable). A job whose project root is itself missing is dropped without
// purging: that looks like an unmounted volume, and the deleted-project sweep
// decides whether the whole project is gone.
func dropStaleEmbedJob(filePath, projectPath string) bool {
	if FileGone(filePath) {
		if fi, err := os.Stat(projectPath); err != nil || !fi.IsDir() {
			logger.Debug("Skipping embed job, project root unavailable", "file", filePath, "project", projectPath)
			return true
		}
		if err := PurgeFile(filePath, projectPath); err != nil {
			logger.Warn("Failed to purge missing file", "file", filePath, "error", err)
		} else {
			logger.Debug("Purged index rows for file no longer on disk", "file", filePath)
		}
		return true
	}
	if target, alias := SymlinkAlias(filePath, projectPath); alias {
		if err := PurgeFile(filePath, projectPath); err == nil {
			logger.Debug("Purged index rows for symlink alias", "file", filePath, "target", target)
		}
		return true
	}
	return false
}
