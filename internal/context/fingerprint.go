package context

import (
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
)

const (
	fingerprintBytes = 16
	// Prefer an exact fqn match; fall back to the bare name so a symbol whose fqn shape
	// changed (or a caller that only knows the name) still resolves.
	selectSymbolByFQNOrNameQuery = `SELECT file, name, COALESCE(fqn,''), kind, COALESCE(start_line,0), COALESCE(end_line,0), COALESCE(project_path,'')
		FROM symbols WHERE file = ? AND project_path = ? AND ((? != '' AND fqn = ?) OR name = ?)
		ORDER BY (fqn = ?) DESC, start_line LIMIT 1`
)

// SymbolRow is a symbol's current location in the index.
type SymbolRow struct {
	File        string `json:"file"`
	FileRel     string `json:"file_rel"`
	ProjectPath string `json:"project_path"`
	Name        string `json:"name"`
	FQN         string `json:"fqn,omitempty"`
	Kind        string `json:"kind"`
	StartLine   int    `json:"start_line"`
	EndLine     int    `json:"end_line"`
}

// SymbolFingerprint hashes lines startLine..endLine of file as they are on disk now: the hex
// of the first 16 bytes of their sha256. Handoff pointers store it at snapshot time and compare
// it later to tell a fresh symbol from a modified one. It fails with CodeNotFound when the
// file is gone or no longer has those lines.
func SymbolFingerprint(file string, startLine, endLine int) (string, error) {
	if _, err := os.Stat(file); err != nil {
		return "", errs.WrapCode(errs.CodeNotFound, err, "file", file)
	}
	text := indexer.ReadSourceRange(file, startLine, endLine, map[string][]string{})
	if text == "" {
		return "", errs.NewCode(errs.CodeNotFound, "line range not in file", "file", file, "start_line", startLine, "end_line", endLine)
	}
	sum := sha256.Sum256([]byte(text))
	return hex.EncodeToString(sum[:fingerprintBytes]), nil
}

// LookupSymbol finds the symbol fqn (or, failing that, name) in fileRel under projectPath, in
// whichever project owns the file (a linked child project's index, when there is one). fileRel
// may also be absolute. It fails with CodeNotFound when the index has no such symbol.
func LookupSymbol(projectPath, fileRel, fqn, name string) (*SymbolRow, error) {
	if fileRel == "" || (fqn == "" && name == "") {
		return nil, errs.NewCode(errs.CodeInvalidInput, "file and fqn or name required")
	}
	file := fileRel
	if !filepath.IsAbs(file) {
		file = filepath.Join(projectPath, fileRel)
	}
	file = projectlinks.NormalizePath(file)
	owner := projectlinks.OwningProject(file, projectPath)
	conn, err := db.IndexReader()
	if err != nil {
		return nil, errs.WrapMessage("failed to look up symbol", err, "file", file)
	}
	var r SymbolRow
	err = conn.QueryRow(selectSymbolByFQNOrNameQuery, file, owner, fqn, fqn, name, fqn).
		Scan(&r.File, &r.Name, &r.FQN, &r.Kind, &r.StartLine, &r.EndLine, &r.ProjectPath)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, errs.NewCode(errs.CodeNotFound, "symbol not in index", "file", file, "fqn", fqn, "name", name)
	}
	if err != nil {
		return nil, errs.WrapMessage("failed to look up symbol", err, "file", file)
	}
	r.FileRel = db.RelPath(r.File, projectlinks.NormalizePath(projectPath))
	return &r, nil
}
