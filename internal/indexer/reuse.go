package indexer

import (
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

// A WTG space keeps one checkout of a repo per space, each on its own branch, so
// most files are byte-identical across checkouts. Indexing a fresh checkout can
// therefore copy an already-indexed sibling's parsed symbols, edges and vectors
// for every unchanged file instead of re-running tree-sitter and the embedder.

// ReuseSource is an already-indexed checkout of the same repo to copy rows from.
type ReuseSource struct {
	ProjectPath string
}

type reuseSymbol struct {
	id                 int64
	name, kind         string
	startLine, endLine int
	code, fqn          string
	skeleton, embed    string
}

type reuseEdge struct {
	sourceSymbol string
	target, kind string
}

type reuseVector struct {
	symbolID   int64
	hash       string
	blob       []byte
	name, kind string
}

// FindReuseSource picks the best already-indexed sibling checkout of the same
// repository to copy from. It returns nil when projectPath already has an index
// of its own (incremental re-index handles that) or when no sibling is indexed.
func FindReuseSource(projectPath string) *ReuseSource {
	projectPath = projectlinks.NormalizePath(projectPath)
	if projectPath == "" {
		return nil
	}
	conn, err := db.IndexReader()
	if err != nil {
		return nil
	}
	var seen int
	if conn.QueryRow(`SELECT 1 FROM indexed_files WHERE project_path = ? LIMIT 1`, projectPath).Scan(&seen) == nil {
		return nil
	}
	best := ""
	bestFiles := 0
	for _, sib := range projectlinks.RepoSiblings(projectPath) {
		var n int
		if conn.QueryRow(`SELECT COUNT(*) FROM indexed_files WHERE project_path = ?`, sib).Scan(&n) != nil {
			continue
		}
		if n > bestFiles {
			best, bestFiles = sib, n
		}
	}
	if best == "" {
		return nil
	}
	log.Printf("index: %s can reuse from sibling checkout %s (%d indexed files)", projectPath, best, bestFiles)
	return &ReuseSource{ProjectPath: best}
}

// ReuseFile copies the sibling's index rows for filePath when the sibling holds a
// byte-identical file at the same repo-relative path. It returns the number of
// symbols copied and whether reuse happened; callers fall back to IndexFile when
// it reports false.
func ReuseFile(filePath, projectPath string, src *ReuseSource) (int, bool) {
	if src == nil {
		return 0, false
	}
	filePath = filepath.Clean(filePath)
	projectPath = projectlinks.NormalizePath(projectPath)
	rel, err := filepath.Rel(projectPath, filePath)
	if err != nil || rel == "." || strings.HasPrefix(rel, "..") {
		return 0, false
	}
	sibFile := filepath.Join(src.ProjectPath, rel)
	if !sameFileContent(filePath, sibFile) {
		return 0, false
	}

	conn, err := db.IndexReader()
	if err != nil {
		return 0, false
	}
	var indexed int
	if conn.QueryRow(`SELECT 1 FROM indexed_files WHERE file = ? AND project_path = ?`,
		sibFile, src.ProjectPath).Scan(&indexed) != nil {
		return 0, false
	}

	symbols, err := readReuseSymbols(conn, sibFile, src.ProjectPath)
	if err != nil {
		return 0, false
	}
	edges, err := readReuseEdges(conn, sibFile, src.ProjectPath)
	if err != nil {
		return 0, false
	}
	vectors, err := readReuseVectors(conn, sibFile, src.ProjectPath)
	if err != nil {
		return 0, false
	}

	// Identical bytes mean identical parse output: code, fqn (basename-derived),
	// skeleton and embed_hash (content-derived) all carry over verbatim; only the
	// file path and project_path differ.
	copied := 0
	newIDs := make(map[int64]int64, len(symbols))
	err = db.IndexWrite(func(tx *sql.Tx) error {
		copied = 0
		if err := deleteCodeVectorsTx(tx, filePath, projectPath); err != nil {
			return err
		}
		if _, err := tx.Exec("DELETE FROM symbols WHERE file = ? AND project_path = ?", filePath, projectPath); err != nil {
			return err
		}
		if _, err := tx.Exec("DELETE FROM edges WHERE source_file = ? AND project_path = ?", filePath, projectPath); err != nil {
			return err
		}
		for _, s := range symbols {
			res, err := tx.Exec("INSERT INTO symbols (name, kind, file, start_line, end_line, code, fqn, project_path, skeleton, embed_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
				s.name, s.kind, filePath, s.startLine, s.endLine, s.code, s.fqn, projectPath, s.skeleton, s.embed)
			if err != nil {
				return err
			}
			if id, err := res.LastInsertId(); err == nil {
				newIDs[s.id] = id
			}
			copied++
		}
		for _, e := range edges {
			if _, err := tx.Exec("INSERT INTO edges (source_file, source_symbol, target, kind, project_path) VALUES (?, ?, ?, ?, ?)",
				filePath, e.sourceSymbol, e.target, e.kind, projectPath); err != nil {
				return err
			}
		}
		return db.UpsertIndexedFileWith(tx, filePath, projectPath, time.Now())
	})
	if err != nil {
		return 0, false
	}

	search.Cache.DeleteByFile(filePath, projectPath)
	copyReuseVectors(vectors, newIDs, filePath, projectPath)
	db.InvalidateSummariesForFile(filePath, projectPath)
	notifyIndexCommitted()
	return copied, true
}

func copyReuseVectors(vectors []reuseVector, newIDs map[int64]int64, filePath, projectPath string) {
	if len(vectors) == 0 {
		return
	}
	entries := make([]search.VectorEntry, 0, len(vectors))
	for _, v := range vectors {
		vec := search.DecodeVector(v.blob)
		if len(vec) != search.VectorDims {
			continue
		}
		entries = append(entries, search.VectorEntry{
			SymbolID:    newIDs[v.symbolID],
			ContentHash: v.hash,
			Vector:      vec,
			DocType:     "code",
			SourceFile:  filePath,
			Name:        v.name,
			Kind:        v.kind,
			ProjectPath: projectPath,
		})
	}
	if len(entries) == 0 {
		return
	}
	if err := search.Cache.Upsert(entries); err != nil {
		log.Printf("index: copy vectors for %s: %v", filePath, err)
	}
}

func readReuseSymbols(conn *sql.DB, file, projectPath string) ([]reuseSymbol, error) {
	rows, err := conn.Query(`SELECT id, name, kind, COALESCE(start_line,0), COALESCE(end_line,0),
		COALESCE(code,''), COALESCE(fqn,''), COALESCE(skeleton,''), COALESCE(embed_hash,'')
		FROM symbols WHERE file = ? AND project_path = ? ORDER BY id`, file, projectPath)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []reuseSymbol
	for rows.Next() {
		var s reuseSymbol
		if rows.Scan(&s.id, &s.name, &s.kind, &s.startLine, &s.endLine, &s.code, &s.fqn, &s.skeleton, &s.embed) != nil {
			continue
		}
		out = append(out, s)
	}
	return out, rows.Err()
}

func readReuseEdges(conn *sql.DB, file, projectPath string) ([]reuseEdge, error) {
	rows, err := conn.Query(`SELECT COALESCE(source_symbol,''), target, kind
		FROM edges WHERE source_file = ? AND project_path = ? ORDER BY id`, file, projectPath)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []reuseEdge
	for rows.Next() {
		var e reuseEdge
		if rows.Scan(&e.sourceSymbol, &e.target, &e.kind) != nil {
			continue
		}
		out = append(out, e)
	}
	return out, rows.Err()
}

func readReuseVectors(conn *sql.DB, file, projectPath string) ([]reuseVector, error) {
	rows, err := conn.Query(`SELECT COALESCE(symbol_id,0), content_hash, vector, COALESCE(name,''), COALESCE(kind,'')
		FROM vectors WHERE source_file = ? AND project_path = ? AND COALESCE(doc_type,'code') = 'code'`,
		file, projectPath)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []reuseVector
	for rows.Next() {
		var v reuseVector
		if rows.Scan(&v.symbolID, &v.hash, &v.blob, &v.name, &v.kind) != nil {
			continue
		}
		out = append(out, v)
	}
	return out, rows.Err()
}

func sameFileContent(a, b string) bool {
	ha, err := fileContentHash(a)
	if err != nil {
		return false
	}
	hb, err := fileContentHash(b)
	if err != nil {
		return false
	}
	return ha == hb
}

func fileContentHash(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
