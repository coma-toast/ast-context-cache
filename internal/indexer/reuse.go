package indexer

import (
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

const (
	selectProjectIndexedQuery       = `SELECT 1 FROM indexed_files WHERE project_path = ? LIMIT 1`
	countProjectIndexedFilesQuery   = `SELECT COUNT(*) FROM indexed_files WHERE project_path = ?`
	selectIndexedFileParserVerQuery = `SELECT COALESCE(parser_version, 0) FROM indexed_files WHERE file = ? AND project_path = ?`
	insertReuseEdgeQuery            = "INSERT INTO edges (source_file, source_symbol, target, kind, project_path) VALUES (?, ?, ?, ?, ?)"
	selectReuseSymbolsQuery         = `SELECT id, name, kind, COALESCE(start_line,0), COALESCE(end_line,0),
		COALESCE(code,''), COALESCE(fqn,''), COALESCE(skeleton,''), COALESCE(embed_hash,'')
		FROM symbols WHERE file = ? AND project_path = ? ORDER BY id`
	selectReuseEdgesQuery = `SELECT COALESCE(source_symbol,''), target, kind
		FROM edges WHERE source_file = ? AND project_path = ? ORDER BY id`
	selectReuseVectorsQuery = `SELECT COALESCE(symbol_id,0), content_hash, vector, COALESCE(name,''), COALESCE(kind,'')
		FROM vectors WHERE source_file = ? AND project_path = ? AND COALESCE(doc_type,'code') = 'code'`
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
	if conn.QueryRow(selectProjectIndexedQuery, projectPath).Scan(&seen) == nil {
		return nil
	}
	best := ""
	bestFiles := 0
	for _, sib := range projectlinks.RepoSiblings(projectPath) {
		var n int
		if conn.QueryRow(countProjectIndexedFilesQuery, sib).Scan(&n) != nil {
			continue
		}
		if n > bestFiles {
			best, bestFiles = sib, n
		}
	}
	if best == "" {
		return nil
	}
	logger.Info("Found sibling checkout to reuse", "project", projectPath, "sibling", best, "indexed_files", bestFiles)
	return &ReuseSource{ProjectPath: best}
}

// ReuseFile copies the sibling's index rows for filePath when the sibling holds a
// byte-identical file at the same repo-relative path. It returns the number of
// symbols copied and whether reuse happened; callers fall back to IndexFile when
// it reports false.
func ReuseFile(filePath, projectPath string, src *ReuseSource) (int, bool) {
	n, _, ok := reuseFile(filePath, projectPath, src)
	return n, ok
}

// Outcome is what IndexOrReuse did for one file.
type Outcome struct {
	Symbols, FullTokens, SkeletonTokens int
	// Reused is true when the rows were copied from a sibling checkout.
	Reused bool
	// NeedsEmbed is true when the file still lacks current code vectors: always
	// after a fresh parse, and after reuse only when the sibling had not finished
	// embedding it.
	NeedsEmbed bool
}

// IndexOrReuse copies filePath's rows from src when the sibling holds a
// byte-identical copy and falls back to IndexFile otherwise. Every first-index
// path (directory jobs, watcher catch-up) goes through it so they agree on when
// reuse applies.
func IndexOrReuse(filePath, projectPath string, src *ReuseSource) (Outcome, error) {
	if n, needsEmbed, ok := reuseFile(filePath, projectPath, src); ok {
		return Outcome{Symbols: n, Reused: true, NeedsEmbed: needsEmbed}, nil
	}
	n, full, skel, err := IndexFile(filePath, projectPath)
	return Outcome{Symbols: n, FullTokens: full, SkeletonTokens: skel, NeedsEmbed: err == nil}, err
}

func reuseFile(filePath, projectPath string, src *ReuseSource) (int, bool, bool) {
	if src == nil {
		return 0, false, false
	}
	filePath = filepath.Clean(filePath)
	projectPath = projectlinks.NormalizePath(projectPath)
	rel, err := filepath.Rel(projectPath, filePath)
	if err != nil || rel == "." || strings.HasPrefix(rel, "..") {
		return 0, false, false
	}
	sibFile := filepath.Join(src.ProjectPath, rel)
	if !sameFileContent(filePath, sibFile) {
		return 0, false, false
	}

	conn, err := db.IndexReader()
	if err != nil {
		return 0, false, false
	}
	// The sibling's rows are only worth copying if today's parser produced them.
	var version int
	if conn.QueryRow(selectIndexedFileParserVerQuery, sibFile, src.ProjectPath).Scan(&version) != nil || version < ParserVersion(filePath) {
		return 0, false, false
	}

	symbols, err := readReuseSymbols(conn, sibFile, src.ProjectPath)
	if err != nil {
		return 0, false, false
	}
	edges, err := readReuseEdges(conn, sibFile, src.ProjectPath)
	if err != nil {
		return 0, false, false
	}
	vectors, err := readReuseVectors(conn, sibFile, src.ProjectPath)
	if err != nil {
		return 0, false, false
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
		if _, err := tx.Exec(deleteFileSymbolsQuery, filePath, projectPath); err != nil {
			return err
		}
		if _, err := tx.Exec(deleteFileEdgesQuery, filePath, projectPath); err != nil {
			return err
		}
		for _, s := range symbols {
			res, err := tx.Exec(insertSymbolQuery,
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
			if _, err := tx.Exec(insertReuseEdgeQuery,
				filePath, e.sourceSymbol, e.target, e.kind, projectPath); err != nil {
				return err
			}
		}
		return db.UpsertIndexedFileWith(tx, filePath, projectPath, time.Now())
	})
	if err != nil {
		return 0, false, false
	}

	search.Cache.DeleteByFile(filePath, projectPath)
	copiedAll := copyReuseVectors(vectors, newIDs, filePath, projectPath)
	db.InvalidateSummariesForFile(filePath, projectPath)
	notifyIndexCommitted(projectPath)
	return copied, !copiedAll || !vectorsCover(symbols, vectors), true
}

// vectorsCover reports whether every symbol has a code vector for its current
// embed_hash, the same test embedqueue uses to find files missing vectors.
func vectorsCover(symbols []reuseSymbol, vectors []reuseVector) bool {
	have := make(map[int64]string, len(vectors))
	for _, v := range vectors {
		have[v.symbolID] = v.hash
	}
	for _, s := range symbols {
		h, ok := have[s.id]
		if !ok || (s.embed != "" && h != s.embed) {
			return false
		}
	}
	return true
}

// copyReuseVectors stores the sibling's vectors under the new symbol ids and
// reports whether every one of them was copied.
func copyReuseVectors(vectors []reuseVector, newIDs map[int64]int64, filePath, projectPath string) bool {
	if len(vectors) == 0 {
		return true
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
		return false
	}
	if err := search.Cache.Upsert(entries); err != nil {
		logger.Warn("Failed to copy reused vectors", "file", filePath, "error", err)
		return false
	}
	return len(entries) == len(vectors)
}

func readReuseSymbols(conn *sql.DB, file, projectPath string) ([]reuseSymbol, error) {
	rows, err := conn.Query(selectReuseSymbolsQuery, file, projectPath)
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
	rows, err := conn.Query(selectReuseEdgesQuery, file, projectPath)
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
	rows, err := conn.Query(selectReuseVectorsQuery, file, projectPath)
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
