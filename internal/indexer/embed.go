package indexer

import (
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

const (
	selectProjectSymbolFilesQuery = "SELECT DISTINCT file FROM symbols WHERE project_path = ?"
	selectFileEmbedSymbolsQuery   = "SELECT id, name, kind, start_line, end_line FROM symbols WHERE file = ? AND project_path = ?"
)

// maxEmbedSymbolsPerFile caps onnx/remote work per file so one generated
// protobuf (thousands of symbols) cannot monopolize the shared embedder and
// stall the rest of the queue. Remaining symbols stay searchable via BM25.
const maxEmbedSymbolsPerFile = 256

// embedBatchSize is how many texts to embed+upsert at a time (releases
// between batches for fairness and bounds peak memory).
const embedBatchSize = 64

// errIndexQuiesced stops an embed batch when the index db is paused for maintenance.
var errIndexQuiesced = errs.New("index db quiesced for maintenance")

func EmbedDirectorySymbols(emb embedder.Interface, dirPath, projectPath string) {
	conn, err := db.IndexReader()
	if err != nil {
		logger.Warn("Failed to query files to embed", "project", projectPath, "error", err)
		return
	}
	rows, err := conn.Query(selectProjectSymbolFilesQuery, projectPath)
	if err != nil {
		logger.Warn("Failed to query files to embed", "project", projectPath, "error", err)
		return
	}
	defer rows.Close()

	var files []string
	for rows.Next() {
		var f string
		rows.Scan(&f)
		files = append(files, f)
	}

	for _, f := range files {
		EmbedFileSymbols(emb, f, projectPath)
	}
	logger.Info("Finished embedding all symbols", "project", projectPath, "files", len(files))
}

func EmbedFileSymbols(emb embedder.Interface, filePath, projectPath string) error {
	conn, err := db.IndexReader()
	if err != nil {
		return err
	}
	if ShouldSkipEmbed(filePath) {
		return nil
	}
	if dropStaleEmbedJob(filePath, projectPath) {
		return nil
	}
	rows, err := conn.Query(selectFileEmbedSymbolsQuery, filePath, projectPath)
	if err != nil {
		logger.Warn("Failed to query symbols to embed", "file", filePath, "error", err)
		return err
	}
	defer rows.Close()

	type symInfo struct {
		id                 int64
		name, kind         string
		startLine, endLine int
	}
	var symbols []symInfo
	for rows.Next() {
		var s symInfo
		rows.Scan(&s.id, &s.name, &s.kind, &s.startLine, &s.endLine)
		symbols = append(symbols, s)
	}

	if len(symbols) == 0 {
		return nil
	}
	if n := len(symbols); n > maxEmbedSymbolsPerFile {
		logger.Info("Capping embedded symbols to keep queue moving", "file", filePath, "cap", maxEmbedSymbolsPerFile, "symbols", n)
		symbols = symbols[:maxEmbedSymbolsPerFile]
	}

	fileCache := map[string][]string{}
	total := 0
	for start := 0; start < len(symbols); start += embedBatchSize {
		if db.IndexReadQuiesced() {
			return errIndexQuiesced
		}
		end := start + embedBatchSize
		if end > len(symbols) {
			end = len(symbols)
		}
		batch := symbols[start:end]
		var texts []string
		var entries []search.VectorEntry
		for _, s := range batch {
			src := ReadSourceRange(filePath, s.startLine, s.endLine, fileCache)
			text := BuildEmbedText(s.kind, s.name, src)
			entries = append(entries, search.VectorEntry{
				SymbolID:    s.id,
				ContentHash: search.ContentHash(text),
				DocType:     "code",
				SourceFile:  filePath,
				Name:        s.name,
				Kind:        s.kind,
				ProjectPath: projectPath,
			})
			texts = append(texts, text)
		}
		embeddings, err := emb.Embed(texts)
		if err != nil {
			logger.Warn("Failed to generate embeddings", "file", filePath, "error", err)
			return err
		}
		if db.IndexReadQuiesced() {
			return errIndexQuiesced
		}
		for i := range entries {
			entries[i].Vector = embeddings[i]
		}
		if err := search.Cache.Upsert(entries); err != nil {
			logger.Warn("Failed to upsert vectors", "file", filePath, "error", err)
			return err
		}
		total += len(entries)
	}

	logger.Debug("Embedded symbols", "file", filePath, "symbols", total)
	return nil
}
