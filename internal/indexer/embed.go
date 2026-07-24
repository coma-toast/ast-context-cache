package indexer

import (
	"fmt"
	"log"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

// maxEmbedSymbolsPerFile caps onnx/remote work per file so one generated
// protobuf (thousands of symbols) cannot monopolize the shared embedder and
// stall the rest of the queue. Remaining symbols stay searchable via BM25.
const maxEmbedSymbolsPerFile = 256

// embedBatchSize is how many texts to embed+upsert at a time (releases
// between batches for fairness and bounds peak memory).
const embedBatchSize = 64

func EmbedDirectorySymbols(emb embedder.Interface, dirPath, projectPath string) {
	rows, err := db.IndexDB.Query(
		"SELECT DISTINCT file FROM symbols WHERE project_path = ?", projectPath)
	if err != nil {
		log.Printf("embed: query files for %s: %v", projectPath, err)
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
	log.Printf("Finished embedding all symbols for %s (%d files)", projectPath, len(files))
}

func EmbedFileSymbols(emb embedder.Interface, filePath, projectPath string) error {
	if db.IndexReadQuiesced() {
		return fmt.Errorf("index db quiesced for maintenance")
	}
	if ShouldSkipEmbed(filePath) {
		return nil
	}
	rows, err := db.IndexDB.Query(
		"SELECT id, name, kind, start_line, end_line FROM symbols WHERE file = ? AND project_path = ?",
		filePath, projectPath)
	if err != nil {
		log.Printf("embed: query symbols for %s: %v", filePath, err)
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
		log.Printf("embed: capping %s at %d/%d symbols (skipping remainder to keep queue moving)",
			filePath, maxEmbedSymbolsPerFile, n)
		symbols = symbols[:maxEmbedSymbolsPerFile]
	}

	fileCache := map[string][]string{}
	total := 0
	for start := 0; start < len(symbols); start += embedBatchSize {
		if db.IndexReadQuiesced() {
			return fmt.Errorf("index db quiesced for maintenance")
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
			if len(src) > 500 {
				src = src[:500]
			}
			hash := ExpectedEmbedHash(s.kind, s.name, filePath, s.startLine, s.endLine)
			text := s.kind + " " + s.name + ": " + src
			texts = append(texts, text)
			entries = append(entries, search.VectorEntry{
				SymbolID:    s.id,
				ContentHash: hash,
				DocType:     "code",
				SourceFile:  filePath,
				Name:        s.name,
				Kind:        s.kind,
				ProjectPath: projectPath,
			})
		}
		embeddings, err := emb.Embed(texts)
		if err != nil {
			log.Printf("embed: generate embeddings for %s: %v", filePath, err)
			return err
		}
		if db.IndexReadQuiesced() {
			return fmt.Errorf("index db quiesced for maintenance")
		}
		for i := range entries {
			entries[i].Vector = embeddings[i]
		}
		if err := search.Cache.Upsert(entries); err != nil {
			log.Printf("embed: upsert vectors for %s: %v", filePath, err)
			return err
		}
		total += len(entries)
	}

	log.Printf("Embedded %d symbols from %s", total, filePath)
	return nil
}
