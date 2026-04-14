package indexer

import (
	"log"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

func EmbedDirectorySymbols(emb embedder.Interface, dirPath, projectPath string) {
	rows, err := db.DB.Query(
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

func EmbedFileSymbols(emb embedder.Interface, filePath, projectPath string) {
	if ShouldSkipEmbed(filePath) {
		return
	}
	rows, err := db.DB.Query(
		"SELECT id, name, kind, start_line, end_line FROM symbols WHERE file = ? AND project_path = ?",
		filePath, projectPath)
	if err != nil {
		log.Printf("embed: query symbols for %s: %v", filePath, err)
		return
	}
	defer rows.Close()

	type symInfo struct {
		id                    int64
		name, kind            string
		startLine, endLine    int
	}
	var symbols []symInfo
	for rows.Next() {
		var s symInfo
		rows.Scan(&s.id, &s.name, &s.kind, &s.startLine, &s.endLine)
		symbols = append(symbols, s)
	}

	if len(symbols) == 0 {
		return
	}

	fileCache := map[string][]string{}
	var texts []string
	var entries []search.VectorEntry

	for _, s := range symbols {
		src := ReadSourceRange(filePath, s.startLine, s.endLine, fileCache)
		if len(src) > 500 {
			src = src[:500]
		}
		text := s.kind + " " + s.name + ": " + src
		hash := search.ContentHash(text)

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
		return
	}

	for i := range entries {
		entries[i].Vector = embeddings[i]
	}

	if err := search.Cache.Upsert(entries); err != nil {
		log.Printf("embed: upsert vectors for %s: %v", filePath, err)
		return
	}

	log.Printf("Embedded %d symbols from %s", len(entries), filePath)
}
