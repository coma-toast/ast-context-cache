package docs

import (
	"fmt"
	"log"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

var docEmbedder embedder.Interface

// SetEmbedder registers the embedder used for doc section vectors.
func SetEmbedder(e embedder.Interface) {
	docEmbedder = e
}

// Embedder returns the configured doc embedder, if any.
func Embedder() embedder.Interface {
	return docEmbedder
}

// EmbedAllSources embeds every cached doc source (startup backfill).
func EmbedAllSources() {
	if docEmbedder == nil {
		return
	}
	sources, err := ListSources()
	if err != nil {
		return
	}
	for _, s := range sources {
		EmbedSource(s.ID)
	}
}

// EmbedSource embeds all sections for one doc source.
func EmbedSource(sourceID int) {
	if docEmbedder == nil {
		return
	}
	deleteDocVectors(sourceID)
	entries, err := ListEntriesBySource(sourceID)
	if err != nil || len(entries) == 0 {
		return
	}
	var sourceName string
	db.DB.QueryRow("SELECT name FROM doc_sources WHERE id = ?", sourceID).Scan(&sourceName)
	const batchSize = 32
	for i := 0; i < len(entries); i += batchSize {
		end := i + batchSize
		if end > len(entries) {
			end = len(entries)
		}
		if err := embedBatch(sourceID, entries[i:end]); err != nil {
			log.Printf("docs: embed source %d: %v", sourceID, err)
			return
		}
	}
	log.Printf("docs: embedded %d sections for source %d (%s)", len(entries), sourceID, sourceName)
}

func embedBatch(sourceID int, entries []DocEntry) error {
	texts := make([]string, len(entries))
	vecs := make([]search.VectorEntry, len(entries))
	for i, e := range entries {
		text := e.Title + ": " + e.Content
		if len(text) > 800 {
			text = text[:800]
		}
		texts[i] = text
		vecs[i] = search.VectorEntry{
			ContentHash: search.ContentHash("doc:" + docVectorKey(sourceID, e.ID) + ":" + text),
			DocType:     "doc",
			SourceFile:  docVectorKey(sourceID, e.ID),
			Name:        e.Title,
			Kind:        "documentation",
			ProjectPath: "",
		}
	}
	embeddings, err := docEmbedder.Embed(texts)
	if err != nil {
		return err
	}
	for i := range vecs {
		vecs[i].Vector = embeddings[i]
	}
	return search.Cache.Upsert(vecs)
}

func docVectorKey(sourceID, entryID int) string {
	return fmt.Sprintf("doc:%d:%d", sourceID, entryID)
}

func deleteDocVectors(sourceID int) {
	prefix := fmt.Sprintf("doc:%d:%%", sourceID)
	db.DB.Exec("DELETE FROM vectors WHERE doc_type = 'doc' AND source_file LIKE ?", prefix)
	search.Cache.DeleteDocByPrefix(prefix)
}
