package memory

import (
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

func memoryVectorKey(ref string) string {
	return "mem:" + ref
}

// EmbedEntry embeds structured memory for semantic recall_memory.
func EmbedEntry(ref string, sessionID string, line string, emb embedder.Interface) {
	if emb == nil || ref == "" || line == "" {
		return
	}
	if len(line) > 400 {
		line = line[:400]
	}
	vec, err := emb.EmbedSingle(line)
	if err != nil {
		return
	}
	entry := search.VectorEntry{
		ContentHash: search.ContentHash("memory:" + ref + ":" + line),
		DocType:     "memory",
		SourceFile:  memoryVectorKey(ref),
		Name:        line,
		Kind:        "structured_memory",
		ProjectPath: sessionID,
		Vector:      vec,
	}
	_ = search.Cache.Upsert([]search.VectorEntry{entry})
}
