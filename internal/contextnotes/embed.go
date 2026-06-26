package contextnotes

import (
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

func noteVectorKey(ref string) string {
	return "note:" + ref
}

// EmbedNote embeds one stored note for semantic search.
func EmbedNote(ref, sessionID, label, content string, emb embedder.Interface) {
	if emb == nil || ref == "" {
		return
	}
	text := label + ": " + content
	if len(text) > 800 {
		text = text[:800]
	}
	vec, err := emb.EmbedSingle(text)
	if err != nil {
		return
	}
	entry := search.VectorEntry{
		ContentHash: search.ContentHash("note:" + ref + ":" + text),
		DocType:     "note",
		SourceFile:  noteVectorKey(ref),
		Name:        label,
		Kind:        "context_note",
		ProjectPath: sessionID,
		Vector:      vec,
	}
	_ = search.Cache.Upsert([]search.VectorEntry{entry})
}

func deleteNoteVector(ref string) {
	key := noteVectorKey(ref)
	db.IndexDB.Exec(`DELETE FROM vectors WHERE doc_type = 'note' AND source_file = ?`, key)
	search.Cache.DeleteNoteByRef(key)
}
