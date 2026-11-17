package memory

import (
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
)

const (
	insertEntryFTSQuery   = `INSERT INTO structured_memory_fts (ref, subject, predicate, object, rule) VALUES (?, ?, ?, ?, ?)`
	deleteEntryFTSQuery   = `DELETE FROM structured_memory_fts WHERE ref = ?`
	insertEntryKBFTSQuery = `INSERT INTO structured_memory_kb_fts (ref, title, subject, predicate, object, rule, body, tags, target, before_value, after_value, reason, category, actor) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
	deleteEntryKBFTSQuery = `DELETE FROM structured_memory_kb_fts WHERE ref = ?`
)

func indexFTS(ref, subject, predicate, object, rule string) {
	if err := indexEntryFTS(Entry{
		Ref:       ref,
		Subject:   subject,
		Predicate: predicate,
		Object:    object,
		Rule:      rule,
	}); err != nil {
		logger.Warn("Failed to index memory FTS", "ref", ref, "error", err)
	}
}

// indexEntryFTS updates both structured_memory_fts and structured_memory_kb_fts
// for e, deleting any prior rows for e.Ref first so edits do not duplicate.
func indexEntryFTS(e Entry) error {
	if db.ContextDB == nil {
		return nil
	}
	if _, err := db.ContextDB.Exec(deleteEntryFTSQuery, e.Ref); err != nil {
		return errs.WrapCodeMessage(errs.CodeInternal, "failed to clear structured_memory_fts", err, "ref", e.Ref)
	}
	if _, err := db.ContextDB.Exec(deleteEntryKBFTSQuery, e.Ref); err != nil {
		return errs.WrapCodeMessage(errs.CodeInternal, "failed to clear structured_memory_kb_fts", err, "ref", e.Ref)
	}
	if _, err := db.ContextDB.Exec(insertEntryFTSQuery,
		e.Ref, e.Subject, e.Predicate, e.Object, e.Rule); err != nil {
		return errs.WrapCodeMessage(errs.CodeInternal, "failed to index structured_memory_fts", err, "ref", e.Ref)
	}
	tagsText := strings.Join(normalizeTagList(e.Tags), " ")
	if _, err := db.ContextDB.Exec(insertEntryKBFTSQuery,
		e.Ref, e.Title, e.Subject, e.Predicate, e.Object, e.Rule,
		e.Body, tagsText, e.Target, e.BeforeValue, e.AfterValue, e.Reason, e.Category, e.Actor); err != nil {
		return errs.WrapCodeMessage(errs.CodeInternal, "failed to index structured_memory_kb_fts", err, "ref", e.Ref)
	}
	return nil
}

func deleteFTS(ref string) {
	if db.ContextDB == nil {
		return
	}
	if _, err := db.ContextDB.Exec(deleteEntryFTSQuery, ref); err != nil {
		logger.Warn("Failed to delete structured_memory_fts row", "ref", ref, "error", err)
	}
	if _, err := db.ContextDB.Exec(deleteEntryKBFTSQuery, ref); err != nil {
		logger.Warn("Failed to delete structured_memory_kb_fts row", "ref", ref, "error", err)
	}
}
