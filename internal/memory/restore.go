package memory

import (
	"database/sql"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

// RestoreInput configures a restore action in forget_memory.
type RestoreInput struct {
	Refs     []string
	Embedder embedder.Interface // optional: re-embed restored entries asynchronously
}

// RestoreResult describes the outcome of restoring superseded facts.
type RestoreResult struct {
	RestoredRefs []string `json:"restored_refs"`
	Conflicts    []string `json:"conflicts,omitempty"`
}

// Restore makes a superseded/forgotten fact active again, and invalidates the fact
// that superseded it, unless that fact has been superseded again, in which case it
// reports a conflict and changes nothing (MEM-4).
func Restore(in RestoreInput) (*RestoreResult, error) {
	var restored []string
	var conflicts []string
	type toReembed struct {
		ref  string
		line string
	}
	var reembedList []toReembed

	factSupersessionMu.Lock()
	defer factSupersessionMu.Unlock()

	for _, ref := range in.Refs {
		ref = strings.TrimSpace(ref)
		if ref == "" {
			continue
		}

		tx, err := db.ContextDB.Begin()
		if err != nil {
			return nil, err
		}

		// 1. Get the entry to restore (use sql.NullString for all nullable text columns).
		var supersededBy, validUntil sql.NullString
		var subject, predicate, object, rule sql.NullString
		err = tx.QueryRow(
			`SELECT superseded_by, valid_until, subject, predicate, object, rule FROM structured_memory WHERE ref = ?`,
			ref,
		).Scan(&supersededBy, &validUntil, &subject, &predicate, &object, &rule)
		if err == sql.ErrNoRows {
			tx.Rollback()
			conflicts = append(conflicts, ref+": not found")
			continue
		} else if err != nil {
			tx.Rollback()
			return nil, err
		}

		if !validUntil.Valid || validUntil.String == "" {
			tx.Rollback()
			// Already active — treat as success (no-op).
			restored = append(restored, ref)
			continue
		}

		// 2. Check if the fact that superseded it has itself been superseded again (MEM-4).
		// If B was superseded by C, B has superseded_by set (and valid_until set).
		// In that case, report a conflict and change nothing.
		if supersededBy.Valid && supersededBy.String != "" {
			var succSupersededBy sql.NullString
			err = tx.QueryRow(
				`SELECT superseded_by FROM structured_memory WHERE ref = ?`,
				supersededBy.String,
			).Scan(&succSupersededBy)
			if err == nil && succSupersededBy.Valid && succSupersededBy.String != "" {
				// The superseding fact has been superseded again — conflict!
				tx.Rollback()
				conflicts = append(conflicts, ref+": superseding fact "+supersededBy.String+" was superseded again by "+succSupersededBy.String)
				continue
			}
		}

		// 3. Reactivate the target ref: clear valid_until, invalidated_reason, and superseded_by.
		_, err = tx.Exec(
			`UPDATE structured_memory SET valid_until = NULL, invalidated_reason = NULL, superseded_by = NULL, access_count = access_count + 1 WHERE ref = ?`,
			ref,
		)
		if err != nil {
			tx.Rollback()
			return nil, err
		}

		// 4. Invalidate the fact that superseded it (B), setting superseded_by = ref, reason = ReasonRestored.
		if supersededBy.Valid && supersededBy.String != "" {
			_, err = tx.Exec(
				`UPDATE structured_memory SET valid_until = datetime('now'), superseded_by = ?, invalidated_reason = ? WHERE ref = ?`,
				ref, ReasonRestored, supersededBy.String,
			)
			if err != nil {
				tx.Rollback()
				return nil, err
			}
			deleteFTS(supersededBy.String)
			_ = search.Cache.DeleteRefs("memory", []string{memoryVectorKey(supersededBy.String)})
		}

		if err := tx.Commit(); err != nil {
			return nil, err
		}

		// Re-insert the restored entry into FTS so it becomes searchable again.
		indexFTS(ref, subject.String, predicate.String, object.String, rule.String)

		line := FormatLine(Entry{
			Subject:   subject.String,
			Predicate: predicate.String,
			Object:    object.String,
			Rule:      rule.String,
		})
		reembedList = append(reembedList, toReembed{ref: ref, line: line})
		restored = append(restored, ref)
	}

	// Re-embed restored entries asynchronously if an embedder was provided.
	if in.Embedder != nil && len(reembedList) > 0 {
		for _, item := range reembedList {
			item := item
			go func() {
				_, _ = EmbedEntryWithVector(item.ref, "", item.line, in.Embedder)
			}()
		}
	}

	return &RestoreResult{RestoredRefs: restored, Conflicts: conflicts}, nil
}
