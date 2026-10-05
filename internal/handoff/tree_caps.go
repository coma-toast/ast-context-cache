package handoff

import (
	"database/sql"
	"errors"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

const (
	selectTreeUsageQuery = `SELECT tokens_used, entries_used FROM handoff_trees WHERE tree_id = ?`
	chargeTreeQuery      = `UPDATE handoff_trees SET tokens_used = MAX(0, tokens_used + ?), entries_used = MAX(0, entries_used + ?)
		WHERE tree_id = ?`
)

// chargeTreeTx adds tokens and entries to tree's usage (RQ-4). A positive charge that would take
// the tree past handoff_tree_max_tokens or handoff_tree_max_entries fails with
// CodeHandoffTreeLimitExceeded, carrying the usage, caps, and the charge, and changes nothing.
// A negative charge (an evicted entry) always applies, flooring usage at zero.
func (s *realService) chargeTreeTx(tx *sql.Tx, tree TreeID, tokens, entries int) error {
	var usedTokens, usedEntries int
	err := tx.QueryRow(selectTreeUsageQuery, string(tree)).Scan(&usedTokens, &usedEntries)
	if errors.Is(err, sql.ErrNoRows) {
		return errs.NewCode(CodeHandoffNotFound, "handoff tree not found", "tree", string(tree))
	}
	if err != nil {
		return errs.WrapMessage("failed to read handoff tree usage", err, "tree", string(tree))
	}
	l := LoadLimits()
	overTokens := tokens > 0 && usedTokens+tokens > l.TreeMaxTokens
	overEntries := entries > 0 && usedEntries+entries > l.TreeMaxEntries
	if overTokens || overEntries {
		return errs.NewCode(CodeHandoffTreeLimitExceeded, "handoff tree cap exceeded", "tree", string(tree),
			"tokens_used", usedTokens, "tokens_max", l.TreeMaxTokens, "would_add_tokens", tokens,
			"entries_used", usedEntries, "entries_max", l.TreeMaxEntries, "would_add_entries", entries)
	}
	if _, err := tx.Exec(chargeTreeQuery, tokens, entries, string(tree)); err != nil {
		return errs.WrapMessage("failed to update handoff tree usage", err, "tree", string(tree))
	}
	return nil
}
