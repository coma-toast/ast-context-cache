package handoff

import (
	"database/sql"
	"errors"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

const (
	selectTreeUsageQuery = `SELECT tokens_used, entries_used FROM handoff_trees WHERE tree_id = ?`
	chargeTreeQuery      = `UPDATE handoff_trees SET tokens_used = tokens_used + ?, entries_used = entries_used + ? WHERE tree_id = ?`
)

// chargeTreeTx adds tokens and entries to tree's usage, or fails with
// CodeHandoffTreeLimitExceeded and the usage details when that would pass a cap (RQ-4, RQ-5).
// It runs inside the transaction of the insert it accounts for, so usage never drifts from the
// rows it counts.
func (s *realService) chargeTreeTx(tx *sql.Tx, tree TreeID, tokens, entries int) error {
	var usedTokens, usedEntries int
	err := tx.QueryRow(selectTreeUsageQuery, string(tree)).Scan(&usedTokens, &usedEntries)
	if errors.Is(err, sql.ErrNoRows) {
		return errs.NewCode(CodeHandoffNotFound, "handoff tree not found", "tree", string(tree))
	}
	if err != nil {
		return errs.WrapMessage("failed to read handoff tree usage", err, "tree", string(tree))
	}
	lim := LoadLimits()
	if usedTokens+tokens > lim.TreeMaxTokens || usedEntries+entries > lim.TreeMaxEntries {
		return errs.NewCode(CodeHandoffTreeLimitExceeded, "handoff tree limit exceeded", "tree", string(tree),
			"tokens_used", usedTokens, "tokens_max", lim.TreeMaxTokens, "entries_used", usedEntries,
			"entries_max", lim.TreeMaxEntries, "would_add_tokens", tokens, "would_add_entries", entries)
	}
	if _, err := tx.Exec(chargeTreeQuery, tokens, entries, string(tree)); err != nil {
		return errs.WrapMessage("failed to charge handoff tree usage", err, "tree", string(tree))
	}
	return nil
}
