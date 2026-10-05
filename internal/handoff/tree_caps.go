package handoff

import (
	"database/sql"
	"errors"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

const (
	selectTreeUsageQuery = `SELECT tokens_used, entries_used FROM handoff_trees WHERE tree_id = ?`
	updateTreeUsageQuery = `UPDATE handoff_trees SET tokens_used = MAX(0, tokens_used + ?), entries_used = MAX(0, entries_used + ?)
		WHERE tree_id = ?`
)

// treeUsage is a tree's running total against its RQ-4 caps.
type treeUsage struct {
	tokens, entries int
}

// chargeTreeTx adds tokens and entries to tree's usage in tx (RQ-4). A positive charge that
// would take either total past its cap fails with CodeHandoffTreeLimitExceeded and the current
// usage, caps, and requested amounts, and leaves the usage unchanged. Negative amounts credit
// the tree back, as when trail entries are evicted.
func (s *realService) chargeTreeTx(tx *sql.Tx, tree TreeID, tokens, entries int) error {
	u, err := treeUsageTx(tx, tree)
	if err != nil {
		return err
	}
	lim := LoadLimits()
	overTokens := tokens > 0 && u.tokens+tokens > lim.TreeMaxTokens
	overEntries := entries > 0 && u.entries+entries > lim.TreeMaxEntries
	if overTokens || overEntries {
		return errs.NewCode(CodeHandoffTreeLimitExceeded, "handoff tree is at its cap", "tree", string(tree),
			"tokens_used", u.tokens, "tokens_max", lim.TreeMaxTokens, "tokens_requested", tokens,
			"entries_used", u.entries, "entries_max", lim.TreeMaxEntries, "entries_requested", entries)
	}
	if _, err := tx.Exec(updateTreeUsageQuery, tokens, entries, string(tree)); err != nil {
		return errs.WrapMessage("failed to update handoff tree usage", err, "tree", string(tree))
	}
	return nil
}

// treeUsageTx reads tree's usage; a missing tree is CodeHandoffNotFound.
func treeUsageTx(tx *sql.Tx, tree TreeID) (treeUsage, error) {
	var u treeUsage
	err := tx.QueryRow(selectTreeUsageQuery, string(tree)).Scan(&u.tokens, &u.entries)
	if errors.Is(err, sql.ErrNoRows) {
		return u, errs.NewCode(CodeHandoffNotFound, "handoff tree not found", "tree", string(tree))
	}
	if err != nil {
		return u, errs.WrapMessage("failed to read handoff tree usage", err, "tree", string(tree))
	}
	return u, nil
}
