package dashboard

import (
	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/db"
)

const (
	selectLedgerStatsBaseQuery = "SELECT " + compressionSavedSum + ", " + ledgerDedupSavedSum + ", " + conservativeSavedSum + ", " +
		virtualStoredSum + ", " + virtualFetchedSum + ", " + virtualRecalledSum + ", " + estimatedRowsSum + " FROM queries WHERE "
	ledgerCompression   = "compression"
	ledgerDedup         = "dedup"
	ledgerConservative  = "conservative"
	ledgerVirtual       = "virtual"
	ledgerEstimatedNote = "estimated"
)

// baselineDefinitions explains each savings number; the Overview shows them as tooltips.
var baselineDefinitions = map[string]string{
	ledgerCompression:   "Full source of every returned symbol minus the tokens actually returned (search and read tools).",
	ledgerDedup:         "Tokens not re-sent because the session already received those symbols.",
	ledgerConservative:  "Each returned symbol plus 20 lines above and below, merged per file, minus the tokens returned.",
	ledgerVirtual:       "Tokens written to virtual context and memory, and tokens later fetched or recalled. Not counted as savings.",
	ledgerEstimatedNote: "Rows logged before the tokenizer counted tokens as bytes/4 estimates.",
}

// fillLedgerStats fills the ledger fields over the stats window.
func fillLedgerStats(s *components.Stats, projectID string) {
	s.BaselineDefinitions = baselineDefinitions
	if db.DB == nil {
		return
	}
	where, args := statsQueriesWhere(projectID)
	if err := db.DB.QueryRow(selectLedgerStatsBaseQuery+where, args...).Scan(
		&s.CompressionSaved, &s.DedupSaved, &s.ConservativeSaved,
		&s.VirtualStoredTokens, &s.VirtualFetchedTokens, &s.VirtualRecalledTokens, &s.EstimatedRows,
	); err != nil {
		logger.Warn("Failed to read ledger stats", "error", err, "project", projectID)
	}
}

// ledgerTotals is the ledger split for a time window.
type ledgerTotals struct {
	CompressionSaved      int `json:"CompressionSaved"`
	DedupSaved            int `json:"DedupSaved"`
	ConservativeSaved     int `json:"ConservativeSaved"`
	VirtualStoredTokens   int `json:"VirtualStoredTokens"`
	VirtualFetchedTokens  int `json:"VirtualFetchedTokens"`
	VirtualRecalledTokens int `json:"VirtualRecalledTokens"`
}

func queryLedgerWindow(projectID string, days int) ledgerTotals {
	var t ledgerTotals
	if db.DB == nil {
		return t
	}
	where := windowOffsetFilter
	args := []any{fmtDaysOffset(days)}
	if projectID != "" {
		where += projectPathClause
		args = append(args, projectID)
	}
	var estimated int
	if err := db.DB.QueryRow(selectLedgerStatsBaseQuery+where, args...).Scan(
		&t.CompressionSaved, &t.DedupSaved, &t.ConservativeSaved,
		&t.VirtualStoredTokens, &t.VirtualFetchedTokens, &t.VirtualRecalledTokens, &estimated,
	); err != nil {
		logger.Warn("Failed to read ledger window", "error", err, "project", projectID, "days", days)
	}
	return t
}
