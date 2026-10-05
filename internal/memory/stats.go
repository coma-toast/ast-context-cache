package memory

import "github.com/coma-toast/ast-context-cache/internal/db"

const (
	countActiveFactsQuery      = `SELECT COUNT(*) FROM structured_memory WHERE kind = 'fact' AND (valid_until IS NULL OR valid_until = '')`
	countActiveProceduresQuery = `SELECT COUNT(*) FROM structured_memory WHERE kind = 'procedure' AND (valid_until IS NULL OR valid_until = '')`
	sumActiveTokensQuery       = `SELECT COALESCE(SUM(token_est),0) FROM structured_memory WHERE valid_until IS NULL OR valid_until = ''`
	countOrphanEntriesQuery    = `SELECT COUNT(*) FROM structured_memory WHERE (valid_until IS NULL OR valid_until = '') AND (access_count IS NULL OR access_count = 0)`
	countRecalled30dQuery      = `SELECT COUNT(*) FROM memory_access WHERE accessed_at >= datetime('now', '-30 days')`
	countStored30dQuery        = `SELECT COUNT(*) FROM structured_memory WHERE created_at >= datetime('now', '-30 days')`
)

// InventoryStats for dashboard Memory tab.
type InventoryStats struct {
	ActiveFacts      int
	ActiveProcedures int
	ActiveTokens     int
	OrphanCount      int
	Recalled30d      int
	Stored30d        int
}

// Inventory returns current structured memory rollup.
func Inventory() InventoryStats {
	var s InventoryStats
	db.ContextDB.QueryRow(countActiveFactsQuery).Scan(&s.ActiveFacts)
	db.ContextDB.QueryRow(countActiveProceduresQuery).Scan(&s.ActiveProcedures)
	db.ContextDB.QueryRow(sumActiveTokensQuery).Scan(&s.ActiveTokens)
	db.ContextDB.QueryRow(countOrphanEntriesQuery).Scan(&s.OrphanCount)
	db.DB.QueryRow(countRecalled30dQuery).Scan(&s.Recalled30d)
	db.ContextDB.QueryRow(countStored30dQuery).Scan(&s.Stored30d)
	return s
}
