package memory

import "github.com/coma-toast/ast-context-cache/internal/db"

// InventoryStats for dashboard Memory tab.
type InventoryStats struct {
	ActiveFacts       int
	ActiveProcedures  int
	ActiveTokens      int
	OrphanCount       int
	Recalled30d       int
	Stored30d         int
}

// Inventory returns current structured memory rollup.
func Inventory() InventoryStats {
	var s InventoryStats
	db.DB.QueryRow(`SELECT COUNT(*) FROM structured_memory WHERE kind = 'fact' AND (valid_until IS NULL OR valid_until = '')`).Scan(&s.ActiveFacts)
	db.DB.QueryRow(`SELECT COUNT(*) FROM structured_memory WHERE kind = 'procedure' AND (valid_until IS NULL OR valid_until = '')`).Scan(&s.ActiveProcedures)
	db.DB.QueryRow(`SELECT COALESCE(SUM(token_est),0) FROM structured_memory WHERE valid_until IS NULL OR valid_until = ''`).Scan(&s.ActiveTokens)
	db.DB.QueryRow(`SELECT COUNT(*) FROM structured_memory WHERE (valid_until IS NULL OR valid_until = '') AND (access_count IS NULL OR access_count = 0)`).Scan(&s.OrphanCount)
	db.DB.QueryRow(`SELECT COUNT(*) FROM memory_access WHERE accessed_at >= datetime('now', '-30 days')`).Scan(&s.Recalled30d)
	db.DB.QueryRow(`SELECT COUNT(*) FROM structured_memory WHERE created_at >= datetime('now', '-30 days')`).Scan(&s.Stored30d)
	return s
}
