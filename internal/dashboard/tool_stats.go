package dashboard

import (
	"database/sql"

	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/db"
)

const toolStatsSelect = `SELECT tool_name,
	COUNT(*),
	COALESCE(AVG(duration_ms), 0),
	COALESCE(SUM(cpu_ms), 0),
	COALESCE(AVG(cpu_ms), 0),
	COALESCE(SUM(tokens_saved), 0),
	COALESCE(SUM(dedup_tokens_saved), 0),
	COALESCE(SUM(savings_vs_files), 0),
	COALESCE(SUM(symbol_baseline_tokens), 0),
	COALESCE(AVG(output_tokens), 0),
	COALESCE(AVG(result_chars), 0),
	SUM(CASE WHEN COALESCE(error, '') != '' THEN 1 ELSE 0 END)
FROM queries WHERE `

func queryToolStats(projectID string) []components.ToolStat {
	where := excludeWatcherFromToolStats
	args := []any{}
	if projectID != "" {
		where += " AND project_path = ?"
		args = append(args, projectID)
	}
	q := toolStatsSelect + where + " GROUP BY tool_name ORDER BY COUNT(*) DESC"
	var rows *sql.Rows
	var err error
	if len(args) > 0 {
		rows, err = db.DB.Query(q, args...)
	} else {
		rows, err = db.DB.Query(q)
	}
	if err != nil {
		return nil
	}
	defer rows.Close()
	var out []components.ToolStat
	for rows.Next() {
		var s components.ToolStat
		if err := rows.Scan(&s.Name, &s.Calls, &s.AvgDurationMs, &s.TotalCpuMs, &s.AvgCpuMs, &s.TokensSaved, &s.DedupTokensSaved, &s.SavingsVsFiles, &s.SymbolBaseline, &s.AvgOutputTokens, &s.AvgResultChars, &s.Errors); err != nil {
			continue
		}
		if s.SymbolBaseline > 0 {
			s.SavingsRatePct = float64(s.TokensSaved) / float64(s.SymbolBaseline) * 100
		}
		out = append(out, s)
	}
	return out
}
