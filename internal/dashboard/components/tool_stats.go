package components

type ToolStat struct {
	Name             string  `json:"tool_name"`
	Calls            int     `json:"calls"`
	AvgDurationMs    float64 `json:"avg_duration_ms"`
	TotalCpuMs       float64 `json:"total_cpu_ms"`
	AvgCpuMs         float64 `json:"avg_cpu_ms"`
	TokensSaved      int     `json:"tokens_saved"`
	DedupTokensSaved int     `json:"dedup_tokens_saved"`
	SavingsVsFiles   int     `json:"savings_vs_files"`
	SymbolBaseline   int     `json:"symbol_baseline_tokens"`
	SavingsRatePct   float64 `json:"savings_rate_pct"`
	AvgOutputTokens  float64 `json:"avg_output_tokens"`
	AvgResultChars   float64 `json:"avg_result_chars"`
	Errors           int     `json:"errors"`
}
