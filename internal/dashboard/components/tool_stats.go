package components

import "fmt"

type ToolStat struct {
	Name            string  `json:"tool_name"`
	Calls           int     `json:"calls"`
	AvgDurationMs   float64 `json:"avg_duration_ms"`
	TotalCpuMs      float64 `json:"total_cpu_ms"`
	AvgCpuMs        float64 `json:"avg_cpu_ms"`
	TokensSaved     int     `json:"tokens_saved"`
	AvgOutputTokens float64 `json:"avg_output_tokens"`
	AvgResultChars  float64 `json:"avg_result_chars"`
	Errors          int     `json:"errors"`
}

func toolStatBars(stats []ToolStat, kind string) []BarItem {
	var items []BarItem
	for _, s := range stats {
		if len(items) >= 10 {
			break
		}
		val := 0
		switch kind {
		case "calls":
			val = s.Calls
		case "tokens":
			val = s.TokensSaved
		}
		if val <= 0 && kind == "tokens" {
			continue
		}
		items = append(items, BarItem{Label: s.Name, Value: val})
	}
	return items
}

func toolStatFloatBars(stats []ToolStat, kind string) []FloatBarItem {
	var items []FloatBarItem
	for _, s := range stats {
		if len(items) >= 10 {
			break
		}
		var val float64
		switch kind {
		case "cpu":
			val = s.TotalCpuMs
		case "latency":
			val = s.AvgDurationMs
		}
		if val <= 0 {
			continue
		}
		items = append(items, FloatBarItem{Label: s.Name, Value: val})
	}
	return items
}

func fmtToolMs(v float64) string {
	if v <= 0 {
		return "-"
	}
	if v >= 100 {
		return fmt.Sprintf("%.0f", v)
	}
	return fmt.Sprintf("%.1f", v)
}

func fmtToolCpu(v float64) string {
	if v <= 0 {
		return "-"
	}
	if v >= 1000 {
		return fmt.Sprintf("%.1fs", v/1000)
	}
	if v >= 100 {
		return fmt.Sprintf("%.0fms", v)
	}
	return fmt.Sprintf("%.1fms", v)
}

func fmtToolFloat(v float64) string {
	if v <= 0 {
		return "-"
	}
	if v >= 100 {
		return fmt.Sprintf("%.0f", v)
	}
	return fmt.Sprintf("%.1f", v)
}
