package components

import (
	"fmt"
	"strings"
)

func chartTotal(items []BarItem) int {
	t := 0
	for _, item := range items {
		t += item.Value
	}
	return t
}

func barValueLabel(val, total int) string {
	if total <= 0 {
		return fmtInt(val)
	}
	pct := float64(val) / float64(total) * 100
	if pct >= 10 {
		return fmt.Sprintf("%s · %.0f%%", fmtInt(val), pct)
	}
	return fmt.Sprintf("%s · %.1f%%", fmtInt(val), pct)
}

func floatChartTotal(items []FloatBarItem) float64 {
	t := 0.0
	for _, item := range items {
		t += item.Value
	}
	return t
}

func floatBarValueLabel(val, total float64, unit string) string {
	label := formatFloatBarVal(val, unit)
	if total <= 0 {
		return label
	}
	pct := val / total * 100
	if pct >= 10 {
		return fmt.Sprintf("%s · %.0f%%", label, pct)
	}
	return fmt.Sprintf("%s · %.1f%%", label, pct)
}

func formatFloatBarVal(v float64, unit string) string {
	switch unit {
	case "cpu":
		if v >= 1000 {
			return fmt.Sprintf("%.1fs", v/1000)
		}
		if v >= 100 {
			return fmt.Sprintf("%.0fms", v)
		}
		return fmt.Sprintf("%.1fms", v)
	case "ms":
		if v >= 100 {
			return fmt.Sprintf("%.0fms", v)
		}
		return fmt.Sprintf("%.1fms", v)
	default:
		if v >= 100 {
			return fmt.Sprintf("%.0f", v)
		}
		return fmt.Sprintf("%.1f", v)
	}
}

func formatToolLabel(name string) string {
	switch name {
	case "get_context_capsule":
		return "context capsule"
	case "get_file_context":
		return "file context"
	case "get_project_map":
		return "project map"
	case "get_impact_graph":
		return "impact graph"
	case "search_semantic":
		return "semantic search"
	case "index_files":
		return "index files"
	case "index_status":
		return "index status"
	case "cache_summary":
		return "cache summary"
	case "analyze_dead_code":
		return "dead code"
	case "analyze_complexity":
		return "complexity"
	case "search_docs":
		return "search docs"
	case "fetch_doc":
		return "fetch doc"
	case "list_doc_sources":
		return "list docs"
	case "add_doc_source":
		return "add doc"
	case "remove_doc_source":
		return "remove doc"
	case "update_doc_source":
		return "update doc"
	case "export_bundle":
		return "export bundle"
	case "import_bundle":
		return "import bundle"
	case "execute_code":
		return "execute code"
	default:
		return strings.ReplaceAll(name, "_", " ")
	}
}
