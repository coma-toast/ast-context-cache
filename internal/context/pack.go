package context

import (
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

// EffectiveMode resolves auto mode from score rank.
func EffectiveMode(mode string, score, maxScore float64, fullCount int) string {
	if mode != "auto" {
		return mode
	}
	if maxScore > 0 {
		ratio := score / maxScore
		if ratio >= 0.5 || fullCount < 2 {
			return "full"
		}
		if ratio >= 0.2 {
			return "skeleton"
		}
		return "summary"
	}
	if fullCount < 3 {
		return "full"
	}
	return "skeleton"
}

func LoadSummary(file, name, projectPath string) string {
	var summary, storedHash string
	err := db.DB.QueryRow(
		"SELECT summary_text, content_hash FROM summaries WHERE file_path = ? AND symbol_name = ? AND project_path = ?",
		file, name, projectPath).Scan(&summary, &storedHash)
	if err != nil || summary == "" {
		if name != "" {
			db.DB.QueryRow(
				"SELECT summary_text, content_hash FROM summaries WHERE file_path = ? AND (symbol_name IS NULL OR symbol_name = '') AND project_path = ?",
				file, projectPath).Scan(&summary, &storedHash)
		}
	}
	if summary == "" {
		return ""
	}
	current := symbolContentHash(file, name, projectPath)
	if storedHash != "" && current != "" && storedHash != current {
		return ""
	}
	return summary
}

func symbolContentHash(file, name, projectPath string) string {
	var code string
	db.DB.QueryRow(
		"SELECT COALESCE(code,'') FROM symbols WHERE file = ? AND name = ? AND project_path = ? LIMIT 1",
		file, name, projectPath).Scan(&code)
	if code != "" {
		return search.ContentHash(code)
	}
	return ""
}

// ApplyMode mutates data with source/skeleton/summary for the effective mode.
func ApplyMode(data map[string]interface{}, effectiveMode, file, name, projectPath string, startLine, endLine int, fileCache map[string][]string) {
	var fullSrc string
	if startLine > 0 && endLine > 0 {
		fullSrc = indexer.ReadSourceRange(file, startLine, endLine, fileCache)
	}
	kind, _ := data["kind"].(string)
	switch effectiveMode {
	case "skeleton":
		var skeleton string
		db.DB.QueryRow("SELECT COALESCE(skeleton,'') FROM symbols WHERE file = ? AND name = ? AND project_path = ? AND start_line = ? LIMIT 1",
			file, name, projectPath, startLine).Scan(&skeleton)
		if skeleton != "" {
			data["skeleton"] = skeleton
		} else if fullSrc != "" {
			data["skeleton"] = indexer.ExtractSkeleton(fullSrc, indexer.GetLanguage(file), kind)
		}
	case "summary":
		if summary := LoadSummary(file, name, projectPath); summary != "" {
			data["summary"] = summary
		} else {
			var skeleton string
			db.DB.QueryRow("SELECT COALESCE(skeleton,'') FROM symbols WHERE file = ? AND name = ? AND project_path = ? AND start_line = ? LIMIT 1",
				file, name, projectPath, startLine).Scan(&skeleton)
			if skeleton != "" {
				data["skeleton"] = skeleton
				data["_fallback"] = "skeleton"
			}
		}
	default:
		if fullSrc != "" {
			data["source"] = fullSrc
		}
	}
}

// SymbolContentForRetrieve picks chunk text for retrieve (skeleton vs full).
func SymbolContentForRetrieve(file, name, projectPath string, startLine, endLine int, includeSource bool, mode string, score, maxScore float64, fullCount int, fileCache map[string][]string) string {
	var code, skeleton string
	db.DB.QueryRow(
		"SELECT COALESCE(code,''), COALESCE(skeleton,'') FROM symbols WHERE name = ? AND file = ? AND project_path = ? AND start_line = ? LIMIT 1",
		name, file, projectPath, startLine).Scan(&code, &skeleton)
	effective := mode
	if effective == "" {
		effective = "skeleton"
	}
	if effective == "auto" {
		effective = EffectiveMode("auto", score, maxScore, fullCount)
	}
	if includeSource || effective == "full" {
		if startLine > 0 && endLine > 0 {
			if src := indexer.ReadSourceRange(file, startLine, endLine, fileCache); src != "" {
				return src
			}
		}
		if code != "" {
			return code
		}
	}
	if skeleton != "" {
		return skeleton
	}
	if code != "" {
		return code
	}
	return ""
}
