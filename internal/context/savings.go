package context

import (
	"encoding/json"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

// SavingsMeta holds token savings breakdown for analytics and MCP responses.
type SavingsMeta struct {
	TokensUsed       int
	SymbolBaseline   int
	FileBaseline     int
	DedupTokensSaved int
	DedupedCount     int
	TokensSaved      int
	SavingsVsFiles   int
	Mode             string
	CacheHit         bool
}

// ComputeSavings derives aggregate savings from component counts.
func ComputeSavings(tokensUsed, symbolBaseline, fileBaseline, dedupTokens int) SavingsMeta {
	modeSaved := symbolBaseline - tokensUsed
	if modeSaved < 0 {
		modeSaved = 0
	}
	vsFiles := fileBaseline - tokensUsed
	if vsFiles < 0 {
		vsFiles = 0
	}
	return SavingsMeta{
		TokensUsed:       tokensUsed,
		SymbolBaseline:   symbolBaseline,
		FileBaseline:     fileBaseline,
		DedupTokensSaved: dedupTokens,
		TokensSaved:      modeSaved + dedupTokens,
		SavingsVsFiles:   vsFiles + dedupTokens,
	}
}

func (m SavingsMeta) ApplyTo(resp map[string]interface{}) {
	resp["tokens_used"] = m.TokensUsed
	resp["symbol_baseline_tokens"] = m.SymbolBaseline
	resp["file_baseline_tokens"] = m.FileBaseline
	resp["full_baseline_tokens"] = m.SymbolBaseline
	resp["dedup_tokens_saved"] = m.DedupTokensSaved
	resp["tokens_saved"] = m.TokensSaved
	resp["savings_vs_files"] = m.SavingsVsFiles
	if m.DedupedCount > 0 {
		resp["deduped"] = m.DedupedCount
	}
	if m.CacheHit {
		resp["cache_hit"] = true
	}
}

// FullSourceTokens estimates tokens for a symbol's full source body.
func FullSourceTokens(file, name, projectPath string, startLine, endLine int, fileCache map[string][]string) int {
	if startLine > 0 && endLine > 0 {
		if src := indexer.ReadSourceRange(file, startLine, endLine, fileCache); src != "" {
			return db.EstimateTokens(src)
		}
	}
	var code string
	db.DB.QueryRow(
		"SELECT COALESCE(code,'') FROM symbols WHERE file = ? AND name = ? AND project_path = ? AND start_line = ? LIMIT 1",
		file, name, projectPath, startLine).Scan(&code)
	return db.EstimateTokens(code)
}

// WouldSendTokens estimates payload tokens if a symbol were returned in the given mode.
func WouldSendTokens(file, name, projectPath, mode string, startLine, endLine int, score, maxScore float64, fullCount int, fileCache map[string][]string) int {
	sym := map[string]interface{}{
		"name":       name,
		"start_line": startLine,
		"end_line":   endLine,
	}
	if kind := symbolKind(file, name, projectPath, startLine); kind != "" {
		sym["kind"] = kind
	}
	effective := EffectiveMode(mode, score, maxScore, fullCount)
	ApplyMode(sym, effective, file, name, projectPath, startLine, endLine, fileCache)
	sym["file"] = db.RelPath(file, projectPath)
	b, _ := json.Marshal(sym)
	return db.EstimateTokens(string(b))
}

func symbolKind(file, name, projectPath string, startLine int) string {
	var kind string
	db.DB.QueryRow(
		"SELECT COALESCE(kind,'') FROM symbols WHERE file = ? AND name = ? AND project_path = ? AND start_line = ? LIMIT 1",
		file, name, projectPath, startLine).Scan(&kind)
	return kind
}

// FileBaselineTokens sums whole-file token estimates for matched files.
func FileBaselineTokens(matchedFiles map[string]bool, fileCache map[string][]string) int {
	total := 0
	for f := range matchedFiles {
		if lines, ok := fileCache[f]; ok {
			total += db.EstimateTokens(strings.Join(lines, "\n"))
		}
	}
	return total
}

// ParseSavingsMeta reads savings fields from a tool response map.
func ParseSavingsMeta(parsed map[string]interface{}, mode string, cacheHit bool) SavingsMeta {
	m := SavingsMeta{Mode: mode, CacheHit: cacheHit}
	if v, ok := parsed["tokens_used"].(float64); ok {
		m.TokensUsed = int(v)
	}
	if v, ok := parsed["symbol_baseline_tokens"].(float64); ok {
		m.SymbolBaseline = int(v)
	} else if v, ok := parsed["full_baseline_tokens"].(float64); ok {
		m.SymbolBaseline = int(v)
	}
	if v, ok := parsed["file_baseline_tokens"].(float64); ok {
		m.FileBaseline = int(v)
	}
	if v, ok := parsed["dedup_tokens_saved"].(float64); ok {
		m.DedupTokensSaved = int(v)
	}
	if v, ok := parsed["tokens_saved"].(float64); ok {
		m.TokensSaved = int(v)
	}
	if v, ok := parsed["savings_vs_files"].(float64); ok {
		m.SavingsVsFiles = int(v)
	}
	if v, ok := parsed["deduped"].(float64); ok {
		m.DedupedCount = int(v)
	}
	if m.TokensSaved == 0 && m.SymbolBaseline > 0 {
		computed := ComputeSavings(m.TokensUsed, m.SymbolBaseline, m.FileBaseline, m.DedupTokensSaved)
		m.TokensSaved = computed.TokensSaved
		m.SavingsVsFiles = computed.SavingsVsFiles
	}
	return m
}

// PackHit holds hybrid search hit context for savings loops.
type PackHit struct {
	Data      map[string]interface{}
	Score     float64
	StartLine int
	EndLine   int
}

func hitFromScored(r search.ScoredResult, projectPath string) PackHit {
	data := r.Data
	file, _ := data["file"].(string)
	name, _ := data["name"].(string)
	startLine, _ := data["start_line"].(int)
	endLine, _ := data["end_line"].(int)
	if startLine == 0 {
		db.DB.QueryRow("SELECT COALESCE(start_line,0), COALESCE(end_line,0) FROM symbols WHERE file = ? AND name = ? AND project_path = ? LIMIT 1",
			file, name, projectPath).Scan(&startLine, &endLine)
		data["start_line"] = startLine
		data["end_line"] = endLine
	}
	return PackHit{Data: data, Score: r.Score, StartLine: startLine, EndLine: endLine}
}
