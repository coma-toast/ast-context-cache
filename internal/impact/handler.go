package impact

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

type impactEntryOut struct {
	File   string `json:"file"`
	Target string `json:"target"`
	Kind   string `json:"kind"`
}

func HandleImpactGraph(args map[string]interface{}, projectPath string) string {
	symbol, _ := args["symbol"].(string)
	if projectPath == "" {
		return `{"error": "project_path required"}`
	}
	if symbol == "" {
		return `{"error": "symbol required"}`
	}

	symbolLower := strings.ToLower(symbol)

	symbolRows, err := db.DB.Query(
		"SELECT DISTINCT file FROM symbols WHERE project_path = ? AND LOWER(name) = ?",
		projectPath, symbolLower)
	if err != nil {
		return fmt.Sprintf(`{"error": "%s"}`, err.Error())
	}
	defer symbolRows.Close()

	symbolFiles := map[string]bool{}
	for symbolRows.Next() {
		var f string
		symbolRows.Scan(&f)
		symbolFiles[f] = true
	}

	type impactEntry struct {
		File   string `json:"file"`
		Target string `json:"target"`
		Kind   string `json:"kind"`
	}
	var impacts []impactEntry

	edgeRows, err := db.DB.Query(
		"SELECT source_file, target, kind FROM edges WHERE project_path = ? AND (LOWER(target) LIKE ? OR LOWER(target) LIKE ?)",
		projectPath, "%"+symbolLower+"%", "%/"+symbolLower)
	if err != nil {
		return fmt.Sprintf(`{"error": "%s"}`, err.Error())
	}
	defer edgeRows.Close()

	seen := map[string]bool{}
	for edgeRows.Next() {
		var srcFile, target, kind string
		edgeRows.Scan(&srcFile, &target, &kind)
		if !seen[srcFile] {
			seen[srcFile] = true
			impacts = append(impacts, impactEntry{File: srcFile, Target: target, Kind: kind})
		}
	}

	for f := range symbolFiles {
		depRows, _ := db.DB.Query(
			"SELECT source_file, target, kind FROM edges WHERE project_path = ? AND LOWER(target) LIKE ?",
			projectPath, "%"+strings.ToLower(filepath.Base(f))+"%")
		if depRows != nil {
			for depRows.Next() {
				var srcFile, target, kind string
				depRows.Scan(&srcFile, &target, &kind)
				if !seen[srcFile] {
					seen[srcFile] = true
					impacts = append(impacts, impactEntry{File: srcFile, Target: target, Kind: kind})
				}
			}
			depRows.Close()
		}
	}

	keys := make([]string, 0, len(symbolFiles))
	for k := range symbolFiles {
		keys = append(keys, db.RelPath(k, projectPath))
	}

	relImpacts := make([]impactEntryOut, len(impacts))
	for i, imp := range impacts {
		relImpacts[i] = impactEntryOut{
			File:   db.RelPath(imp.File, projectPath),
			Target: imp.Target,
			Kind:   imp.Kind,
		}
	}

	data, _ := json.Marshal(map[string]interface{}{
		"symbol":      symbol,
		"defined_in":  keys,
		"impacted_by": relImpacts,
		"total_files": len(seen),
	})
	return string(data)
}
