package mcp

import (
	"encoding/json"
	"path/filepath"
	"sort"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
)

func handleProjectMap(projectPath string, depth int) string {
	type fileEntry struct {
		path    string
		symbols []map[string]string
	}

	rows, err := db.DB.Query(
		"SELECT file, name, kind FROM symbols WHERE project_path = ? ORDER BY file, start_line",
		projectPath)
	if err != nil {
		data, _ := json.Marshal(map[string]string{"error": err.Error()})
		return string(data)
	}
	defer rows.Close()

	files := map[string][]map[string]string{}
	for rows.Next() {
		var file, name, kind string
		rows.Scan(&file, &name, &kind)
		files[file] = append(files[file], map[string]string{"name": name, "kind": kind})
	}

	sortedFiles := make([]string, 0, len(files))
	for f := range files {
		sortedFiles = append(sortedFiles, f)
	}
	sort.Strings(sortedFiles)

	type dirNode struct {
		Name    string                 `json:"name"`
		Type    string                 `json:"type"`
		Files   []map[string]interface{} `json:"files,omitempty"`
		Dirs    []*dirNode             `json:"dirs,omitempty"`
	}

	dirs := map[string]*dirNode{}
	var topDirs []string

	for _, f := range sortedFiles {
		rel, err := filepath.Rel(projectPath, f)
		if err != nil {
			continue
		}
		dir := filepath.Dir(rel)
		if _, ok := dirs[dir]; !ok {
			dirs[dir] = &dirNode{Name: dir, Type: "directory"}
			topDirs = append(topDirs, dir)
		}

		if depth >= 2 {
			fileInfo := map[string]interface{}{
				"name":    filepath.Base(f),
				"symbols": len(files[f]),
			}
			if depth >= 3 {
				fileInfo["symbol_list"] = files[f]
			}
			dirs[dir].Files = append(dirs[dir].Files, fileInfo)
		}
	}

	sort.Strings(topDirs)

	result := map[string]interface{}{
		"project_path": projectPath,
		"depth":        depth,
		"total_files":  len(sortedFiles),
	}

	if depth == 1 {
		dirNames := make([]string, len(topDirs))
		for i, d := range topDirs {
			dirNames[i] = d
		}
		result["directories"] = dirNames
	} else {
		tree := make([]map[string]interface{}, 0)
		for _, d := range topDirs {
			dn := dirs[d]
			entry := map[string]interface{}{
				"directory": dn.Name,
			}
			if dn.Files != nil {
				entry["files"] = dn.Files
			}
			tree = append(tree, entry)
		}
		result["tree"] = tree
	}

	data, _ := json.Marshal(result)
	return string(data)
}

func handleFileContext(file, projectPath, mode string) string {
	rows, err := db.DB.Query(
		"SELECT name, kind, start_line, end_line, COALESCE(skeleton,''), COALESCE(code,'') FROM symbols WHERE file = ? AND project_path = ? ORDER BY start_line",
		file, projectPath)
	if err != nil {
		data, _ := json.Marshal(map[string]string{"error": err.Error()})
		return string(data)
	}
	defer rows.Close()

	fileCache := map[string][]string{}
	var symbols []map[string]interface{}

	for rows.Next() {
		var name, kind, skeleton, code string
		var startLine, endLine int
		rows.Scan(&name, &kind, &startLine, &endLine, &skeleton, &code)

		sym := map[string]interface{}{
			"name":       name,
			"kind":       kind,
			"start_line": startLine,
			"end_line":   endLine,
		}

		switch mode {
		case "skeleton":
			if skeleton != "" {
				sym["skeleton"] = skeleton
			} else if src := indexer.ReadSourceRange(file, startLine, endLine, fileCache); src != "" {
				lang := indexer.GetLanguage(file)
				sym["skeleton"] = indexer.ExtractSkeleton(src, lang, kind)
			}
		case "summary":
			var summary string
			db.DB.QueryRow("SELECT summary_text FROM summaries WHERE file_path = ? AND symbol_name = ? AND project_path = ?",
				file, name, projectPath).Scan(&summary)
			if summary != "" {
				sym["summary"] = summary
			} else if skeleton != "" {
				sym["skeleton"] = skeleton
				sym["_fallback"] = "skeleton"
			}
		default: // "full"
			if src := indexer.ReadSourceRange(file, startLine, endLine, fileCache); src != "" {
				sym["source"] = src
			}
		}

		symbols = append(symbols, sym)
	}

	lang := ""
	ext := strings.ToLower(filepath.Ext(file))
	switch ext {
	case ".py":
		lang = "Python"
	case ".go":
		lang = "Go"
	case ".js", ".jsx":
		lang = "JavaScript"
	case ".ts":
		lang = "TypeScript"
	case ".tsx":
		lang = "TSX"
	case ".sh":
		lang = "Bash"
	case ".fish":
		lang = "Fish"
	}

	data, _ := json.Marshal(map[string]interface{}{
		"file":     file,
		"language": lang,
		"mode":     mode,
		"symbols":  symbols,
		"total":    len(symbols),
	})
	return string(data)
}
