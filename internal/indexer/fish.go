package indexer

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func IndexFishFile(filePath, projectPath string) (int, error) {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return 0, err
	}

	db.DB.Exec("DELETE FROM symbols WHERE file = ? AND project_path = ?", filePath, projectPath)
	db.DB.Exec("DELETE FROM edges WHERE source_file = ? AND project_path = ?", filePath, projectPath)

	lines := strings.Split(string(content), "\n")
	count := 0

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "source ") {
			parts := strings.Fields(trimmed)
			if len(parts) >= 2 {
				db.DB.Exec("INSERT INTO edges (source_file, target, kind, project_path) VALUES (?, ?, 'import', ?)",
					filePath, parts[1], projectPath)
			}
		}
	}

	type funcInfo struct {
		name  string
		line  int
		depth int
	}

	var funcStack []funcInfo
	depth := 0

	for i, line := range lines {
		trimmed := strings.TrimSpace(line)

		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}

		if strings.HasPrefix(trimmed, "function ") {
			parts := strings.Fields(trimmed)
			if len(parts) >= 2 {
				funcStack = append(funcStack, funcInfo{name: parts[1], line: i, depth: depth})
			}
			depth++
			continue
		}

		if !strings.HasPrefix(trimmed, "else") {
			for _, kw := range []string{"if ", "for ", "while ", "switch "} {
				if strings.HasPrefix(trimmed, kw) {
					depth++
					break
				}
			}
			if trimmed == "begin" {
				depth++
			}
		}

		if trimmed == "end" || strings.HasPrefix(trimmed, "end;") || strings.HasPrefix(trimmed, "end #") {
			depth--
			if len(funcStack) > 0 && funcStack[len(funcStack)-1].depth == depth {
				fs := funcStack[len(funcStack)-1]
				funcStack = funcStack[:len(funcStack)-1]
				code := strings.TrimSpace(lines[fs.line])
				fqn := fmt.Sprintf("%s.%s", filepath.Base(filePath), fs.name)
				db.DB.Exec("INSERT INTO symbols (name, kind, file, start_line, end_line, code, fqn, project_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
					fs.name, "function", filePath, fs.line+1, i+1, code, fqn, projectPath)
				count++
			}
		}
	}
	db.UpsertIndexedFile(filePath, projectPath, time.Now())
	return count, nil
}
