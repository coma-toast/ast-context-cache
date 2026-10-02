package db

import (
	"encoding/json"
	"strings"
)

const settingProjectIndexExcludes = "project_index_excludes"

// GetProjectIndexExcludes returns project path → per-project exclude patterns
// (gitignore syntax, relative to the project root). Stored as one JSON map setting,
// like project_display_names.
func GetProjectIndexExcludes() map[string][]string {
	out := map[string][]string{}
	if err := json.Unmarshal([]byte(GetSetting(settingProjectIndexExcludes, "{}")), &out); err != nil {
		return map[string][]string{}
	}
	return out
}

// ProjectIndexExcludes returns the per-project exclude patterns for projectPath.
func ProjectIndexExcludes(projectPath string) []string {
	projectPath = normalizeProjectPath(projectPath)
	if projectPath == "" {
		return nil
	}
	return GetProjectIndexExcludes()[projectPath]
}

// SetProjectIndexExcludes replaces the per-project exclude patterns for projectPath.
// Blank lines are dropped; an empty list clears the entry.
func SetProjectIndexExcludes(projectPath string, patterns []string) error {
	projectPath = normalizeProjectPath(projectPath)
	if projectPath == "" {
		return nil
	}
	var cleaned []string
	for _, p := range patterns {
		if p = strings.TrimSpace(p); p != "" {
			cleaned = append(cleaned, p)
		}
	}
	all := GetProjectIndexExcludes()
	if len(cleaned) == 0 {
		delete(all, projectPath)
	} else {
		all[projectPath] = cleaned
	}
	b, err := json.Marshal(all)
	if err != nil {
		return err
	}
	return SetSetting(settingProjectIndexExcludes, string(b))
}
