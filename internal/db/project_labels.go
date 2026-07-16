package db

import (
	"encoding/json"
	"path/filepath"
	"strings"
)

const settingProjectDisplayNames = "project_display_names"

// GetProjectDisplayNames returns path → custom dashboard label overrides.
func GetProjectDisplayNames() map[string]string {
	s := GetSetting(settingProjectDisplayNames, "{}")
	out := map[string]string{}
	if err := json.Unmarshal([]byte(s), &out); err != nil {
		return map[string]string{}
	}
	return out
}

// ProjectDisplayName returns the custom label for a project, or empty if unset.
func ProjectDisplayName(projectPath string) string {
	projectPath = normalizeProjectPath(projectPath)
	if projectPath == "" {
		return ""
	}
	return strings.TrimSpace(GetProjectDisplayNames()[projectPath])
}

// SetProjectDisplayName sets or clears a custom display label for a project path.
// Empty label clears the override (auto label from repo/branch/workspace is used again).
func SetProjectDisplayName(projectPath, label string) error {
	projectPath = normalizeProjectPath(projectPath)
	if projectPath == "" {
		return nil
	}
	names := GetProjectDisplayNames()
	label = strings.TrimSpace(label)
	if label == "" {
		delete(names, projectPath)
	} else {
		names[projectPath] = label
	}
	b, err := json.Marshal(names)
	if err != nil {
		return err
	}
	return SetSetting(settingProjectDisplayNames, string(b))
}

func normalizeProjectPath(projectPath string) string {
	projectPath = strings.TrimSpace(projectPath)
	if projectPath == "" {
		return ""
	}
	abs, err := filepath.Abs(projectPath)
	if err != nil {
		return filepath.Clean(projectPath)
	}
	return filepath.Clean(abs)
}
