package db

import (
	"encoding/json"
	"sort"
)

const settingPinnedProjects = "pinned_projects"

// GetPinnedProjects returns sorted absolute project paths marked as pinned (warm priority).
func GetPinnedProjects() []string {
	s := GetSetting(settingPinnedProjects, "[]")
	var paths []string
	if err := json.Unmarshal([]byte(s), &paths); err != nil {
		return nil
	}
	sort.Strings(paths)
	return paths
}

// SetPinnedProjects replaces the full pinned list.
func SetPinnedProjects(paths []string) error {
	if paths == nil {
		paths = []string{}
	}
	b, err := json.Marshal(paths)
	if err != nil {
		return err
	}
	return SetSetting(settingPinnedProjects, string(b))
}

// IsPinnedProject reports whether a project path is pinned.
func IsPinnedProject(projectPath string) bool {
	for _, p := range GetPinnedProjects() {
		if p == projectPath {
			return true
		}
	}
	return false
}

// TogglePinnedProject adds or removes a project from the pinned set.
// PinnedProjectCount returns how many projects are pinned.
func PinnedProjectCount() int {
	return len(GetPinnedProjects())
}

// TogglePinnedProject adds or removes a project from the pinned set.
func TogglePinnedProject(projectPath string, pin bool) error {
	paths := GetPinnedProjects()
	var out []string
	seen := false
	for _, p := range paths {
		if p == projectPath {
			seen = true
			if !pin {
				continue
			}
		}
		out = append(out, p)
	}
	if pin && !seen {
		out = append(out, projectPath)
		sort.Strings(out)
	}
	return SetPinnedProjects(out)
}
