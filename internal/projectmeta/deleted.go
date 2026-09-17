package projectmeta

import (
	"encoding/json"
	"sync"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/watcher"
)

// Deleted projects must not reappear via passive filesystem discovery after the
// user explicitly deletes them from the dashboard — only an explicit tool call
// against the exact path (index_files, or any MCP tool that touches it) should
// bring one back.
const deletedSettingKey = "deleted_project_paths"

var deletedMu sync.Mutex

// MarkDeleted records projectPath so DiscoverPaths stops surfacing it until an
// explicit tool call re-indexes it.
func MarkDeleted(projectPath string) {
	projectPath = watcher.NormalizeProjectPath(projectPath)
	if projectPath == "" {
		return
	}
	deletedMu.Lock()
	defer deletedMu.Unlock()
	paths := loadDeletedLocked()
	for _, p := range paths {
		if p == projectPath {
			return
		}
	}
	paths = append(paths, projectPath)
	saveDeletedLocked(paths)
}

// ClearDeleted removes projectPath from the deleted-paths tombstone, e.g. after
// an explicit re-index makes the entry moot.
func ClearDeleted(projectPath string) {
	projectPath = watcher.NormalizeProjectPath(projectPath)
	if projectPath == "" {
		return
	}
	deletedMu.Lock()
	defer deletedMu.Unlock()
	paths := loadDeletedLocked()
	out := paths[:0]
	changed := false
	for _, p := range paths {
		if p == projectPath {
			changed = true
			continue
		}
		out = append(out, p)
	}
	if changed {
		saveDeletedLocked(out)
	}
}

// WasDeleted reports whether projectPath was explicitly deleted and not since
// re-indexed.
func WasDeleted(projectPath string) bool {
	projectPath = watcher.NormalizeProjectPath(projectPath)
	if projectPath == "" {
		return false
	}
	deletedMu.Lock()
	defer deletedMu.Unlock()
	for _, p := range loadDeletedLocked() {
		if p == projectPath {
			return true
		}
	}
	return false
}

func loadDeletedLocked() []string {
	raw := db.GetSetting(deletedSettingKey, "[]")
	var out []string
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		return nil
	}
	return out
}

func saveDeletedLocked(paths []string) {
	if paths == nil {
		paths = []string{}
	}
	if b, err := json.Marshal(paths); err == nil {
		db.SetSetting(deletedSettingKey, string(b))
	}
}
