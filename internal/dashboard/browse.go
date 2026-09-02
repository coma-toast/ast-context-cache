package dashboard

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
)

type browseDirEntry struct {
	Name string `json:"name"`
	Path string `json:"path"`
}

// handleBrowseDir lists the subdirectories of ?path=, for the Settings "Move data
// directory" folder picker. Files are omitted — this is a directory picker, not a
// general file browser. Only reads directory listings; never creates, modifies, or
// deletes anything on disk.
func handleBrowseDir(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(map[string]string{"error": "GET required"})
		return
	}

	path := strings.TrimSpace(r.URL.Query().Get("path"))
	if path == "" {
		if home, err := os.UserHomeDir(); err == nil {
			path = home
		} else {
			path = "/"
		}
	}
	path = filepath.Clean(path)

	info, err := os.Stat(path)
	if err != nil || !info.IsDir() {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"path":    path,
			"error":   "not a directory or not accessible",
			"entries": []browseDirEntry{},
		})
		return
	}

	dirEntries, err := os.ReadDir(path)
	if err != nil {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"path":    path,
			"parent":  parentDir(path),
			"error":   "cannot read directory: " + err.Error(),
			"entries": []browseDirEntry{},
		})
		return
	}

	entries := make([]browseDirEntry, 0, len(dirEntries))
	for _, e := range dirEntries {
		if !e.IsDir() {
			continue
		}
		// Skip symlinks whose target isn't a directory (or is broken), so navigating
		// into them can't fail confusingly one level down.
		full := filepath.Join(path, e.Name())
		if e.Type()&os.ModeSymlink != 0 {
			target, err := os.Stat(full)
			if err != nil || !target.IsDir() {
				continue
			}
		}
		entries = append(entries, browseDirEntry{Name: e.Name(), Path: full})
	}
	sort.Slice(entries, func(i, j int) bool { return strings.ToLower(entries[i].Name) < strings.ToLower(entries[j].Name) })

	json.NewEncoder(w).Encode(map[string]interface{}{
		"path":      path,
		"parent":    parentDir(path),
		"entries":   entries,
		"shortcuts": browseShortcuts(),
	})
}

func parentDir(path string) string {
	parent := filepath.Dir(path)
	if parent == path {
		return ""
	}
	return parent
}

// browseShortcuts gives the picker a few useful jump-off points, notably the common
// external-drive mount root on each platform — the whole point of this picker is
// usually finding a USB drive.
func browseShortcuts() []browseDirEntry {
	var out []browseDirEntry
	if home, err := os.UserHomeDir(); err == nil {
		out = append(out, browseDirEntry{Name: "Home", Path: home})
	}
	switch runtime.GOOS {
	case "darwin":
		if st, err := os.Stat("/Volumes"); err == nil && st.IsDir() {
			out = append(out, browseDirEntry{Name: "Volumes", Path: "/Volumes"})
		}
	case "linux":
		for _, p := range []string{"/media", "/mnt"} {
			if st, err := os.Stat(p); err == nil && st.IsDir() {
				out = append(out, browseDirEntry{Name: p, Path: p})
			}
		}
	}
	out = append(out, browseDirEntry{Name: "/", Path: "/"})
	return out
}
