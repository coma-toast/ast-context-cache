package logretention

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

type logFile struct {
	path    string
	size    int64
	modTime time.Time
}

// RunOnce deletes .log files under configured absolute roots when enabled. Only .log; roots must be absolute directories.
func RunOnce() {
	if strings.ToLower(strings.TrimSpace(db.GetSetting("log_retention_enabled", "false"))) != "true" {
		return
	}
	raw := db.GetSetting("log_retention_roots", "[]")
	var roots []string
	if err := json.Unmarshal([]byte(raw), &roots); err != nil {
		logger.Warn("Failed to parse log_retention_roots JSON", "error", err)
		return
	}
	maxAgeDays, _ := strconv.Atoi(strings.TrimSpace(db.GetSetting("log_retention_max_age_days", "0")))
	maxMib, _ := strconv.Atoi(strings.TrimSpace(db.GetSetting("log_retention_max_total_mib", "0")))
	maxTotal := int64(maxMib) * 1024 * 1024
	dry := strings.ToLower(strings.TrimSpace(db.GetSetting("log_retention_dry_run", "false"))) == "true"

	if maxAgeDays <= 0 && maxTotal <= 0 {
		logger.Warn("Log retention enabled without limits; set log_retention_max_age_days and/or log_retention_max_total_mib")
		return
	}

	var absRoots []string
	for _, r := range roots {
		r = strings.TrimSpace(r)
		if r == "" {
			continue
		}
		r = filepath.Clean(r)
		if !filepath.IsAbs(r) {
			logger.Warn("Skipping non-absolute log retention root", "root", r)
			continue
		}
		st, err := os.Stat(r)
		if err != nil || !st.IsDir() {
			logger.Warn("Skipping missing or non-directory log retention root", "root", r)
			continue
		}
		absRoots = append(absRoots, r)
	}
	if len(absRoots) == 0 {
		logger.Warn("No valid log retention roots")
		return
	}

	var all []logFile
	for _, root := range absRoots {
		_ = filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
			if err != nil || info.IsDir() {
				return nil
			}
			if !strings.EqualFold(filepath.Ext(path), ".log") {
				return nil
			}
			all = append(all, logFile{path: path, size: info.Size(), modTime: info.ModTime()})
			return nil
		})
	}

	cutoff := time.Time{}
	if maxAgeDays > 0 {
		cutoff = time.Now().Add(-time.Duration(maxAgeDays) * 24 * time.Hour)
	}

	toDelete := map[string]struct{}{}
	if maxAgeDays > 0 {
		for _, f := range all {
			if f.modTime.Before(cutoff) {
				toDelete[f.path] = struct{}{}
			}
		}
	}

	if maxTotal > 0 {
		var survivors []logFile
		var total int64
		for _, f := range all {
			if _, gone := toDelete[f.path]; gone {
				continue
			}
			survivors = append(survivors, f)
			total += f.size
		}
		if total > maxTotal {
			sort.Slice(survivors, func(i, j int) bool {
				return survivors[i].modTime.Before(survivors[j].modTime)
			})
			over := total - maxTotal
			var freed int64
			for _, f := range survivors {
				if freed >= over {
					break
				}
				toDelete[f.path] = struct{}{}
				freed += f.size
			}
		}
	}

	deleted := 0
	var deletedBytes int64
	for p := range toDelete {
		var sz int64
		for _, f := range all {
			if f.path == p {
				sz = f.size
				break
			}
		}
		if dry {
			logger.Info("Log retention dry run would delete file", "path", p)
			deleted++
			deletedBytes += sz
			continue
		}
		if err := os.Remove(p); err != nil {
			logger.Warn("Failed to remove log file", "path", p, "error", err)
			continue
		}
		deleted++
		deletedBytes += sz
	}

	meta, _ := json.Marshal(map[string]interface{}{
		"at":            time.Now().UTC().Format(time.RFC3339),
		"deleted_files": deleted,
		"bytes_freed":   deletedBytes,
		"dry_run":       dry,
	})
	_ = db.SetSetting("log_retention_last_run", string(meta))
	realtime.Notify(realtime.SettingsChanged)
	if deleted > 0 || dry {
		logger.Info("Finished log retention run", "deleted", deleted, "bytes_freed", deletedBytes, "dry_run", dry)
	}
}
