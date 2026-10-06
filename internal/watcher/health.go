package watcher

import (
	"math"
	"os"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/sys"
)

// catchUpAfterLostEvents is how long scheduleCatchUp waits for a burst of
// overflow reports to settle before rescanning (a var so tests can shorten it).
var catchUpAfterLostEvents = 2 * time.Second

// scheduleCatchUp rescans a project after its backend reported lost events
// (fsnotify.ErrEventOverflow). Keyed under the project path so
// cancelDebounceTimersForProject cancels it along with the file timers.
func scheduleCatchUp(projectPath string) {
	key := projectPath + string(os.PathSeparator) + "\x00catch-up"
	debounceMu.Lock()
	defer debounceMu.Unlock()
	if t, ok := debounceTimers[key]; ok {
		stopDebounce(t)
	}
	bg.Add(1)
	var t *time.Timer
	t = time.AfterFunc(catchUpAfterLostEvents, func() {
		defer bg.Done()
		debounceMu.Lock()
		if debounceTimers[key] == t {
			delete(debounceTimers, key)
		}
		debounceMu.Unlock()
		if !IsActive(projectPath) {
			return
		}
		logger.Info("Watcher rescanning (events lost, or a directory moved in or out)", "project", projectPath)
		trackedCatchUp(projectPath)
	})
	debounceTimers[key] = t
}

// FDStatus is the server's file-descriptor usage, for index_status and the
// dashboard.
func FDStatus() map[string]interface{} {
	return fdStatusMap(sys.FileDescriptorUsage())
}

func fdStatusMap(u sys.FDUsage) map[string]interface{} {
	out := map[string]interface{}{
		"available":  u.Available,
		"soft_limit": u.SoftLimit,
		"hard_limit": u.HardLimit,
		"level":      u.Level(),
	}
	if u.Available {
		out["open"] = u.Open
		out["usage_pct"] = math.Round(u.Pct()*10) / 10
	}
	return out
}

// ResourceStatus summarizes descriptor usage and what the watchers hold.
func ResourceStatus() map[string]interface{} {
	mu.Lock()
	active, watches := len(activeWatchers), 0
	for _, w := range activeWatchers {
		watches += w.OSWatches()
	}
	mu.Unlock()
	return map[string]interface{}{
		"file_descriptors": FDStatus(),
		"watch_backend":    DefaultBackendName(),
		"active_watchers":  active,
		"os_watches":       watches,
	}
}

// ProjectStatus describes one project's watcher, including why it isn't
// running when it was refused.
func ProjectStatus(projectPath string) map[string]interface{} {
	projectPath = NormalizeProjectPath(projectPath)
	out := map[string]interface{}{"active": false}
	mu.Lock()
	w, running := activeWatchers[projectPath]
	if running {
		out["active"] = true
		out["backend"] = w.Name()
		out["os_watches"] = w.OSWatches()
	}
	if t, ok := lastActivity[projectPath]; ok {
		out["last_activity"] = t.Format(time.RFC3339)
	}
	mu.Unlock()
	if !running {
		if reason := WatchRefusal(projectPath); reason != "" {
			out["blocked_reason"] = reason
		}
	}
	return out
}

// lastFDLevel is only touched by idleLoop's goroutine.
var lastFDLevel = sys.FDLevelOK

// checkFDPressure logs when descriptor usage crosses a level, so creeping
// exhaustion shows in the log before calls start failing with "too many open
// files".
func checkFDPressure() {
	u := sys.FileDescriptorUsage()
	level := u.Level()
	if level == lastFDLevel {
		return
	}
	prev := lastFDLevel
	lastFDLevel = level
	switch level {
	case sys.FDLevelWarning, sys.FDLevelCritical:
		st := ResourceStatus()
		logger.Warn("File descriptor usage high", "open", u.Open, "soft_limit", u.SoftLimit, "pct", math.Round(u.Pct()), "level", level,
			"active_watchers", st["active_watchers"], "watch_backend", st["watch_backend"], "os_watches", st["os_watches"])
	case sys.FDLevelUnknown:
		// Counting needs a descriptor itself, so failing to count usually
		// means the process is already at its limit.
		if sys.FDCountSupported() {
			logger.Warn("Cannot count open file descriptors; the process may be out of them", "soft_limit", u.SoftLimit)
		}
	case sys.FDLevelOK:
		logger.Info("File descriptor usage back to normal", "previous_level", prev, "open", u.Open, "soft_limit", u.SoftLimit)
	}
}
