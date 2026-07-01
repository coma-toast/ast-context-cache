package dashboard

import (
	"context"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/db"
)

const recentQueryTimeout = 5 * time.Second

var (
	cachedLogOpts   components.LogViewOpts
	cachedLogOptsMu sync.RWMutex
	logOptsLoaded   bool
)

func defaultLogViewOpts() components.LogViewOpts {
	return components.LogViewOpts{TailLines: 200, MaxLineChars: 500}
}

// RefreshLogViewOptsFromDB reloads log display settings from usage.db (call after settings change).
func RefreshLogViewOptsFromDB() {
	cachedLogOptsMu.Lock()
	defer cachedLogOptsMu.Unlock()
	cachedLogOpts = loadLogViewOptsFromDB()
	logOptsLoaded = true
}

func logViewOptsFast() components.LogViewOpts {
	cachedLogOptsMu.RLock()
	if logOptsLoaded {
		opts := cachedLogOpts
		cachedLogOptsMu.RUnlock()
		return opts
	}
	cachedLogOptsMu.RUnlock()
	cachedLogOptsMu.Lock()
	defer cachedLogOptsMu.Unlock()
	if !logOptsLoaded {
		cachedLogOpts = loadLogViewOptsFromDB()
		logOptsLoaded = true
	}
	return cachedLogOpts
}

func loadLogViewOptsFromDB() components.LogViewOpts {
	opts := defaultLogViewOpts()
	if db.DB == nil {
		return opts
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	var tail, chars string
	err := db.DB.QueryRowContext(ctx, "SELECT value FROM settings WHERE key = ?", "dashboard_log_tail_lines").Scan(&tail)
	if err == nil {
		if n, e := strconv.Atoi(strings.TrimSpace(tail)); e == nil && n > 0 {
			opts.TailLines = clampInt(n, 50, 500)
		}
	}
	err = db.DB.QueryRowContext(ctx, "SELECT value FROM settings WHERE key = ?", "dashboard_log_line_chars").Scan(&chars)
	if err == nil {
		if n, e := strconv.Atoi(strings.TrimSpace(chars)); e == nil && n > 0 {
			opts.MaxLineChars = clampInt(n, 80, 8000)
		}
	}
	return opts
}

func clampInt(n, lo, hi int) int {
	if n < lo {
		return lo
	}
	if n > hi {
		return hi
	}
	return n
}
