package dashboard

import (
	"bytes"
	"context"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
)

const recentPartialCacheTTL = 10 * time.Minute

var (
	recentPartialCacheMu sync.RWMutex
	recentPartialCache   string
	recentPartialCacheAt time.Time
)

func renderRecentPanelHTML(mcp, indexing []components.RecentQuery) string {
	logOpts := components.LogViewOpts{TailLines: 200, MaxLineChars: 500}
	var buf bytes.Buffer
	components.RecentPanel(mcp, indexing, nil, "", false, logOpts).Render(context.Background(), &buf)
	return buf.String()
}

func storeRecentPartialCache(html string) {
	if html == "" {
		return
	}
	recentPartialCacheMu.Lock()
	recentPartialCache = html
	recentPartialCacheAt = time.Now()
	recentPartialCacheMu.Unlock()
}

func loadRecentPartialCache() string {
	recentPartialCacheMu.RLock()
	defer recentPartialCacheMu.RUnlock()
	if recentPartialCache == "" || time.Since(recentPartialCacheAt) > recentPartialCacheTTL {
		return ""
	}
	return recentPartialCache
}
