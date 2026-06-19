package dashboard

import (
	"os"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

const logNotifyInterval = 2 * time.Second

func initLogNotifyBridge() {
	go watchServerLog()
}

func watchServerLog() {
	ticker := time.NewTicker(logNotifyInterval)
	var (
		lastPath string
		lastSize int64 = -1
		lastMod  time.Time
	)
	for range ticker.C {
		path := db.ResolveServerLogPath()
		if path != lastPath {
			lastPath = path
			lastSize = -1
			lastMod = time.Time{}
		}
		st, err := os.Stat(path)
		if err != nil {
			continue
		}
		if st.Size() == lastSize && st.ModTime().Equal(lastMod) {
			continue
		}
		lastSize = st.Size()
		lastMod = st.ModTime()
		realtime.Notify(realtime.Recent)
	}
}
