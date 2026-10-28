package dashboard

import (
	"net/http"
	"strconv"

	"github.com/coma-toast/ast-context-cache/internal/embedqueue"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

// saveQueueCapSetting is the settings POST path for an embed queue soft limit: it validates
// 1..hard max (400), refuses an EMBED_QUEUE_*_CAP env-locked key (409), and applies live.
func saveQueueCapSetting(w http.ResponseWriter, key, value string) {
	n, err := embedqueue.SetQueueCap(key, value)
	if err != nil {
		writeFlagsError(w, networkErrorStatus(err), err.Error())
		return
	}
	writeSettingsOK(w, map[string]string{"key": key, "value": strconv.Itoa(n)}, false, realtime.SettingsChanged|realtime.IndexHealth|realtime.HealthBar)
}
