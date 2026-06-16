package dashboard

import (
	"encoding/json"
	"log"
	"net/http"

	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

func appendReloadFields(out map[string]interface{}) {
	if err := embedder.Reload(); err != nil {
		log.Printf("embedder reload: %v", err)
		out["reload_error"] = err.Error()
		return
	}
	wb, wm, _, _, _ := embedder.WiredSnapshot()
	out["reloaded"] = true
	out["embed_backend"] = wb
	out["embed_model"] = wm
}

func writeSettingsOK(w http.ResponseWriter, extra map[string]string, reloadEmbed bool, notify realtime.Reason) {
	out := map[string]interface{}{"status": "ok"}
	for k, v := range extra {
		out[k] = v
	}
	if reloadEmbed {
		appendReloadFields(out)
		notify |= realtime.IndexHealth | realtime.HealthBar
	}
	realtime.Notify(notify)
	json.NewEncoder(w).Encode(out)
}

func writeEmbedSettingsOK(w http.ResponseWriter) {
	out := map[string]interface{}{"status": "ok"}
	appendReloadFields(out)
	realtime.Notify(realtime.SettingsChanged | realtime.IndexHealth | realtime.HealthBar)
	json.NewEncoder(w).Encode(out)
}
