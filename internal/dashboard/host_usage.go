package dashboard

import (
	"encoding/json"
	"net/http"

	"github.com/coma-toast/ast-context-cache/internal/transcripts"
)

const hostUsageWindowDays = 30

// hostUsageResponse is the transcript usage series (TL-5); Days is empty while the ingest
// setting is off.
type hostUsageResponse struct {
	Enabled    bool                `json:"Enabled"`
	WindowDays int                 `json:"WindowDays"`
	Days       []transcripts.Usage `json:"Days"`
}

func handleDashboardHostUsageJSON(w http.ResponseWriter, r *http.Request) {
	resp := hostUsageResponse{Enabled: transcripts.Enabled(), WindowDays: hostUsageWindowDays, Days: []transcripts.Usage{}}
	if resp.Enabled {
		days, err := transcripts.DailySeries(hostUsageWindowDays)
		if err != nil {
			logger.Warn("Failed to read host usage", "error", err)
		}
		resp.Days = days
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// validBoolSetting reports whether value is "true" or "false".
func validBoolSetting(value string) bool {
	return value == "true" || value == "false"
}
