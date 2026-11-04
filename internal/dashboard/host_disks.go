package dashboard

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/hostdisk"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
	"github.com/coma-toast/ast-context-cache/internal/sys"
)

// hostDisks polls host_disk_health_url; a change refreshes the disk card.
var hostDisks = hostdisk.NewSampler(func() string { return db.GetSetting(hostdisk.SettingKey, "") }, func() {
	invalidateIndexHealthCache()
	realtime.Notify(realtime.IndexHealth)
})

// StartHostDiskSampler polls the host's smartctl_exporter until ctx is done; saving the
// setting polls again at once.
func StartHostDiskSampler(ctx context.Context) {
	hostDisks.Run(ctx, hostdisk.DefaultInterval)
}

// applyHostDisks copies the host disk sample and guest state onto the disk card.
func applyHostDisks(h *components.IndexHealth) {
	snap := hostDisks.Snapshot()
	h.Disks, h.HostDiskURLSet, h.HostDiskError, h.HostDiskGuest = snap.Disks, snap.URLSet, snap.Error, sys.RunsAsGuest()
	h.HostDiskSampledUnix = 0
	if !snap.SampledAt.IsZero() {
		h.HostDiskSampledUnix = snap.SampledAt.Unix()
	}
}

// handleHostDiskTest checks a host disk health URL from Settings before it is saved:
// POST {"url": "..."} returns a hostdisk.TestResult. It never changes the setting or the card.
func handleHostDiskTest(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method != http.MethodPost {
		writeFlagsError(w, http.StatusMethodNotAllowed, "POST required")
		return
	}
	var req struct {
		URL string `json:"url"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 8<<10)).Decode(&req); err != nil {
		writeFlagsError(w, http.StatusBadRequest, "invalid request: "+err.Error())
		return
	}
	json.NewEncoder(w).Encode(hostdisk.Test(r.Context(), req.URL))
}
