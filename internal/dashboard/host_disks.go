package dashboard

import (
	"context"

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
