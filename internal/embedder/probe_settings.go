package embedder

import (
	"strconv"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

const (
	probeIntervalSetting       = "embed_probe_interval_seconds"
	defaultProbeInterval       = 10 * time.Second
	minProbeInterval           = 5 * time.Second
	maxProbeInterval           = 600 * time.Second
)

// ProbeIntervalSettingKey is the dashboard DB key for connectivity probe cadence.
const ProbeIntervalSettingKey = probeIntervalSetting

// ProbeInterval returns the configured healthy-state connectivity probe interval.
func ProbeInterval() time.Duration {
	raw := db.GetSetting(probeIntervalSetting, "")
	if raw == "" {
		return defaultProbeInterval
	}
	sec, err := strconv.Atoi(raw)
	if err != nil || sec < int(minProbeInterval/time.Second) {
		return defaultProbeInterval
	}
	d := time.Duration(sec) * time.Second
	if d > maxProbeInterval {
		return maxProbeInterval
	}
	return d
}

// RestartConnectivityProbe applies the current probe interval (restarts the background loop).
func RestartConnectivityProbe() {
	StartConnectivityProbe(Raw())
}
