package dashboard

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/hostdisk"
)

// getJSON GETs a dashboard JSON endpoint into out.
func getJSON(t *testing.T, h http.Handler, path string, out any) {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, path, nil)
	req.Host = "127.0.0.1:7830"
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	require.Equal(t, http.StatusOK, rr.Code, rr.Body.String())
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), out), rr.Body.String())
}

func TestHostDisksOnDiskCard(t *testing.T) {
	fixture, err := os.ReadFile("../hostdisk/testdata/smartctl_exporter.prom")
	require.NoError(t, err)
	var status atomic.Int32
	status.Store(http.StatusOK)
	exporter := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(int(status.Load()))
		w.Write(fixture)
	}))
	t.Cleanup(exporter.Close)
	h, _ := setupSettingsBatch(t, "")
	t.Cleanup(func() {
		_ = db.SetSetting(hostdisk.SettingKey, "")
		hostDisks.Poll(context.Background())
		invalidateIndexHealthCache()
	})
	type card struct {
		Disks []struct {
			Device, Model, SmartStatus, SmartSource, WarningLevel string
			Host                                                  bool
			TemperatureC                                          *float64
		}
		HostDiskURLSet      bool
		HostDiskError       string
		HostDiskSampledUnix int64
	}
	poll := func() card {
		hostDisks.Poll(context.Background())
		invalidateIndexHealthCache()
		var c card
		getJSON(t, h, "/api/dashboard/index-health", &c)
		return c
	}

	assert.Empty(t, poll().Disks, "no URL, no host disks")

	code, out := postBatch(t, h, http.MethodPost, `{"settings":{"host_disk_health_url":"`+exporter.URL+`/metrics"}}`)
	require.Equal(t, http.StatusOK, code, out.Error)
	var settings struct{ HostDiskHealthURL string }
	getJSON(t, h, "/api/dashboard/settings", &settings)
	assert.Equal(t, exporter.URL+"/metrics", settings.HostDiskHealthURL)

	c := poll()
	require.Len(t, c.Disks, 5)
	assert.True(t, c.HostDiskURLSet)
	assert.Empty(t, c.HostDiskError)
	assert.NotZero(t, c.HostDiskSampledUnix)
	nvme := c.Disks[0]
	assert.Equal(t, "nvme0", nvme.Device)
	assert.True(t, nvme.Host)
	assert.Equal(t, hostdisk.Source, nvme.SmartSource)
	assert.Equal(t, "FAILED", nvme.SmartStatus)
	assert.Equal(t, "critical", nvme.WarningLevel)
	require.NotNil(t, nvme.TemperatureC)
	assert.Equal(t, 37.0, *nvme.TemperatureC)

	status.Store(http.StatusServiceUnavailable)
	failed := poll()
	assert.Len(t, failed.Disks, 5, "the last good sample stays on the card")
	assert.Equal(t, c.HostDiskSampledUnix, failed.HostDiskSampledUnix)
	assert.Contains(t, failed.HostDiskError, "503")

	code, out = postBatch(t, h, http.MethodPost, `{"settings":{"host_disk_health_url":""}}`)
	require.Equal(t, http.StatusOK, code, out.Error)
	cleared := poll()
	assert.Empty(t, cleared.Disks)
	assert.False(t, cleared.HostDiskURLSet)
	assert.Empty(t, cleared.HostDiskError)
}
