package dashboard

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedqueue"
)

// setupQueueCaps clears the cap env vars and restores the default soft limits afterwards.
func setupQueueCaps(t *testing.T) http.Handler {
	t.Helper()
	for _, c := range embedqueue.QueueCaps() {
		t.Setenv(c.Env, "")
	}
	t.Cleanup(embedqueue.LoadQueueCaps)
	h := setupFlagsAPI(t)
	embedqueue.LoadQueueCaps()
	return h
}

func getSettingsMap(t *testing.T, h http.Handler) map[string]string {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, "/api/settings", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	var out map[string]string
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &out))
	return out
}

func queueCapState(t *testing.T, key string) embedqueue.QueueCapState {
	t.Helper()
	for _, st := range buildSettingsData(settingsBuildOpts{}).EmbedQueueCaps {
		if st.Key == key {
			return st
		}
	}
	t.Fatalf("no queue cap %s in settings data", key)
	return embedqueue.QueueCapState{}
}

func TestSettingsQueueCapsRoundTrip(t *testing.T) {
	h := setupQueueCaps(t)
	settings := getSettingsMap(t, h)
	for _, c := range embedqueue.QueueCaps() {
		assert.Equal(t, strconv.Itoa(c.Default), settings[c.Key], "default for %s", c.Key)
		assert.Equal(t, c.Default, queueCapState(t, c.Key).Value)
	}
	code, out := serveFlags(t, h, http.MethodPost, "/api/settings", `{"key":"embed_queue_low_cap","value":" 4096 "}`)
	require.Equal(t, http.StatusOK, code, out.Error)
	assert.Equal(t, "4096", db.GetSetting(embedqueue.LowCapSetting, ""))
	assert.Equal(t, "4096", getSettingsMap(t, h)[embedqueue.LowCapSetting])
	assert.Equal(t, 4096, embedqueue.Snapshot().LowCap, "applied live")
	assert.Equal(t, 4096, queueCapState(t, embedqueue.LowCapSetting).Value)
	assert.Equal(t, 4096, buildIndexHealthFresh("").EmbedLowCap)
	assert.Equal(t, 4096, buildHealthData().QueueLowCap)
}

func TestSettingsQueueCapsValidation(t *testing.T) {
	h := setupQueueCaps(t)
	tests := []struct {
		key, value string
	}{
		{"embed_queue_pending_cap", "0"},
		{"embed_queue_pending_cap", strconv.Itoa(embedqueue.PendingCapHardMax + 1)},
		{"embed_queue_high_cap", "-1"},
		{"embed_queue_high_cap", "lots"},
		{"embed_queue_low_cap", strconv.Itoa(embedqueue.LowCapHardMax + 1)},
		{"embed_queue_low_cap", ""},
	}
	for _, tc := range tests {
		t.Run(tc.key+"="+tc.value, func(t *testing.T) {
			code, out := serveFlags(t, h, http.MethodPost, "/api/settings", `{"key":"`+tc.key+`","value":"`+tc.value+`"}`)
			assert.Equal(t, http.StatusBadRequest, code)
			assert.Contains(t, out.Error, tc.key+" must be an integer 1–")
			assert.Empty(t, db.GetSetting(tc.key, ""), "rejected values are not stored")
		})
	}
}

func TestSettingsQueueCapsEnvLocked(t *testing.T) {
	h := setupQueueCaps(t)
	t.Setenv("EMBED_QUEUE_HIGH_CAP", "64")
	embedqueue.LoadQueueCaps()
	code, out := serveFlags(t, h, http.MethodPost, "/api/settings", `{"key":"embed_queue_high_cap","value":"512"}`)
	assert.Equal(t, http.StatusConflict, code)
	assert.Contains(t, out.Error, "EMBED_QUEUE_HIGH_CAP")
	assert.Empty(t, db.GetSetting(embedqueue.HighCapSetting, ""))
	st := queueCapState(t, embedqueue.HighCapSetting)
	assert.True(t, st.EnvLocked)
	assert.Equal(t, 64, st.Value)
	assert.False(t, queueCapState(t, embedqueue.LowCapSetting).EnvLocked)
}

func TestMetricsReportQueueCaps(t *testing.T) {
	setupQueueCaps(t)
	_, err := embedqueue.SetQueueCap(embedqueue.PendingCapSetting, "77")
	require.NoError(t, err)
	h := NewHandler("")
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	body := rr.Body.String()
	assert.Contains(t, body, `astcache_embed_queue_cap{queue="pending"} 77`)
	assert.Contains(t, body, `astcache_embed_queue_cap{queue="high"} 128`)
	assert.Contains(t, body, `astcache_embed_queue_cap{queue="low"} 2048`)
}
