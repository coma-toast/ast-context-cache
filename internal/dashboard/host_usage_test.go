package dashboard

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/transcripts"
)

const testInsertHostUsageQuery = `INSERT INTO host_usage_daily (day, project_dir, input, output, cache_read, cache_write) VALUES (date('now','localtime'), ?, ?, 1, 2, 3)`

func getHostUsage(t *testing.T) hostUsageResponse {
	t.Helper()
	rec := httptest.NewRecorder()
	handleDashboardHostUsageJSON(rec, httptest.NewRequest(http.MethodGet, "/api/dashboard/host-usage", nil))
	require.Equal(t, http.StatusOK, rec.Code)
	var out hostUsageResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &out))
	return out
}

func TestDashboardHostUsage(t *testing.T) {
	testEmbedDB(t)
	_, err := db.DB.Exec(testInsertHostUsageQuery, "a", 10)
	require.NoError(t, err)
	_, err = db.DB.Exec(testInsertHostUsageQuery, "b", 5)
	require.NoError(t, err)
	out := getHostUsage(t)
	assert.False(t, out.Enabled)
	assert.Empty(t, out.Days, "nothing is shown while the ingest is off")
	require.NoError(t, db.SetSetting(transcripts.SettingKey, "true"))
	out = getHostUsage(t)
	assert.True(t, out.Enabled)
	assert.Equal(t, hostUsageWindowDays, out.WindowDays)
	require.Len(t, out.Days, 1)
	assert.Equal(t, int64(15), out.Days[0].Input)
}

func TestTranscriptSettingValidation(t *testing.T) {
	testEmbedDB(t)
	post := func(value string) *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, "/api/settings", strings.NewReader(`{"key":"`+transcripts.SettingKey+`","value":"`+value+`"}`))
		req.Header.Set("Content-Type", "application/json")
		handleSettings(rec, req)
		return rec
	}
	assert.Equal(t, http.StatusBadRequest, post("yes").Code)
	assert.Equal(t, http.StatusOK, post("true").Code)
	assert.True(t, transcripts.Enabled())
	rec := httptest.NewRecorder()
	require.NoError(t, db.SetSetting(transcripts.SettingKey, "false"))
	handleSettings(rec, httptest.NewRequest(http.MethodGet, "/api/settings", nil))
	var settings map[string]string
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &settings))
	assert.Equal(t, "false", settings[transcripts.SettingKey])
}
