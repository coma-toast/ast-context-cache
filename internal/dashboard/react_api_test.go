package dashboard

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/mcp"
)

// handleDashboardMCPTierJSON used to re-derive the tier from os.Getenv itself,
// defaulting an unset AST_MCP_TIER to "extended" — but the MCP server's own
// real default (mcp.DefaultConfig) is TierComplete, so this endpoint reported
// the wrong tier whenever the env var was unset. It now reads the server's
// actual live config instead of guessing.
func TestHandleDashboardMCPTierJSONReportsRealConfig(t *testing.T) {
	orig := mcp.GetConfig()
	t.Cleanup(func() { mcp.SetConfig(orig) })

	mcp.SetConfig(mcp.ServerConfig{
		ActiveTier: mcp.TierComplete,
		CodeMode:   true,
		ToolConfigs: map[string]*mcp.ToolConfig{
			"execute_code": {Enabled: false, Tier: mcp.TierCore},
		},
	})

	req := httptest.NewRequest(http.MethodGet, "/api/dashboard/mcp-tier", nil)
	rec := httptest.NewRecorder()
	handleDashboardMCPTierJSON(rec, req)

	var out struct {
		Tier          string `json:"tier"`
		CodeMode      bool   `json:"code_mode"`
		ToolOverrides map[string]struct {
			Enabled bool   `json:"enabled"`
			Tier    string `json:"tier"`
		} `json:"tool_overrides"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("unmarshal: %v (body=%s)", err, rec.Body.String())
	}
	if out.Tier != "complete" {
		t.Fatalf("tier=%q want complete", out.Tier)
	}
	if !out.CodeMode {
		t.Fatal("code_mode should be true")
	}
	ov, ok := out.ToolOverrides["execute_code"]
	if !ok || ov.Enabled || ov.Tier != "core" {
		t.Fatalf("tool_overrides[execute_code]=%+v want {enabled:false tier:core}", ov)
	}
}

// BF-3: the mcp-tier view reports the tools.json the server loads, so a custom
// AST_MCP_TOOLS_CONFIG is shown instead of the hard-coded ~/.astcache/tools.json.
func TestHandleDashboardMCPTierJSONToolsConfigPath(t *testing.T) {
	custom := filepath.Join(t.TempDir(), "custom-tools.json")
	tests := []struct {
		name       string
		env        string
		write      bool
		wantPath   string
		wantExists bool
	}{
		{name: "env path exists", env: custom, write: true, wantPath: custom, wantExists: true},
		{name: "env path missing", env: custom + ".missing", wantPath: custom + ".missing"},
		{name: "default path", env: "", wantPath: filepath.Join(t.TempDir(), ".astcache", "tools.json")},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.env == "" {
				home := filepath.Dir(filepath.Dir(tt.wantPath))
				t.Setenv("HOME", home)
			}
			t.Setenv("AST_MCP_TOOLS_CONFIG", tt.env)
			if tt.write {
				require.NoError(t, os.WriteFile(tt.env, []byte("{}"), 0o644))
			}
			rec := httptest.NewRecorder()
			handleDashboardMCPTierJSON(rec, httptest.NewRequest(http.MethodGet, "/api/dashboard/mcp-tier", nil))
			var out struct {
				Path   string `json:"tools_json_path"`
				Exists bool   `json:"tools_json_exists"`
			}
			require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &out))
			assert.Equal(t, tt.wantPath, out.Path)
			assert.Equal(t, tt.wantExists, out.Exists)
		})
	}
}

// BF-10: Tokens saved counts compression + dedup from search/read tools only;
// virtual context writes (store_context) are reported separately.
func TestDashboardStatsExcludesVirtual(t *testing.T) {
	testEmbedDB(t)
	now := time.Now().UTC().Format(time.RFC3339)
	const insertQuery = `INSERT INTO queries (timestamp, tool_name, session_id, project_path, tokens_saved, dedup_tokens_saved, savings_vs_files, duration_ms)
		VALUES (?, ?, 'sess-v', '/proj-v', ?, ?, ?, 1)`
	_, err := db.DB.Exec(insertQuery, now, "get_context_capsule", 1200, 100, 300)
	require.NoError(t, err)
	_, err = db.DB.Exec(insertQuery, now, "store_context", 9000, 900, 9000)
	require.NoError(t, err)
	rec := httptest.NewRecorder()
	handleDashboardStatsJSON(rec, httptest.NewRequest(http.MethodGet, "/api/dashboard/stats?project_id=/proj-v", nil))
	require.Equal(t, http.StatusOK, rec.Code)
	var out struct {
		TotalQueries     int
		TokensSaved      int
		DedupTokensSaved int
		SavingsVsFiles   int
	}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &out))
	assert.Equal(t, 2, out.TotalQueries)
	assert.Equal(t, 1200, out.TokensSaved)
	assert.Equal(t, 100, out.DedupTokensSaved)
	assert.Equal(t, 300, out.SavingsVsFiles)
}

func TestDashboardStatsLedgers(t *testing.T) {
	testEmbedDB(t)
	now := time.Now().UTC().Format(time.RFC3339)
	const insertQuery = `INSERT INTO queries (timestamp, tool_name, session_id, project_path, tokens_saved, dedup_tokens_saved,
		tokens_used, conservative_baseline_tokens, ledger, estimate_method, duration_ms)
		VALUES (?, ?, 'sess-l', '/proj-l', ?, ?, ?, ?, ?, ?, 1)`
	for _, r := range []struct {
		tool                             string
		saved, dedup, used, conservative int
		ledger, method                   string
	}{
		{"get_context_capsule", 1200, 200, 300, 700, "compression", "o200k_base"},
		{"search_semantic", 500, 0, 100, 0, "", "bytes4"},
		{"store_context", 9000, 0, 0, 0, "virtual", "o200k_base"},
		{"store_memory", 40, 0, 0, 0, "virtual", "o200k_base"},
		{"fetch_context", 0, 0, 800, 0, "virtual", "o200k_base"},
		{"recall_memory", 0, 0, 60, 0, "virtual", "o200k_base"},
	} {
		_, err := db.DB.Exec(insertQuery, now, r.tool, r.saved, r.dedup, r.used, r.conservative, r.ledger, r.method)
		require.NoError(t, err)
	}
	rec := httptest.NewRecorder()
	handleDashboardStatsJSON(rec, httptest.NewRequest(http.MethodGet, "/api/dashboard/stats?project_id=/proj-l", nil))
	require.Equal(t, http.StatusOK, rec.Code)
	var out struct {
		TokensSaved           int
		CompressionSaved      int
		DedupSaved            int
		ConservativeSaved     int
		VirtualStoredTokens   int
		VirtualFetchedTokens  int
		VirtualRecalledTokens int
		EstimatedRows         int
		BaselineDefinitions   map[string]string
	}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &out))
	assert.Equal(t, 1700, out.TokensSaved)
	assert.Equal(t, 1500, out.CompressionSaved, "the legacy ledger-less search row counts as compression")
	assert.Equal(t, 200, out.DedupSaved)
	assert.Equal(t, out.TokensSaved, out.CompressionSaved+out.DedupSaved)
	assert.Equal(t, 400, out.ConservativeSaved)
	assert.Equal(t, 9040, out.VirtualStoredTokens)
	assert.Equal(t, 800, out.VirtualFetchedTokens)
	assert.Equal(t, 60, out.VirtualRecalledTokens)
	assert.Equal(t, 1, out.EstimatedRows)
	assert.Contains(t, out.BaselineDefinitions, "conservative")
	digest := buildWeeklyDigest("/proj-l")
	assert.Equal(t, 1500, digest.CompressionSaved)
	assert.Equal(t, 400, digest.ConservativeSaved)
	assert.Equal(t, 800, digest.VirtualFetchedTokens)
}

// TestHandlePinnedNoteRoundTrip covers PIN-4: GET returns the current note (or zero
// state when no note exists), PUT stores/updates the note, and subsequent GET returns
// the updated content.
func TestHandlePinnedNoteRoundTrip(t *testing.T) {
	dbtest.Init(t)

	// Seed one project so /api/projects returns it.
	if _, err := db.DB.Exec(`INSERT INTO queries (tool_name, project_path, timestamp) VALUES ('t', '/tmp/proj-pn', datetime('now'))`); err != nil {
		t.Fatal(err)
	}

	// --- GET with no stored note → zero-state response ---
	req := httptest.NewRequest(http.MethodGet, "/api/pinned-note?project=/tmp/proj-pn", nil)
	rec := httptest.NewRecorder()
	handlePinnedNote(rec, req)
	require.Equal(t, http.StatusOK, rec.Code)
	var getOut struct {
		Ref            string `json:"ref"`
		Content        string `json:"content"`
		SessionID      string `json:"session_id"`
		ProjectPath    string `json:"project_path"`
		Label          string `json:"label"`
		Tags           string `json:"tags"`
		Kind           string `json:"kind"`
		MetadataJSON   string `json:"metadata_json"`
		TokenEst       int    `json:"token_est"`
		AccessCount    int    `json:"access_count"`
		CreatedAt      string `json:"created_at"`
		LastAccessedAt string `json:"last_accessed_at"`
		ProjectName    string `json:"project_name"`
	}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &getOut))
	// Pinned() returns a zero-value ContextNote when nothing is stored.
	assert.Empty(t, getOut.Ref)
	assert.Empty(t, getOut.Content)
	assert.Empty(t, getOut.SessionID)
	assert.Equal(t, "/tmp/proj-pn", getOut.ProjectPath)

	// --- PUT: store a note ---
	body := []byte(`{"project_path":"/tmp/proj-pn","content":"hello world","origin":"dashboard"}`)
	req = httptest.NewRequest(http.MethodPut, "/api/pinned-note", nil)
	req.Body = nopCloser(body)
	req.Header.Set("Content-Type", "application/json")
	rec = httptest.NewRecorder()
	handlePinnedNote(rec, req)
	require.Equal(t, http.StatusOK, rec.Code)
	var putOut struct {
		Ref         string `json:"ref"`
		Content     string `json:"content"`
		ProjectPath string `json:"project_path"`
	}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &putOut))
	assert.NotEmpty(t, putOut.Ref, "PUT should return the created ref")
	assert.Equal(t, "hello world", putOut.Content)

	// --- GET after PUT: should return the stored note ---
	req = httptest.NewRequest(http.MethodGet, "/api/pinned-note?project=/tmp/proj-pn", nil)
	rec = httptest.NewRecorder()
	handlePinnedNote(rec, req)
	require.Equal(t, http.StatusOK, rec.Code)
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &getOut))
	assert.Equal(t, "hello world", getOut.Content)
	assert.NotEmpty(t, getOut.Ref)
	assert.Equal(t, "pinned", getOut.Kind)
}

// TestHandlePinnedNoteEmptyBody covers PIN-4: GET with no project param returns an error.
func TestHandlePinnedNoteMissingProject(t *testing.T) {
	dbtest.Init(t)
	req := httptest.NewRequest(http.MethodGet, "/api/pinned-note", nil)
	rec := httptest.NewRecorder()
	handlePinnedNote(rec, req)
	assert.Equal(t, http.StatusOK, rec.Code)
	var out map[string]string
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &out))
	assert.Contains(t, out["error"], "project query parameter required")
}

// TestHandlePinnedNoteOverCap rejects content over the 500-token cap (PIN-1).
func TestHandlePinnedNoteOverCap(t *testing.T) {
	dbtest.Init(t)
	over := strings.Repeat("word ", 700)
	body := []byte(fmt.Sprintf(`{"project_path":"/tmp/proj-pn","content":"%s"}`, over))
	req := httptest.NewRequest(http.MethodPut, "/api/pinned-note", nil)
	req.Body = nopCloser(body)
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	handlePinnedNote(rec, req)
	require.Equal(t, http.StatusOK, rec.Code)
	var out map[string]string
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &out))
	assert.Contains(t, out["error"], "max 500")
}

// handlePinnedNote appears in /api/projects responses — verify it flows through
// getPinnedNoteContent so the UI sees the note on load.
func TestHandlePinnedNoteFlowsThroughProjectsList(t *testing.T) {
	dbtest.Init(t)

	// Seed the project in the queries table so it appears in handleProjects.
	if _, err := db.DB.Exec(`INSERT INTO queries (tool_name, project_path, timestamp) VALUES ('t', '/tmp/proj-pn2', datetime('now'))`); err != nil {
		t.Fatal(err)
	}
	// Store a pinned note.
	if _, err := db.ContextDB.Exec(
		`INSERT INTO context_notes (ref,session_id,project_path,label,content,content_hash,tags,kind,metadata_json,token_est) VALUES ('p','pinned','/tmp/proj-pn2','my note','from projects list','','','pinned','{}',0)`,
	); err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest(http.MethodGet, "/api/projects", nil)
	rec := httptest.NewRecorder()
	handleProjects(rec, req)
	require.Equal(t, http.StatusOK, rec.Code)
	var projects []map[string]interface{}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &projects))
	require.Len(t, projects, 1)
	gotNote, ok := projects[0]["PinnedNote"]
	require.True(t, ok, "PinnedNote key must be present")
	assert.Equal(t, "from projects list", gotNote)
}

func nopCloser(b []byte) io.ReadCloser {
	return io.NopCloser(bytes.NewReader(b))
}
