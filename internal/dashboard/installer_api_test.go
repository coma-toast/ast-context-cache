package dashboard

import (
	"encoding/json"
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
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/flags"
	"github.com/coma-toast/ast-context-cache/internal/installer"
)

// No t.Parallel: the db pools, the flags snapshot, and the installer service are package globals.

const testCursorConfig = `{
  "mcpServers": {
    "other": {"url": "http://example.test/mcp"}
  }
}
`

type installerTestEnv struct {
	h              http.Handler
	svc            installer.Service
	home           string
	now            time.Time
	desktopRunning bool
}

// setupInstallerAPI opens a fresh database under a temp HOME and injects an installer rooted
// there, so no test touches the real home directory's host configs.
func setupInstallerAPI(t *testing.T) *installerTestEnv {
	t.Helper()
	for _, f := range flags.All() {
		t.Setenv(f.Env, "")
	}
	t.Cleanup(flags.Reload)
	env := &installerTestEnv{home: dbtest.Init(t), now: time.Now()}
	flags.Reload()
	env.newInstaller(t)
	t.Cleanup(func() { SetInstaller(nil) })
	env.h = NewHandler("127.0.0.1")
	return env
}

// newInstaller (re)creates the injected service, which reruns its one-time legacy check.
func (e *installerTestEnv) newInstaller(t *testing.T) {
	t.Helper()
	svc, err := installer.New(installer.Config{
		Home:                 e.home,
		MCPURL:               installer.MCPURL(7821),
		Executable:           "/opt/ast/bin/ast-mcp",
		GOOS:                 "darwin",
		HooksEnabled:         func() bool { return false },
		LookPath:             func(name string) (string, error) { return "", errs.New("not found", "name", name) },
		ClaudeDesktopRunning: func() bool { return e.desktopRunning },
		Now:                  func() time.Time { return e.now },
	})
	require.NoError(t, err)
	e.svc = svc
	SetInstaller(svc)
}

// serve sends an Origin-less request, as a CLI or curl would, and decodes the JSON reply into out.
func (e *installerTestEnv) serve(t *testing.T, method, path, body string, out any) int {
	t.Helper()
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	req.Host = "127.0.0.1:7830"
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	e.h.ServeHTTP(rr, req)
	if out != nil {
		require.NoError(t, json.Unmarshal(rr.Body.Bytes(), out), rr.Body.String())
	}
	return rr.Code
}

func (e *installerTestEnv) preview(t *testing.T, body string) installer.Plan {
	t.Helper()
	var plan installer.Plan
	code := e.serve(t, http.MethodPost, "/api/dashboard/installer/preview", body, &plan)
	require.Equal(t, http.StatusOK, code)
	require.NotEmpty(t, plan.ID)
	return plan
}

func (e *installerTestEnv) overview(t *testing.T) installerOverview {
	t.Helper()
	var out installerOverview
	require.Equal(t, http.StatusOK, e.serve(t, http.MethodGet, "/api/dashboard/installer", "", &out))
	return out
}

func (e *installerTestEnv) writeHomeFile(t *testing.T, rel, content string) string {
	t.Helper()
	path := filepath.Join(e.home, rel)
	require.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
	require.NoError(t, os.WriteFile(path, []byte(content), 0o644))
	return path
}

func componentStatus(t *testing.T, ov installerOverview, target installer.Target, comp installer.Component) installerComponent {
	t.Helper()
	for _, tg := range ov.Targets {
		if tg.ID != target {
			continue
		}
		for _, c := range tg.Components {
			if c.Component == comp {
				return c
			}
		}
	}
	t.Fatalf("no %s/%s in overview", target, comp)
	return installerComponent{}
}

func TestInstallerAPIOverview(t *testing.T) {
	env := setupInstallerAPI(t)
	ov := env.overview(t)
	require.Len(t, ov.Targets, 7)
	assert.Equal(t, installer.AllTargets()[0], ov.Targets[0].ID, "display order")
	assert.NotNil(t, ov.LegacyWarnings)
	assert.True(t, ov.HooksEnabled, "feature_search_nudge is on by default, so the Hooks row shows")
	for _, tg := range ov.Targets {
		assert.Len(t, tg.Components, 4, tg.ID)
		assert.NotEmpty(t, tg.Name, tg.ID)
	}
	mcp := componentStatus(t, ov, installer.TargetCursor, installer.ComponentMCP)
	assert.True(t, mcp.Supported)
	assert.Equal(t, installer.StatusNotInstalled, mcp.Status)
	assert.Equal(t, filepath.Join(env.home, ".cursor", "mcp.json"), mcp.Path)
	jb := componentStatus(t, ov, installer.TargetJetBrains, installer.ComponentMCP)
	assert.False(t, jb.Supported)
	assert.Equal(t, installer.StatusUnsupported, jb.Status)
	assert.NotEmpty(t, jb.Reason)
}

// TestInstallerAPIPreviewApply covers W9 steps 2–3: the preview shows the diff, apply merges into
// the file under the temp HOME, and the status becomes Installed.
func TestInstallerAPIPreviewApply(t *testing.T) {
	env := setupInstallerAPI(t)
	path := env.writeHomeFile(t, ".cursor/mcp.json", testCursorConfig)
	plan := env.preview(t, `{"targets":["cursor"],"components":["mcp"],"action":"install"}`)
	require.Len(t, plan.Changes, 1)
	ch := plan.Changes[0]
	assert.Equal(t, path, ch.Path)
	assert.Equal(t, installer.KindModify, ch.Kind)
	assert.False(t, ch.Skipped)
	assert.Contains(t, ch.Diff, `+    "ast-context-cache"`)
	assert.Empty(t, plan.Errors)
	assert.True(t, plan.ExpiresAt.After(env.now))
	before, err := os.ReadFile(path)
	require.NoError(t, err)
	assert.Equal(t, testCursorConfig, string(before), "preview writes nothing")
	var res installer.ApplyResult
	code := env.serve(t, http.MethodPost, "/api/dashboard/installer/apply", `{"plan_id":"`+plan.ID+`"}`, &res)
	require.Equal(t, http.StatusOK, code)
	assert.Equal(t, []string{path}, res.Written)
	require.Len(t, res.Backups, 1)
	after, err := os.ReadFile(path)
	require.NoError(t, err)
	assert.Contains(t, string(after), `"other"`, "other servers kept")
	assert.Contains(t, string(after), `"ast-context-cache"`)
	assert.Equal(t, installer.StatusInstalled, componentStatus(t, env.overview(t), installer.TargetCursor, installer.ComponentMCP).Status)
}

// TestInstallerAPIApplyRepreview covers IN-5: a stale, expired, or unknown plan answers 409 with
// repreview and writes nothing.
func TestInstallerAPIApplyRepreview(t *testing.T) {
	tests := []struct {
		name     string
		mutate   func(t *testing.T, env *installerTestEnv)
		planID   string
		wantCode errs.Code
	}{
		{
			name: "file changed since preview",
			mutate: func(t *testing.T, env *installerTestEnv) {
				env.writeHomeFile(t, ".cursor/mcp.json", "{}\n")
			},
			wantCode: errs.CodeConflict,
		},
		{
			name:     "plan expired",
			mutate:   func(t *testing.T, env *installerTestEnv) { env.now = env.now.Add(11 * time.Minute) },
			wantCode: errs.CodeExpired,
		},
		{
			name:     "unknown plan",
			planID:   "plan_unknown",
			wantCode: errs.CodeNotFound,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			env := setupInstallerAPI(t)
			path := env.writeHomeFile(t, ".cursor/mcp.json", testCursorConfig)
			plan := env.preview(t, `{"targets":["cursor"],"components":["mcp"]}`)
			if tt.mutate != nil {
				tt.mutate(t, env)
			}
			id := plan.ID
			if tt.planID != "" {
				id = tt.planID
			}
			before, err := os.ReadFile(path)
			require.NoError(t, err)
			var out installerErrorBody
			code := env.serve(t, http.MethodPost, "/api/dashboard/installer/apply", `{"plan_id":"`+id+`"}`, &out)
			require.Equal(t, http.StatusConflict, code)
			assert.True(t, out.Repreview)
			assert.Equal(t, tt.wantCode, out.Code)
			assert.NotEmpty(t, out.Error)
			after, err := os.ReadFile(path)
			require.NoError(t, err)
			assert.Equal(t, string(before), string(after), "nothing written")
		})
	}
}

// TestInstallerAPIBackupsRestore covers IN-4: an apply's backup is listed and restores the file.
func TestInstallerAPIBackupsRestore(t *testing.T) {
	env := setupInstallerAPI(t)
	path := env.writeHomeFile(t, ".cursor/mcp.json", testCursorConfig)
	var empty struct {
		Backups []installer.Backup `json:"backups"`
	}
	require.Equal(t, http.StatusOK, env.serve(t, http.MethodGet, "/api/dashboard/installer/backups", "", &empty))
	assert.NotNil(t, empty.Backups)
	assert.Empty(t, empty.Backups)
	plan := env.preview(t, `{"targets":["cursor"],"components":["mcp"]}`)
	require.Equal(t, http.StatusOK, env.serve(t, http.MethodPost, "/api/dashboard/installer/apply", `{"plan_id":"`+plan.ID+`"}`, nil))
	var list struct {
		Backups []installer.Backup `json:"backups"`
	}
	require.Equal(t, http.StatusOK, env.serve(t, http.MethodGet, "/api/dashboard/installer/backups", "", &list))
	require.Len(t, list.Backups, 1)
	assert.Equal(t, path, list.Backups[0].Path)
	var restored map[string]string
	code := env.serve(t, http.MethodPost, "/api/dashboard/installer/restore", `{"backup_id":"`+list.Backups[0].ID+`"}`, &restored)
	require.Equal(t, http.StatusOK, code)
	assert.Equal(t, "restored", restored["status"])
	got, err := os.ReadFile(path)
	require.NoError(t, err)
	assert.Equal(t, testCursorConfig, string(got))
	var notFound installerErrorBody
	code = env.serve(t, http.MethodPost, "/api/dashboard/installer/restore", `{"backup_id":"20200101-000000/missing"}`, &notFound)
	assert.Equal(t, http.StatusNotFound, code)
	assert.False(t, notFound.Repreview)
}

func TestInstallerAPIRejectsBadRequests(t *testing.T) {
	env := setupInstallerAPI(t)
	tests := []struct {
		name   string
		method string
		path   string
		body   string
		want   int
	}{
		{"preview invalid json", http.MethodPost, "/api/dashboard/installer/preview", `{`, http.StatusBadRequest},
		{"preview no targets", http.MethodPost, "/api/dashboard/installer/preview", `{"targets":[]}`, http.StatusBadRequest},
		{"preview unknown target", http.MethodPost, "/api/dashboard/installer/preview", `{"targets":["emacs"]}`, http.StatusBadRequest},
		{"preview unknown action", http.MethodPost, "/api/dashboard/installer/preview", `{"targets":["cursor"],"action":"upgrade"}`, http.StatusBadRequest},
		{"preview unknown component", http.MethodPost, "/api/dashboard/installer/preview", `{"targets":["cursor"],"components":["themes"]}`, http.StatusBadRequest},
		{"preview wrong method", http.MethodGet, "/api/dashboard/installer/preview", ``, http.StatusMethodNotAllowed},
		{"apply missing plan", http.MethodPost, "/api/dashboard/installer/apply", `{}`, http.StatusBadRequest},
		{"restore missing id", http.MethodPost, "/api/dashboard/installer/restore", `{}`, http.StatusBadRequest},
		{"restore traversal", http.MethodPost, "/api/dashboard/installer/restore", `{"backup_id":"../x"}`, http.StatusBadRequest},
		{"overview wrong method", http.MethodPost, "/api/dashboard/installer", `{}`, http.StatusMethodNotAllowed},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var out installerErrorBody
			assert.Equal(t, tt.want, env.serve(t, tt.method, tt.path, tt.body, &out))
			assert.NotEmpty(t, out.Error)
		})
	}
}

func TestInstallerAPIUnavailable(t *testing.T) {
	env := setupInstallerAPI(t)
	SetInstaller(nil)
	var out installerErrorBody
	assert.Equal(t, http.StatusServiceUnavailable, env.serve(t, http.MethodGet, "/api/dashboard/installer", "", &out))
	assert.Equal(t, installerUnavailableMsg, out.Error)
}

// TestInstallerAPIGuardsWrites checks the installer writes sit behind the dashboard's Origin
// guard: a cross-site preview or apply never reaches the installer.
func TestInstallerAPIGuardsWrites(t *testing.T) {
	env := setupInstallerAPI(t)
	plan := env.preview(t, `{"targets":["cursor"],"components":["mcp"]}`)
	for _, path := range []string{"/api/dashboard/installer/preview", "/api/dashboard/installer/apply", "/api/dashboard/installer/restore"} {
		t.Run(path, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(`{"targets":["cursor"],"plan_id":"`+plan.ID+`","backup_id":"x/y"}`))
			req.Host = "127.0.0.1:7830"
			req.Header.Set("Origin", "http://evil.example")
			req.Header.Set("Content-Type", "application/json")
			rr := httptest.NewRecorder()
			env.h.ServeHTTP(rr, req)
			assert.Equal(t, http.StatusForbidden, rr.Code, rr.Body.String())
		})
	}
	_, err := os.Stat(filepath.Join(env.home, ".cursor", "mcp.json"))
	assert.True(t, os.IsNotExist(err), "the guarded apply wrote nothing")
}

// The legacy warnings stay on the card until dismissed, but showing them there counts as their
// one-time display, so a later CLI install stays quiet.
func TestInstallerAPILegacyWarningsDismiss(t *testing.T) {
	env := setupInstallerAPI(t)
	assert.Empty(t, env.overview(t).LegacyWarnings)
	_, err := db.DB.Exec("INSERT INTO agent_configs (agent_type, install_path, is_global, instructions_hash) VALUES ('cursor', '~/.cursor/mcp.json', 1, 'h')")
	require.NoError(t, err)
	require.NoError(t, db.SetSetting("installer_legacy_migrated", ""))
	env.newInstaller(t)
	require.Len(t, env.overview(t).LegacyWarnings, 1)
	assert.Empty(t, env.svc.ClaimLegacyWarnings(), "the dashboard already showed them")
	require.Len(t, env.overview(t).LegacyWarnings, 1, "still shown until dismissed")
	assert.Equal(t, http.StatusMethodNotAllowed, env.serve(t, http.MethodGet, "/api/dashboard/installer/legacy/dismiss", "", nil))
	var out map[string]string
	require.Equal(t, http.StatusOK, env.serve(t, http.MethodPost, "/api/dashboard/installer/legacy/dismiss", "", &out))
	assert.Equal(t, "dismissed", out["status"])
	assert.Empty(t, env.overview(t).LegacyWarnings)
}

// Applying while Claude Desktop runs returns the components to recheck; when the app drops the
// entry, the recheck sees it and the overview explains the missing status.
func TestInstallerAPIRecheckSeesDroppedDesktopEntry(t *testing.T) {
	env := setupInstallerAPI(t)
	prev := installerRecheckDelays
	installerRecheckDelays = nil
	t.Cleanup(func() { installerRecheckDelays = prev })
	env.desktopRunning = true
	const desktopConfig = "{\"mcpServers\": {}}\n"
	path := env.writeHomeFile(t, "Library/Application Support/Claude/claude_desktop_config.json", desktopConfig)
	plan := env.preview(t, `{"targets":["claude_desktop"],"components":["mcp"]}`)
	assert.Contains(t, strings.Join(plan.Warnings, "\n"), "Claude Desktop is running and rewrites claude_desktop_config.json from memory, which would undo this change: quit Claude Desktop, then apply")
	var res installer.ApplyResult
	require.Equal(t, http.StatusOK, env.serve(t, http.MethodPost, "/api/dashboard/installer/apply", `{"plan_id":"`+plan.ID+`"}`, &res))
	require.Len(t, res.Recheck, 1)
	assert.Equal(t, installer.StatusInstalled, res.Recheck[0].Status)
	require.NoError(t, os.WriteFile(path, []byte(desktopConfig), 0o644))
	recheckInstall(env.svc, res.Recheck, []time.Duration{0})
	mcp := componentStatus(t, env.overview(t), installer.TargetClaudeDesktop, installer.ComponentMCP)
	assert.Equal(t, installer.StatusMissing, mcp.Status)
	assert.Contains(t, mcp.StatusReason, "quit Claude Desktop, then install again")
}
