package installer

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoopbackURL(t *testing.T) {
	t.Parallel()
	tests := []struct {
		in, want string
	}{
		{in: "http://localhost:7821/mcp", want: "http://127.0.0.1:7821/mcp"},
		{in: "http://127.0.0.1:7821/mcp", want: "http://127.0.0.1:7821/mcp"},
		{in: "http://[::1]:7821/mcp", want: "http://127.0.0.1:7821/mcp"},
		{in: "https://localhost/mcp?x=1", want: "https://127.0.0.1/mcp?x=1"},
		{in: "http://example.com:7821/mcp", want: "http://example.com:7821/mcp"},
		{in: "http://192.168.1.5:7821/mcp", want: "http://192.168.1.5:7821/mcp"},
		{in: "localhost:7821", want: "localhost:7821"},
		{in: "bridge", want: "bridge"},
	}
	for _, tt := range tests {
		t.Run(tt.in, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.want, loopbackURL(tt.in))
		})
	}
}

func TestEntryHashesTreatLoopbackSpellingsAlike(t *testing.T) {
	t.Parallel()
	hash := func(url string) string {
		b, err := canonicalJSON(jsonObj{{"command", "mcp-local"}, {"args", []string{"bridge", url}}})
		require.NoError(t, err)
		return desiredEntryHash(b)
	}
	want := hash("http://127.0.0.1:7821/mcp")
	assert.Equal(t, want, hash("http://localhost:7821/mcp"), "nested URLs are normalized too")
	assert.Equal(t, want, hash("http://[::1]:7821/mcp"))
	assert.NotEqual(t, want, hash("http://localhost:7800/mcp"), "a different port is a different server")
	assert.NotEqual(t, want, hash("http://localhost:7821/other"), "a different path is a different server")
}

// An entry registered as localhost (what mcp-local writes) is the installer's own loopback
// registration: verify reports it installed and install leaves the file alone.
func TestLoopbackSpellingIsInstalledAndNotRewritten(t *testing.T) {
	for _, url := range []string{"http://localhost:7821/mcp", "http://[::1]:7821/mcp"} {
		t.Run(url, func(t *testing.T) {
			home := newTestHome(t)
			cfg := filepath.Join(home, ".cursor", "mcp.json")
			original := `{"mcpServers": {"ast-context-cache": {"url": "` + url + `"}}}`
			writeFile(t, cfg, original)
			s := newTestService(t, home, testOpts{})
			assert.Equal(t, StatusInstalled, statusOf(t, s, TargetCursor, ComponentMCP).Status)
			p, err := s.Plan(PlanRequest{Targets: []Target{TargetCursor}, Components: []Component{ComponentMCP}, Action: ActionInstall})
			require.NoError(t, err)
			require.Len(t, p.Changes, 1)
			assert.True(t, p.Changes[0].Skipped, "install would rewrite %s", cfg)
			assert.Equal(t, original, readFile(t, cfg))
		})
	}
}

func TestLoopbackDifferentPortIsStillReplaced(t *testing.T) {
	home := newTestHome(t)
	cfg := filepath.Join(home, ".cursor", "mcp.json")
	writeFile(t, cfg, `{"mcpServers": {"ast-context-cache": {"url": "http://localhost:7800/mcp"}}}`)
	s := newTestService(t, home, testOpts{})
	assert.Equal(t, StatusModifiedByUser, statusOf(t, s, TargetCursor, ComponentMCP).Status)
	applyPlan(t, s, PlanRequest{Targets: []Target{TargetCursor}, Components: []Component{ComponentMCP}, Action: ActionInstall})
	assert.Contains(t, readFile(t, cfg), `"http://127.0.0.1:7821/mcp"`)
}
