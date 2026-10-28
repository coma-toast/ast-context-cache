package installer

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/hooks"
)

func hooksReq(targets ...Target) PlanRequest {
	return PlanRequest{Targets: targets, Components: []Component{ComponentHooks}}
}

func TestClaudeHooksFollowFlags(t *testing.T) {
	tests := []struct {
		name      string
		hooks     bool
		nudge     bool
		wantSubs  []string
		wantNoSub []string
	}{
		{name: "nudge only", nudge: true, wantSubs: []string{"search-nudge"}, wantNoSub: []string{"session-start", "pre-tool-use-agent"}},
		{name: "handoff only", hooks: true, wantSubs: []string{"session-start", "pre-tool-use-agent"}, wantNoSub: []string{"search-nudge"}},
		{name: "both", hooks: true, nudge: true, wantSubs: []string{"session-start", "subagent-start", "subagent-stop", "pre-tool-use-agent", "search-nudge"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			home := newTestHome(t)
			s := newTestService(t, home, testOpts{hooks: tt.hooks, nudge: tt.nudge})
			applyPlan(t, s, hooksReq(TargetClaudeCode))
			got := readFile(t, filepath.Join(home, ".claude", "settings.json"))
			for _, sub := range tt.wantSubs {
				assert.Equal(t, 1, strings.Count(got, testExe+" hook "+sub), sub)
			}
			for _, sub := range tt.wantNoSub {
				assert.NotContains(t, got, " hook "+sub)
			}
			if tt.nudge {
				assert.Contains(t, got, `"matcher": "Grep|Glob|Read"`)
			}
			assert.Equal(t, StatusInstalled, statusOf(t, s, TargetClaudeCode, ComponentHooks).Status)
			p, err := s.Plan(hooksReq(TargetClaudeCode))
			require.NoError(t, err)
			for _, c := range p.Changes {
				assert.True(t, c.Skipped, "re-run is a no-op")
			}
		})
	}
}

func TestClaudeHooksUpgradeAddsNudge(t *testing.T) {
	home := newTestHome(t)
	path := filepath.Join(home, ".claude", "settings.json")
	applyPlan(t, newTestService(t, home, testOpts{hooks: true}), hooksReq(TargetClaudeCode))
	s := newTestService(t, home, testOpts{hooks: true, nudge: true})
	assert.Equal(t, StatusOutdated, statusOf(t, s, TargetClaudeCode, ComponentHooks).Status, "a new spec makes the install outdated")
	applyPlan(t, s, hooksReq(TargetClaudeCode))
	got := readFile(t, path)
	assert.Equal(t, 1, strings.Count(got, " hook search-nudge"))
	assert.Equal(t, 1, strings.Count(got, " hook pre-tool-use-agent"))
	assert.Equal(t, StatusInstalled, statusOf(t, s, TargetClaudeCode, ComponentHooks).Status)
	off := newTestService(t, home, testOpts{hooks: true})
	assert.Equal(t, StatusOutdated, statusOf(t, off, TargetClaudeCode, ComponentHooks).Status, "turning the flag off leaves an outdated entry")
	applyPlan(t, off, hooksReq(TargetClaudeCode))
	assert.NotContains(t, readFile(t, path), "search-nudge", "reinstalling with the flag off removes the nudge")
	applyPlan(t, off, PlanRequest{Targets: []Target{TargetClaudeCode}, Components: []Component{ComponentHooks}, Action: ActionUninstall})
	assert.NoFileExists(t, path, "the settings.json the installer created is deleted once empty")
}

func TestNudgeHooksRequireFlag(t *testing.T) {
	home := newTestHome(t)
	s := newTestService(t, home, testOpts{})
	for _, target := range []Target{TargetCursor, TargetOpenCode} {
		st := statusOf(t, s, target, ComponentHooks)
		assert.Equal(t, StatusUnsupported, st.Status, target)
		assert.Contains(t, st.Reason, "feature_search_nudge", target)
	}
	for _, info := range s.Targets() {
		for _, ci := range info.Components {
			if ci.Component == ComponentHooks && (info.ID == TargetCursor || info.ID == TargetOpenCode || info.ID == TargetClaudeCode) {
				assert.False(t, ci.Supported, info.ID)
			}
		}
	}
	on := newTestService(t, home, testOpts{nudge: true})
	for _, info := range on.Targets() {
		for _, ci := range info.Components {
			if ci.Component == ComponentHooks && (info.ID == TargetCursor || info.ID == TargetOpenCode || info.ID == TargetClaudeCode) {
				assert.True(t, ci.Supported, info.ID)
			}
		}
	}
}

func TestOpenCodeNudgePlugin(t *testing.T) {
	home := newTestHome(t)
	s := newTestService(t, home, testOpts{nudge: true, url: MCPURL(7999)})
	path := filepath.Join(home, ".config", "opencode", "plugins", openCodeNudgeFile)
	applyPlan(t, s, hooksReq(TargetOpenCode))
	got := readFile(t, path)
	assert.Contains(t, got, `const HEALTH_URL = "http://127.0.0.1:7999/health"`)
	assert.Contains(t, got, `"tool.execute.before"`)
	assert.NotContains(t, got, "__", "every placeholder is filled")
	for _, msg := range hooks.NudgeMessages {
		assert.Contains(t, got, strings.ReplaceAll(msg, "<", `<`))
	}
	assert.Equal(t, StatusInstalled, statusOf(t, s, TargetOpenCode, ComponentHooks).Status)
	writeFile(t, path, got+"// mine\n")
	assert.Equal(t, StatusModifiedByUser, statusOf(t, s, TargetOpenCode, ComponentHooks).Status)
	p, err := s.Plan(PlanRequest{Targets: []Target{TargetOpenCode}, Components: []Component{ComponentHooks}, Action: ActionUninstall})
	require.NoError(t, err)
	assert.Len(t, p.Warnings, 1, "deleting an edited plugin warns")
	_, err = s.Apply(p.ID)
	require.NoError(t, err)
	assert.NoFileExists(t, path)
}

func TestOpenCodeNudgePluginExternal(t *testing.T) {
	home := newTestHome(t)
	path := filepath.Join(home, ".config", "opencode", "plugins", openCodeNudgeFile)
	writeFile(t, path, "// someone else's\n")
	s := newTestService(t, home, testOpts{nudge: true})
	assert.Equal(t, StatusExternallyManaged, statusOf(t, s, TargetOpenCode, ComponentHooks).Status)
	applyPlan(t, s, hooksReq(TargetOpenCode))
	assert.Equal(t, "// someone else's\n", readFile(t, path), "a file the installer did not create is left alone")
	applyPlan(t, s, PlanRequest{Targets: []Target{TargetOpenCode}, Components: []Component{ComponentHooks}, Action: ActionUninstall})
	assert.FileExists(t, path)
}

func TestCursorNudgeHookCreatedFileIsRemoved(t *testing.T) {
	home := newTestHome(t)
	path := filepath.Join(home, ".cursor", "hooks.json")
	s := newTestService(t, home, testOpts{nudge: true})
	applyPlan(t, s, hooksReq(TargetCursor))
	assert.Contains(t, readFile(t, path), testExe+" hook cursor-search-nudge")
	assert.Equal(t, StatusInstalled, statusOf(t, s, TargetCursor, ComponentHooks).Status)
	applyPlan(t, s, PlanRequest{Targets: []Target{TargetCursor}, Components: []Component{ComponentHooks}, Action: ActionUninstall})
	_, err := os.Stat(path)
	assert.True(t, os.IsNotExist(err), "a hooks.json the installer created is deleted on uninstall")
}

func TestCursorNudgeHookKeepsUserHooks(t *testing.T) {
	home := newTestHome(t)
	path := filepath.Join(home, ".cursor", "hooks.json")
	user := `{"version": 1, "hooks": {"preToolUse": [{"command": "./mine.sh"}, {"command": "/old/ast-mcp hook cursor-search-nudge", "matcher": "Read"}]}}` + "\n"
	writeFile(t, path, user)
	s := newTestService(t, home, testOpts{nudge: true})
	assert.Equal(t, StatusOutdated, statusOf(t, s, TargetCursor, ComponentHooks).Status)
	applyPlan(t, s, hooksReq(TargetCursor))
	got := readFile(t, path)
	assert.Contains(t, got, "./mine.sh")
	assert.NotContains(t, got, "/old/ast-mcp")
	assert.Equal(t, 1, strings.Count(got, "cursor-search-nudge"))
	applyPlan(t, s, PlanRequest{Targets: []Target{TargetCursor}, Components: []Component{ComponentHooks}, Action: ActionUninstall})
	got = readFile(t, path)
	assert.Contains(t, got, "./mine.sh")
	assert.NotContains(t, got, "ast-mcp")
}
