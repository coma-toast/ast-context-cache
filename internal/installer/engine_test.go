package installer

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/instructions"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
)

const cursorWithServers = `{
  // my servers
  "mcpServers": {
    "one": { "url": "https://one.example/mcp" }, // first
    /* second */
    "two": { "command": "two" },
    "three": { "command": "three", "args": ["x"] },
  },
}
`

func statusOf(t *testing.T, s *realService, target Target, c Component) ComponentStatus {
	t.Helper()
	all, err := s.Verify([]Target{target})
	require.NoError(t, err)
	for _, st := range all {
		if st.Component == c {
			return st
		}
	}
	t.Fatalf("no status for %s/%s", target, c)
	return ComponentStatus{}
}

// AC31: three foreign servers and comments survive, our entry is added, and a backup exists.
func TestInstallKeepsForeignServersAndBacksUp(t *testing.T) {
	home := newTestHome(t)
	path := filepath.Join(home, ".cursor", "mcp.json")
	writeFile(t, path, cursorWithServers)
	s := newTestService(t, home, testOpts{})
	_, res := applyPlan(t, s, PlanRequest{Targets: []Target{TargetCursor}, Components: []Component{ComponentMCP}})
	got := readFile(t, path)
	for _, keep := range []string{`"one": { "url": "https://one.example/mcp" }, // first`, "/* second */", `"two": { "command": "two" }`, `"three": { "command": "three", "args": ["x"] }`, "// my servers"} {
		assert.Contains(t, got, keep)
	}
	assert.Contains(t, got, `"ast-context-cache": {`)
	require.Len(t, res.Backups, 1)
	assert.Equal(t, path, res.Backups[0].Path)
	assert.Equal(t, cursorWithServers, readFile(t, filepath.Join(s.backupRoot, res.Backups[0].ID)))
	assert.Regexp(t, `^\d{8}-\d{6}/`, res.Backups[0].ID)
	assert.Equal(t, StatusInstalled, statusOf(t, s, TargetCursor, ComponentMCP).Status)
}

// AC32: a malformed TOML aborts the target and leaves the file byte-identical.
func TestMalformedTOMLAborts(t *testing.T) {
	home := newTestHome(t)
	path := filepath.Join(home, ".codex", "config.toml")
	bad := "model = \"o3\"\n[mcp_servers\nurl = "
	writeFile(t, path, bad)
	s := newTestService(t, home, testOpts{})
	p, err := s.Plan(PlanRequest{Targets: []Target{TargetCodex}})
	require.NoError(t, err)
	require.Len(t, p.Errors, 1)
	assert.Equal(t, errs.CodeInvalidInput, p.Errors[0].Code)
	for _, c := range p.Changes {
		assert.True(t, c.Skipped, "aborted target still writes %s", c.Path)
	}
	_, err = s.Apply(p.ID)
	require.NoError(t, err)
	assert.Equal(t, bad, readFile(t, path))
	_, err = os.Stat(filepath.Join(home, ".codex", "AGENTS.md"))
	assert.True(t, os.IsNotExist(err), "no other codex file is written when the target aborts")
}

// AC33: uninstall removes only our registration, our skill files, our block, and our hooks, and
// deletes no file the installer didn't create.
func TestClaudeCodeUninstallIsSurgical(t *testing.T) {
	home := newTestHome(t)
	claudeJSON := "{\n  \"theme\": \"dark\",\n  \"mcpServers\": {\n    \"other\": {\"type\": \"http\", \"url\": \"https://x\"}\n  }\n}\n"
	claudeMD := "# Mine\n\nKeep me.\n"
	userSkill := filepath.Join(home, ".claude", "skills", "my-skill", "SKILL.md")
	writeFile(t, filepath.Join(home, ".claude.json"), claudeJSON)
	writeFile(t, filepath.Join(home, ".claude", "CLAUDE.md"), claudeMD)
	writeFile(t, userSkill, "mine\n")
	s := newTestService(t, home, testOpts{hooks: true})
	req := PlanRequest{Targets: []Target{TargetClaudeCode}}
	applyPlan(t, s, req)
	settings := filepath.Join(home, ".claude", "settings.json")
	require.FileExists(t, settings)
	// The user adds their own file to one of our skill directories after the install.
	extra := filepath.Join(home, ".claude", "skills", "ast-context-cache-usage", "notes.md")
	writeFile(t, extra, "user notes\n")
	req.Action = ActionUninstall
	applyPlan(t, s, req)
	assert.Equal(t, claudeJSON, readFile(t, filepath.Join(home, ".claude.json")))
	assert.Equal(t, claudeMD, readFile(t, filepath.Join(home, ".claude", "CLAUDE.md")))
	assert.FileExists(t, userSkill)
	assert.FileExists(t, extra)
	assert.NoFileExists(t, filepath.Join(home, ".claude", "skills", "ast-context-cache-usage", "SKILL.md"))
	assert.NoDirExists(t, filepath.Join(home, ".claude", "skills", "ast-context-cache-agents"))
	assert.NoFileExists(t, settings, "settings.json was created by the installer and is empty again")
	for _, c := range []Component{ComponentMCP, ComponentSkills, ComponentRules, ComponentHooks} {
		assert.Equal(t, StatusNotInstalled, statusOf(t, s, TargetClaudeCode, c).Status, c)
	}
}

// AC34: a symlinked skill dir outside ~/.astcache is externally managed and skipped by default;
// replacing it takes a backup that restores the link.
func TestExternallyManagedSkills(t *testing.T) {
	home := newTestHome(t)
	ext := filepath.Join(t.TempDir(), "configsync", "ast-context-cache")
	writeFile(t, filepath.Join(ext, "SKILL.md"), "external skill\n")
	link := filepath.Join(home, ".claude", "skills", "ast-context-cache")
	require.NoError(t, os.MkdirAll(filepath.Dir(link), 0o755))
	require.NoError(t, os.Symlink(ext, link))
	s := newTestService(t, home, testOpts{})
	st := statusOf(t, s, TargetClaudeCode, ComponentSkills)
	assert.Equal(t, StatusExternallyManaged, st.Status)
	assert.Contains(t, st.Reason, "symlink")
	req := PlanRequest{Targets: []Target{TargetClaudeCode}, Components: []Component{ComponentSkills}}
	p, err := s.Plan(req)
	require.NoError(t, err)
	require.Len(t, p.Changes, 1)
	assert.True(t, p.Changes[0].Skipped)
	assert.Contains(t, p.Changes[0].Reason, "externally managed")

	req.ReplaceExternal = true
	_, res := applyPlan(t, s, req)
	fi, err := os.Lstat(link)
	assert.True(t, os.IsNotExist(err), "the external symlink is removed: %v", fi)
	assert.Equal(t, "external skill\n", readFile(t, filepath.Join(ext, "SKILL.md")), "the link target is untouched")
	assert.Equal(t, StatusInstalled, statusOf(t, s, TargetClaudeCode, ComponentSkills).Status)
	require.Len(t, res.Backups, 1)
	assert.True(t, res.Backups[0].Symlink)
	require.NoError(t, s.Restore(res.Backups[0].ID))
	target, err := os.Readlink(link)
	require.NoError(t, err)
	assert.Equal(t, ext, target)
}

// AC35: an edited block reads Modified by user; a block from an older version reads Outdated.
func TestBlockStatusFromDisk(t *testing.T) {
	home := newTestHome(t)
	path := filepath.Join(home, ".claude", "CLAUDE.md")
	s := newTestService(t, home, testOpts{})
	req := PlanRequest{Targets: []Target{TargetClaudeCode}, Components: []Component{ComponentRules}}
	applyPlan(t, s, req)
	assert.Equal(t, StatusInstalled, statusOf(t, s, TargetClaudeCode, ComponentRules).Status)
	content := readFile(t, path)
	writeFile(t, path, strings.Replace(content, "Prefer the", "Always prefer the", 1))
	assert.Equal(t, StatusModifiedByUser, statusOf(t, s, TargetClaudeCode, ComponentRules).Status)
	writeFile(t, path, "# Mine\n\n"+renderBlock("3.9.0", instructions.AgentsBlock, "\n")+"\n")
	assert.Equal(t, StatusOutdated, statusOf(t, s, TargetClaudeCode, ComponentRules).Status)
	os.Remove(path)
	assert.Equal(t, StatusMissing, statusOf(t, s, TargetClaudeCode, ComponentRules).Status)
}

// AC36: a file changed between preview and apply fails with CodeConflict and nothing is written.
func TestApplyConflictWritesNothing(t *testing.T) {
	home := newTestHome(t)
	mcp := filepath.Join(home, ".cursor", "mcp.json")
	writeFile(t, mcp, "{}\n")
	s := newTestService(t, home, testOpts{})
	p, err := s.Plan(PlanRequest{Targets: []Target{TargetCursor}})
	require.NoError(t, err)
	writeFile(t, mcp, "{\"mcpServers\": {}}\n")
	_, err = s.Apply(p.ID)
	require.Error(t, err)
	assert.True(t, errs.HasCode(err, errs.CodeConflict))
	assert.Equal(t, "{\"mcpServers\": {}}\n", readFile(t, mcp))
	assert.NoFileExists(t, filepath.Join(home, ".cursor", "rules", "ast-context-cache.mdc"))
	assert.NoDirExists(t, filepath.Join(home, ".agents", "skills"))
	backups, err := s.Backups()
	require.NoError(t, err)
	assert.Empty(t, backups)
	_, err = s.Apply(p.ID)
	assert.True(t, errs.HasCode(err, errs.CodeNotFound), "a conflicted plan must be previewed again")
}

func TestPlanExpires(t *testing.T) {
	home := newTestHome(t)
	now := time.Now()
	s := newTestService(t, home, testOpts{now: func() time.Time { return now }})
	p, err := s.Plan(PlanRequest{Targets: []Target{TargetCursor}})
	require.NoError(t, err)
	now = now.Add(planTTL + time.Second)
	_, err = s.Apply(p.ID)
	assert.True(t, errs.HasCode(err, errs.CodeExpired))
}

// AC37: every registration uses the configured port.
func TestNonDefaultPortEverywhere(t *testing.T) {
	home := newTestHome(t)
	url := MCPURL(9911)
	s := newTestService(t, home, testOpts{url: url})
	applyPlan(t, s, PlanRequest{Targets: AllTargets(), Components: []Component{ComponentMCP}})
	files := []string{
		".claude.json",
		".cursor/mcp.json",
		".config/opencode/opencode.json",
		".codex/config.toml",
		"Library/Application Support/Claude/claude_desktop_config.json",
		"Library/Application Support/Code/User/mcp.json",
	}
	for _, f := range files {
		got := readFile(t, filepath.Join(home, f))
		assert.Contains(t, got, "http://127.0.0.1:9911/mcp", f)
		assert.NotContains(t, got, "7821", f)
	}
	for _, st := range mustVerify(t, s) {
		if st.Component == ComponentMCP && st.Target != TargetJetBrains {
			assert.Equal(t, StatusInstalled, st.Status, st.Target)
		}
	}
	jb := statusOf(t, s, TargetJetBrains, ComponentMCP)
	assert.Equal(t, StatusUnsupported, jb.Status)
	assert.Contains(t, jb.Reason, "http://127.0.0.1:9911/mcp")
}

func mustVerify(t *testing.T, s *realService) []ComponentStatus {
	t.Helper()
	out, err := s.Verify(nil)
	require.NoError(t, err)
	return out
}

func TestHooksAppendKeepsUserHooks(t *testing.T) {
	home := newTestHome(t)
	path := filepath.Join(home, ".claude", "settings.json")
	user := `{
  "hooks": {
    "SessionStart": [
      {"matcher": "startup", "hooks": [{"type": "command", "command": "~/bin/mine.sh"}]}
    ],
    "PreToolUse": [
      {"matcher": "Bash", "hooks": [{"type": "command", "command": "~/bin/guard.sh"}, {"type": "command", "command": "/old/ast-mcp hook pre-tool-use-agent"}]}
    ]
  }
}
`
	writeFile(t, path, user)
	s := newTestService(t, home, testOpts{hooks: true})
	req := PlanRequest{Targets: []Target{TargetClaudeCode}, Components: []Component{ComponentHooks}}
	applyPlan(t, s, req)
	got := readFile(t, path)
	assert.Contains(t, got, `"command": "~/bin/mine.sh"`)
	assert.Contains(t, got, `"command": "~/bin/guard.sh"`)
	assert.NotContains(t, got, "/old/ast-mcp", "a stale hook of ours inside a user group is replaced")
	for _, sub := range []string{"session-start", "subagent-start", "subagent-stop", "pre-tool-use-agent"} {
		assert.Equal(t, 1, strings.Count(got, testExe+" hook "+sub), sub)
	}
	assert.Equal(t, StatusInstalled, statusOf(t, s, TargetClaudeCode, ComponentHooks).Status)
	req.Action = ActionUninstall
	applyPlan(t, s, req)
	got = readFile(t, path)
	assert.Contains(t, got, `"command": "~/bin/mine.sh"`)
	assert.Contains(t, got, `"command": "~/bin/guard.sh"`)
	assert.NotContains(t, got, "ast-mcp")
}

func TestHooksRequireFlag(t *testing.T) {
	home := newTestHome(t)
	s := newTestService(t, home, testOpts{})
	st := statusOf(t, s, TargetClaudeCode, ComponentHooks)
	assert.Equal(t, StatusUnsupported, st.Status)
	assert.Contains(t, st.Reason, "feature_handoff_hooks")
	p, err := s.Plan(PlanRequest{Targets: []Target{TargetClaudeCode}})
	require.NoError(t, err)
	for _, c := range p.Changes {
		assert.NotEqual(t, ComponentHooks, c.Component, "hooks are not offered while the flag is off")
	}
	for _, info := range s.Targets() {
		if info.ID != TargetClaudeCode {
			continue
		}
		for _, ci := range info.Components {
			if ci.Component == ComponentHooks {
				assert.False(t, ci.Supported)
			}
		}
	}
}

func TestSharedAgentsSkills(t *testing.T) {
	home := newTestHome(t)
	s := newTestService(t, home, testOpts{})
	skill := filepath.Join(home, ".agents", "skills", "ast-context-cache-usage", "SKILL.md")
	p, _ := applyPlan(t, s, PlanRequest{Targets: []Target{TargetCursor, TargetCodex}, Components: []Component{ComponentSkills}})
	writes := 0
	for _, c := range p.Changes {
		if c.Path == skill && !c.Skipped {
			writes++
		}
	}
	assert.Equal(t, 1, writes, "a shared skill file is written once")
	assert.Equal(t, StatusInstalled, statusOf(t, s, TargetCodex, ComponentSkills).Status)
	assert.Equal(t, StatusInstalled, statusOf(t, s, TargetCursor, ComponentSkills).Status)
	assert.Equal(t, StatusCovered, statusOf(t, s, TargetOpenCode, ComponentSkills).Status)
	applyPlan(t, s, PlanRequest{Targets: []Target{TargetCursor}, Components: []Component{ComponentSkills}, Action: ActionUninstall})
	assert.FileExists(t, skill, "codex still uses the shared skill")
	applyPlan(t, s, PlanRequest{Targets: []Target{TargetCodex}, Components: []Component{ComponentSkills}, Action: ActionUninstall})
	assert.NoFileExists(t, skill)
}

func TestCursorSkillsCoveredByClaudeSkills(t *testing.T) {
	home := newTestHome(t)
	s := newTestService(t, home, testOpts{})
	applyPlan(t, s, PlanRequest{Targets: []Target{TargetClaudeCode}, Components: []Component{ComponentSkills}})
	st := statusOf(t, s, TargetCursor, ComponentSkills)
	assert.Equal(t, StatusCovered, st.Status)
	assert.Contains(t, st.Reason, "~/.claude/skills/")
	p, err := s.Plan(PlanRequest{Targets: []Target{TargetCursor}, Components: []Component{ComponentSkills}})
	require.NoError(t, err)
	require.Len(t, p.Changes, 1)
	assert.True(t, p.Changes[0].Skipped)
}

func TestCursorRuleFileNotOurs(t *testing.T) {
	home := newTestHome(t)
	path := filepath.Join(home, ".cursor", "rules", "ast-context-cache.mdc")
	writeFile(t, path, "---\nalwaysApply: true\n---\n\nmy own rule\n")
	s := newTestService(t, home, testOpts{})
	assert.Equal(t, StatusExternallyManaged, statusOf(t, s, TargetCursor, ComponentRules).Status)
	p, err := s.Plan(PlanRequest{Targets: []Target{TargetCursor}, Components: []Component{ComponentRules}})
	require.NoError(t, err)
	require.True(t, p.Changes[0].Skipped)
	_, res := applyPlan(t, s, PlanRequest{Targets: []Target{TargetCursor}, Components: []Component{ComponentRules}, ReplaceExternal: true})
	require.Len(t, res.Backups, 1)
	assert.Equal(t, StatusInstalled, statusOf(t, s, TargetCursor, ComponentRules).Status)
}

func TestVSCodeParseErrorGivesManualSteps(t *testing.T) {
	home := newTestHome(t)
	path := filepath.Join(home, "Library", "Application Support", "Code", "User", "mcp.json")
	bad := "{\n\t\"servers\": {\n\t\t\"a\": {\"url\": \"x\"},,\n\t}\n}\n"
	writeFile(t, path, bad)
	s := newTestService(t, home, testOpts{})
	p, err := s.Plan(PlanRequest{Targets: []Target{TargetVSCode}})
	require.NoError(t, err)
	require.Len(t, p.Errors, 1)
	assert.Equal(t, errs.CodeInvalidInput, p.Errors[0].Code)
	assert.Contains(t, p.Errors[0].Message, "MCP: Open User Configuration")
	_, err = s.Apply(p.ID)
	require.NoError(t, err)
	assert.Equal(t, bad, readFile(t, path))
}

func TestClaudeDesktopBridgeFallback(t *testing.T) {
	home := newTestHome(t)
	s := newTestService(t, home, testOpts{noBridge: true})
	p, err := s.Plan(PlanRequest{Targets: []Target{TargetClaudeDesktop}})
	require.NoError(t, err)
	require.NotEmpty(t, p.Changes)
	assert.Contains(t, string(p.Changes[0].After), `"command": "npx"`)
	assert.Contains(t, string(p.Changes[0].After), `"mcp-remote"`)
	assert.Len(t, p.Warnings, 1)
	s = newTestService(t, home, testOpts{noBridge: true, noNpx: true})
	p, err = s.Plan(PlanRequest{Targets: []Target{TargetClaudeDesktop}})
	require.NoError(t, err)
	assert.Len(t, p.Warnings, 2)
	assert.Contains(t, p.Warnings[1], "npx is not on PATH")
}

func TestLinuxPaths(t *testing.T) {
	home := newTestHome(t)
	s := newTestService(t, home, testOpts{goos: "linux"})
	assert.Equal(t, filepath.Join(home, ".config", "Claude", "claude_desktop_config.json"), statusOf(t, s, TargetClaudeDesktop, ComponentMCP).Path)
	assert.Equal(t, filepath.Join(home, ".config", "Code", "User", "mcp.json"), statusOf(t, s, TargetVSCode, ComponentMCP).Path)
}

func TestWindowsUnsupported(t *testing.T) {
	home := newTestHome(t)
	s := newTestService(t, home, testOpts{goos: "windows"})
	for _, st := range mustVerify(t, s) {
		assert.Equal(t, StatusUnsupported, st.Status)
		assert.Contains(t, st.Reason, "Windows")
	}
}

func TestBackupPruneAndRestore(t *testing.T) {
	home := newTestHome(t)
	require.NoError(t, db.SetSetting(backupKeepSetting, "2"))
	now := time.Date(2026, 10, 5, 12, 0, 0, 0, time.Local)
	s := newTestService(t, home, testOpts{now: func() time.Time { return now }})
	path := filepath.Join(home, ".cursor", "mcp.json")
	req := PlanRequest{Targets: []Target{TargetCursor}, Components: []Component{ComponentMCP}}
	for i := range 4 {
		writeFile(t, path, "{\"n\": "+string(rune('0'+i))+"}\n")
		applyPlan(t, s, req)
		now = now.Add(time.Minute)
	}
	backups, err := s.Backups()
	require.NoError(t, err)
	require.Len(t, backups, 2)
	assert.Equal(t, "{\"n\": 3}\n", readFile(t, filepath.Join(s.backupRoot, backups[0].ID)), "newest first")
	require.NoError(t, s.Restore(backups[1].ID))
	assert.Equal(t, "{\"n\": 2}\n", readFile(t, path))
	assert.Error(t, s.Restore("../../etc/passwd"))
	assert.True(t, errs.HasCode(s.Restore("20261005-120000/nope"), errs.CodeNotFound))
}

func TestAtomicWriteFollowsSymlinkAndKeepsMode(t *testing.T) {
	dir := t.TempDir()
	real := filepath.Join(dir, "real", "CLAUDE.md")
	writeFile(t, real, "old\n")
	require.NoError(t, os.Chmod(real, 0o600))
	link := filepath.Join(dir, "CLAUDE.md")
	require.NoError(t, os.Symlink(real, link))
	require.NoError(t, atomicWrite(link, []byte("new\n"), 0o644))
	assert.Equal(t, "new\n", readFile(t, real))
	fi, err := os.Lstat(link)
	require.NoError(t, err)
	assert.NotZero(t, fi.Mode()&os.ModeSymlink, "the link survives")
	fi, err = os.Stat(real)
	require.NoError(t, err)
	assert.Equal(t, os.FileMode(0o600), fi.Mode().Perm())
}

func TestLegacyMigration(t *testing.T) {
	home := newTestHome(t)
	_, err := db.DB.Exec("INSERT INTO agent_configs (agent_type, install_path, is_global, instructions_hash) VALUES ('claude_code', '~/.claude.json', 1, 'h'), ('cursor', '.cursor/mcp.json', 0, 'h')")
	require.NoError(t, err)
	writeFile(t, filepath.Join(home, ".claude.json"), "# Code Context Instructions\n")
	s := newTestService(t, home, testOpts{})
	w := s.LegacyWarnings()
	require.Len(t, w, 2)
	joined := strings.Join(w, "\n")
	assert.Contains(t, joined, "~/.claude/backups/")
	assert.Contains(t, joined, "not valid JSON")
	assert.Contains(t, joined, "project-scope")
	assert.Equal(t, "1", db.GetSetting(legacyMigratedSetting, ""))
	_, err = db.DB.Exec("DELETE FROM agent_configs")
	require.NoError(t, err)
	s = newTestService(t, home, testOpts{})
	assert.Len(t, s.LegacyWarnings(), 2, "the check runs once; its findings persist")
}

func TestPlanValidation(t *testing.T) {
	home := newTestHome(t)
	s := newTestService(t, home, testOpts{})
	_, err := s.Plan(PlanRequest{})
	assert.True(t, errs.HasCode(err, errs.CodeInvalidInput))
	_, err = s.Plan(PlanRequest{Targets: []Target{"emacs"}})
	assert.True(t, errs.HasCode(err, errs.CodeInvalidInput))
	_, err = s.Plan(PlanRequest{Targets: []Target{TargetCursor}, Components: []Component{"themes"}})
	assert.True(t, errs.HasCode(err, errs.CodeInvalidInput))
	_, err = s.Apply("plan_missing")
	assert.True(t, errs.HasCode(err, errs.CodeNotFound))
}

func TestUnsupportedComponentExplained(t *testing.T) {
	home := newTestHome(t)
	s := newTestService(t, home, testOpts{})
	p, err := s.Plan(PlanRequest{Targets: []Target{TargetJetBrains}, Components: []Component{ComponentMCP}})
	require.NoError(t, err)
	require.Len(t, p.Changes, 1)
	assert.True(t, p.Changes[0].Skipped)
	assert.Contains(t, p.Changes[0].Reason, "Settings → Tools → AI Assistant")
	require.Len(t, p.Status, 1)
	assert.Equal(t, StatusUnsupported, p.Status[0].Status)
}

// A symlink at one of our own skill paths is replaced by a real directory, and restoring its
// backup puts the link back after saving the SKILL.md written in its place.
func TestReplaceAndRestoreOwnNamedSkillSymlink(t *testing.T) {
	home := newTestHome(t)
	ext := filepath.Join(t.TempDir(), "usage")
	writeFile(t, filepath.Join(ext, "SKILL.md"), "external usage\n")
	link := filepath.Join(home, ".claude", "skills", "ast-context-cache-usage")
	require.NoError(t, os.MkdirAll(filepath.Dir(link), 0o755))
	require.NoError(t, os.Symlink(ext, link))
	s := newTestService(t, home, testOpts{})
	assert.Equal(t, StatusExternallyManaged, statusOf(t, s, TargetClaudeCode, ComponentSkills).Status)
	req := PlanRequest{Targets: []Target{TargetClaudeCode}, Components: []Component{ComponentSkills}, ReplaceExternal: true}
	_, res := applyPlan(t, s, req)
	fi, err := os.Lstat(link)
	require.NoError(t, err)
	assert.True(t, fi.IsDir(), "the link is now a real directory")
	assert.Equal(t, "external usage\n", readFile(t, filepath.Join(ext, "SKILL.md")))
	assert.Equal(t, StatusInstalled, statusOf(t, s, TargetClaudeCode, ComponentSkills).Status)
	require.Len(t, res.Backups, 1)
	require.NoError(t, s.Restore(res.Backups[0].ID))
	target, err := os.Readlink(link)
	require.NoError(t, err)
	assert.Equal(t, ext, target)
	backups, err := s.Backups()
	require.NoError(t, err)
	assert.Len(t, backups, 2, "the SKILL.md the link replaced was backed up")
}
