package installer

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// goldenScenarios are the AC38 fixtures per target under testdata/<target>/<scenario>/:
//   - empty: install into a HOME with no host config.
//   - existing: install into the user's existing configs (comments, foreign servers, CRLF/BOM).
//   - upgrade: install over an older install (seeded by state.json).
//   - uninstall: install, then uninstall, starting from the same tree as existing.
//
// in/ is the starting HOME and golden/ the expected HOME. Golden files use placeholders for the
// canonical assets ({{SKILL:usage}}, {{AGENTS_BLOCK}}, ...). Regenerate with
// `make test TEST_PKGS="-run TestGoldenFixtures ./internal/installer/ -args -update"` and review the diff.
var goldenScenarios = []string{"empty", "existing", "upgrade", "uninstall"}

func TestGoldenFixtures(t *testing.T) {
	for _, target := range AllTargets() {
		for _, scenario := range goldenScenarios {
			t.Run(string(target)+"/"+scenario, func(t *testing.T) {
				runGolden(t, target, scenario)
			})
		}
	}
}

func runGolden(t *testing.T, target Target, scenario string) {
	dir := filepath.Join("testdata", string(target), scenario)
	home := newTestHome(t)
	copyTree(t, filepath.Join(dir, "in"), home)
	seedState(t, home, filepath.Join(dir, "state.json"))
	before := snapshotTree(t, home)
	s := newTestService(t, home, testOpts{hooks: true})
	req := PlanRequest{Targets: []Target{target}, Action: ActionInstall}
	applyPlan(t, s, req)
	if scenario == "uninstall" {
		req.Action = ActionUninstall
		applyPlan(t, s, req)
	}
	got := snapshotTree(t, home)
	assertBackedUp(t, s, home, before, got)
	goldenDir := filepath.Join(dir, "golden")
	if *updateGolden {
		writeGolden(t, goldenDir, got)
		return
	}
	want := map[string]string{}
	for rel, content := range snapshotTree(t, goldenDir) {
		want[rel] = expandGolden(t, content)
	}
	require.Equal(t, want, got)
	// Re-running the same action is a no-op.
	p, err := s.Plan(req)
	require.NoError(t, err)
	for _, c := range p.Changes {
		assert.True(t, c.Skipped, "re-run would write %s (%s)", c.Path, c.Kind)
	}
}

// assertBackedUp checks NFR-8: every pre-existing file the run changed or deleted has a backup
// holding its original content.
func assertBackedUp(t *testing.T, s *realService, home string, before, after map[string]string) {
	t.Helper()
	backups, err := s.Backups()
	require.NoError(t, err)
	for rel, orig := range before {
		if cur, ok := after[rel]; ok && cur == orig {
			continue
		}
		path := filepath.Join(home, filepath.FromSlash(rel))
		found := false
		for _, b := range backups {
			if b.Path == path && readFile(t, filepath.Join(s.backupRoot, filepath.FromSlash(b.ID))) == orig {
				found = true
			}
		}
		assert.True(t, found, "no backup of the original %s", rel)
	}
}

func writeGolden(t *testing.T, dir string, files map[string]string) {
	t.Helper()
	require.NoError(t, os.RemoveAll(dir))
	if len(files) == 0 {
		writeFile(t, filepath.Join(dir, ".keep"), "")
		return
	}
	for rel, content := range files {
		writeFile(t, filepath.Join(dir, filepath.FromSlash(rel)), collapseGolden(t, content))
	}
}
