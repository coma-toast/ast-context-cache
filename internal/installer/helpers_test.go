package installer

import (
	"encoding/json"
	"flag"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/instructions"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/version"
	"github.com/coma-toast/ast-context-cache/rules"
	"github.com/coma-toast/ast-context-cache/skills"
)

const (
	testVersion = "4.0.0"
	testExe     = "/opt/ast/bin/ast-mcp"
	testBridge  = "/usr/local/bin/mcp-local"
)

var updateGolden = flag.Bool("update", false, "rewrite installer golden fixtures")

// TestMain pins the version stamp so goldens don't change with every release. DB-backed tests
// don't use t.Parallel: the db pools are package globals (dbtest.Init swaps them per test).
func TestMain(m *testing.M) {
	version.Version = testVersion
	os.Exit(m.Run())
}

type testOpts struct {
	url      string
	goos     string
	hooks    bool
	noBridge bool
	noNpx    bool
	now      func() time.Time
}

// newTestHome gives the test a temp HOME with an open database.
func newTestHome(t *testing.T) string {
	t.Helper()
	return dbtest.Init(t)
}

func newTestService(t *testing.T, home string, o testOpts) *realService {
	t.Helper()
	if o.url == "" {
		o.url = MCPURL(DefaultMCPPort)
	}
	if o.goos == "" {
		o.goos = "darwin"
	}
	cfg := Config{
		Home:         home,
		MCPURL:       o.url,
		Executable:   testExe,
		GOOS:         o.goos,
		HooksEnabled: func() bool { return o.hooks },
		Now:          o.now,
		LookPath: func(name string) (string, error) {
			switch {
			case name == "mcp-local" && !o.noBridge:
				return testBridge, nil
			case name == "npx" && !o.noNpx:
				return "/usr/local/bin/npx", nil
			}
			return "", errs.New("not found", "name", name)
		},
	}
	svc, err := New(cfg)
	require.NoError(t, err)
	return svc.(*realService)
}

// install plans and applies req, failing the test on plan errors.
func applyPlan(t *testing.T, s *realService, req PlanRequest) (*Plan, *ApplyResult) {
	t.Helper()
	p, err := s.Plan(req)
	require.NoError(t, err)
	require.Empty(t, p.Errors)
	res, err := s.Apply(p.ID)
	require.NoError(t, err)
	return p, res
}

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	require.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
	require.NoError(t, os.WriteFile(path, []byte(content), 0o644))
}

func readFile(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	require.NoError(t, err)
	return string(b)
}

// copyTree copies a fixture tree into home, skipping .keep placeholders.
func copyTree(t *testing.T, src, dst string) {
	t.Helper()
	err := filepath.WalkDir(src, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || d.Name() == ".keep" {
			return err
		}
		rel, _ := filepath.Rel(src, p)
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		writeFile(t, filepath.Join(dst, rel), string(b))
		return nil
	})
	require.NoError(t, err)
}

// snapshotTree reads every file under root except the installer's own data directory.
func snapshotTree(t *testing.T, root string) map[string]string {
	t.Helper()
	out := map[string]string{}
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(root, p)
		if d.IsDir() && rel == ".astcache" {
			return filepath.SkipDir
		}
		if d.IsDir() || d.Name() == ".keep" {
			return nil
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		out[filepath.ToSlash(rel)] = string(b)
		return nil
	})
	require.NoError(t, err)
	return out
}

// goldenPlaceholders maps canonical asset content to placeholders, so goldens stay valid when
// skills or instruction text change. Longest first, so whole files win over blocks inside them.
func goldenPlaceholders(t *testing.T) [][2]string {
	t.Helper()
	all, err := skills.All()
	require.NoError(t, err)
	front, body := splitFrontmatter(rules.CursorRule)
	cursorBlock := renderBlock(testVersion, body, "\n")
	out := [][2]string{
		{front + "\n" + cursorBlock + "\n", "{{CURSOR_RULE_FILE}}"},
		{cursorBlock, "{{CURSOR_BLOCK}}"},
		{renderBlock(testVersion, instructions.AgentsBlock, "\n"), "{{AGENTS_BLOCK}}"},
	}
	for _, sk := range all {
		out = append(out, [2]string{sk.Content(), "{{SKILL:" + sk.Name + "}}"})
	}
	sort.SliceStable(out, func(i, j int) bool { return len(out[i][0]) > len(out[j][0]) })
	return out
}

func expandGolden(t *testing.T, s string) string {
	t.Helper()
	for _, p := range goldenPlaceholders(t) {
		s = strings.ReplaceAll(s, p[1], p[0])
	}
	return s
}

func collapseGolden(t *testing.T, s string) string {
	t.Helper()
	for _, p := range goldenPlaceholders(t) {
		s = strings.ReplaceAll(s, p[0], p[1])
	}
	return s
}

// fixtureState is testdata/<target>/<scenario>/state.json: installer_state rows (paths relative
// to HOME) seeded before the scenario runs, as a previous install would have left them.
type fixtureState struct {
	Target    string `json:"target"`
	Component string `json:"component"`
	Path      string `json:"path"`
	// Content, when set, is hashed the way the installer hashes what it wrote.
	Content   string `json:"content"`
	EntryHash string `json:"entry_hash"`
	Created   int    `json:"created"`
}

func seedState(t *testing.T, home, file string) {
	t.Helper()
	b, err := os.ReadFile(file)
	if os.IsNotExist(err) {
		return
	}
	require.NoError(t, err)
	var rows []fixtureState
	require.NoError(t, json.Unmarshal(b, &rows))
	var ups []stateRow
	for _, r := range rows {
		hash := r.EntryHash
		if r.Content != "" {
			hash = shortHash([]byte(r.Content))
		}
		ups = append(ups, stateRow{Target: Target(r.Target), Component: Component(r.Component), Path: filepath.Join(home, r.Path), EntryHash: hash, Version: "3.9.0", Created: r.Created})
	}
	require.NoError(t, applyState(ups, nil))
}
