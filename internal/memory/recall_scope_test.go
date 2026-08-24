package memory

import (
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

func TestProjectMatch(t *testing.T) {
	if frag, args := projectMatch("project_path", nil); frag != "" || args != nil {
		t.Fatalf("empty: frag=%q args=%v", frag, args)
	}
	if frag, args := projectMatch("project_path", []string{"/a"}); frag != "project_path = ?" || len(args) != 1 {
		t.Fatalf("single: frag=%q args=%v", frag, args)
	}
	frag, args := projectMatch("sm.project_path", []string{"/a", "/b"})
	if frag != "sm.project_path IN (?,?)" || len(args) != 2 {
		t.Fatalf("multi: frag=%q args=%v", frag, args)
	}
}

func TestScopeClauseSiblingsAreOptIn(t *testing.T) {
	in := RecallInput{ProjectPath: "/a", Scope: ScopeProject}
	frag, args := scopeClause(in)
	if frag != ` AND scope = 'project' AND project_path = ?` || len(args) != 1 {
		t.Fatalf("default frag=%q args=%v", frag, args)
	}
	if frag, _ := scopeClauseFor(in, "sm."); frag != ` AND sm.scope = 'project' AND sm.project_path = ?` {
		t.Fatalf("prefixed frag=%q", frag)
	}
}

// A memory stored while working in one WTG worktree must be recallable from a
// sibling worktree of the same repo sitting on a different branch.
func TestRecallCrossesRepoSiblings(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if err := db.Init(); err != nil {
		t.Fatal(err)
	}
	alpha := filepath.Join(home, "spaces", "alpha", "repo")
	bravo := filepath.Join(home, "spaces", "bravo", "repo")
	git := func(args ...string) {
		if out, err := exec.Command("git", args...).CombinedOutput(); err != nil {
			t.Skipf("git unavailable: %v: %s", err, out)
		}
	}
	git("init", "-q", alpha)
	git("-C", alpha, "config", "user.email", "test@example.com")
	git("-C", alpha, "config", "user.name", "test")
	git("-C", alpha, "commit", "-q", "--allow-empty", "-m", "init")
	git("-C", alpha, "worktree", "add", "-q", "-b", "feature", bravo)

	// Both worktrees are indexed, which is what makes them known siblings.
	db.IndexDB.Exec(`INSERT INTO symbols (name, kind, file, start_line, end_line, project_path) VALUES ('X','function',?,1,1,?)`, filepath.Join(alpha, "a.go"), alpha)
	db.IndexDB.Exec(`INSERT INTO symbols (name, kind, file, start_line, end_line, project_path) VALUES ('X','function',?,1,1,?)`, filepath.Join(bravo, "a.go"), bravo)

	if _, err := Store(StoreInput{
		Kind: KindFact, Scope: ScopeProject, ProjectPath: alpha,
		Subject: "login.form", Predicate: "matches_by", Object: "internal user id",
	}); err != nil {
		t.Fatal(err)
	}

	from := RecallInput{ProjectPath: bravo, Scope: ScopeProject, Limit: 10}
	res, err := Recall(from, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Lines) != 0 {
		t.Fatalf("sibling recall must stay opt-in, got %v", res.Lines)
	}

	from.IncludeRepoSiblings = true
	res, err = Recall(from, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Lines) != 1 {
		t.Fatalf("lines=%v want the alpha memory recalled from bravo", res.Lines)
	}
}
