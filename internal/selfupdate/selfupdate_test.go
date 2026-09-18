package selfupdate

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// waitForIdle blocks until the background update goroutine (if any) finishes,
// so a test doesn't return while runUpdate is still writing to the shared
// package-level snapshot in the background — which would race with the next
// test's own set(Snapshot{}) reset.
func waitForIdle(t *testing.T) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if !GetSnapshot().Active {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("update did not finish within timeout")
}

func runGit(t *testing.T, dir string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(),
		"GIT_AUTHOR_NAME=test", "GIT_AUTHOR_EMAIL=test@example.com",
		"GIT_COMMITTER_NAME=test", "GIT_COMMITTER_EMAIL=test@example.com",
	)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v\n%s", args, err, out)
	}
	return strings.TrimSpace(string(out))
}

// setupClone creates a bare "origin" repo with one commit on main and a
// clone of it (also on main, clean), returning the clone's directory. Tests
// mutate origin and/or the clone to exercise Check/Start against a real git
// checkout rather than mocking git out entirely.
func setupClone(t *testing.T) (cloneDir string) {
	t.Helper()
	tmp := t.TempDir()
	originDir := filepath.Join(tmp, "origin.git")
	runGit(t, tmp, "init", "--bare", "--initial-branch=main", originDir)

	seed := filepath.Join(tmp, "seed")
	if err := os.Mkdir(seed, 0o755); err != nil {
		t.Fatal(err)
	}
	runGit(t, seed, "init", "--initial-branch=main", "-q")
	if err := os.WriteFile(filepath.Join(seed, "f.txt"), []byte("v1"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGit(t, seed, "add", "f.txt")
	runGit(t, seed, "commit", "-q", "-m", "initial")
	runGit(t, seed, "remote", "add", "origin", originDir)
	runGit(t, seed, "push", "-q", "origin", "main")

	cloneDir = filepath.Join(tmp, "clone")
	runGit(t, tmp, "clone", "-q", originDir, cloneDir)
	runGit(t, cloneDir, "checkout", "-q", "main")
	return cloneDir
}

func advanceOrigin(t *testing.T, cloneDir string) {
	t.Helper()
	// Push a second commit from a fresh clone of the same origin so the
	// "clone" fixture stays untouched — Check/Start must fetch this via
	// origin/main, not see it already sitting in the local checkout.
	remote := runGit(t, cloneDir, "remote", "get-url", "origin")
	tmp := t.TempDir()
	other := filepath.Join(tmp, "other")
	runGit(t, tmp, "clone", "-q", remote, other)
	if err := os.WriteFile(filepath.Join(other, "f.txt"), []byte("v2"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGit(t, other, "commit", "-q", "-am", "second")
	runGit(t, other, "push", "-q", "origin", "main")
}

func TestCheckReportsNoUpdateWhenUpToDate(t *testing.T) {
	dir := setupClone(t)
	result := Check(dir)
	if result.Error != "" {
		t.Fatalf("unexpected error: %s", result.Error)
	}
	if result.UpdateAvailable {
		t.Fatalf("expected no update available, got %+v", result)
	}
	if !result.Clean {
		t.Fatalf("expected clean working tree, got %+v", result)
	}
	if result.Branch != "main" {
		t.Fatalf("expected branch main, got %q", result.Branch)
	}
}

func TestCheckReportsUpdateAvailable(t *testing.T) {
	dir := setupClone(t)
	advanceOrigin(t, dir)

	result := Check(dir)
	if result.Error != "" {
		t.Fatalf("unexpected error: %s", result.Error)
	}
	if !result.UpdateAvailable {
		t.Fatalf("expected update available, got %+v", result)
	}
	if result.CommitsBehind != 1 {
		t.Fatalf("expected 1 commit behind, got %d", result.CommitsBehind)
	}
}

func TestStartRefusesWhenNotOnMain(t *testing.T) {
	dir := setupClone(t)
	advanceOrigin(t, dir)
	runGit(t, dir, "checkout", "-q", "-b", "feature")

	set(Snapshot{})
	started, errMsg := Start(dir)
	if started {
		t.Fatalf("expected Start to refuse off main")
	}
	if !strings.Contains(errMsg, "not on main") {
		t.Fatalf("expected 'not on main' error, got %q", errMsg)
	}
}

func TestStartRefusesWhenDirty(t *testing.T) {
	dir := setupClone(t)
	advanceOrigin(t, dir)
	if err := os.WriteFile(filepath.Join(dir, "f.txt"), []byte("local edit"), 0o644); err != nil {
		t.Fatal(err)
	}

	set(Snapshot{})
	started, errMsg := Start(dir)
	if started {
		t.Fatalf("expected Start to refuse a dirty working tree")
	}
	if !strings.Contains(errMsg, "uncommitted changes") {
		t.Fatalf("expected 'uncommitted changes' error, got %q", errMsg)
	}
}

func TestStartRefusesWhenAlreadyUpToDate(t *testing.T) {
	dir := setupClone(t)

	set(Snapshot{})
	started, errMsg := Start(dir)
	if started {
		t.Fatalf("expected Start to refuse when already up to date")
	}
	if !strings.Contains(errMsg, "up to date") {
		t.Fatalf("expected 'up to date' error, got %q", errMsg)
	}
}

// Start used the same check-then-act shape StartDataDirMove once had (see
// TestStartDataDirMoveIsExclusiveUnderConcurrency in internal/db): claiming
// Active must happen atomically, or many concurrent calls could all pass the
// check and all launch runUpdate against the same checkout.
func TestStartIsExclusiveUnderConcurrency(t *testing.T) {
	dir := setupClone(t)
	advanceOrigin(t, dir)

	set(Snapshot{})
	const n = 20
	var wg sync.WaitGroup
	var startedCount int32
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			started, _ := Start(dir)
			if started {
				atomic.AddInt32(&startedCount, 1)
			}
		}()
	}
	wg.Wait()
	if startedCount != 1 {
		t.Fatalf("started count=%d want exactly 1", startedCount)
	}
	// The one winner's runUpdate is now running in the background (it will fail
	// at the "make build" step, since this fixture has no Makefile — that's
	// fine, just wait for it so it doesn't outlive the test).
	waitForIdle(t)
}
