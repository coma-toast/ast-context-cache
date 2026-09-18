// Package selfupdate lets the dashboard pull the latest main, rebuild, and
// restart ast-mcp in place — the same "update" flow an operator would run by
// hand (git pull, make build, restart), just triggered from the UI.
package selfupdate

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

// Snapshot is a point-in-time view of an update for the dashboard.
type Snapshot struct {
	Active     bool
	Done       bool
	Phase      string
	Error      string
	StartedAt  time.Time
	FinishedAt time.Time
	FromCommit string
	ToCommit   string
}

// CheckResult reports whether origin/main has commits beyond the current
// checkout, without changing anything.
type CheckResult struct {
	Branch          string
	Clean           bool
	CurrentCommit   string
	LatestCommit    string
	CommitsBehind   int
	UpdateAvailable bool
	Error           string
}

var (
	mu       sync.Mutex
	snapshot Snapshot
)

// GetSnapshot returns the current update state for the dashboard.
func GetSnapshot() Snapshot {
	mu.Lock()
	defer mu.Unlock()
	return snapshot
}

func set(s Snapshot) {
	mu.Lock()
	snapshot = s
	mu.Unlock()
	realtime.Notify(realtime.Settings)
}

func fail(err error) {
	s := GetSnapshot()
	s.Active = false
	s.Done = false
	s.Phase = "error"
	s.Error = err.Error()
	s.FinishedAt = time.Now()
	set(s)
}

func git(repoDir string, args ...string) (string, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = repoDir
	out, err := cmd.CombinedOutput()
	return strings.TrimSpace(string(out)), err
}

// Check reports whether origin/main is ahead of the current checkout at
// repoDir. Safe to call anytime — it only fetches, never mutates the
// checkout — so the dashboard can poll it to render the "Update available"
// state before anyone clicks anything.
func Check(repoDir string) CheckResult {
	branch, err := git(repoDir, "rev-parse", "--abbrev-ref", "HEAD")
	if err != nil {
		return CheckResult{Error: fmt.Sprintf("not a git checkout: %v", err)}
	}
	if out, err := git(repoDir, "fetch", "origin", "main", "--quiet"); err != nil {
		return CheckResult{Branch: branch, Error: fmt.Sprintf("git fetch: %v (%s)", err, out)}
	}
	current, err := git(repoDir, "rev-parse", "HEAD")
	if err != nil {
		return CheckResult{Branch: branch, Error: err.Error()}
	}
	latest, err := git(repoDir, "rev-parse", "origin/main")
	if err != nil {
		return CheckResult{Branch: branch, Error: err.Error()}
	}
	statusOut, _ := git(repoDir, "status", "--porcelain")
	behindOut, _ := git(repoDir, "rev-list", "--count", current+"..origin/main")
	behind, _ := strconv.Atoi(behindOut)
	return CheckResult{
		Branch:          branch,
		Clean:           statusOut == "",
		CurrentCommit:   current,
		LatestCommit:    latest,
		CommitsBehind:   behind,
		UpdateAvailable: current != latest,
	}
}

// tryClaim atomically checks and sets Active, so two near-simultaneous Start
// calls can't both launch runUpdate.
func tryClaim() bool {
	mu.Lock()
	defer mu.Unlock()
	if snapshot.Active {
		return false
	}
	snapshot = Snapshot{Active: true}
	return true
}

// Start pulls origin/main, rebuilds, and restarts in the background. Refuses
// unless the checkout is on main with a clean working tree — this is ast-mcp's
// own source checkout, quite possibly mid-development, so an update must
// never silently discard local commits or uncommitted work.
func Start(repoDir string) (started bool, errMsg string) {
	if !tryClaim() {
		return false, "an update is already in progress"
	}
	check := Check(repoDir)
	if check.Error != "" {
		set(Snapshot{})
		return false, check.Error
	}
	if check.Branch != "main" {
		set(Snapshot{})
		return false, fmt.Sprintf("not on main (currently on %s) — switch to main to update", check.Branch)
	}
	if !check.Clean {
		set(Snapshot{})
		return false, "working tree has uncommitted changes — commit or stash before updating"
	}
	if !check.UpdateAvailable {
		set(Snapshot{})
		return false, "already up to date"
	}

	set(Snapshot{
		Active:     true,
		Phase:      "pulling",
		StartedAt:  time.Now(),
		FromCommit: check.CurrentCommit,
		ToCommit:   check.LatestCommit,
	})
	go runUpdate(repoDir)
	return true, ""
}

func runUpdate(repoDir string) {
	if out, err := git(repoDir, "pull", "--ff-only", "origin", "main"); err != nil {
		fail(fmt.Errorf("git pull: %w (%s)", err, out))
		return
	}

	s := GetSnapshot()
	s.Phase = "building"
	set(s)

	buildCmd := exec.Command("make", "build")
	buildCmd.Dir = repoDir
	if out, err := buildCmd.CombinedOutput(); err != nil {
		fail(fmt.Errorf("make build: %w (%s)", err, truncate(string(out), 4000)))
		return
	}

	s = GetSnapshot()
	s.Phase = "built"
	s.Done = true
	s.FinishedAt = time.Now()
	set(s)

	// Deliberately does not call db.RestartProcess() here — see its doc comment
	// for why restarting in place isn't triggered automatically yet. The dashboard
	// surfaces a "Restart now" button once Done is true instead.
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[len(s)-n:]
}
