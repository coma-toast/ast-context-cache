package installer

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// unit is one target × component: it knows its files, computes status from disk, and plans the
// file changes for an install or uninstall.
type unit interface {
	path() string
	status(e *env) ComponentStatus
	plan(e *env) ([]FileChange, []string, error)
}

// targetSpec is a target's units keyed by component; every component has a unit, unsupported
// ones included, so the status table is always complete.
type targetSpec struct {
	id    Target
	name  string
	units map[Component]unit
}

// env carries one unit evaluation's context.
type env struct {
	s               *realService
	target          Target
	component       Component
	action          Action
	replaceExternal bool
	state           stateIndex
}

// unsupportedUnit is a component the host has no supported location for.
type unsupportedUnit struct {
	reason string
}

const windowsReason = "Windows is not supported yet; the installer supports macOS and Linux"

// spec returns the spec for t.
func (s *realService) spec(t Target) (targetSpec, bool) {
	for _, spec := range s.specs() {
		if spec.id == t {
			return spec, true
		}
	}
	return targetSpec{}, false
}

// specs builds every target's units for the configured home, OS, and MCP URL.
func (s *realService) specs() []targetSpec {
	specs := []targetSpec{
		s.claudeCodeSpec(),
		s.cursorSpec(),
		s.openCodeSpec(),
		s.codexSpec(),
		s.claudeDesktopSpec(),
		s.vsCodeSpec(),
		s.jetBrainsSpec(),
	}
	if s.cfg.GOOS != "darwin" && s.cfg.GOOS != "linux" {
		for i := range specs {
			for _, c := range allComponents {
				specs[i].units[c] = unsupportedUnit{reason: windowsReason}
			}
		}
	}
	return specs
}

// homePath joins elements under the home directory.
func (s *realService) homePath(elem ...string) string {
	return filepath.Join(append([]string{s.home}, elem...)...)
}

// displayPath abbreviates the home directory as ~ for messages.
func (s *realService) displayPath(p string) string {
	if rest, ok := strings.CutPrefix(p, s.home+string(filepath.Separator)); ok {
		return "~/" + filepath.ToSlash(rest)
	}
	return p
}

// external reports whether path exists as a symlink resolving outside ~/.astcache (IN-9), and
// returns its link target.
func (s *realService) external(path string) (string, bool) {
	fi, err := os.Lstat(path)
	if err != nil || fi.Mode()&os.ModeSymlink == 0 {
		return "", false
	}
	link, _ := os.Readlink(path)
	resolved, err := filepath.EvalSymlinks(path)
	if err == nil && (resolved == s.astcache || strings.HasPrefix(resolved, s.astcache+string(filepath.Separator))) {
		return "", false
	}
	return link, true
}

func (e *env) row(path string) *stateRow {
	return e.state.get(e.target, e.component, path)
}

func (e *env) cs(st Status, path, reason string) ComponentStatus {
	return ComponentStatus{Target: e.target, Component: e.component, Status: st, Path: path, Reason: reason}
}

// change builds a pending change with its preview hash.
func (e *env) change(path string, kind ChangeKind, before, after []byte) FileChange {
	return FileChange{Target: e.target, Component: e.component, Path: path, Kind: kind, Before: before, After: after, BeforeHash: contentHash(before)}
}

// skip builds a change that writes nothing.
func (e *env) skip(path, reason string) FileChange {
	return FileChange{Target: e.target, Component: e.component, Path: path, Kind: KindNone, Skipped: true, Reason: reason}
}

// stateRow builds the row recorded for path once the change is applied.
func (e *env) stateRow(path, hash string, created int) stateRow {
	return stateRow{Target: e.target, Component: e.component, Path: path, EntryHash: hash, Version: currentVersion(), Created: created}
}

func (e *env) stateKey(path string) stateKey {
	return stateKey{e.target, e.component, path}
}

// forget is a skipped change that drops path's state row, for an uninstall with nothing on disk.
func (e *env) forget(path, reason string) FileChange {
	c := e.skip(path, reason)
	if e.row(path) != nil {
		c.deletes = []stateKey{e.stateKey(path)}
	}
	return c
}

func (u unsupportedUnit) path() string {
	return ""
}

func (u unsupportedUnit) status(e *env) ComponentStatus {
	return e.cs(StatusUnsupported, "", u.reason)
}

func (u unsupportedUnit) plan(e *env) ([]FileChange, []string, error) {
	return []FileChange{e.skip("", "unsupported: "+u.reason)}, nil, nil
}

// unsupportedReason reports whether u can't be installed, and why.
func unsupportedReason(u unit, hooksEnabled bool) (string, bool) {
	switch v := u.(type) {
	case unsupportedUnit:
		return v.reason, true
	case hooksUnit:
		if !hooksEnabled {
			return hooksDisabledReason, true
		}
	}
	return "", false
}

// lookPath wraps exec.LookPath so tests can stub it through Config.
func lookPath(name string) (string, error) {
	return exec.LookPath(name)
}
