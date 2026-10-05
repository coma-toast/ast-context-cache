// Package installer registers ast-context-cache with agent hosts (IN-1..IN-15): MCP server
// entries, skills, rule and instruction blocks, and Claude Code hooks. Every change is previewed
// as a per-file diff, merged into the user's files rather than overwriting them, backed up
// before writing, and re-checked against the preview before it is applied.
package installer

import (
	"crypto/rand"
	"encoding/hex"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/flags"
	"github.com/coma-toast/ast-context-cache/internal/version"
)

// Target is an agent host the installer can configure.
type Target string

// Supported targets (IN-1).
const (
	TargetClaudeCode    Target = "claude_code"
	TargetCursor        Target = "cursor"
	TargetOpenCode      Target = "opencode"
	TargetCodex         Target = "codex"
	TargetClaudeDesktop Target = "claude_desktop"
	TargetVSCode        Target = "vscode"
	TargetJetBrains     Target = "jetbrains"
)

// Component is one installable piece of a target (IN-2).
type Component string

// Components, in display and apply order.
const (
	ComponentMCP    Component = "mcp"
	ComponentSkills Component = "skills"
	ComponentRules  Component = "rules"
	ComponentHooks  Component = "hooks"
)

// Action is what a plan does.
type Action string

// Plan actions.
const (
	ActionInstall   Action = "install"
	ActionUninstall Action = "uninstall"
)

// DefaultMCPPort is the MCP server's default port.
const DefaultMCPPort = 7821

const (
	serverName = "ast-context-cache"
	planTTL    = 10 * time.Minute
)

var (
	allTargets    = []Target{TargetClaudeCode, TargetCursor, TargetOpenCode, TargetCodex, TargetClaudeDesktop, TargetVSCode, TargetJetBrains}
	allComponents = []Component{ComponentMCP, ComponentSkills, ComponentRules, ComponentHooks}
)

// Service is the installer API used by the CLI and the dashboard.
type Service interface {
	// Targets lists every target with its components and where each would be written.
	Targets() []TargetInfo
	// Plan previews an install or uninstall. Nothing is written until Apply.
	Plan(req PlanRequest) (*Plan, error)
	// Apply writes a previewed plan. It fails with errs.CodeConflict, writing nothing, when any
	// file changed since the preview, and with errs.CodeExpired after the plan's TTL.
	Apply(planID string) (*ApplyResult, error)
	// Verify computes status from disk for the given targets (all when empty).
	Verify(targets []Target) ([]ComponentStatus, error)
	// Backups lists saved backups, newest first.
	Backups() ([]Backup, error)
	// Restore writes a backup back to its original path, backing up the current file first.
	Restore(backupID string) error
	// LegacyWarnings returns the findings of the one-time pre-4.0 install-record check (IN-13).
	LegacyWarnings() []string
}

// Config holds the installer's environment. Zero fields take defaults from the process.
type Config struct {
	// Home is the user's home directory; defaults to $HOME.
	Home string
	// MCPURL is the URL every registration points at; defaults to MCPURL(DefaultMCPPort).
	MCPURL string
	// Executable is the absolute ast-mcp path written into hook commands; defaults to os.Executable.
	Executable string
	// GOOS selects per-OS config paths; defaults to runtime.GOOS.
	GOOS string
	// LookPath finds bridge commands for Claude Desktop; defaults to exec.LookPath.
	LookPath func(string) (string, error)
	// HooksEnabled gates the Claude Code hooks component; defaults to the feature_handoff_hooks flag.
	HooksEnabled func() bool
	// Now is the clock for backups and plan expiry.
	Now func() time.Time
}

// TargetInfo describes a target for the dashboard and CLI.
type TargetInfo struct {
	ID         Target          `json:"id"`
	Name       string          `json:"name"`
	Components []ComponentInfo `json:"components"`
}

// ComponentInfo says whether a target supports a component and where it is written.
type ComponentInfo struct {
	Component Component `json:"component"`
	Supported bool      `json:"supported"`
	Path      string    `json:"path,omitempty"`
	Reason    string    `json:"reason,omitempty"`
}

// PlanRequest selects what to preview. Empty Components means every supported component.
type PlanRequest struct {
	Targets         []Target    `json:"targets"`
	Components      []Component `json:"components"`
	Action          Action      `json:"action"`
	ReplaceExternal bool        `json:"replace_external"`
}

// Plan is a previewed set of file changes, held in memory for planTTL under ID.
type Plan struct {
	ID        string            `json:"plan_id"`
	Action    Action            `json:"action"`
	Changes   []FileChange      `json:"changes"`
	Status    []ComponentStatus `json:"status"`
	Warnings  []string          `json:"warnings"`
	Errors    []PlanError       `json:"errors,omitempty"`
	ExpiresAt time.Time         `json:"expires_at"`

	targets []Target
}

// PlanError is a component whose file could not be edited safely (for example a parse error).
// Every change for that target is skipped (IN-3).
type PlanError struct {
	Target    Target    `json:"target"`
	Component Component `json:"component"`
	Code      errs.Code `json:"code"`
	Message   string    `json:"message"`
}

// ApplyResult reports what Apply wrote.
type ApplyResult struct {
	PlanID   string            `json:"plan_id"`
	Written  []string          `json:"written"`
	Backups  []Backup          `json:"backups"`
	Status   []ComponentStatus `json:"status"`
	Warnings []string          `json:"warnings"`
}

type realService struct {
	cfg        Config
	home       string
	astcache   string
	backupRoot string

	mu    sync.Mutex // guards plans
	plans map[string]*Plan
}

// New returns the installer service. With the usage database open it also runs the one-time
// legacy install-record check (IN-13).
func New(cfg Config) (Service, error) {
	if cfg.Home == "" {
		cfg.Home = os.Getenv("HOME")
	}
	if cfg.Home == "" {
		return nil, errs.NewCode(errs.CodeInvalidInput, "home directory unknown")
	}
	if cfg.MCPURL == "" {
		cfg.MCPURL = MCPURL(DefaultMCPPort)
	}
	if cfg.GOOS == "" {
		cfg.GOOS = runtime.GOOS
	}
	if cfg.Executable == "" {
		cfg.Executable = defaultExecutable()
	}
	if cfg.LookPath == nil {
		cfg.LookPath = lookPath
	}
	if cfg.HooksEnabled == nil {
		cfg.HooksEnabled = func() bool { return flags.Enabled(flags.KeyHandoffHooks) }
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	s := &realService{
		cfg:        cfg,
		home:       cfg.Home,
		astcache:   filepath.Join(cfg.Home, ".astcache"),
		backupRoot: filepath.Join(cfg.Home, ".astcache", "backups"),
		plans:      map[string]*Plan{},
	}
	s.migrateLegacy()
	return s, nil
}

// MCPURL is the loopback MCP endpoint for port (IN-7).
func MCPURL(port int) string {
	return "http://127.0.0.1:" + strconv.Itoa(port) + "/mcp"
}

// AllTargets lists every target in display order.
func AllTargets() []Target {
	return slices.Clone(allTargets)
}

// ParseTarget validates a target name.
func ParseTarget(s string) (Target, error) {
	t := Target(strings.TrimSpace(s))
	if !slices.Contains(allTargets, t) {
		return "", errs.NewCode(errs.CodeInvalidInput, "unknown target", "target", s)
	}
	return t, nil
}

// ParseComponent validates a component name.
func ParseComponent(s string) (Component, error) {
	c := Component(strings.TrimSpace(s))
	if !slices.Contains(allComponents, c) {
		return "", errs.NewCode(errs.CodeInvalidInput, "unknown component", "component", s)
	}
	return c, nil
}

// HasErrors reports whether any target was aborted.
func (p *Plan) HasErrors() bool {
	return len(p.Errors) > 0
}

// Targets lists every target with its components.
func (s *realService) Targets() []TargetInfo {
	var out []TargetInfo
	for _, spec := range s.specs() {
		info := TargetInfo{ID: spec.id, Name: spec.name}
		for _, c := range allComponents {
			u := spec.units[c]
			ci := ComponentInfo{Component: c, Path: u.path()}
			if r, ok := unsupportedReason(u, s.cfg.HooksEnabled()); ok {
				ci.Reason = r
			} else {
				ci.Supported = true
			}
			info.Components = append(info.Components, ci)
		}
		out = append(out, info)
	}
	return out
}

// Plan previews req and stores the result for Apply.
func (s *realService) Plan(req PlanRequest) (*Plan, error) {
	if req.Action == "" {
		req.Action = ActionInstall
	}
	if req.Action != ActionInstall && req.Action != ActionUninstall {
		return nil, errs.NewCode(errs.CodeInvalidInput, "unknown action", "action", req.Action)
	}
	if len(req.Targets) == 0 {
		return nil, errs.NewCode(errs.CodeInvalidInput, "no targets selected")
	}
	for _, c := range req.Components {
		if _, err := ParseComponent(string(c)); err != nil {
			return nil, err
		}
	}
	st, err := loadState()
	if err != nil {
		return nil, err
	}
	p := &Plan{Action: req.Action, Changes: []FileChange{}, Status: []ComponentStatus{}, Warnings: []string{}}
	for _, t := range dedupe(req.Targets) {
		spec, ok := s.spec(t)
		if !ok {
			return nil, errs.NewCode(errs.CodeInvalidInput, "unknown target", "target", t)
		}
		p.targets = append(p.targets, t)
		s.planTarget(p, spec, req, st)
	}
	dedupeChanges(p.Changes)
	for i := range p.Changes {
		c := &p.Changes[i]
		if !c.Skipped && c.Diff == "" {
			c.Diff = unifiedDiff(c.Path, c.Before, c.After)
		}
	}
	id, err := newPlanID()
	if err != nil {
		return nil, err
	}
	p.ID, p.ExpiresAt = id, s.cfg.Now().Add(planTTL)
	s.mu.Lock()
	s.prunePlansLocked()
	s.plans[id] = p
	s.mu.Unlock()
	return p, nil
}

// Apply re-checks every file against the preview, then backs up and writes them in order.
func (s *realService) Apply(planID string) (*ApplyResult, error) {
	s.mu.Lock()
	p, ok := s.plans[planID]
	delete(s.plans, planID)
	s.mu.Unlock()
	if !ok {
		return nil, errs.NewCode(errs.CodeNotFound, "plan not found; preview again", "plan_id", planID)
	}
	if s.cfg.Now().After(p.ExpiresAt) {
		return nil, errs.NewCode(errs.CodeExpired, "plan expired; preview again", "plan_id", planID)
	}
	if db.DB == nil {
		return nil, errs.NewCode(errs.CodeInternal, "usage database not open")
	}
	// Verify every file before writing any, so a conflict leaves everything untouched.
	for _, c := range p.Changes {
		if c.Skipped || c.underLink() {
			continue
		}
		h, err := currentHash(c.Path, c.linkTarget != "")
		if err != nil {
			return nil, err
		}
		if h != c.BeforeHash {
			return nil, errs.NewCode(errs.CodeConflict, "file changed since preview", "path", c.Path)
		}
	}
	res := &ApplyResult{PlanID: planID, Written: []string{}, Backups: []Backup{}, Warnings: p.Warnings}
	for _, c := range p.Changes {
		if err := s.applyChange(c, res); err != nil {
			return res, err
		}
	}
	status, err := s.Verify(p.targets)
	res.Status = status
	logger.Info("Installer plan applied", "plan_id", planID, "action", p.Action, "written", len(res.Written), "backups", len(res.Backups))
	return res, err
}

// Verify computes every component's status from disk plus installer_state.
func (s *realService) Verify(targets []Target) ([]ComponentStatus, error) {
	if len(targets) == 0 {
		targets = allTargets
	}
	st, err := loadState()
	if err != nil {
		return nil, err
	}
	out := []ComponentStatus{}
	for _, t := range dedupe(targets) {
		spec, ok := s.spec(t)
		if !ok {
			return nil, errs.NewCode(errs.CodeInvalidInput, "unknown target", "target", t)
		}
		for _, c := range allComponents {
			e := &env{s: s, target: t, component: c, action: ActionInstall, state: st}
			out = append(out, spec.units[c].status(e))
		}
	}
	return out, nil
}

// Backups lists saved backups, newest first.
func (s *realService) Backups() ([]Backup, error) {
	out, err := listBackups(s.backupRoot)
	if out == nil {
		out = []Backup{}
	}
	return out, err
}

// Restore writes a backup back to its original path. The current file is backed up first, so a
// restore can itself be undone.
func (s *realService) Restore(backupID string) error {
	parts := strings.Split(backupID, "/")
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" || parts[0] == ".." || parts[1] == ".." || strings.ContainsAny(parts[1], `\`) {
		return errs.NewCode(errs.CodeInvalidInput, "invalid backup id", "backup_id", backupID)
	}
	if _, ok := parseBackupDir(parts[0]); !ok {
		return errs.NewCode(errs.CodeInvalidInput, "invalid backup id", "backup_id", backupID)
	}
	src := filepath.Join(s.backupRoot, parts[0], parts[1])
	data, err := os.ReadFile(src)
	if os.IsNotExist(err) {
		return errs.NewCode(errs.CodeNotFound, "backup not found", "backup_id", backupID)
	}
	if err != nil {
		return errs.WrapMessage("failed to read backup", err, "backup_id", backupID)
	}
	path, link := decodeBackupName(parts[1])
	if !filepath.IsAbs(path) || !strings.HasPrefix(path, s.home+string(filepath.Separator)) {
		return errs.NewCode(errs.CodeInvalidInput, "backup path is outside the home directory", "path", path)
	}
	fi, statErr := os.Lstat(path)
	switch {
	case statErr != nil:
		// Nothing there to back up.
	case fi.IsDir():
		// One of our skill directories: keep its SKILL.md before the symlink replaces it.
		if _, err := backupFile(s.backupRoot, filepath.Join(path, "SKILL.md"), false, s.cfg.Now()); err != nil {
			return err
		}
	default:
		// A content restore writes through a symlinked path, so back up the content it replaces.
		asLink := link && fi.Mode()&os.ModeSymlink != 0
		if _, err := backupFile(s.backupRoot, path, asLink, s.cfg.Now()); err != nil {
			return err
		}
	}
	if link {
		return restoreSymlink(path, string(data))
	}
	if err := atomicWrite(path, data, defaultCreatedMode); err != nil {
		return err
	}
	logger.Info("Installer backup restored", "backup_id", backupID, "path", path)
	return nil
}

// planTarget appends one target's statuses, changes, warnings, and errors to p.
func (s *realService) planTarget(p *Plan, spec targetSpec, req PlanRequest, st stateIndex) {
	comps := req.Components
	explicit := len(comps) > 0
	if !explicit {
		comps = allComponents
	}
	var changes []FileChange
	var failed *PlanError
	for _, c := range dedupe(comps) {
		u := spec.units[c]
		e := &env{s: s, target: spec.id, component: c, action: req.Action, replaceExternal: req.ReplaceExternal, state: st}
		cs := u.status(e)
		if _, unsupported := unsupportedReason(u, s.cfg.HooksEnabled()); unsupported && !explicit && cs.Status == StatusUnsupported {
			continue
		}
		p.Status = append(p.Status, cs)
		ch, warn, err := u.plan(e)
		if err != nil && failed == nil {
			failed = &PlanError{Target: spec.id, Component: c, Code: errs.CodeOf(err), Message: err.Error()}
			if failed.Code == "" {
				failed.Code = errs.CodeInternal
			}
		}
		changes = append(changes, ch...)
		p.Warnings = append(p.Warnings, warn...)
	}
	if failed != nil {
		// Abort the whole target: nothing for it is written (IN-3).
		p.Errors = append(p.Errors, *failed)
		for i := range changes {
			if !changes[i].Skipped {
				changes[i].Skipped, changes[i].Reason = true, "aborted: "+failed.Message
			}
			changes[i].upserts, changes[i].deletes = nil, nil
		}
		changes = append(changes, FileChange{Target: spec.id, Component: failed.Component, Kind: KindNone, Skipped: true, Reason: failed.Message})
	}
	p.Changes = append(p.Changes, changes...)
}

// applyChange backs up and writes one change, then records its state.
func (s *realService) applyChange(c FileChange, res *ApplyResult) error {
	if !c.Skipped {
		asLink := c.linkTarget != ""
		b, err := backupFile(s.backupRoot, c.Path, asLink, s.cfg.Now())
		if err != nil {
			return err
		}
		if b != nil {
			res.Backups = append(res.Backups, *b)
		}
		if err := writeChange(c); err != nil {
			return err
		}
		res.Written = append(res.Written, c.Path)
	}
	return applyState(c.upserts, c.deletes)
}

func (s *realService) prunePlansLocked() {
	now := s.cfg.Now()
	for id, p := range s.plans {
		if now.After(p.ExpiresAt) {
			delete(s.plans, id)
		}
	}
}

// writeChange performs one change on disk.
func writeChange(c FileChange) error {
	switch c.Kind {
	case KindCreate, KindModify, KindRemoveBlock:
		return atomicWrite(c.Path, c.After, defaultCreatedMode)
	case KindDelete:
		if err := os.Remove(c.Path); err != nil && !os.IsNotExist(err) {
			return errs.WrapMessage("failed to delete file", err, "path", c.Path)
		}
		// Our skill directories hold only SKILL.md; Remove fails harmlessly if the user added files.
		if dir := filepath.Dir(c.Path); strings.HasPrefix(filepath.Base(dir), serverName+"-") {
			os.Remove(dir)
		}
		return nil
	default:
		return nil
	}
}

// restoreSymlink puts a replaced external symlink back. Our own skill directory at that path is
// removed first; a directory holding anything besides SKILL.md makes the restore fail untouched.
func restoreSymlink(path, target string) error {
	if fi, err := os.Lstat(path); err == nil {
		if err := clearForSymlink(path, fi); err != nil {
			return err
		}
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return errs.WrapMessage("failed to create directory", err, "path", path)
	}
	if err := os.Symlink(target, path); err != nil {
		return errs.WrapMessage("failed to restore symlink", err, "path", path)
	}
	logger.Info("Installer symlink restored", "path", path, "target", target)
	return nil
}

func clearForSymlink(path string, fi os.FileInfo) error {
	if fi.IsDir() {
		entries, err := os.ReadDir(path)
		if err != nil {
			return errs.WrapMessage("failed to read directory", err, "path", path)
		}
		for _, e := range entries {
			if e.Name() != "SKILL.md" {
				return errs.NewCode(errs.CodeConflict, "directory holds other files; move them before restoring the symlink", "path", path, "file", e.Name())
			}
		}
		if err := os.RemoveAll(path); err != nil {
			return errs.WrapMessage("failed to clear path for symlink restore", err, "path", path)
		}
		return nil
	}
	if err := os.Remove(path); err != nil {
		return errs.WrapMessage("failed to clear path for symlink restore", err, "path", path)
	}
	return nil
}

// dedupeChanges skips a second write to a path another target already writes (shared skills in
// ~/.agents/skills), moving its state rows onto the first so both targets record ownership.
func dedupeChanges(changes []FileChange) {
	first := map[string]int{}
	for i := range changes {
		c := &changes[i]
		if c.Skipped || c.Path == "" {
			continue
		}
		j, seen := first[c.Path]
		if !seen {
			first[c.Path] = i
			continue
		}
		changes[j].upserts = append(changes[j].upserts, c.upserts...)
		changes[j].deletes = append(changes[j].deletes, c.deletes...)
		c.upserts, c.deletes = nil, nil
		c.Skipped, c.Reason = true, "same file as "+string(changes[j].Target)
	}
}

func dedupe[T comparable](in []T) []T {
	var out []T
	for _, v := range in {
		if !slices.Contains(out, v) {
			out = append(out, v)
		}
	}
	return out
}

func newPlanID() (string, error) {
	b := make([]byte, 12)
	if _, err := rand.Read(b); err != nil {
		return "", errs.WrapMessage("failed to generate plan id", err)
	}
	return "plan_" + hex.EncodeToString(b), nil
}

// defaultExecutable is this binary's absolute, symlink-resolved path.
func defaultExecutable() string {
	exe, err := os.Executable()
	if err != nil {
		return "ast-mcp"
	}
	if real, err := filepath.EvalSymlinks(exe); err == nil {
		return real
	}
	return exe
}

// currentVersion is the version stamped into blocks and state.
func currentVersion() string {
	return version.Version
}
