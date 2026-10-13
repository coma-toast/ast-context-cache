// Package flags is the feature-flag registry. Each flag resolves env > setting > default (a
// non-empty, parseable env value locks it), and feature_handoff is a master switch that forces
// every feature_handoff_* child off. Resolved values are cached in a snapshot so hot paths such
// as tools/list never read the settings table; Set and Reload rebuild it and notify subscribers.
package flags

import (
	"maps"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// Sources a flag's value can come from, reported in FlagState.Source.
const (
	SourceEnv     = "env"
	SourceSetting = "setting"
	SourceDefault = "default"
)

// Flag is one registry entry.
type Flag struct {
	Key, Env, Description string
	Default               bool
	// Tools are hidden from tools/list and rejected when the flag is off.
	Tools []string
	// Actions maps a tool to the actions it rejects when the flag is off.
	Actions map[string][]string
}

// FlagState is a flag's resolved state for the settings API and dashboard. Enabled is the
// effective value, so a child reads false while feature_handoff is off; Source is where the
// flag's own value came from. AffectsTools marks flags whose toggle changes tools/list or the
// actions a tool accepts.
type FlagState struct {
	Key          string `json:"key"`
	Description  string `json:"description"`
	Source       string `json:"source"`
	Env          string `json:"env"`
	Enabled      bool   `json:"enabled"`
	Default      bool   `json:"default"`
	Locked       bool   `json:"locked"`
	AffectsTools bool   `json:"affects_tools"`
}

type snapshot struct {
	states    []FlagState
	effective map[string]bool
}

type change struct {
	key string
	on  bool
}

var (
	mu          sync.Mutex // serializes rebuilds and guards subscribers
	current     atomic.Pointer[snapshot]
	subscribers []func(key string, on bool)
)

// Enabled reports the effective value of key. Unknown keys are off.
func Enabled(key string) bool {
	return load().effective[key]
}

// Set persists key's value to the settings table and applies it immediately. It fails with
// CodeNotFound for an unknown key and CodeConflict when the env var locks the flag.
// Subscribers are called, after locks are released, for every flag whose effective value
// changed, including children implied by the master switch.
func Set(key string, on bool) error {
	f, ok := lookup(key)
	if !ok {
		return errs.NewCode(errs.CodeNotFound, "unknown feature flag", "key", key)
	}
	if _, locked := envValue(f); locked {
		return errs.NewCode(errs.CodeConflict, "feature flag locked by environment", "key", key, "env", f.Env)
	}
	if db.DB == nil {
		return errs.NewCode(errs.CodeInternal, "settings database not open", "key", key)
	}
	mu.Lock()
	old := loadLocked()
	if err := db.SetSetting(key, strconv.FormatBool(on)); err != nil {
		mu.Unlock()
		return errs.WrapMessage("failed to save feature flag", err, "key", key)
	}
	next := build()
	storeLocked(next)
	subs := slices.Clone(subscribers)
	mu.Unlock()
	logger.Info("Feature flag changed", "key", key, "enabled", on, "effective", next.effective[key], "source", SourceSetting)
	notify(diff(old, next), subs)
	return nil
}

// Reload rebuilds the snapshot from env and the settings table, notifying subscribers of any
// effective change. Call it after db.Init at startup: a snapshot is only cached while the
// database is open, so lookups before then see env and defaults.
func Reload() {
	mu.Lock()
	old := current.Load()
	next := build()
	storeLocked(next)
	subs := slices.Clone(subscribers)
	mu.Unlock()
	if old != nil {
		notify(diff(old, next), subs)
	}
}

// State lists every flag's resolved state in registry order.
func State() []FlagState {
	return slices.Clone(load().states)
}

// OnChange registers fn to be called with each flag whose effective value changes.
func OnChange(fn func(key string, on bool)) {
	mu.Lock()
	defer mu.Unlock()
	subscribers = append(subscribers, fn)
}

// AffectsTools reports whether toggling key can change tools/list or which actions a tool
// accepts, counting the children the master switch implies.
func AffectsTools(key string) bool {
	for _, f := range registry {
		if f.Key != key && parentOf(f.Key) != key {
			continue
		}
		if len(f.Tools) > 0 || len(f.Actions) > 0 {
			return true
		}
	}
	return false
}

// ToolEnabled reports whether every flag gating tool is on. Tools no flag lists are enabled.
func ToolEnabled(tool string) bool {
	return ToolDisabledBy(tool) == ""
}

// ToolDisabledBy returns the first flag (in registry order) that is off and lists tool, or ""
// when the tool is enabled.
func ToolDisabledBy(tool string) string {
	s := load()
	for _, f := range registry {
		if !s.effective[f.Key] && slices.Contains(f.Tools, tool) {
			return f.Key
		}
	}
	return ""
}

// ActionEnabled reports whether tool is enabled and no flag that is off disables action on it.
func ActionEnabled(tool, action string) bool {
	if !ToolEnabled(tool) {
		return false
	}
	s := load()
	for _, f := range registry {
		if !s.effective[f.Key] && slices.Contains(f.Actions[tool], action) {
			return false
		}
	}
	return true
}

// All returns a copy of the registry in order.
func All() []Flag {
	out := make([]Flag, 0, len(registry))
	for _, f := range registry {
		f.Tools = slices.Clone(f.Tools)
		if f.Actions != nil {
			actions := maps.Clone(f.Actions)
			for tool, a := range actions {
				actions[tool] = slices.Clone(a)
			}
			f.Actions = actions
		}
		out = append(out, f)
	}
	return out
}

func lookup(key string) (Flag, bool) {
	for _, f := range registry {
		if f.Key == key {
			return f, true
		}
	}
	return Flag{}, false
}

// parentOf returns the master switch that implies key, or "" for a top-level flag.
func parentOf(key string) string {
	if strings.HasPrefix(key, handoffChildKeyPrefix) {
		return KeyHandoff
	}
	return ""
}

func load() *snapshot {
	if s := current.Load(); s != nil {
		return s
	}
	mu.Lock()
	defer mu.Unlock()
	return loadLocked()
}

func loadLocked() *snapshot {
	if s := current.Load(); s != nil {
		return s
	}
	s := build()
	storeLocked(s)
	return s
}

// storeLocked caches s only while the database is open, so a lookup made before db.Init
// cannot pin defaults over the stored settings.
func storeLocked(s *snapshot) {
	if db.DB == nil {
		current.Store(nil)
		return
	}
	current.Store(s)
}

func build() *snapshot {
	s := &snapshot{states: make([]FlagState, 0, len(registry)), effective: make(map[string]bool, len(registry))}
	for _, f := range registry {
		st := resolve(f)
		s.states = append(s.states, st)
		s.effective[f.Key] = st.Enabled
	}
	for i := range s.states {
		st := &s.states[i]
		if p := parentOf(st.Key); p != "" && !s.effective[p] {
			st.Enabled = false
			s.effective[st.Key] = false
		}
	}
	return s
}

func resolve(f Flag) FlagState {
	st := FlagState{Key: f.Key, Description: f.Description, Env: f.Env, Default: f.Default, Enabled: f.Default, Source: SourceDefault, AffectsTools: AffectsTools(f.Key)}
	if on, ok := envValue(f); ok {
		st.Enabled, st.Source, st.Locked = on, SourceEnv, true
		return st
	}
	v := db.GetSetting(f.Key, "")
	if v == "" {
		return st
	}
	on, err := strconv.ParseBool(v)
	if err != nil {
		logger.Warn("Ignoring invalid feature flag setting", "key", f.Key, "value", v, "error", err)
		return st
	}
	st.Enabled, st.Source = on, SourceSetting
	return st
}

// envValue returns the env override for f. An unparseable value is ignored (and does not lock
// the flag) rather than silently forcing it off.
func envValue(f Flag) (on, ok bool) {
	v := strings.TrimSpace(os.Getenv(f.Env))
	if v == "" {
		return false, false
	}
	on, err := strconv.ParseBool(v)
	if err != nil {
		logger.Warn("Ignoring invalid feature flag environment value", "key", f.Key, "env", f.Env, "value", v, "error", err)
		return false, false
	}
	return on, true
}

func diff(old, next *snapshot) []change {
	var out []change
	for _, f := range registry {
		if old.effective[f.Key] != next.effective[f.Key] {
			out = append(out, change{key: f.Key, on: next.effective[f.Key]})
		}
	}
	return out
}

func notify(changes []change, subs []func(key string, on bool)) {
	for _, c := range changes {
		for _, fn := range subs {
			fn(c.key, c.on)
		}
	}
}
