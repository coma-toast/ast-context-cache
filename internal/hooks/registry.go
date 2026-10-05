package hooks

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

const (
	// registryTTL drops pending handoffs and agent mappings no hook consumed: a denied Agent
	// call leaves its pending entry behind, and SubagentStop never fires for a killed session.
	registryTTL = time.Hour
	// staleLockAge breaks a lock file left by a hook that was killed while holding it. Hooks
	// hold the lock for a file read and write, far less than this.
	staleLockAge     = 5 * time.Second
	lockRetry        = 5 * time.Millisecond
	sessionKeyHexLen = 32
)

// Registry is the hooks' local state, one JSON file per parent session: handoffs a
// pre-tool-use-agent hook created that no subagent-start hook has opened yet, and the child
// session each subagent-start hook opened, keyed by agent id. Hooks for one session can run in
// parallel (several subagents start at once), so every update holds a per-file lock.
type Registry struct {
	dir string
	now func() time.Time
}

// PendingHandoff is a handoff created for an Agent call whose subagent hasn't started yet.
type PendingHandoff struct {
	Ref       string    `json:"ref"`
	ToolUseID string    `json:"tool_use_id,omitempty"`
	CreatedAt time.Time `json:"created_at"`
}

// AgentHandoff is the handoff child a subagent-start hook opened for one subagent.
type AgentHandoff struct {
	Ref       string    `json:"ref"`
	SessionID string    `json:"session_id"`
	CreatedAt time.Time `json:"created_at"`
}

// sessionState is one registry file.
type sessionState struct {
	Pending []PendingHandoff        `json:"pending,omitempty"`
	Agents  map[string]AgentHandoff `json:"agents,omitempty"`
}

// NewRegistry returns a registry stored in dir.
func NewRegistry(dir string) *Registry {
	return &Registry{dir: dir, now: time.Now}
}

// DefaultRegistryDir is ~/.astcache/hooks, or the user cache dir when there is no home.
func DefaultRegistryDir() string {
	if home := os.Getenv("HOME"); home != "" {
		return filepath.Join(home, ".astcache", "hooks")
	}
	if d, err := os.UserCacheDir(); err == nil {
		return filepath.Join(d, "ast-context-cache", "hooks")
	}
	return filepath.Join(os.TempDir(), "ast-context-cache-hooks")
}

// AddPending appends p to the session's pending queue.
func (r *Registry) AddPending(ctx context.Context, sessionID string, p PendingHandoff) error {
	if p.CreatedAt.IsZero() {
		p.CreatedAt = r.now()
	}
	return r.update(ctx, sessionID, func(st *sessionState) {
		st.Pending = append(st.Pending, p)
	})
}

// TakePending removes and returns the oldest pending handoff. Parallel spawns start their
// subagents in spawn order, so FIFO matches each subagent to its own handoff.
func (r *Registry) TakePending(ctx context.Context, sessionID string) (PendingHandoff, bool, error) {
	var out PendingHandoff
	var ok bool
	err := r.update(ctx, sessionID, func(st *sessionState) {
		if len(st.Pending) == 0 {
			return
		}
		out, ok = st.Pending[0], true
		st.Pending = slices.Delete(st.Pending, 0, 1)
	})
	return out, ok, err
}

// DropPending removes the pending handoff ref, if queued.
func (r *Registry) DropPending(ctx context.Context, sessionID, ref string) error {
	return r.update(ctx, sessionID, func(st *sessionState) {
		st.Pending = slices.DeleteFunc(st.Pending, func(p PendingHandoff) bool { return p.Ref == ref })
	})
}

// SetAgent records the handoff child opened for agentID.
func (r *Registry) SetAgent(ctx context.Context, sessionID, agentID string, a AgentHandoff) error {
	if a.CreatedAt.IsZero() {
		a.CreatedAt = r.now()
	}
	return r.update(ctx, sessionID, func(st *sessionState) {
		if st.Agents == nil {
			st.Agents = map[string]AgentHandoff{}
		}
		st.Agents[agentID] = a
	})
}

// Agent returns the handoff child opened for agentID.
func (r *Registry) Agent(ctx context.Context, sessionID, agentID string) (AgentHandoff, bool, error) {
	var out AgentHandoff
	var ok bool
	err := r.update(ctx, sessionID, func(st *sessionState) {
		out, ok = st.Agents[agentID]
	})
	return out, ok, err
}

// TakeAgent removes and returns the handoff child opened for agentID.
func (r *Registry) TakeAgent(ctx context.Context, sessionID, agentID string) (AgentHandoff, bool, error) {
	var out AgentHandoff
	var ok bool
	err := r.update(ctx, sessionID, func(st *sessionState) {
		if out, ok = st.Agents[agentID]; ok {
			delete(st.Agents, agentID)
		}
	})
	return out, ok, err
}

// update applies fn to the session's state under the file lock, dropping expired entries first.
// A missing or unreadable file reads as empty; an emptied state removes the file.
func (r *Registry) update(ctx context.Context, sessionID string, fn func(*sessionState)) error {
	if sessionID == "" {
		return errs.New("session_id required")
	}
	if err := os.MkdirAll(r.dir, 0o700); err != nil {
		return errs.WrapMessage("failed to create hook registry dir", err, "path", r.dir)
	}
	path := filepath.Join(r.dir, sessionKey(sessionID)+".json")
	unlock, err := r.lock(ctx, path+".lock")
	if err != nil {
		return err
	}
	defer unlock()
	var st sessionState
	if data, err := os.ReadFile(path); err == nil {
		// A corrupt file is replaced: the registry only steers hooks, it is never the record.
		_ = json.Unmarshal(data, &st)
	}
	st.prune(r.now().Add(-registryTTL))
	fn(&st)
	if len(st.Pending) == 0 && len(st.Agents) == 0 {
		if err := os.Remove(path); err != nil && !errors.Is(err, fs.ErrNotExist) {
			return errs.WrapMessage("failed to remove hook registry file", err, "path", path)
		}
		return nil
	}
	return writeFileAtomic(path, st)
}

// lock takes an O_EXCL lock file, breaking one older than staleLockAge, until ctx ends.
func (r *Registry) lock(ctx context.Context, path string) (func(), error) {
	for {
		f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
		if err == nil {
			_ = f.Close()
			return func() { _ = os.Remove(path) }, nil
		}
		if !errors.Is(err, fs.ErrExist) {
			return nil, errs.WrapMessage("failed to create hook registry lock", err, "path", path)
		}
		if info, err := os.Stat(path); err == nil && time.Since(info.ModTime()) > staleLockAge {
			_ = os.Remove(path)
			continue
		}
		select {
		case <-ctx.Done():
			return nil, errs.WrapMessage("timed out waiting for hook registry lock", ctx.Err(), "path", path)
		case <-time.After(lockRetry):
		}
	}
}

func (st *sessionState) prune(cutoff time.Time) {
	st.Pending = slices.DeleteFunc(st.Pending, func(p PendingHandoff) bool { return p.CreatedAt.Before(cutoff) })
	for id, a := range st.Agents {
		if a.CreatedAt.Before(cutoff) {
			delete(st.Agents, id)
		}
	}
}

// sessionKey names a session's file. Host session ids are uuids today, but hashing keeps any
// id a safe file name.
func sessionKey(sessionID string) string {
	sum := sha256.Sum256([]byte(sessionID))
	return hex.EncodeToString(sum[:])[:sessionKeyHexLen]
}

// writeFileAtomic writes v as JSON through a temp file and rename, so a reader never sees a
// partial file.
func writeFileAtomic(path string, v any) error {
	data, err := json.Marshal(v)
	if err != nil {
		return errs.WrapMessage("failed to encode hook registry", err, "path", path)
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".*.tmp")
	if err != nil {
		return errs.WrapMessage("failed to create hook registry temp file", err, "path", path)
	}
	_, werr := tmp.Write(data)
	cerr := tmp.Close()
	if err := errors.Join(werr, cerr); err != nil {
		_ = os.Remove(tmp.Name())
		return errs.WrapMessage("failed to write hook registry", err, "path", path)
	}
	if err := os.Rename(tmp.Name(), path); err != nil {
		_ = os.Remove(tmp.Name())
		return errs.WrapMessage("failed to replace hook registry", err, "path", path)
	}
	return nil
}
