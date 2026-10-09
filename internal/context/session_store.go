package context

import (
	"context"
	"maps"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/flags"
)

const (
	sessionIdleTTL        = 30 * time.Minute
	sessionEvictInterval  = 5 * time.Minute
	seededReturnedModeTag = "seed"
)

// ReturnedSymbol is one symbol delivered to (or seeded into) a session.
type ReturnedSymbol struct {
	File        string
	Name        string
	ProjectPath string
	Mode        string
	StartLine   int
	Tokens      int
}

// sessionSet is a session's returned-symbol keys and the richest mode each was delivered in. The sessions table is written
// through a batching buffer (flushed every few seconds), so reading it alone let a
// symbol returned moments ago be sent again; the in-memory set is updated before
// the call that returned the symbol responds.
type sessionSet struct {
	keys     map[string]struct{}
	modes    map[string]string
	hydrated bool
	evicted  bool
	lastUsed time.Time

	mu sync.Mutex
}

// sessions maps session id to *sessionSet.
var sessions sync.Map

// ReturnedKeys returns a copy of the dedup keys (SymbolDedupKey) already returned
// to sessionID, hydrating them once from the database. It returns an empty map for
// no session, so callers can add a list's own keys to it to skip in-list duplicates.
func ReturnedKeys(sessionID string) map[string]struct{} {
	if sessionID == "" {
		return map[string]struct{}{}
	}
	s := lockedSession(sessionID)
	defer s.mu.Unlock()
	return maps.Clone(s.keys)
}

// ReturnedModes returns a copy of the keys already returned to sessionID mapped to the
// richest mode each was delivered in ("" or "seed" when unknown), hydrating once from the
// database. It returns an empty map for no session.
func ReturnedModes(sessionID string) map[string]string {
	if sessionID == "" {
		return map[string]string{}
	}
	s := lockedSession(sessionID)
	defer s.mu.Unlock()
	return maps.Clone(s.modes)
}

// DedupCovers reports whether a symbol already returned (returned from ReturnedModes) covers a
// request for it in mode want, so it is skipped. With feature_mode_v2 on, an earlier delivery
// covers the request only when its mode ranks at least as high as want (a skeleton does not
// cover a later full request); a seeded or legacy row with no mode covers everything. With
// the flag off any earlier delivery covers it.
func DedupCovers(returned map[string]string, key, want string) bool {
	prev, ok := returned[key]
	if !ok {
		return false
	}
	if !flags.Enabled(flags.KeyModeV2) || prev == "" || prev == seededReturnedModeTag {
		return true
	}
	return ModeRank(prev) >= ModeRank(want)
}

// mergeReturnedMode records key as returned in mode, keeping the richer of mode and any
// earlier mode ("" and "seed" rank above everything, since they cover any request).
func mergeReturnedMode(modes map[string]string, key, mode string) {
	prev, ok := modes[key]
	if !ok || prev == "" || prev == seededReturnedModeTag {
		if !ok {
			modes[key] = mode
		}
		return
	}
	if mode == "" || mode == seededReturnedModeTag || ModeRank(mode) > ModeRank(prev) {
		modes[key] = mode
	}
}

// MarkReturned records syms as returned to sessionID: the in-memory set at once,
// then the sessions table through the write buffer.
func MarkReturned(sessionID string, syms ...ReturnedSymbol) {
	if sessionID == "" || len(syms) == 0 {
		return
	}
	addReturned(sessionID, syms)
	for _, r := range syms {
		symbolID := LookupSymbolID(r.File, r.Name, r.ProjectPath, r.StartLine)
		db.EnqueueSessionReturned(sessionID, symbolID, r.Name, r.StartLine, r.File, r.Mode, r.Tokens)
	}
}

// SeedReturned marks syms as already seen by sessionID without them having been
// delivered in one of its calls (a fork inheriting its parent's context). Rows are
// persisted like MarkReturned's, tagged mode "seed" unless one is given.
func SeedReturned(sessionID string, syms []ReturnedSymbol) {
	seeded := make([]ReturnedSymbol, len(syms))
	for i, r := range syms {
		if r.Mode == "" {
			r.Mode = seededReturnedModeTag
		}
		seeded[i] = r
	}
	MarkReturned(sessionID, seeded...)
}

// StartSessionStoreEviction drops in-memory sets idle for over 30 minutes until ctx
// is done. An evicted session rehydrates from the database on its next call.
func StartSessionStoreEviction(ctx context.Context) {
	go sessionEvictionLoop(ctx)
}

func sessionEvictionLoop(ctx context.Context) {
	logger.Debug("Starting session store eviction loop")
	defer logger.Debug("Stopped session store eviction loop")
	for {
		select {
		case <-ctx.Done():
			return
		case <-time.After(sessionEvictInterval):
		}
		if n := evictIdleSessions(time.Now().Add(-sessionIdleTTL)); n > 0 {
			logger.Debug("Evicted idle session sets", "sessions", n)
		}
	}
}

// evictIdleSessions drops sets last used before cutoff and returns how many.
func evictIdleSessions(cutoff time.Time) int {
	n := 0
	sessions.Range(func(k, v any) bool {
		s := v.(*sessionSet)
		s.mu.Lock()
		if s.lastUsed.Before(cutoff) {
			s.evicted = true
			sessions.CompareAndDelete(k, s)
			n++
		}
		s.mu.Unlock()
		return true
	})
	return n
}

func addReturned(sessionID string, syms []ReturnedSymbol) {
	s := lockedSession(sessionID)
	defer s.mu.Unlock()
	for _, r := range syms {
		key := SymbolDedupKey(r.File, r.Name, r.StartLine)
		s.keys[key] = struct{}{}
		mergeReturnedMode(s.modes, key, r.Mode)
	}
}

// lockedSession returns sessionID's set, locked, hydrated and marked used. A set
// evicted between the lookup and the lock is skipped: writes to it would be lost.
func lockedSession(sessionID string) *sessionSet {
	for {
		v, _ := sessions.LoadOrStore(sessionID, &sessionSet{keys: map[string]struct{}{}, modes: map[string]string{}})
		s := v.(*sessionSet)
		s.mu.Lock()
		if s.evicted {
			s.mu.Unlock()
			continue
		}
		s.lastUsed = time.Now()
		if !s.hydrated {
			s.hydrate(sessionID)
		}
		return s
	}
}

// hydrate merges the persisted keys into the set. On a read error the set stays
// unhydrated so the next call retries; keys recorded in memory are still served.
func (s *sessionSet) hydrate(sessionID string) {
	if db.DB == nil {
		return
	}
	persisted, err := loadReturned(sessionID)
	if err != nil {
		logger.Warn("Failed to load returned symbols for session", "session_id", sessionID, "error", err)
		return
	}
	for key, mode := range persisted {
		s.keys[key] = struct{}{}
		mergeReturnedMode(s.modes, key, mode)
	}
	s.hydrated = true
}
