package context

import (
	"context"
	"maps"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
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

// sessionSet is a session's returned-symbol keys. The sessions table is written
// through a batching buffer (flushed every few seconds), so reading it alone let a
// symbol returned moments ago be sent again; the in-memory set is updated before
// the call that returned the symbol responds.
type sessionSet struct {
	keys     map[string]struct{}
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
		s.keys[SymbolDedupKey(r.File, r.Name, r.StartLine)] = struct{}{}
	}
}

// lockedSession returns sessionID's set, locked, hydrated and marked used. A set
// evicted between the lookup and the lock is skipped: writes to it would be lost.
func lockedSession(sessionID string) *sessionSet {
	for {
		v, _ := sessions.LoadOrStore(sessionID, &sessionSet{keys: map[string]struct{}{}})
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
	persisted, err := loadReturnedKeys(sessionID)
	if err != nil {
		logger.Warn("Failed to load returned symbols for session", "session_id", sessionID, "error", err)
		return
	}
	maps.Copy(s.keys, persisted)
	s.hydrated = true
}
