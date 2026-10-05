// Package trail records the searches each session runs — query, normalized filters, hit count
// and top hits — so a handoff can tell a child what its parent already looked for. Entries are
// kept in a per-session in-memory ring for read-your-writes and persisted to usage.db
// search_trail through a batched writer.
package trail

import (
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
)

const (
	// MaxTopHits caps Entry.TopHits.
	MaxTopHits = 5
	// ringCap is how many entries per session stay in memory.
	ringCap = 200
)

var errNoDB = errs.NewCode(errs.CodeInternal, "usage database not open")

// Entry is one search a session ran.
type Entry struct {
	SessionID   string `json:"session_id"`
	Tool        string `json:"tool"`
	Query       string `json:"query"`
	QueryNorm   string `json:"query_norm"`
	FiltersKey  string `json:"filters_key,omitempty"`
	Mode        string `json:"mode,omitempty"`
	DocType     string `json:"doc_type,omitempty"`
	ProjectPath string `json:"project_path,omitempty"`
	// HitCount is the candidate count before session dedup, so a repeated search still
	// reports what it found rather than what was left to deliver.
	HitCount int  `json:"hit_count"`
	ZeroHit  bool `json:"zero_hit"`
	// TopHits holds up to MaxTopHits "relfile#name@line" refs (see HitRef), best first.
	TopHits []string  `json:"top_hits,omitempty"`
	At      time.Time `json:"at"`
	// CandidateHits lists every pre-dedup candidate in the window the call considered, in
	// TopHits' format, for repeat-search accounting. Subscribers see it; it is never stored.
	CandidateHits []string `json:"-"`
}

// ring keeps a session's most recent entries, oldest first.
type ring struct {
	mu      sync.Mutex
	entries []Entry
}

// MatchKey identifies a repeat of the same search: tool|queryNorm|filtersKey|docType.
func (e Entry) MatchKey() string {
	return e.Tool + "|" + e.QueryNorm + "|" + e.FiltersKey + "|" + e.DocType
}

// NormalizeQuery lowercases q, collapses runs of whitespace to one space, and trims it.
func NormalizeQuery(q string) string {
	return strings.Join(strings.Fields(strings.ToLower(q)), " ")
}

// HitRef formats one hit as "relfile#name@line" for TopHits and CandidateHits.
func HitRef(relFile, name string, line int) string {
	return relFile + "#" + name + "@" + strconv.Itoa(line)
}

var (
	rings sync.Map // session id -> *ring

	subMu       sync.Mutex
	subscribers []func(Entry)
)

// Record adds e to its session's trail. It fills QueryNorm and At when empty, derives ZeroHit
// from HitCount, caps TopHits, and then calls every subscriber synchronously. Entries without
// a session are dropped.
func Record(e Entry) {
	if e.SessionID == "" {
		return
	}
	if e.QueryNorm == "" {
		e.QueryNorm = NormalizeQuery(e.Query)
	}
	if e.At.IsZero() {
		e.At = time.Now()
	}
	// UTC with no monotonic reading, so the ring copy equals the copy read back from the DB.
	e.At = e.At.UTC().Round(0)
	e.ZeroHit = e.HitCount == 0
	if len(e.TopHits) > MaxTopHits {
		e.TopHits = e.TopHits[:MaxTopHits]
	}
	e.TopHits = slices.Clone(e.TopHits)
	stored := e
	stored.TopHits = slices.Clone(e.TopHits)
	stored.CandidateHits = nil
	ringFor(e.SessionID).add(stored)
	db.EnqueueTrail(toRow(stored))
	notify(e)
}

// ForSession returns up to limit of sid's entries, newest first (limit <= 0 means 200). It
// merges the in-memory ring, which holds writes still buffered for the DB, with persisted rows.
func ForSession(sid string, limit int) []Entry {
	if sid == "" {
		return nil
	}
	if limit <= 0 {
		limit = ringCap
	}
	var mem []Entry
	if r, ok := rings.Load(sid); ok {
		mem = r.(*ring).newestFirst()
	}
	if db.DB == nil {
		return mergeEntries(mem, nil, limit)
	}
	persisted, err := loadSession(sid, limit)
	if err != nil {
		logger.Warn("Failed to load search trail", "error", err, "session", sid)
	}
	return mergeEntries(mem, persisted, limit)
}

// Lookup returns sid's newest entry whose MatchKey equals matchKey.
func Lookup(sid, matchKey string) (Entry, bool) {
	if sid == "" {
		return Entry{}, false
	}
	// The ring holds the session's newest entries, so a ring match is the newest overall.
	if r, ok := rings.Load(sid); ok {
		for _, e := range r.(*ring).newestFirst() {
			if e.MatchKey() == matchKey {
				return e, true
			}
		}
	}
	if db.DB == nil {
		return Entry{}, false
	}
	e, ok, err := loadMatch(sid, matchKey)
	if err != nil {
		logger.Warn("Failed to look up search trail", "error", err, "session", sid)
	}
	return e, ok
}

// PruneOlderThan deletes trail entries recorded more than d ago, from memory and the DB, and
// returns how many DB rows it deleted.
func PruneOlderThan(d time.Duration) (int64, error) {
	cutoff := time.Now().Add(-d).UTC()
	pruneMemory(cutoff)
	if db.DB == nil {
		return 0, errNoDB
	}
	n, err := deleteBefore(cutoff)
	if err != nil {
		return 0, errs.WrapMessage("failed to prune search trail", err, "older_than", d.String())
	}
	if n > 0 {
		logger.Info("Pruned search trail", "rows", n, "older_than", d.String())
	}
	return n, nil
}

// Subscribe registers fn to be called synchronously after each Record, with the recorded entry.
func Subscribe(fn func(Entry)) {
	subMu.Lock()
	defer subMu.Unlock()
	subscribers = append(subscribers, fn)
}

func notify(e Entry) {
	subMu.Lock()
	subs := slices.Clone(subscribers)
	subMu.Unlock()
	for _, fn := range subs {
		fn(e)
	}
}

func mergeEntries(mem, persisted []Entry, limit int) []Entry {
	all := append(slices.Clone(mem), persisted...)
	sort.SliceStable(all, func(i, j int) bool { return all[i].At.After(all[j].At) })
	seen := make(map[string]struct{}, len(all))
	out := make([]Entry, 0, min(len(all), limit))
	for _, e := range all {
		key := e.MatchKey() + "|" + strconv.FormatInt(e.At.UnixNano(), 10)
		if _, dup := seen[key]; dup {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, e)
		if len(out) == limit {
			break
		}
	}
	return out
}

func pruneMemory(cutoff time.Time) {
	rings.Range(func(k, v any) bool {
		if v.(*ring).dropBefore(cutoff) == 0 {
			rings.Delete(k)
		}
		return true
	})
}

func ringFor(sid string) *ring {
	if r, ok := rings.Load(sid); ok {
		return r.(*ring)
	}
	r, _ := rings.LoadOrStore(sid, &ring{})
	return r.(*ring)
}

// resetMemory forgets every in-memory ring, as a restart would; tests use it to prove
// persisted rows are read back.
func resetMemory() {
	rings.Range(func(k, _ any) bool {
		rings.Delete(k)
		return true
	})
}

func (r *ring) add(e Entry) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.entries) == ringCap {
		r.entries = slices.Delete(r.entries, 0, 1)
	}
	r.entries = append(r.entries, e)
}

func (r *ring) newestFirst() []Entry {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]Entry, len(r.entries))
	for i, e := range r.entries {
		e.TopHits = slices.Clone(e.TopHits)
		out[len(r.entries)-1-i] = e
	}
	return out
}

// dropBefore removes entries recorded before cutoff and returns how many remain.
func (r *ring) dropBefore(cutoff time.Time) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.entries = slices.DeleteFunc(r.entries, func(e Entry) bool { return e.At.Before(cutoff) })
	return len(r.entries)
}
