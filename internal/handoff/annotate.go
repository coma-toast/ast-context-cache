package handoff

import (
	"database/sql"
	"encoding/json"
	"errors"
	"path/filepath"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/trail"
)

const (
	// searchCounterFlushInterval coalesces a child's search_calls/repeat_calls writes.
	searchCounterFlushInterval = 2 * time.Second
	// annotateCacheMax bounds the cached snapshots; past it the cache starts over.
	annotateCacheMax = 256

	selectHandoffProjectQuery = `SELECT COALESCE(project_path, '') FROM handoffs WHERE ref = ?`
	selectSiblingTrailQuery   = `SELECT id, author_session_id, COALESCE(refs_json, '') FROM scratchpad_entries
		WHERE tree_id = ? AND type = '` + string(EntryTypeTrail) + `' AND author_session_id != ? AND retracted_at IS NULL
			AND json_extract(refs_json, '$.match_key') = ?
		ORDER BY id DESC LIMIT 1`
	addSearchCallsQuery = `UPDATE handoff_children SET search_calls = search_calls + ?, repeat_calls = repeat_calls + ?
		WHERE child_session_id = ?`
)

// annotateSnapshot is the part of a handoff's snapshot that Annotate matches searches against.
// Snapshots are immutable (HO-3), so it is cached for the life of the process.
type annotateSnapshot struct {
	// trail maps a match key to the parent's newest entry with it.
	trail map[string]trailContent
	// manifest holds the parent's returned symbols as "relfile#name@line" (trail.HitRef),
	// relative to the snapshot project, so they compare across sibling worktrees.
	manifest map[string]struct{}
	project  string
}

// searchCounter is a child's not-yet-written search and repeat counts (OB-1).
type searchCounter struct {
	searches  int
	repeats   int
	flushedAt time.Time
}

// annotator caches snapshots and coalesces search-count writes for Annotate.
type annotator struct {
	mu        sync.Mutex
	snapshots map[HandoffRef]*annotateSnapshot
	counters  map[SessionID]*searchCounter
}

// siblingTrailRefs is a live trail scratchpad entry's refs_json (written from the trail
// subscription, SP-5).
type siblingTrailRefs struct {
	MatchKey string   `json:"match_key"`
	HitCount int      `json:"hit_count"`
	ZeroHit  bool     `json:"zero_hit"`
	TopHits  []string `json:"top_hits,omitempty"`
}

func newAnnotator() *annotator {
	return &annotator{snapshots: map[HandoffRef]*annotateSnapshot{}, counters: map[SessionID]*searchCounter{}}
}

// SearchEventFor is the event Annotate matches for a search recorded as e.
func SearchEventFor(e trail.Entry) SearchEvent {
	if e.QueryNorm == "" {
		e.QueryNorm = trail.NormalizeQuery(e.Query)
	}
	return SearchEvent{
		Tool: e.Tool, Query: e.Query, MatchKey: e.MatchKey(), Hits: e.HitCount, TopKeys: e.TopHits, ZeroHit: e.HitCount == 0,
		ProjectPath: e.ProjectPath, CandidateKeys: e.CandidateHits,
	}
}

// Annotate matches a tree session's search against its parent's snapshot and its siblings' live
// trail (OP-9, OP-10, SP-6) and counts a child's searches and repeats (OB-1). In a fresh child
// it sets "parent_explored": true on results the parent was already returned; a fork child's
// dedup already skipped those. It returns the response-level fields to merge, or nil. A session
// outside every tree costs one map lookup (NFR-2).
func (s *realService) Annotate(sid SessionID, ev SearchEvent, results []map[string]any) map[string]any {
	e, ok := s.trees.lookup(sid)
	if !ok {
		return nil
	}
	out := map[string]any{}
	if e.isChild && e.handoff != "" {
		snap, err := s.annot.snapshot(e.handoff)
		if err != nil {
			s.logger.Warn("Failed to load handoff snapshot for annotations", e.handoff.Attr(), sid.Attr(), "error", err)
		} else {
			tc, matched := snap.trail[ev.MatchKey]
			if matched {
				out["parent_trail_match"] = map[string]any{"query": tc.Query, "hit_count": tc.HitCount, "zero_hit": tc.ZeroHit, "top_hits": tc.TopHits}
			}
			if e.mode == ModeFresh {
				snap.markExplored(results, ev.ProjectPath)
			}
			s.countSearch(sid, matched || snap.overlaps(ev.CandidateKeys))
		}
	}
	if m := s.siblingTrailMatch(e.tree, sid, ev.MatchKey); m != nil {
		out["sibling_trail_match"] = m
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// countSearch adds one search (and maybe a repeat) to sid's counters, writing them through at
// most every searchCounterFlushInterval.
func (s *realService) countSearch(sid SessionID, repeat bool) {
	countChildSearch(repeat)
	a := s.annot
	a.mu.Lock()
	c := a.counters[sid]
	if c == nil {
		c = &searchCounter{}
		a.counters[sid] = c
	}
	c.searches++
	if repeat {
		c.repeats++
	}
	now := nowFunc()
	due := now.Sub(c.flushedAt) >= searchCounterFlushInterval
	var searches, repeats int
	if due {
		searches, repeats = c.searches, c.repeats
		c.searches, c.repeats, c.flushedAt = 0, 0, now
	}
	a.mu.Unlock()
	if due {
		s.writeSearchCounts(sid, searches, repeats)
	}
}

// flushSearchCounters writes sid's pending search counts now. Touch calls it so a child's last
// searches are counted on its next tool call.
func (s *realService) flushSearchCounters(sid SessionID) {
	a := s.annot
	a.mu.Lock()
	c := a.counters[sid]
	if c == nil || (c.searches == 0 && c.repeats == 0) {
		a.mu.Unlock()
		return
	}
	searches, repeats := c.searches, c.repeats
	c.searches, c.repeats, c.flushedAt = 0, 0, nowFunc()
	a.mu.Unlock()
	s.writeSearchCounts(sid, searches, repeats)
}

func (s *realService) writeSearchCounts(sid SessionID, searches, repeats int) {
	err := db.HandoffTx(func(tx *sql.Tx) error {
		_, err := tx.Exec(addSearchCallsQuery, searches, repeats, string(sid))
		return err
	})
	if err != nil {
		s.logger.Warn("Failed to record handoff child search counts", sid.Attr(), "searches", searches, "repeats", repeats, "error", err)
	}
}

// siblingTrailMatch is the newest live trail entry another session of the tree recorded for the
// same search (SP-6), or nil.
func (s *realService) siblingTrailMatch(tree TreeID, sid SessionID, matchKey string) map[string]any {
	if matchKey == "" || db.ContextDB == nil {
		return nil
	}
	var id int64
	var author SessionID
	var refsJSON string
	err := db.ContextDB.QueryRow(selectSiblingTrailQuery, string(tree), string(sid), matchKey).Scan(&id, &author, &refsJSON)
	if errors.Is(err, sql.ErrNoRows) {
		return nil
	}
	if err != nil {
		s.logger.Warn("Failed to match sibling trail", tree.Attr(), sid.Attr(), "error", err)
		return nil
	}
	var refs siblingTrailRefs
	_ = json.Unmarshal([]byte(refsJSON), &refs)
	return map[string]any{"author": string(author), "entry": id, "hit_count": refs.HitCount, "zero_hit": refs.ZeroHit, "top_hits": refs.TopHits}
}

// snapshot returns ref's cached match data, loading it on first use.
func (a *annotator) snapshot(ref HandoffRef) (*annotateSnapshot, error) {
	a.mu.Lock()
	snap, ok := a.snapshots[ref]
	a.mu.Unlock()
	if ok {
		return snap, nil
	}
	snap, err := loadAnnotateSnapshot(ref)
	if err != nil {
		return nil, err
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if len(a.snapshots) >= annotateCacheMax {
		a.snapshots = map[HandoffRef]*annotateSnapshot{}
	}
	a.snapshots[ref] = snap
	return snap, nil
}

func loadAnnotateSnapshot(ref HandoffRef) (*annotateSnapshot, error) {
	if db.ContextDB == nil {
		return nil, errNoContextDB
	}
	snap := &annotateSnapshot{trail: map[string]trailContent{}, manifest: map[string]struct{}{}}
	if err := db.ContextDB.QueryRow(selectHandoffProjectQuery, string(ref)).Scan(&snap.project); err != nil {
		return nil, errs.WrapMessage("failed to read handoff project", err, "handoff", string(ref))
	}
	items, err := loadSnapshotItems(ref, SectionTrail)
	if err != nil {
		return nil, err
	}
	for _, it := range items {
		if _, seen := snap.trail[it.key]; seen {
			continue
		}
		var tc trailContent
		if json.Unmarshal([]byte(it.content), &tc) == nil {
			snap.trail[it.key] = tc
		}
	}
	if items, err = loadSnapshotItems(ref, SectionManifest); err != nil {
		return nil, err
	}
	for _, it := range items {
		snap.manifest[trail.HitRef(it.fileRel, it.label, it.startLine)] = struct{}{}
	}
	return snap, nil
}

// markExplored sets "parent_explored" on results whose symbol is in the parent's manifest
// (OP-10). Result files are relative to the search project, or absolute.
func (a *annotateSnapshot) markExplored(results []map[string]any, project string) {
	if len(a.manifest) == 0 {
		return
	}
	for _, r := range results {
		file, _ := r["file"].(string)
		name, _ := r["name"].(string)
		if file == "" || name == "" {
			continue
		}
		if filepath.IsAbs(file) && project != "" {
			file = db.RelPath(file, project)
		}
		if filepath.IsAbs(file) {
			file = db.RelPath(file, a.project)
		}
		if _, ok := a.manifest[trail.HitRef(file, name, intField(r["start_line"]))]; ok {
			r["parent_explored"] = true
		}
	}
}

// overlaps reports whether at least half of a search's pre-dedup keys are in the parent's
// manifest: OB-1's second repeat rule.
func (a *annotateSnapshot) overlaps(keys []string) bool {
	if len(keys) == 0 || len(a.manifest) == 0 {
		return false
	}
	n := 0
	for _, k := range keys {
		if _, ok := a.manifest[k]; ok {
			n++
		}
	}
	return n*2 >= len(keys)
}

// intField reads a result's numeric field, which may be any JSON-ish number type.
func intField(v any) int {
	switch n := v.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		return int(n)
	case json.Number:
		i, _ := n.Int64()
		return int(i)
	}
	return 0
}
