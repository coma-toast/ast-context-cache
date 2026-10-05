package handoff

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	astcontext "github.com/coma-toast/ast-context-cache/internal/context"
	"github.com/coma-toast/ast-context-cache/internal/contextnotes"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/memory"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
	"github.com/coma-toast/ast-context-cache/internal/trail"
)

const (
	// snapshotTrailLimit is how many of the parent's newest trail entries a snapshot copies.
	snapshotTrailLimit = 200
	// pointerKindFile marks a pointer to a whole file rather than a symbol.
	pointerKindFile = "file"

	selectSymbolFileByFQNQueryPrefix = `SELECT file FROM symbols WHERE fqn = ? AND `
	selectSymbolFileByFQNQuerySuffix = ` ORDER BY start_line LIMIT 1`
	selectSnapshotItemsQuery         = `SELECT id, section, ord, COALESCE(item_key,''), COALESCE(label,''), COALESCE(content,''),
		COALESCE(file_rel,''), COALESCE(fqn,''), COALESCE(kind,''), COALESCE(start_line,0), COALESCE(end_line,0),
		COALESCE(fingerprint,''), token_est
		FROM handoff_snapshot_items WHERE handoff_ref = ? AND section = ? ORDER BY ord, id`
)

// snapshotItem is one handoff_snapshot_items row. For pointers, content holds the symbol's bare
// name (empty for a file pointer) and label the parent's note.
type snapshotItem struct {
	id          int64
	section     Section
	ord         int
	key         string
	label       string
	content     string
	fileRel     string
	fqn         string
	kind        string
	startLine   int
	endLine     int
	fingerprint string
	// tokenEst is what the item makes available to a child (OB-2): its content, or for a
	// pointer the source at auto mode.
	tokenEst int
	// charge is what the item counts against the tree cap (RQ-4): the bytes it stores.
	charge int
}

// snapshot is everything Create copies from the parent, gathered before its transaction.
type snapshot struct {
	items     []snapshotItem
	breakdown Breakdown
	entries   int
}

// trailContent is a snapshot trail item's content.
type trailContent struct {
	Tool     string   `json:"tool"`
	Query    string   `json:"query"`
	HitCount int      `json:"hit_count"`
	ZeroHit  bool     `json:"zero_hit"`
	TopHits  []string `json:"top_hits,omitempty"`
}

// memoryContent is a snapshot memory item's content: enough to clone the entry into a child.
type memoryContent struct {
	Kind        memory.Kind `json:"kind"`
	ProjectPath string      `json:"project_path,omitempty"`
	Subject     string      `json:"subject,omitempty"`
	Predicate   string      `json:"predicate,omitempty"`
	Object      string      `json:"object,omitempty"`
	Rule        string      `json:"rule,omitempty"`
}

// gatherSnapshot copies the parent's manifest, trail, notes, memory, and pointer fingerprints
// (HO-2, HO-4). It only reads, so nothing is stored when it fails (HO-6).
func gatherSnapshot(req CreateRequest, project string) (*snapshot, error) {
	sid := string(req.SessionID)
	snap := &snapshot{breakdown: Breakdown{Counts: map[Section]int{}}}
	if req.IncludeManifest == nil || *req.IncludeManifest {
		snap.addManifest(sid, project)
	}
	if !req.ExcludeAllTrail {
		snap.addTrail(sid, req.ExcludeTrail, req.ExcludeTrailQuery)
	}
	if err := snap.addNotes(req.CtxRefs); err != nil {
		return nil, err
	}
	if err := snap.addMemory(sid, req.MemRefs); err != nil {
		return nil, err
	}
	if err := snap.addPointers(project, req.Pointers); err != nil {
		return nil, err
	}
	return snap, nil
}

// addManifest copies the keys returned to the parent. The manifest counts as one cap entry
// however many keys it holds.
func (s *snapshot) addManifest(sid, project string) {
	keys := make([]string, 0)
	for k := range astcontext.ReturnedKeys(sid) {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	added := false
	for _, k := range keys {
		file, name, line, ok := parseDedupKey(k)
		if !ok {
			continue
		}
		rel := db.RelPath(file, project)
		cost := max(1, db.EstimateTokens(trail.HitRef(rel, name, line)))
		s.add(snapshotItem{section: SectionManifest, key: k, label: name, fileRel: rel, startLine: line, tokenEst: cost, charge: cost})
		added = true
	}
	if added {
		s.entries++
	}
}

// addTrail copies the parent's trail, newest first, minus the pruned entries. Indexes in
// exclude count from 0 at the newest entry.
func (s *snapshot) addTrail(sid string, exclude []int, excludeQuery string) {
	skip := make(map[int]bool, len(exclude))
	for _, i := range exclude {
		skip[i] = true
	}
	excludeQuery = strings.ToLower(strings.TrimSpace(excludeQuery))
	for i, e := range trail.ForSession(sid, snapshotTrailLimit) {
		if skip[i] {
			continue
		}
		if excludeQuery != "" && strings.Contains(strings.ToLower(e.Query), excludeQuery) {
			continue
		}
		body, _ := json.Marshal(trailContent{Tool: e.Tool, Query: e.Query, HitCount: e.HitCount, ZeroHit: e.ZeroHit, TopHits: e.TopHits})
		cost := db.EstimateTokens(string(body))
		s.add(snapshotItem{section: SectionTrail, key: e.MatchKey(), label: e.Query, content: string(body), tokenEst: cost, charge: cost})
		s.entries++
	}
}

func (s *snapshot) addNotes(refs []string) error {
	seen := map[string]bool{}
	for _, ref := range refs {
		ref = strings.TrimSpace(ref)
		if ref == "" || seen[ref] {
			continue
		}
		seen[ref] = true
		n, err := contextnotes.Peek(ref)
		if err != nil {
			return errs.WrapMessage("failed to snapshot context note", err, "ref", ref)
		}
		cost := db.EstimateTokens(n.Content)
		s.add(snapshotItem{section: SectionNote, key: n.Ref, label: n.Label, content: n.Content, tokenEst: cost, charge: cost})
		s.entries++
	}
	return nil
}

// addMemory copies the explicit refs and then every active session-scoped entry of the parent.
func (s *snapshot) addMemory(sid string, refs []string) error {
	var entries []memory.Entry
	for _, ref := range refs {
		if ref = strings.TrimSpace(ref); ref == "" {
			continue
		}
		e, err := memory.Peek(ref)
		if err != nil {
			return errs.WrapMessage("failed to snapshot memory entry", err, "ref", ref)
		}
		entries = append(entries, *e)
	}
	active, err := memory.ActiveForSession(sid)
	if err != nil {
		return errs.WrapMessage("failed to snapshot session memory", err, "session_id", sid)
	}
	seen := map[string]bool{}
	for _, e := range append(entries, active...) {
		if seen[e.Ref] {
			continue
		}
		seen[e.Ref] = true
		body, _ := json.Marshal(memoryContent{
			Kind: e.Kind, ProjectPath: e.ProjectPath, Subject: e.Subject, Predicate: e.Predicate, Object: e.Object, Rule: e.Rule,
		})
		line := memory.FormatLine(e)
		cost := max(1, db.EstimateTokens(line))
		s.add(snapshotItem{section: SectionMemory, key: e.Ref, label: line, content: string(body), tokenEst: cost, charge: cost})
		s.entries++
	}
	return nil
}

func (s *snapshot) addPointers(project string, pointers []PointerInput) error {
	fileCache := map[string][]string{}
	for _, p := range pointers {
		item, err := resolvePointer(project, p, fileCache)
		if err != nil {
			return err
		}
		s.add(*item)
		s.entries++
	}
	return nil
}

func (s *snapshot) add(item snapshotItem) {
	item.ord = s.breakdown.Counts[item.section]
	s.items = append(s.items, item)
	s.breakdown.Counts[item.section]++
	switch item.section {
	case SectionManifest:
		s.breakdown.Manifest += item.charge
	case SectionTrail:
		s.breakdown.Trail += item.charge
	case SectionNote:
		s.breakdown.Notes += item.charge
	case SectionMemory:
		s.breakdown.Memory += item.charge
	case SectionPointer:
		s.breakdown.Pointers += item.charge
	}
	s.breakdown.Total += item.charge
}

// resolvePointer turns a pointer key into a fingerprinted snapshot item. The key is a symbol
// key ("file|name", optionally "|line", where name may be qualified as Class.method), a
// project-relative or absolute file path, or an fqn ("<file basename>.<qualified name>").
func resolvePointer(project string, p PointerInput, fileCache map[string][]string) (*snapshotItem, error) {
	key := strings.TrimSpace(p.Key)
	if key == "" {
		return nil, errs.NewCode(errs.CodeInvalidInput, "pointer key required")
	}
	if !strings.Contains(key, "|") {
		if abs := absUnder(project, key); abs != "" {
			if fi, err := os.Stat(abs); err == nil && fi.Mode().IsRegular() {
				return filePointer(project, key, abs, p.Note)
			}
		}
	}
	file, fqn, name, err := pointerSymbol(project, key)
	if err != nil {
		return nil, err
	}
	if project == "" && !filepath.IsAbs(file) {
		return nil, errs.NewCode(errs.CodeInvalidInput, "project_path required for relative pointers", "pointer", key)
	}
	row, err := astcontext.LookupSymbol(project, file, fqn, name)
	if err != nil {
		return nil, errs.WrapMessage("failed to resolve pointer", err, "pointer", key)
	}
	fp, err := astcontext.SymbolFingerprint(row.File, row.StartLine, row.EndLine)
	if err != nil {
		return nil, errs.WrapMessage("failed to fingerprint pointer", err, "pointer", key)
	}
	cost := db.EstimateTokens(key + p.Note + row.FQN + row.FileRel)
	return &snapshotItem{
		section: SectionPointer, key: key, label: p.Note, content: row.Name, fileRel: row.FileRel, fqn: row.FQN, kind: row.Kind,
		startLine: row.StartLine, endLine: row.EndLine, fingerprint: fp, charge: max(1, cost),
		tokenEst: astcontext.FullSourceTokens(row.File, row.Name, row.ProjectPath, row.StartLine, row.EndLine, fileCache),
	}, nil
}

// pointerSymbol splits a symbol pointer key into the file, fqn, and bare name LookupSymbol takes.
func pointerSymbol(project, key string) (file, fqn, name string, err error) {
	if parts := strings.Split(key, "|"); len(parts) >= 2 {
		file, qualified := parts[0], parts[1]
		return file, filepath.Base(file) + "." + qualified, bareName(qualified), nil
	}
	if project == "" {
		return "", "", "", errs.NewCode(errs.CodeInvalidInput, "project_path required to resolve an fqn pointer", "pointer", key)
	}
	conn, err := db.IndexReader()
	if err != nil {
		return "", "", "", errs.WrapMessage("failed to resolve pointer", err, "pointer", key)
	}
	scope, args := projectlinks.ScopeSQL("", project)
	err = conn.QueryRow(selectSymbolFileByFQNQueryPrefix+scope+selectSymbolFileByFQNQuerySuffix, append([]any{key}, args...)...).Scan(&file)
	if errors.Is(err, sql.ErrNoRows) {
		return "", "", "", errs.NewCode(errs.CodeNotFound, "pointer not found: not a file, symbol key, or indexed fqn", "pointer", key)
	}
	if err != nil {
		return "", "", "", errs.WrapMessage("failed to resolve pointer", err, "pointer", key)
	}
	return file, key, bareName(key), nil
}

// filePointer fingerprints a whole file.
func filePointer(project, key, abs, note string) (*snapshotItem, error) {
	data, err := os.ReadFile(abs)
	if err != nil {
		return nil, errs.WrapCodeMessage(errs.CodeNotFound, "failed to read pointer file", err, "pointer", key)
	}
	lines := lineCount(data)
	fp, err := astcontext.SymbolFingerprint(abs, 1, lines)
	if err != nil {
		return nil, errs.WrapMessage("failed to fingerprint pointer", err, "pointer", key)
	}
	rel := db.RelPath(abs, project)
	return &snapshotItem{
		section: SectionPointer, key: key, label: note, fileRel: rel, kind: pointerKindFile, startLine: 1, endLine: lines,
		fingerprint: fp, charge: max(1, db.EstimateTokens(key+note+rel)), tokenEst: db.EstimateTokens(string(data)),
	}, nil
}

// loadSnapshotItems reads one section of a handoff's snapshot in order.
func loadSnapshotItems(ref HandoffRef, section Section) ([]snapshotItem, error) {
	if db.ContextDB == nil {
		return nil, errNoContextDB
	}
	rows, err := db.ContextDB.Query(selectSnapshotItemsQuery, string(ref), string(section))
	if err != nil {
		return nil, errs.WrapMessage("failed to read handoff snapshot", err, "handoff", string(ref), "section", string(section))
	}
	defer rows.Close()
	var out []snapshotItem
	for rows.Next() {
		var it snapshotItem
		if err := rows.Scan(&it.id, &it.section, &it.ord, &it.key, &it.label, &it.content, &it.fileRel, &it.fqn, &it.kind,
			&it.startLine, &it.endLine, &it.fingerprint, &it.tokenEst); err != nil {
			return nil, errs.WrapMessage("failed to read handoff snapshot item", err, "handoff", string(ref))
		}
		out = append(out, it)
	}
	return out, rows.Err()
}

// parseDedupKey splits a SymbolDedupKey ("file|name|line") from the right, so a file path
// containing "|" still parses.
func parseDedupKey(k string) (file, name string, line int, ok bool) {
	i := strings.LastIndex(k, "|")
	if i <= 0 {
		return "", "", 0, false
	}
	line, err := strconv.Atoi(k[i+1:])
	if err != nil {
		return "", "", 0, false
	}
	j := strings.LastIndex(k[:i], "|")
	if j <= 0 {
		return "", "", 0, false
	}
	return k[:j], k[j+1 : i], line, true
}

// absUnder returns path as an absolute path, resolving a relative one under project; "" when
// a relative path has no project to resolve against.
func absUnder(project, path string) string {
	if filepath.IsAbs(path) {
		return filepath.Clean(path)
	}
	if project == "" {
		return ""
	}
	return filepath.Join(project, path)
}

// bareName strips the qualification from Class.method (or an fqn), leaving the symbol name.
func bareName(qualified string) string {
	if i := strings.LastIndex(qualified, "."); i >= 0 && i < len(qualified)-1 {
		return qualified[i+1:]
	}
	return qualified
}

// lineCount counts lines as ReadSourceRange numbers them: a trailing newline ends the last line.
func lineCount(data []byte) int {
	n := bytes.Count(data, []byte("\n"))
	if len(data) > 0 && data[len(data)-1] != '\n' {
		n++
	}
	return max(1, n)
}
