package search

import (
	"crypto/sha256"
	"database/sql"
	"encoding/binary"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

const (
	selectAllVectorsQuery        = "SELECT id, COALESCE(symbol_id,0), content_hash, vector, COALESCE(doc_type,'code'), COALESCE(source_file,''), COALESCE(name,''), COALESCE(kind,''), COALESCE(project_path,'') FROM vectors ORDER BY id"
	selectSymbolRowByIDQuery     = "SELECT COALESCE(start_line,0), COALESCE(end_line,0), COALESCE(fqn,'') FROM symbols WHERE id = ?"
	selectSymbolRowByNameQuery   = "SELECT COALESCE(start_line,0), COALESCE(end_line,0), COALESCE(fqn,'') FROM symbols WHERE file = ? AND name = ? AND project_path = ? ORDER BY start_line LIMIT 1"
	upsertVectorQuery            = `INSERT OR REPLACE INTO vectors (content_hash, vector, doc_type, source_file, name, kind, project_path, symbol_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
	deleteVectorRefQuery         = "DELETE FROM vectors WHERE doc_type = ? AND source_file = ?"
	deleteDocVectorsLikeQuery    = "DELETE FROM vectors WHERE doc_type = 'doc' AND source_file LIKE ?"
	deleteOrphanCodeVectorsQuery = `
		DELETE FROM vectors
		WHERE COALESCE(doc_type, 'code') = 'code'
		  AND symbol_id > 0
		  AND symbol_id NOT IN (SELECT id FROM symbols)`
	selectSymbolIDsQuery    = `SELECT id FROM symbols`
	countVectorsQuery       = "SELECT COUNT(*) FROM vectors"
	countScopedVectorsQuery = "SELECT COUNT(*) FROM vectors WHERE "
)

const VectorDims = 768

// memoryCandidatePool is the least number of memory vector candidates SearchMemory
// returns when it skips the session filter.
const memoryCandidatePool = 50

type VectorEntry struct {
	ID          int64
	SymbolID    int64
	ContentHash string
	Vector      []float32
	DocType     string
	SourceFile  string
	Name        string
	Kind        string
	ProjectPath string
}

type VectorCache struct {
	mu       sync.RWMutex
	entries  []VectorEntry
	loaded   bool
	lastUsed time.Time
	stopIdle chan struct{}
}

var Cache = &VectorCache{stopIdle: make(chan struct{})}

// OnVectorsUpserted, when set (the candidate cache sets it), runs once per project
// after Upsert commits vectors for it, so cached rankings that predate them are dropped.
var OnVectorsUpserted func(projectPath string)

func init() {
	go Cache.idleLoop()
}

func (vc *VectorCache) ensureLoaded() {
	if db.IndexReadQuiesced() {
		return
	}
	vc.mu.RLock()
	if vc.loaded {
		vc.lastUsed = time.Now()
		vc.mu.RUnlock()
		return
	}
	vc.mu.RUnlock()

	vc.mu.Lock()
	defer vc.mu.Unlock()
	if vc.loaded {
		vc.lastUsed = time.Now()
		return
	}
	if db.IndexReadQuiesced() || db.IndexDB == nil {
		return
	}
	vc.loadFromDB()
}

func (vc *VectorCache) loadFromDB() {
	conn, err := db.IndexReader()
	if err != nil {
		return
	}
	rows, err := conn.Query(selectAllVectorsQuery)
	if err != nil {
		logger.Warn("Failed to load vectors", "error", err)
		return
	}
	defer rows.Close()

	var entries []VectorEntry
	for rows.Next() {
		var e VectorEntry
		var blob []byte
		rows.Scan(&e.ID, &e.SymbolID, &e.ContentHash, &blob, &e.DocType, &e.SourceFile, &e.Name, &e.Kind, &e.ProjectPath)
		e.Vector = blobToFloat32(blob)
		if len(e.Vector) == VectorDims {
			entries = append(entries, e)
		}
	}

	vc.entries = entries
	vc.loaded = true
	vc.lastUsed = time.Now()
	logger.Info("Loaded vectors into memory", "vectors", len(entries), "mb", math.Round(float64(len(entries)*VectorDims*4)/(1024*1024)*10)/10)
}

func (vc *VectorCache) Unload() {
	vc.mu.Lock()
	defer vc.mu.Unlock()
	if !vc.loaded {
		return
	}
	n := len(vc.entries)
	vc.entries = nil
	vc.loaded = false
	logger.Info("Vector cache unloaded", "entries_freed", n)
	realtime.Notify(realtime.IndexHealth)
}

func (vc *VectorCache) idleTimeout() time.Duration {
	if !db.PoolsReady() {
		return time.Minute
	}
	val := db.GetSetting("idle_unload_minutes", "1")
	mins, err := strconv.Atoi(val)
	if err != nil || mins < 0 {
		mins = 1
	}
	if mins == 0 {
		return 0
	}
	d := time.Duration(mins) * time.Minute
	// Tiered "warm": keep vectors longer when any project is pinned (reduces reload churn).
	if db.PinnedProjectCount() > 0 {
		d *= 3
	}
	return d
}

func (vc *VectorCache) idleLoop() {
	ticker := time.NewTicker(15 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			vc.idleTick()
		case <-vc.stopIdle:
			return
		}
	}
}

// idleTick unloads the cache once it has sat unused past idleTimeout.
func (vc *VectorCache) idleTick() {
	// Nothing loaded means nothing to unload, so don't touch the db. This
	// loop starts in init() and runs in every binary that imports search,
	// while tests open and close db's package-global pools with no lock
	// this loop could share.
	vc.mu.RLock()
	loaded := vc.loaded
	vc.mu.RUnlock()
	if !loaded {
		return
	}
	timeout := vc.idleTimeout()
	if timeout == 0 {
		return
	}
	vc.mu.Lock()
	if vc.loaded && time.Since(vc.lastUsed) > timeout {
		n := len(vc.entries)
		vc.entries = nil
		vc.loaded = false
		logger.Info("Vector cache unloaded after idle timeout", "timeout", timeout, "entries_freed", n)
		vc.mu.Unlock()
		realtime.Notify(realtime.IndexHealth)
		return
	}
	vc.mu.Unlock()
}

func (vc *VectorCache) Load() error {
	vc.ensureLoaded()
	return nil
}

func (vc *VectorCache) Search(query []float32, projectPath string, docType string, limit int, filters *SearchFilters) []ScoredResult {
	if len(query) != VectorDims {
		return nil
	}
	vc.ensureLoaded()
	var scope projectlinks.ScopeSet
	if projectPath != "" {
		scope = projectlinks.ResolveScopeSet(projectPath)
	}
	results := vc.topMatches(query, projectPath, scope, docType, limit, filters)
	// symbolRowFromEntry hits index.db per result; run it after RUnlock so pool waits
	// can't pin vc.mu (a queued Upsert writer would then block every RLock caller).
	out := make([]ScoredResult, len(results))
	for i, r := range results {
		startLine, endLine, fqn := symbolRowFromEntry(r.entry)
		data := symbolResult(r.entry.Name, r.entry.Kind, r.entry.SourceFile, fqn, startLine, endLine)
		data["similarity"] = r.sim
		data["content_hash"] = r.entry.ContentHash
		out[i] = ScoredResult{Data: data, Score: r.sim}
	}
	return out
}

type scoredEntry struct {
	entry VectorEntry
	sim   float64
}

// topMatches scores entries under RLock and returns the top-limit copies. It must not touch
// the database: scope is resolved by the caller before the lock is taken. Candidates are
// tracked by index so a full-cache scan doesn't copy every VectorEntry.
func (vc *VectorCache) topMatches(query []float32, projectPath string, scope projectlinks.ScopeSet, docType string, limit int, filters *SearchFilters) []scoredEntry {
	vc.mu.RLock()
	defer vc.mu.RUnlock()
	type scoredIdx struct {
		idx int
		sim float64
	}
	var results []scoredIdx
	for i := range vc.entries {
		e := &vc.entries[i]
		if e.DocType == "doc" {
			if docType != "doc" {
				continue
			}
		} else if projectPath != "" && !scope.Contains(e.ProjectPath) {
			continue
		}
		if docType != "" && e.DocType != docType {
			continue
		}
		if filters != nil && !filters.Empty() && e.DocType != "doc" {
			if !filters.MatchesSymbol(e.SourceFile, e.Kind, projectPath) {
				continue
			}
		}
		results = append(results, scoredIdx{idx: i, sim: cosineSimilarity(query, e.Vector)})
	}
	// Sort whatever the length, so callers always get LessVector order.
	sort.SliceStable(results, func(i, j int) bool {
		return LessVector(&vc.entries[results[i].idx], results[i].sim, &vc.entries[results[j].idx], results[j].sim)
	})
	results = results[:min(max(limit, 0), len(results))]
	out := make([]scoredEntry, len(results))
	for i, r := range results {
		out[i] = scoredEntry{entry: vc.entries[r.idx], sim: r.sim}
	}
	return out
}

// rankEntries sorts results into LessVector order, whatever their count, and
// keeps the first limit.
func rankEntries(results []scoredEntry, limit int) []scoredEntry {
	sort.SliceStable(results, func(i, j int) bool {
		return LessVector(&results[i].entry, results[i].sim, &results[j].entry, results[j].sim)
	})
	return results[:min(max(limit, 0), len(results))]
}

// symbolRowFromEntry returns the lines and fqn of the symbol a vector was
// embedded from.
func symbolRowFromEntry(e VectorEntry) (start, end int, fqn string) {
	conn, err := db.IndexReader()
	if err != nil {
		return start, end, fqn
	}
	if e.SymbolID > 0 {
		conn.QueryRow(selectSymbolRowByIDQuery, e.SymbolID).Scan(&start, &end, &fqn)
	}
	if start == 0 {
		conn.QueryRow(selectSymbolRowByNameQuery, e.SourceFile, e.Name, e.ProjectPath).Scan(&start, &end, &fqn)
	}
	return start, end, fqn
}

func (vc *VectorCache) Upsert(entries []VectorEntry) error {
	vc.ensureLoaded()
	err := db.IndexWrite(func(tx *sql.Tx) error {
		stmt, err := tx.Prepare(upsertVectorQuery)
		if err != nil {
			return err
		}
		defer stmt.Close()
		for _, e := range entries {
			blob := float32ToBlob(e.Vector)
			if _, err := stmt.Exec(e.ContentHash, blob, e.DocType, e.SourceFile, e.Name, e.Kind, e.ProjectPath, e.SymbolID); err != nil {
				return errs.WrapMessage("failed to insert vector", err, "name", e.Name)
			}
		}
		return nil
	})
	if err != nil {
		return err
	}
	vc.upsertMemory(entries)
	if OnVectorsUpserted != nil {
		notified := map[string]bool{}
		for _, e := range entries {
			if e.ProjectPath != "" && !notified[e.ProjectPath] {
				notified[e.ProjectPath] = true
				OnVectorsUpserted(e.ProjectPath)
			}
		}
	}
	return nil
}

func (vc *VectorCache) upsertMemory(entries []VectorEntry) {
	vc.mu.Lock()
	defer vc.mu.Unlock()
	hashMap := make(map[string]int, len(vc.entries))
	for i, e := range vc.entries {
		hashMap[e.ContentHash+"|"+e.ProjectPath] = i
	}
	for _, e := range entries {
		key := e.ContentHash + "|" + e.ProjectPath
		if idx, ok := hashMap[key]; ok {
			vc.entries[idx] = e
		} else {
			vc.entries = append(vc.entries, e)
		}
	}
}

// SearchDoc returns top doc-section vector matches (doc_type=doc only).
func (vc *VectorCache) SearchDoc(query []float32, limit int) []ScoredResult {
	if len(query) != VectorDims {
		return nil
	}
	vc.ensureLoaded()
	vc.mu.RLock()
	defer vc.mu.RUnlock()
	var results []scoredEntry
	for _, e := range vc.entries {
		if e.DocType != "doc" {
			continue
		}
		results = append(results, scoredEntry{entry: e, sim: cosineSimilarity(query, e.Vector)})
	}
	results = rankEntries(results, limit)
	out := make([]ScoredResult, len(results))
	for i, r := range results {
		out[i] = ScoredResult{
			Data: map[string]interface{}{
				"name":       r.entry.Name,
				"kind":       r.entry.Kind,
				"file":       r.entry.SourceFile,
				"similarity": r.sim,
				"doc_id":     docEntryIDFromSource(r.entry.SourceFile),
				"doc_type":   "doc",
			},
			Score: r.sim,
		}
	}
	return out
}

// SearchNote returns top note vector matches (doc_type=note only), optionally filtered by session_id in project_path.
func (vc *VectorCache) SearchNote(query []float32, sessionID string, limit int) []ScoredResult {
	if len(query) != VectorDims {
		return nil
	}
	vc.ensureLoaded()
	vc.mu.RLock()
	defer vc.mu.RUnlock()
	var results []scoredEntry
	for _, e := range vc.entries {
		if e.DocType != "note" {
			continue
		}
		if sessionID != "" && e.ProjectPath != sessionID {
			continue
		}
		results = append(results, scoredEntry{entry: e, sim: cosineSimilarity(query, e.Vector)})
	}
	results = rankEntries(results, limit)
	out := make([]ScoredResult, len(results))
	for i, r := range results {
		ref := strings.TrimPrefix(r.entry.SourceFile, "note:")
		out[i] = ScoredResult{
			Data: map[string]interface{}{
				"ref":        ref,
				"name":       r.entry.Name,
				"similarity": r.sim,
				"doc_type":   "note",
			},
			Score: r.sim,
		}
	}
	return out
}

// SearchMemory returns top structured-memory vector matches (doc_type=memory),
// best first. Memory vectors carry the storing session in ProjectPath. With
// includeSessionless (the caller's scope is not session-only) no session filter
// applies and up to max(limit*5, memoryCandidatePool) candidates come back, since
// the caller's SQL re-select is what scopes them. Otherwise only vectors stored by
// sessionID (any session when it is empty) pass, up to limit. Callers must still
// re-check validity and scope against the rows.
func (vc *VectorCache) SearchMemory(query []float32, sessionID string, includeSessionless bool, limit int) []ScoredResult {
	if len(query) != VectorDims {
		return nil
	}
	vc.ensureLoaded()
	vc.mu.RLock()
	defer vc.mu.RUnlock()
	var results []scoredEntry
	for _, e := range vc.entries {
		if e.DocType != "memory" {
			continue
		}
		if !includeSessionless && (e.ProjectPath == "" || (sessionID != "" && e.ProjectPath != sessionID)) {
			continue
		}
		results = append(results, scoredEntry{entry: e, sim: cosineSimilarity(query, e.Vector)})
	}
	if includeSessionless {
		limit = max(limit*5, memoryCandidatePool)
	}
	results = rankEntries(results, limit)
	out := make([]ScoredResult, len(results))
	for i, r := range results {
		ref := strings.TrimPrefix(r.entry.SourceFile, "mem:")
		out[i] = ScoredResult{
			Data: map[string]interface{}{
				"ref":        ref,
				"name":       r.entry.Name,
				"similarity": r.sim,
				"doc_type":   "memory",
			},
			Score: r.sim,
		}
	}
	return out
}

// DeleteRefs deletes the docType vectors stored under sourceFiles (note and
// memory vectors are keyed "note:<ref>" / "mem:<ref>") in one index write, then
// drops them from memory. It fails rather than skipping the delete while the
// index is quiesced, so callers must delete the vectors before the rows that own
// them: a failure then leaves both in place for a retry instead of orphaning the
// vectors.
func (vc *VectorCache) DeleteRefs(docType string, sourceFiles []string) error {
	if len(sourceFiles) == 0 {
		return nil
	}
	err := db.IndexWrite(func(tx *sql.Tx) error {
		for _, f := range sourceFiles {
			if _, err := tx.Exec(deleteVectorRefQuery, docType, f); err != nil {
				return errs.WrapMessage("failed to delete vector", err, "doc_type", docType, "source_file", f)
			}
		}
		return nil
	})
	if err != nil {
		return err
	}
	vc.DeleteBySourceFiles(docType, sourceFiles)
	return nil
}

func docEntryIDFromSource(sourceFile string) int {
	var sourceID, entryID int
	if _, err := fmt.Sscanf(sourceFile, "doc:%d:%d", &sourceID, &entryID); err != nil {
		return 0
	}
	return entryID
}

// DeleteDocByPrefix deletes the doc vectors whose source file matches the LIKE
// pattern prefix (e.g. "doc:12:%"), then drops them from memory. Like DeleteRefs
// it fails while the index is quiesced instead of skipping the delete.
func (vc *VectorCache) DeleteDocByPrefix(prefix string) error {
	err := db.IndexWrite(func(tx *sql.Tx) error {
		_, err := tx.Exec(deleteDocVectorsLikeQuery, prefix)
		return err
	})
	if err != nil {
		return err
	}
	p := strings.TrimSuffix(prefix, "%")
	vc.mu.Lock()
	defer vc.mu.Unlock()
	if !vc.loaded {
		return nil
	}
	n := 0
	for _, e := range vc.entries {
		if e.DocType == "doc" && strings.HasPrefix(e.SourceFile, p) {
			continue
		}
		vc.entries[n] = e
		n++
	}
	vc.entries = vc.entries[:n]
	return nil
}

// PurgeOrphanCodeVectors removes code vectors whose symbol_id no longer exists.
func PurgeOrphanCodeVectors() int {
	conn, err := db.IndexReader()
	if err != nil {
		return 0
	}
	res, err := conn.Exec(deleteOrphanCodeVectorsQuery)
	if err != nil {
		return 0
	}
	n, _ := res.RowsAffected()
	if n > 0 {
		Cache.purgeOrphansFromMemory()
	}
	return int(n)
}

func (vc *VectorCache) purgeOrphansFromMemory() {
	conn, err := db.IndexReader()
	if err != nil {
		return
	}
	rows, err := conn.Query(selectSymbolIDsQuery)
	if err != nil {
		return
	}
	defer rows.Close()
	valid := map[int64]struct{}{}
	for rows.Next() {
		var id int64
		if rows.Scan(&id) == nil {
			valid[id] = struct{}{}
		}
	}
	vc.mu.Lock()
	defer vc.mu.Unlock()
	if !vc.loaded {
		return
	}
	out := vc.entries[:0]
	for _, e := range vc.entries {
		if e.DocType != "" && e.DocType != "code" {
			out = append(out, e)
			continue
		}
		if e.SymbolID > 0 {
			if _, ok := valid[e.SymbolID]; !ok {
				continue
			}
		}
		out = append(out, e)
	}
	vc.entries = out
}

// DeleteByFile drops a file's in-memory vectors. Callers are responsible for
// deleting the matching database rows, in the same index write that replaces or
// removes the file's symbols.
func (vc *VectorCache) DeleteByFile(filePath, projectPath string) {
	vc.mu.Lock()
	defer vc.mu.Unlock()
	if !vc.loaded {
		return
	}
	n := 0
	for _, e := range vc.entries {
		if e.SourceFile == filePath && e.ProjectPath == projectPath {
			continue
		}
		vc.entries[n] = e
		n++
	}
	vc.entries = vc.entries[:n]
}

// DeleteByProject drops every in-memory vector belonging to a project. Callers
// are responsible for deleting the matching database rows.
func (vc *VectorCache) DeleteByProject(projectPath string) {
	vc.mu.Lock()
	defer vc.mu.Unlock()
	if !vc.loaded {
		return
	}
	n := 0
	for _, e := range vc.entries {
		if e.ProjectPath == projectPath {
			continue
		}
		vc.entries[n] = e
		n++
	}
	vc.entries = vc.entries[:n]
}

// DeleteBySourceFiles drops in-memory vectors of docType whose source file is in
// sourceFiles. Callers are responsible for deleting the matching database rows.
func (vc *VectorCache) DeleteBySourceFiles(docType string, sourceFiles []string) {
	if len(sourceFiles) == 0 {
		return
	}
	drop := make(map[string]bool, len(sourceFiles))
	for _, f := range sourceFiles {
		drop[f] = true
	}
	vc.mu.Lock()
	defer vc.mu.Unlock()
	if !vc.loaded {
		return
	}
	n := 0
	for _, e := range vc.entries {
		if e.DocType == docType && drop[e.SourceFile] {
			continue
		}
		vc.entries[n] = e
		n++
	}
	vc.entries = vc.entries[:n]
}

func (vc *VectorCache) Count(projectPath string) int {
	var scope projectlinks.ScopeSet
	if projectPath != "" {
		scope = projectlinks.ResolveScopeSet(projectPath)
	}
	vc.mu.RLock()
	if vc.loaded {
		defer vc.mu.RUnlock()
		if projectPath == "" {
			return len(vc.entries)
		}
		count := 0
		for _, e := range vc.entries {
			if scope.Contains(e.ProjectPath) {
				count++
			}
		}
		return count
	}
	vc.mu.RUnlock()
	var count int
	conn, err := db.IndexReader()
	if err != nil {
		return count
	}
	if projectPath == "" {
		conn.QueryRow(countVectorsQuery).Scan(&count)
	} else {
		frag, args := projectlinks.ScopeSQL("", projectPath)
		conn.QueryRow(countScopedVectorsQuery+frag, args...).Scan(&count)
	}
	return count
}

func (vc *VectorCache) MemoryMB() float64 {
	vc.mu.RLock()
	defer vc.mu.RUnlock()
	if !vc.loaded {
		return 0
	}
	return float64(len(vc.entries)*VectorDims*4) / (1024 * 1024)
}

func (vc *VectorCache) IsLoaded() bool {
	vc.mu.RLock()
	defer vc.mu.RUnlock()
	return vc.loaded
}

func (vc *VectorCache) GetAll(projectPath string) []VectorEntry {
	vc.ensureLoaded()
	vc.mu.RLock()
	defer vc.mu.RUnlock()
	if projectPath == "" {
		result := make([]VectorEntry, len(vc.entries))
		copy(result, vc.entries)
		return result
	}
	var result []VectorEntry
	for _, e := range vc.entries {
		if e.ProjectPath == projectPath {
			result = append(result, e)
		}
	}
	return result
}

func ContentHash(text string) string {
	h := sha256.Sum256([]byte(text))
	return fmt.Sprintf("%x", h[:16])
}

func cosineSimilarity(a, b []float32) float64 {
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}

func float32ToBlob(v []float32) []byte {
	buf := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}
	return buf
}

// DecodeVector turns a stored vector blob back into floats (nil when malformed).
func DecodeVector(b []byte) []float32 { return blobToFloat32(b) }

func blobToFloat32(b []byte) []float32 {
	if len(b)%4 != 0 {
		return nil
	}
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return v
}
