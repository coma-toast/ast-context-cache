package search

import (
	"crypto/sha256"
	"database/sql"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

const VectorDims = 768

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
	if db.IndexDB == nil {
		return
	}
	rows, err := db.IndexDB.Query("SELECT id, COALESCE(symbol_id,0), content_hash, vector, COALESCE(doc_type,'code'), COALESCE(source_file,''), COALESCE(name,''), COALESCE(kind,''), COALESCE(project_path,'') FROM vectors")
	if err != nil {
		log.Printf("WARNING: load vectors: %v", err)
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
	log.Printf("Loaded %d vectors into memory (%.1f MB)", len(entries), float64(len(entries)*VectorDims*4)/(1024*1024))
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
	log.Printf("Vector cache unloaded (%d entries freed)", n)
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
			timeout := vc.idleTimeout()
			if timeout == 0 {
				continue
			}
			vc.mu.Lock()
			if vc.loaded && time.Since(vc.lastUsed) > timeout {
				n := len(vc.entries)
				vc.entries = nil
				vc.loaded = false
				log.Printf("Vector cache unloaded after %v idle (%d entries freed)", timeout, n)
				vc.mu.Unlock()
				realtime.Notify(realtime.IndexHealth)
				continue
			}
			vc.mu.Unlock()
		case <-vc.stopIdle:
			return
		}
	}
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

	vc.mu.RLock()
	defer vc.mu.RUnlock()

	type scored struct {
		entry VectorEntry
		sim   float64
	}
	var results []scored

	for _, e := range vc.entries {
		if e.DocType == "doc" {
			if docType != "doc" {
				continue
			}
		} else if projectPath != "" && !projectlinks.ScopeContains(projectPath, e.ProjectPath) {
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
		sim := cosineSimilarity(query, e.Vector)
		results = append(results, scored{entry: e, sim: sim})
	}

	// Partial sort: find top-limit by score
	if len(results) > limit {
		for i := 0; i < limit; i++ {
			maxIdx := i
			for j := i + 1; j < len(results); j++ {
				if results[j].sim > results[maxIdx].sim {
					maxIdx = j
				}
			}
			results[i], results[maxIdx] = results[maxIdx], results[i]
		}
		results = results[:limit]
	}

	out := make([]ScoredResult, len(results))
	for i, r := range results {
		startLine, endLine := symbolLinesFromEntry(r.entry)
		out[i] = ScoredResult{
			Data: map[string]interface{}{
				"name":         r.entry.Name,
				"kind":         r.entry.Kind,
				"file":         r.entry.SourceFile,
				"start_line":   startLine,
				"end_line":     endLine,
				"similarity":   r.sim,
				"content_hash": r.entry.ContentHash,
			},
			Score: r.sim,
		}
	}
	return out
}

func symbolLinesFromEntry(e VectorEntry) (start, end int) {
	if e.SymbolID > 0 {
		db.IndexDB.QueryRow("SELECT COALESCE(start_line,0), COALESCE(end_line,0) FROM symbols WHERE id = ?", e.SymbolID).Scan(&start, &end)
	}
	if start == 0 {
		db.IndexDB.QueryRow(
			"SELECT COALESCE(start_line,0), COALESCE(end_line,0) FROM symbols WHERE file = ? AND name = ? AND project_path = ? ORDER BY start_line LIMIT 1",
			e.SourceFile, e.Name, e.ProjectPath).Scan(&start, &end)
	}
	return start, end
}

func (vc *VectorCache) Upsert(entries []VectorEntry) error {
	vc.ensureLoaded()
	err := db.IndexWrite(func(tx *sql.Tx) error {
		stmt, err := tx.Prepare(`INSERT OR REPLACE INTO vectors (content_hash, vector, doc_type, source_file, name, kind, project_path, symbol_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`)
		if err != nil {
			return err
		}
		defer stmt.Close()
		for _, e := range entries {
			blob := float32ToBlob(e.Vector)
			if _, err := stmt.Exec(e.ContentHash, blob, e.DocType, e.SourceFile, e.Name, e.Kind, e.ProjectPath, e.SymbolID); err != nil {
				return fmt.Errorf("insert vector for %s: %w", e.Name, err)
			}
		}
		return nil
	})
	if err != nil {
		return err
	}
	// Update in-memory cache
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
	return nil
}

// SearchDoc returns top doc-section vector matches (doc_type=doc only).
func (vc *VectorCache) SearchDoc(query []float32, limit int) []ScoredResult {
	if len(query) != VectorDims {
		return nil
	}
	vc.ensureLoaded()
	vc.mu.RLock()
	defer vc.mu.RUnlock()
	type scored struct {
		entry VectorEntry
		sim   float64
	}
	var results []scored
	for _, e := range vc.entries {
		if e.DocType != "doc" {
			continue
		}
		results = append(results, scored{entry: e, sim: cosineSimilarity(query, e.Vector)})
	}
	if len(results) > limit {
		for i := 0; i < limit; i++ {
			maxIdx := i
			for j := i + 1; j < len(results); j++ {
				if results[j].sim > results[maxIdx].sim {
					maxIdx = j
				}
			}
			results[i], results[maxIdx] = results[maxIdx], results[i]
		}
		results = results[:limit]
	}
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
	type scored struct {
		entry VectorEntry
		sim   float64
	}
	var results []scored
	for _, e := range vc.entries {
		if e.DocType != "note" {
			continue
		}
		if sessionID != "" && e.ProjectPath != sessionID {
			continue
		}
		results = append(results, scored{entry: e, sim: cosineSimilarity(query, e.Vector)})
	}
	if len(results) > limit {
		for i := 0; i < limit; i++ {
			maxIdx := i
			for j := i + 1; j < len(results); j++ {
				if results[j].sim > results[maxIdx].sim {
					maxIdx = j
				}
			}
			results[i], results[maxIdx] = results[maxIdx], results[i]
		}
		results = results[:limit]
	}
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

// SearchMemory returns top structured-memory vector matches (doc_type=memory).
func (vc *VectorCache) SearchMemory(query []float32, sessionID string, limit int) []ScoredResult {
	if len(query) != VectorDims {
		return nil
	}
	vc.ensureLoaded()
	vc.mu.RLock()
	defer vc.mu.RUnlock()
	type scored struct {
		entry VectorEntry
		sim   float64
	}
	var results []scored
	for _, e := range vc.entries {
		if e.DocType != "memory" {
			continue
		}
		if sessionID != "" && e.ProjectPath != sessionID && e.ProjectPath != "" {
			continue
		}
		results = append(results, scored{entry: e, sim: cosineSimilarity(query, e.Vector)})
	}
	if len(results) > limit {
		for i := 0; i < limit; i++ {
			maxIdx := i
			for j := i + 1; j < len(results); j++ {
				if results[j].sim > results[maxIdx].sim {
					maxIdx = j
				}
			}
			results[i], results[maxIdx] = results[maxIdx], results[i]
		}
		results = results[:limit]
	}
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

func (vc *VectorCache) DeleteNoteByRef(sourceFile string) {
	db.IndexDB.Exec("DELETE FROM vectors WHERE doc_type = 'note' AND source_file = ?", sourceFile)
	vc.mu.Lock()
	defer vc.mu.Unlock()
	if !vc.loaded {
		return
	}
	n := 0
	for _, e := range vc.entries {
		if e.DocType == "note" && e.SourceFile == sourceFile {
			continue
		}
		vc.entries[n] = e
		n++
	}
	vc.entries = vc.entries[:n]
}

func docEntryIDFromSource(sourceFile string) int {
	var sourceID, entryID int
	if _, err := fmt.Sscanf(sourceFile, "doc:%d:%d", &sourceID, &entryID); err != nil {
		return 0
	}
	return entryID
}

func (vc *VectorCache) DeleteDocByPrefix(prefix string) {
	p := strings.TrimSuffix(prefix, "%")
	db.IndexDB.Exec("DELETE FROM vectors WHERE doc_type = 'doc' AND source_file LIKE ?", prefix)
	vc.mu.Lock()
	defer vc.mu.Unlock()
	if !vc.loaded {
		return
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
}

// PurgeOrphanCodeVectors removes code vectors whose symbol_id no longer exists.
func PurgeOrphanCodeVectors() int {
	if db.IndexDB == nil {
		return 0
	}
	res, err := db.IndexDB.Exec(`
		DELETE FROM vectors
		WHERE COALESCE(doc_type, 'code') = 'code'
		  AND symbol_id > 0
		  AND symbol_id NOT IN (SELECT id FROM symbols)`)
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
	rows, err := db.IndexDB.Query(`SELECT id FROM symbols`)
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

func (vc *VectorCache) DeleteByFile(filePath, projectPath string) {
	db.IndexDB.Exec("DELETE FROM vectors WHERE source_file = ? AND project_path = ?", filePath, projectPath)

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

func (vc *VectorCache) Count(projectPath string) int {
	vc.mu.RLock()
	if vc.loaded {
		defer vc.mu.RUnlock()
		if projectPath == "" {
			return len(vc.entries)
		}
		count := 0
		for _, e := range vc.entries {
			if projectlinks.ScopeContains(projectPath, e.ProjectPath) {
				count++
			}
		}
		return count
	}
	vc.mu.RUnlock()
	var count int
	if projectPath == "" {
		db.IndexDB.QueryRow("SELECT COUNT(*) FROM vectors").Scan(&count)
	} else {
		frag, args := projectlinks.ScopeSQL("", projectPath)
		db.IndexDB.QueryRow("SELECT COUNT(*) FROM vectors WHERE "+frag, args...).Scan(&count)
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
