package search

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"strconv"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
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
	vc.loadFromDB()
}

func (vc *VectorCache) loadFromDB() {
	rows, err := db.DB.Query("SELECT id, COALESCE(symbol_id,0), content_hash, vector, COALESCE(doc_type,'code'), COALESCE(source_file,''), COALESCE(name,''), COALESCE(kind,''), COALESCE(project_path,'') FROM vectors")
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
}

func (vc *VectorCache) idleTimeout() time.Duration {
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
		if projectPath != "" && e.ProjectPath != projectPath {
			continue
		}
		if docType != "" && e.DocType != docType {
			continue
		}
		if filters != nil && !filters.Empty() {
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
		out[i] = ScoredResult{
			Data: map[string]interface{}{
				"name":         r.entry.Name,
				"kind":         r.entry.Kind,
				"file":         r.entry.SourceFile,
				"similarity":   r.sim,
				"content_hash": r.entry.ContentHash,
			},
			Score: r.sim,
		}
	}
	return out
}

func (vc *VectorCache) Upsert(entries []VectorEntry) error {
	vc.ensureLoaded()
	tx, err := db.DB.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	stmt, err := tx.Prepare(`INSERT OR REPLACE INTO vectors (content_hash, vector, doc_type, source_file, name, kind, project_path, symbol_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		return err
	}
	defer stmt.Close()

	for _, e := range entries {
		blob := float32ToBlob(e.Vector)
		_, err := stmt.Exec(e.ContentHash, blob, e.DocType, e.SourceFile, e.Name, e.Kind, e.ProjectPath, e.SymbolID)
		if err != nil {
			return fmt.Errorf("insert vector for %s: %w", e.Name, err)
		}
	}

	if err := tx.Commit(); err != nil {
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

func (vc *VectorCache) DeleteByFile(filePath, projectPath string) {
	db.DB.Exec("DELETE FROM vectors WHERE source_file = ? AND project_path = ?", filePath, projectPath)

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
			if e.ProjectPath == projectPath {
				count++
			}
		}
		return count
	}
	vc.mu.RUnlock()
	var count int
	if projectPath == "" {
		db.DB.QueryRow("SELECT COUNT(*) FROM vectors").Scan(&count)
	} else {
		db.DB.QueryRow("SELECT COUNT(*) FROM vectors WHERE project_path = ?", projectPath).Scan(&count)
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
