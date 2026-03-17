package cache

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
	"sync"
	"time"
)

type QueryCache struct {
	mu      sync.RWMutex
	entries map[string]*cacheEntry
	maxAge  time.Duration
	maxSize int
}

type cacheEntry struct {
	result    string
	timestamp time.Time
}

var GlobalCache = NewQueryCache(5*time.Minute, 1000)

func NewQueryCache(maxAge time.Duration, maxSize int) *QueryCache {
	c := &QueryCache{
		entries: make(map[string]*cacheEntry),
		maxAge:  maxAge,
		maxSize: maxSize,
	}
	go c.cleanupLoop()
	return c
}

func (c *QueryCache) cleanupLoop() {
	ticker := time.NewTicker(c.maxAge)
	for range ticker.C {
		c.cleanup()
	}
}

func (c *QueryCache) cleanup() {
	c.mu.Lock()
	defer c.mu.Unlock()
	cutoff := time.Now().Add(-c.maxAge)
	for key, entry := range c.entries {
		if entry.timestamp.Before(cutoff) {
			delete(c.entries, key)
		}
	}
}

func (c *QueryCache) Get(key string) (string, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if entry, ok := c.entries[key]; ok {
		if time.Since(entry.timestamp) < c.maxAge {
			return entry.result, true
		}
	}
	return "", false
}

func (c *QueryCache) Set(key, result string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.entries) >= c.maxSize {
		var oldestKey string
		oldestTime := time.Now()
		for k, v := range c.entries {
			if v.timestamp.Before(oldestTime) {
				oldestTime = v.timestamp
				oldestKey = k
			}
		}
		if oldestKey != "" {
			delete(c.entries, oldestKey)
		}
	}
	c.entries[key] = &cacheEntry{
		result:    result,
		timestamp: time.Now(),
	}
}

func HashQuery(query, projectPath, mode string, limit int) string {
	data := map[string]interface{}{
		"query":        query,
		"project_path": projectPath,
		"mode":         mode,
		"limit":        limit,
	}
	bytes, _ := json.Marshal(data)
	hash := sha256.Sum256(bytes)
	return hex.EncodeToString(hash[:])
}

func (c *QueryCache) ClearProject(projectPath string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	prefix := projectPath + ":"
	for key := range c.entries {
		if strings.HasPrefix(key, prefix) {
			delete(c.entries, key)
		}
	}
}

func (c *QueryCache) Stats() (int, int) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.maxSize, len(c.entries)
}

func (c *QueryCache) ClearAll() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.entries = make(map[string]*cacheEntry)
}
