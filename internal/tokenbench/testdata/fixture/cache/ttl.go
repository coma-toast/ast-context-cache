package cache

import "time"

// TTLCache expires entries a fixed duration after they were written.
type TTLCache struct {
	ttl     time.Duration
	entries map[string]ttlEntry
}

type ttlEntry struct {
	value   any
	expires time.Time
}

// NewTTLCache returns a cache whose entries live for ttl.
func NewTTLCache(ttl time.Duration) *TTLCache {
	return &TTLCache{ttl: ttl, entries: map[string]ttlEntry{}}
}

// Get returns the value under key unless it has expired.
func (c *TTLCache) Get(key string) (any, bool) {
	e, ok := c.entries[key]
	if !ok || time.Now().After(e.expires) {
		return nil, false
	}
	return e.value, true
}

// Put stores value under key with a fresh expiry.
func (c *TTLCache) Put(key string, value any) {
	c.entries[key] = ttlEntry{value: value, expires: time.Now().Add(c.ttl)}
}

// Sweep drops every expired entry.
func (c *TTLCache) Sweep() int {
	n := 0
	for k, e := range c.entries {
		if time.Now().After(e.expires) {
			delete(c.entries, k)
			n++
		}
	}
	return n
}
