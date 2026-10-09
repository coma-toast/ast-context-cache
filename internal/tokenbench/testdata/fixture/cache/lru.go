package cache

import "container/list"

// LRU is a fixed-size least-recently-used cache for catalog lookups.
type LRU struct {
	capacity int
	order    *list.List
	items    map[string]*list.Element
}

type lruEntry struct {
	key   string
	value any
}

// NewLRU returns an empty cache holding at most capacity entries.
func NewLRU(capacity int) *LRU {
	return &LRU{capacity: capacity, order: list.New(), items: map[string]*list.Element{}}
}

// Get returns the cached value and marks it most recently used.
func (c *LRU) Get(key string) (any, bool) {
	el, ok := c.items[key]
	if !ok {
		return nil, false
	}
	c.order.MoveToFront(el)
	return el.Value.(*lruEntry).value, true
}

// Put stores value under key, evicting the oldest entry when full.
func (c *LRU) Put(key string, value any) {
	if el, ok := c.items[key]; ok {
		el.Value.(*lruEntry).value = value
		c.order.MoveToFront(el)
		return
	}
	c.items[key] = c.order.PushFront(&lruEntry{key: key, value: value})
	if c.order.Len() > c.capacity {
		c.evictOldest()
	}
}

func (c *LRU) evictOldest() {
	el := c.order.Back()
	c.order.Remove(el)
	delete(c.items, el.Value.(*lruEntry).key)
}
