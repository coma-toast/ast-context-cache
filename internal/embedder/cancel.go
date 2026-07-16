package embedder

import (
	"context"
	"sync"
)

type requestCanceler struct {
	mu     sync.Mutex
	ctx    context.Context
	cancel context.CancelFunc
}

func newRequestCanceler() *requestCanceler {
	ctx, cancel := context.WithCancel(context.Background())
	return &requestCanceler{ctx: ctx, cancel: cancel}
}

func (c *requestCanceler) Context() context.Context {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.ctx
}

func (c *requestCanceler) CancelInFlight() {
	c.mu.Lock()
	c.cancel()
	c.ctx, c.cancel = context.WithCancel(context.Background())
	c.mu.Unlock()
}
