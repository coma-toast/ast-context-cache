package embedder

import (
	"context"
	"errors"
)

// healthTracker wraps an embedder and records success/failure for dashboard health.
type healthTracker struct {
	inner Interface
}

// TrackHealth returns an Interface that updates embedder health on each call.
func TrackHealth(inner Interface) Interface {
	if inner == nil {
		return nil
	}
	return &healthTracker{inner: inner}
}

func (t *healthTracker) Embed(texts []string) ([][]float32, error) {
	vecs, err := t.inner.Embed(texts)
	switch {
	case err == nil:
		MarkSuccess()
	case !errors.Is(err, context.Canceled):
		MarkError(err)
	}
	return vecs, err
}

func (t *healthTracker) EmbedSingle(text string) ([]float32, error) {
	vec, err := t.inner.EmbedSingle(text)
	switch {
	case err == nil:
		MarkSuccess()
	case !errors.Is(err, context.Canceled):
		MarkError(err)
	}
	return vec, err
}

func (t *healthTracker) CancelInFlight() {
	if c, ok := t.inner.(interface{ CancelInFlight() }); ok {
		c.CancelInFlight()
	}
}
