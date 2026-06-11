package embedder

import (
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

type slowEmbedder struct {
	block time.Duration
	calls atomic.Int32
}

func (s *slowEmbedder) Embed(texts []string) ([][]float32, error) {
	s.calls.Add(1)
	time.Sleep(s.block)
	return nil, errors.New("should not complete")
}

func (s *slowEmbedder) EmbedSingle(text string) ([]float32, error) {
	vecs, err := s.Embed([]string{text})
	if err != nil {
		return nil, err
	}
	return vecs[0], nil
}

func TestConnectivityProbeTimeoutMarksError(t *testing.T) {
	probeTimeout = 50 * time.Millisecond
	defer func() { probeTimeout = 8 * time.Second }()
	MarkReady()
	slow := &slowEmbedder{block: 500 * time.Millisecond}
	runConnectivityProbe(TrackHealth(slow))
	state, _, _ := HealthSnapshot()
	if state != "error" {
		t.Fatalf("state=%q want error", state)
	}
	time.Sleep(600 * time.Millisecond)
	if slow.calls.Load() != 1 {
		t.Fatalf("calls=%d want 1 (stale result ignored)", slow.calls.Load())
	}
	if state, _, _ := HealthSnapshot(); state != "error" {
		t.Fatalf("state after stale completion=%q want error", state)
	}
}
