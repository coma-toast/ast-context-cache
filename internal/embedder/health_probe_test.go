package embedder

import (
	"errors"
	"fmt"
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

func TestProbeTimeoutIgnoredDuringRecentEmbedActivity(t *testing.T) {
	MarkReady()
	MarkSuccess()
	MarkProbeResult(fmt.Errorf("connectivity probe: timeout after 8s"))
	state, _, lastErr := HealthSnapshot()
	if state != "ready" || lastErr != "" {
		t.Fatalf("probe timeout during activity: state=%q err=%q", state, lastErr)
	}
}

func TestMarkSuccessClearsProbeOnlyError(t *testing.T) {
	MarkReady()
	MarkError(errors.New("connectivity probe: timeout after 8s"))
	state, _, lastErr := HealthSnapshot()
	if state != "error" {
		t.Fatalf("state=%q", state)
	}
	MarkSuccess()
	state, _, lastErr = HealthSnapshot()
	if state != "ready" || lastErr != "" {
		t.Fatalf("after worker success: state=%q err=%q", state, lastErr)
	}
}

func TestConnectivityProbeTimeoutMarksError(t *testing.T) {
	probeTimeout = 50 * time.Millisecond
	defer func() { probeTimeout = 8 * time.Second }()
	MarkReady()
	healthMu.Lock()
	healthLastUse = time.Time{}
	healthLastWorkerOK = time.Time{}
	healthMu.Unlock()
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
