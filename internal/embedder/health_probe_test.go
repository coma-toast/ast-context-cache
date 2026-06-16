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

type attemptEmbedder struct {
	failUntil int32
	calls     atomic.Int32
}

func (a *attemptEmbedder) Embed(texts []string) ([][]float32, error) {
	n := a.calls.Add(1)
	if n <= a.failUntil {
		return nil, fmt.Errorf("attempt %d failed", n)
	}
	out := make([][]float32, len(texts))
	for i := range texts {
		out[i] = []float32{1, 2, 3}
	}
	return out, nil
}

func (a *attemptEmbedder) EmbedSingle(text string) ([]float32, error) {
	vec, err := a.Embed([]string{text})
	if err != nil {
		return nil, err
	}
	return vec[0], nil
}

func TestProbeTimeoutIgnoredDuringRecentEmbedActivity(t *testing.T) {
	MarkReady()
	MarkSuccess()
	MarkProbeResult(fmt.Errorf("connectivity probe: timeout after 15s"))
	state, _, lastErr := HealthSnapshot()
	if state != "ready" || lastErr != "" {
		t.Fatalf("probe timeout during activity: state=%q err=%q", state, lastErr)
	}
}

func TestMarkSuccessClearsProbeOnlyError(t *testing.T) {
	MarkReady()
	MarkError(errors.New("connectivity probe: timeout after 15s"))
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
	MarkReady()
	healthMu.Lock()
	healthLastUse = time.Time{}
	healthLastWorkerOK = time.Time{}
	healthMu.Unlock()
	old := probeAttemptTimeouts
	probeAttemptTimeouts = []time.Duration{50 * time.Millisecond}
	defer func() { probeAttemptTimeouts = old }()
	slow := &slowEmbedder{block: 500 * time.Millisecond}
	if runConnectivityProbeCycle(slow) != probeFail {
		t.Fatalf("want cycle probeFail")
	}
	state, _, _ := HealthSnapshot()
	if state != "error" {
		t.Fatalf("state=%q want error", state)
	}
	time.Sleep(600 * time.Millisecond)
	if slow.calls.Load() != 1 {
		t.Fatalf("calls=%d want 1 (stale result ignored)", slow.calls.Load())
	}
}

func TestConnectivityProbeCycleRecoversOnLaterAttempt(t *testing.T) {
	MarkReady()
	healthMu.Lock()
	healthLastUse = time.Time{}
	healthLastWorkerOK = time.Time{}
	healthMu.Unlock()
	old := probeAttemptTimeouts
	probeAttemptTimeouts = []time.Duration{100 * time.Millisecond, 100 * time.Millisecond, 100 * time.Millisecond}
	defer func() { probeAttemptTimeouts = old }()
	emb := &attemptEmbedder{failUntil: 1}
	if runConnectivityProbeCycle(emb) != probeOK {
		t.Fatalf("want recovery on second attempt, calls=%d", emb.calls.Load())
	}
	state, _, lastErr := HealthSnapshot()
	if state != "ready" || lastErr != "" {
		t.Fatalf("state=%q err=%q", state, lastErr)
	}
}

func TestProbeDeferCheckSkipsFailure(t *testing.T) {
	MarkReady()
	healthMu.Lock()
	healthLastUse = time.Time{}
	healthLastWorkerOK = time.Time{}
	healthMu.Unlock()
	oldCheck := probeDeferCheck
	probeDeferCheck = func() bool { return true }
	defer func() { probeDeferCheck = oldCheck }()
	old := probeAttemptTimeouts
	probeAttemptTimeouts = []time.Duration{10 * time.Millisecond}
	defer func() { probeAttemptTimeouts = old }()
	if runConnectivityProbeCycle(&slowEmbedder{block: time.Second}) != probeSkipped {
		t.Fatal("want probeSkipped when defer check active")
	}
	state, _, _ := HealthSnapshot()
	if state != "ready" {
		t.Fatalf("state=%q want ready", state)
	}
}

func TestRecoveryCycleClearsError(t *testing.T) {
	MarkReady()
	MarkError(errors.New("connection refused"))
	if runRecoveryCycle(&stubEmbedder{}) != probeOK {
		t.Fatal("want recovery")
	}
	state, _, lastErr := HealthSnapshot()
	if state != "degraded" && state != "ready" {
		t.Fatalf("state=%q err=%q", state, lastErr)
	}
}

func TestProbeErrorBackoff(t *testing.T) {
	if got := probeErrorBackoff(1); got != probeRecoveryInterval {
		t.Fatalf("probeErrorBackoff=%s want %s", got, probeRecoveryInterval)
	}
	if got := probeErrorBackoff(99); got != probeRecoveryInterval {
		t.Fatalf("probeErrorBackoff=%s want %s", got, probeRecoveryInterval)
	}
}
