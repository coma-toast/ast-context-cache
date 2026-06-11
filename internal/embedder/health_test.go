package embedder

import (
	"errors"
	"testing"
	"time"
)

type stubEmbedder struct {
	err error
}

func (s *stubEmbedder) Embed(texts []string) ([][]float32, error) {
	if s.err != nil {
		return nil, s.err
	}
	out := make([][]float32, len(texts))
	for i := range texts {
		out[i] = []float32{1, 2, 3}
	}
	return out, nil
}

func (s *stubEmbedder) EmbedSingle(text string) ([]float32, error) {
	if s.err != nil {
		return nil, s.err
	}
	return []float32{1, 2, 3}, nil
}

func TestTrackHealthRecordsErrorAndRecovery(t *testing.T) {
	MarkReady()
	stub := &stubEmbedder{err: errors.New("connection refused")}
	tracked := TrackHealth(stub)
	_, err := tracked.Embed([]string{"x"})
	if err == nil {
		t.Fatal("expected error")
	}
	state, _, lastErr := HealthSnapshot()
	if state != "error" || lastErr != "connection refused" {
		t.Fatalf("state=%q err=%q", state, lastErr)
	}
	stub.err = nil
	_, err = tracked.Embed([]string{"y"})
	if err != nil {
		t.Fatal(err)
	}
	state, _, lastErr = HealthSnapshot()
	if state != "degraded" || lastErr != "connection refused" {
		t.Fatalf("after recovery state=%q err=%q want degraded with last error", state, lastErr)
	}
	healthMu.Lock()
	healthRecentErrAt = time.Now().Add(-recentErrorWindow - time.Second)
	healthMu.Unlock()
	state, _, lastErr = HealthSnapshot()
	if state != "ready" || lastErr != "" {
		t.Fatalf("after error window state=%q err=%q", state, lastErr)
	}
}

func TestMarkProbeResultDoesNotClearWorkerError(t *testing.T) {
	prevBackend := ActiveBackend
	ActiveBackend = "onnx"
	defer func() { ActiveBackend = prevBackend }()
	probeFailed.Store(false)
	MarkError(errors.New("worker failed"))
	MarkProbeResult(nil)
	state, _, lastErr := HealthSnapshot()
	if state != "error" || lastErr != "worker failed" {
		t.Fatalf("probe success should not clear worker error: state=%q err=%q", state, lastErr)
	}
}

func TestMarkProbeResultRecoversNetworkBackendFromWorkerError(t *testing.T) {
	prevBackend := ActiveBackend
	ActiveBackend = "docker"
	defer func() { ActiveBackend = prevBackend }()
	probeFailed.Store(false)
	MarkError(errors.New("connection refused"))
	recovered := make(chan struct{}, 1)
	SetOnRecovery(func() { recovered <- struct{}{} })
	MarkProbeResult(nil)
	state, _, lastErr := HealthSnapshot()
	if state != "degraded" || lastErr != "connection refused" {
		t.Fatalf("state=%q err=%q want degraded after connectivity recovery", state, lastErr)
	}
	select {
	case <-recovered:
		t.Fatal("onRecovery should not run until fully ready")
	default:
	}
	healthMu.Lock()
	healthRecentErrAt = time.Now().Add(-recentErrorWindow - time.Second)
	healthMu.Unlock()
	MarkSuccess()
	state, _, lastErr = HealthSnapshot()
	if state != "ready" || lastErr != "" {
		t.Fatalf("state=%q err=%q", state, lastErr)
	}
	select {
	case <-recovered:
	case <-time.After(500 * time.Millisecond):
		t.Fatal("expected onRecovery after probe success on network backend")
	}
}

func TestMarkProbeResultDoesNotClearNonConnectivityWorkerErrorOnDocker(t *testing.T) {
	prevBackend := ActiveBackend
	ActiveBackend = "docker"
	defer func() { ActiveBackend = prevBackend }()
	probeFailed.Store(false)
	MarkError(errors.New("embedding[0] has 384 dimensions (expected 768)"))
	MarkProbeResult(nil)
	state, _, lastErr := HealthSnapshot()
	if state != "error" || lastErr != "embedding[0] has 384 dimensions (expected 768)" {
		t.Fatalf("probe success should not clear non-connectivity worker error: state=%q err=%q", state, lastErr)
	}
}

func TestMarkProbeResultTriggersRecovery(t *testing.T) {
	MarkReady()
	recovered := make(chan struct{}, 1)
	SetOnRecovery(func() { recovered <- struct{}{} })
	MarkProbeResult(errors.New("connection refused"))
	MarkProbeResult(nil)
	state, _, _ := HealthSnapshot()
	if state != "degraded" {
		t.Fatalf("state=%q want degraded after probe reconnect", state)
	}
	healthMu.Lock()
	healthRecentErrAt = time.Now().Add(-recentErrorWindow - time.Second)
	healthMu.Unlock()
	MarkSuccess()
	select {
	case <-recovered:
	case <-time.After(500 * time.Millisecond):
		t.Fatal("expected onRecovery after probe reconnect")
	}
}
