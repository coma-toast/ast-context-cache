package embedder

import (
	"errors"
	"testing"
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
	if state != "ready" || lastErr != "" {
		t.Fatalf("after recovery state=%q err=%q", state, lastErr)
	}
}
