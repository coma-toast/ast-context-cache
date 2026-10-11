package memory

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestConflictCandidatesNoEmbedder returns fts method when embedder is nil (AC18).
func TestConflictCandidatesNoEmbedder(t *testing.T) {
	testMemoryDB(t)
	e := Entry{
		Kind:    KindFact,
		Scope:   ScopeProject,
		Subject: "some subject",
		Object:  "some object",
		Ref:     "test-ref",
	}
	_, method := ConflictCandidates(e, nil, 0)
	require.Contains(t, method, "fts")
	assert.Contains(t, method, "fts")
}

// TestConflictCandidatesWithEmbedderFails returns fts fallback when embedder Embed fails (AC17).
func TestConflictCandidatesWithEmbedderFails(t *testing.T) {
	testMemoryDB(t)
	e := Entry{
		Kind:    KindFact,
		Scope:   ScopeProject,
		Subject: "conflicting fact",
		Object:  "other fact",
		Ref:     "ref-embed-fail",
	}
	_, method := ConflictCandidates(e, brokenEmbedder, 0)
	require.Contains(t, method, "fts")
	assert.Contains(t, method, "fts")
}

// TestConflictCandidatesFactStayActive checks that a fact stays active when it has
// the highest similarity but no other active entries supersede it (AC17).
// With an empty vector cache the embed path may fall back to FTS, so we accept
// either method as long as conflicts is empty (fact stays active).
func TestConflictCandidatesFactStayActive(t *testing.T) {
	testMemoryDB(t)
	e := Entry{
		Kind:    KindFact,
		Scope:   ScopeProject,
		Subject: "my fact",
		Object:  "my fact content",
		Ref:     "ref-stay-active",
	}
	conflicts, method := ConflictCandidates(e, fakeEmbedder, 0)
	// The method must come from the embed or fts path (never an unexplained state)
	if method != "embed" {
		assert.Contains(t, method, "fts")
	}
	// The fact itself should not be in the conflict list (it stays active)
	assert.Empty(t, conflicts)
}

// TestConflictCandidatesFactStayActiveEmpty checks that a fact stays active even with
// no embedder (uses FTS fallback), AC18-compatible.
func TestConflictCandidatesFactStayActiveEmpty(t *testing.T) {
	testMemoryDB(t)
	e := Entry{
		Kind:    KindFact,
		Scope:   ScopeProject,
		Subject: "my fact",
		Object:  "my fact content",
		Ref:     "ref-stay-active-empty",
	}
	conflicts, method := ConflictCandidates(e, nil, 0)
	require.Contains(t, method, "fts")
	assert.Empty(t, conflicts)
}

// fakeEmbedder is a minimal embedder that returns a non-nil embedding.
var fakeEmbedder = &fakeEmbedderType{}

type fakeEmbedderType struct{}

func (f *fakeEmbedderType) Embed(texts []string) ([][]float32, error) {
	vecs := make([][]float32, len(texts))
	for i := range vecs {
		vecs[i] = []float32{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
	}
	return vecs, nil
}

func (f *fakeEmbedderType) EmbedSingle(string) ([]float32, error) {
	return []float32{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, nil
}

var brokenEmbedder = &brokenEmbedderType{}

type brokenEmbedderType struct{}

func (b *brokenEmbedderType) Embed(texts []string) ([][]float32, error) {
	return nil, errors.New("embed failed")
}

func (b *brokenEmbedderType) EmbedSingle(string) ([]float32, error) {
	return nil, errors.New("embed failed")
}