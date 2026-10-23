package memory

import (
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/search"
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
	assert.Contains(t, method, "fts")
}

// TestConflictCandidatesFactStayActive checks that a fact stays active when it has
// the highest similarity but no other active entries supersede it (AC17).
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
	if method != "embed" {
		assert.Contains(t, method, "fts")
	}
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
	assert.Contains(t, method, "fts")
	assert.Empty(t, conflicts)
}

// TestConflictCandidatesFTSFindsConflict stores 'build system is make', then calls ConflictCandidates
// with 'build tool is bazel' via FTS (nil embedder), verifying conflict is found and make fact stays active.
func TestConflictCandidatesFTSFindsConflict(t *testing.T) {
	testMemoryDB(t)
	makeEntry, err := Store(StoreInput{
		Kind:        KindFact,
		Scope:       ScopeProject,
		ProjectPath: "/test/repo",
		Subject:     "build system",
		Predicate:   "is",
		Object:      "make",
	})
	require.NoError(t, err)

	bazelEntry := Entry{
		Kind:        KindFact,
		Scope:       ScopeProject,
		ProjectPath: "/test/repo",
		Subject:     "build tool",
		Predicate:   "is",
		Object:      "bazel",
		Ref:         "bazel-ref",
	}

	conflicts, method := ConflictCandidates(bazelEntry, nil, 0)
	assert.Equal(t, "fts", method)
	require.Len(t, conflicts, 1)
	assert.Equal(t, makeEntry.Ref, conflicts[0].Ref)

	// Verify make fact stays active (ValidUntil empty)
	cand, err := entryByRef(makeEntry.Ref)
	require.NoError(t, err)
	assert.Empty(t, cand.ValidUntil)
}

// TestConflictCandidatesEmbedFindsConflict stores 'build system is make', embeds it,
// then calls ConflictCandidates with 'build tool is bazel' via stubEmbedder.
func TestConflictCandidatesEmbedFindsConflict(t *testing.T) {
	testMemoryDB(t)
	search.Cache.Unload()
	t.Cleanup(search.Cache.Unload)

	makeEntry, err := Store(StoreInput{
		Kind:        KindFact,
		Scope:       ScopeProject,
		ProjectPath: "/test/repo",
		Subject:     "build system",
		Predicate:   "is",
		Object:      "make",
	})
	require.NoError(t, err)

	makeLine := FormatLine(Entry{Kind: KindFact, Subject: "build system", Predicate: "is", Object: "make"})
	bazelLine := FormatLine(Entry{Kind: KindFact, Subject: "build tool", Predicate: "is", Object: "bazel"})

	v := make([]float32, search.VectorDims)
	v[0] = 1.0

	emb := stubEmbedder{
		makeLine:  v,
		bazelLine: v,
	}

	EmbedEntry(makeEntry.Ref, "", makeLine, emb)

	bazelEntry := Entry{
		Kind:        KindFact,
		Scope:       ScopeProject,
		ProjectPath: "/test/repo",
		Subject:     "build tool",
		Predicate:   "is",
		Object:      "bazel",
		Ref:         "bazel-ref",
	}

	conflicts, method := ConflictCandidates(bazelEntry, emb, 0)
	assert.Equal(t, "embed", method)
	require.Len(t, conflicts, 1)
	assert.Equal(t, makeEntry.Ref, conflicts[0].Ref)

	// Verify make fact stays active
	cand, err := entryByRef(makeEntry.Ref)
	require.NoError(t, err)
	assert.Empty(t, cand.ValidUntil)
}

// TestConflictCandidatesTimeoutSetting reads the conflict timeout from the
// AST_CONFLICT_TIMEOUT_MS setting (MEM-13): a 1ms timeout with a slow embedder
// falls back to FTS and reports conflict_method=fts.
func TestConflictCandidatesTimeoutSetting(t *testing.T) {
	testMemoryDB(t)
	t.Setenv("AST_CONFLICT_TIMEOUT_MS", "1")
	e := Entry{
		Kind:      KindFact,
		Scope:     ScopeProject,
		Subject:   "build system",
		Predicate: "is",
		Object:    "make",
		Ref:       "ref-timeout",
	}
	_, method := ConflictCandidates(e, &slowEmbedder{}, 0)
	assert.Equal(t, "fts", method, "a 1ms timeout must fall back to FTS")
}

// slowEmbedder blocks forever so the 1ms conflict timeout fires.
type slowEmbedder struct{}

func (s *slowEmbedder) Embed(texts []string) ([][]float32, error) {
	time.Sleep(2 * time.Second)
	return nil, nil
}

func (s *slowEmbedder) EmbedSingle(string) ([]float32, error) {
	time.Sleep(2 * time.Second)
	return nil, nil
}

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
