package handoff

import (
	"context"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/contextnotes"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/memory"
)

// tokensOf returns text of exactly n o200k tokens ("word", n-2 " word", "."), with no
// surrounding space for Store to trim.
func tokensOf(n int) string {
	return "word" + strings.Repeat(" word", n-2) + "."
}

func TestCompleteTruncatesChildSummary(t *testing.T) {
	s := newFanInService(t)
	h := seedHandoff(t, handoffSeed{root: "parent", label: "auth", children: 1})
	child := h.children[0]
	content := tokensOf(2000)
	wake := s.waiters.wait(h.tree)
	res, err := s.Complete(context.Background(), CompleteRequest{SessionID: child, Content: content, Summary: tokensOf(500)})
	require.NoError(t, err)
	assert.Equal(t, StatusDone, res.Status, "status defaults to done")
	assert.Equal(t, SummarySourceChild, res.SummarySource)
	assert.True(t, res.SummaryTruncated)
	assert.LessOrEqual(t, db.EstimateTokens(res.Summary), 300)
	assert.True(t, strings.HasSuffix(res.Summary, truncationMark))
	assert.GreaterOrEqual(t, res.TokensSaved, 1700, "AC10 return tokens saved")
	assert.Equal(t, h.ref, res.Handoff)
	assert.Equal(t, "[result "+res.ResultRef+" for "+string(h.ref)+"] done — "+res.Summary, res.Stub)
	assert.LessOrEqual(t, db.EstimateTokens(res.Stub), LoadLimits().SummaryMaxTokens+40, "RT-4 stub cap")
	assert.Empty(t, res.SupersededRef)
	assert.True(t, woken(wake), "collect waiters are woken")

	note, err := contextnotes.Peek(res.ResultRef)
	require.NoError(t, err)
	assert.Equal(t, content, note.Content, "the full result is fetchable by ref")
	assert.Equal(t, contextnotes.KindHandoffResult, note.Kind)
	assert.Equal(t, string(child), note.SessionID)
	assert.Equal(t, "result: auth", note.Label)
	assert.JSONEq(t, `{"handoff":"`+string(h.ref)+`","status":"done"}`, note.MetadataJSON)

	assert.Equal(t, StatusDone, childStatus(t, child))
	assert.Equal(t, res.ResultRef, queryString(t, `SELECT result_ref FROM handoff_children WHERE child_session_id = ?`, child))
	assert.Equal(t, res.Summary, queryString(t, `SELECT summary FROM handoff_children WHERE child_session_id = ?`, child))
	assert.Equal(t, "child", queryString(t, `SELECT summary_source FROM handoff_children WHERE child_session_id = ?`, child))
	assert.Equal(t, "done", queryString(t, `SELECT result_status FROM handoff_children WHERE child_session_id = ?`, child))
	assert.Equal(t, 1, count(t, `SELECT summary_truncated FROM handoff_children WHERE child_session_id = ?`, child))
	assert.Equal(t, 2000, count(t, `SELECT tokens_used FROM handoff_trees WHERE tree_id = ?`, h.tree), "the result is charged to the tree")
	assert.Equal(t, 1, count(t, `SELECT entries_used FROM handoff_trees WHERE tree_id = ?`, h.tree))
}

func TestCompleteDerivesSummaryAndPromotesMemory(t *testing.T) {
	s := newFanInService(t)
	h := seedHandoff(t, handoffSeed{root: "parent", children: 1})
	child := h.children[0]
	content := "Traced the retry path end to end.\n\n" +
		"FACT: retry | uses | exponential backoff\n" +
		"The client wraps every call.\n" +
		"- FACT: client timeout is 30s\n"
	res, err := s.Complete(context.Background(), CompleteRequest{
		SessionID: child, Content: content, Status: StatusPartial,
		ChangedFiles: []string{"retry.go"}, OpenQuestions: []string{"jitter?"},
	})
	require.NoError(t, err)
	assert.Equal(t, SummarySourceDerived, res.SummarySource)
	assert.False(t, res.SummaryTruncated)
	assert.Equal(t, "FACT: retry uses exponential backoff\nFACT: client timeout is 30s\n"+
		"Traced the retry path end to end.\nThe client wraps every call.", res.Summary, "facts first, then leading lines")
	assert.Equal(t, StatusPartial, res.Status)
	require.Len(t, res.PromotedMemory, 2)

	recalled, err := memory.Recall(memory.RecallInput{SessionID: string(h.root), Scope: memory.ScopeSession}, nil)
	require.NoError(t, err)
	require.Len(t, recalled.Entries, 2, "the parent's recall finds the promoted facts")
	for _, e := range recalled.Entries {
		assert.Equal(t, res.ResultRef, e.SourceRef)
		assert.Contains(t, res.PromotedMemory, e.Ref)
	}
	childMem, err := memory.ActiveForSession(string(child))
	require.NoError(t, err)
	assert.Empty(t, childMem, "promoted to the parent, not the child")
	note, err := contextnotes.Peek(res.ResultRef)
	require.NoError(t, err)
	assert.JSONEq(t, `{"handoff":"`+string(h.ref)+`","status":"partial","changed_files":["retry.go"],"open_questions":["jitter?"]}`,
		note.MetadataJSON, "RT-8 fields ride in the result metadata")
}

func TestCompleteAgainSupersedes(t *testing.T) {
	s := newFanInService(t)
	h := seedHandoff(t, handoffSeed{root: "parent", children: 1})
	child := h.children[0]
	first, err := s.Complete(context.Background(), CompleteRequest{SessionID: child, Content: "first pass", Summary: "partial", Status: StatusPartial})
	require.NoError(t, err)
	second, err := s.Complete(context.Background(), CompleteRequest{SessionID: child, Content: "second pass", Summary: "all done"})
	require.NoError(t, err)
	assert.Equal(t, first.ResultRef, second.SupersededRef)
	assert.NotEqual(t, first.ResultRef, second.ResultRef)
	assert.Equal(t, StatusDone, childStatus(t, child))
	assert.Equal(t, second.ResultRef, queryString(t, `SELECT result_ref FROM handoff_children WHERE child_session_id = ?`, child))
	assert.Equal(t, 2, count(t, `SELECT COUNT(*) FROM handoff_results WHERE child_session_id = ?`, child))
	assert.Equal(t, first.ResultRef, queryString(t, `SELECT result_ref FROM handoff_results WHERE child_session_id = ? AND superseded_at IS NOT NULL`, child))
	assert.Equal(t, second.ResultRef, queryString(t, `SELECT result_ref FROM handoff_results WHERE child_session_id = ? AND superseded_at IS NULL`, child))
	old, err := contextnotes.Peek(first.ResultRef)
	require.NoError(t, err, "a superseded result stays fetchable (RT-7)")
	assert.Equal(t, "first pass", old.Content)
}

func TestCompleteRejects(t *testing.T) {
	s := newFanInService(t)
	h := seedHandoff(t, handoffSeed{root: "parent", children: 1})
	tests := []struct {
		name string
		req  CompleteRequest
		code errs.Code
	}{
		{"no session", CompleteRequest{Content: "x"}, errs.CodeInvalidInput},
		{"no content", CompleteRequest{SessionID: h.children[0], Content: "  "}, errs.CodeInvalidInput},
		{"open status", CompleteRequest{SessionID: h.children[0], Content: "x", Status: StatusOpen}, errs.CodeInvalidInput},
		{"unknown status", CompleteRequest{SessionID: h.children[0], Content: "x", Status: "great"}, errs.CodeInvalidInput},
		{"not a child", CompleteRequest{SessionID: "stranger", Content: "x"}, CodeHandoffNotFound},
		{"root session", CompleteRequest{SessionID: h.root, Content: "x"}, CodeHandoffNotFound},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := s.Complete(context.Background(), tt.req)
			assert.True(t, errs.HasCode(err, tt.code), "%v", err)
		})
	}
	assert.Equal(t, StatusOpen, childStatus(t, h.children[0]))
}

func TestCompleteOverTreeCapLeavesNothing(t *testing.T) {
	s := newFanInService(t)
	h := seedHandoff(t, handoffSeed{root: "parent", children: 1})
	child := h.children[0]
	require.NoError(t, db.SetSetting(SettingTreeMaxTokens, "50"))
	content := tokensOf(100) + "\nFACT: a is b"
	_, err := s.Complete(context.Background(), CompleteRequest{SessionID: child, Content: content})
	require.True(t, errs.HasCode(err, CodeHandoffTreeLimitExceeded), "%v", err)
	assert.Equal(t, db.EstimateTokens(content), ErrorMap(err)["details"].(map[string]any)["would_add_tokens"])
	assert.Equal(t, StatusOpen, childStatus(t, child))
	assert.Zero(t, count(t, `SELECT COUNT(*) FROM handoff_results WHERE child_session_id = ?`, child))
	assert.Zero(t, count(t, `SELECT COUNT(*) FROM context_notes WHERE session_id = ?`, child), "the stored result is deleted again")
	parentMem, err := memory.ActiveForSession(string(h.root))
	require.NoError(t, err)
	assert.Empty(t, parentMem, "nothing promoted for a failed completion")
	assert.Zero(t, count(t, `SELECT tokens_used FROM handoff_trees WHERE tree_id = ?`, h.tree))
}

func TestTruncateToTokens(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name      string
		in        string
		max       int
		truncated bool
	}{
		{"fits", "short summary", 10, false},
		{"exactly at cap", strings.Repeat("a", 40), 10, false},
		{"words", tokensOf(50), 10, true},
		{"no spaces", strings.Repeat("x", 400), 10, true},
		{"multibyte", strings.Repeat("é", 300), 10, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, truncated := truncateToTokens(tt.in, tt.max)
			assert.Equal(t, tt.truncated, truncated)
			assert.LessOrEqual(t, db.EstimateTokens(got), tt.max)
			assert.True(t, utf8.ValidString(got))
			if !truncated {
				assert.Equal(t, tt.in, got)
				return
			}
			assert.True(t, strings.HasSuffix(got, truncationMark))
			assert.Greater(t, db.EstimateTokens(got), tt.max/2, "a cut keeps most of the cap")
		})
	}
}

const testCountMemoryVectorsQuery = "SELECT COUNT(*) FROM vectors WHERE source_file = ?"

func TestCompletePromotedMemoryIsEmbedded(t *testing.T) {
	s := newFanInService(t)
	s.emb = embedder.NewHashEmbedder(embedder.Dimensions)
	h := seedHandoff(t, handoffSeed{root: "parent", children: 1})
	res, err := s.Complete(context.Background(), CompleteRequest{SessionID: h.children[0], Content: "FACT: retry | uses | exponential backoff"})
	require.NoError(t, err)
	require.Len(t, res.PromotedMemory, 1)
	require.Eventually(t, func() bool {
		var n int
		return db.IndexDB.QueryRow(testCountMemoryVectorsQuery, "mem:"+res.PromotedMemory[0]).Scan(&n) == nil && n == 1
	}, 5*time.Second, 20*time.Millisecond, "promoted memory gets a vector like any other memory")
}
