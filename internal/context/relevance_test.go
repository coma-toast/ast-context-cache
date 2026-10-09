package context

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/flags"
	"github.com/coma-toast/ast-context-cache/internal/render"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

const widgetsTestGo = `package widgets

func WidgetAlpha() int { return 1 }

func TestWidgetHelpers() {}
`

// precisionReply is the capsule response's results plus its precision fields.
type precisionReply struct {
	Results []map[string]any `json:"results"`
	Deduped int              `json:"deduped"`
	// Withheld, NoMatch and Collapsed are absent when nothing was held back.
	Withheld  *Withheld         `json:"withheld"`
	NoMatch   *render.NoMatch   `json:"no_match"`
	Collapsed []render.Collapse `json:"collapsed"`
	Error     string            `json:"error"`
}

func capsuleWith(t *testing.T, project string, args map[string]any) precisionReply {
	t.Helper()
	r := HandleGetContextWithMeta(args, project)
	var out precisionReply
	require.NoError(t, json.Unmarshal([]byte(r.JSON), &out), r.JSON)
	return out
}

func resultNames(rs []map[string]any) []string {
	var out []string
	for _, r := range rs {
		out = append(out, r["name"].(string))
	}
	return out
}

// setFlag sets a feature flag for one test, restoring the default (on) afterwards.
func setFlag(t *testing.T, key string, on bool) {
	t.Helper()
	require.NoError(t, flags.Set(key, on))
	t.Cleanup(func() { require.NoError(t, flags.Set(key, true)) })
}

// AC5: a query whose top hit is weak returns no results and an explicit no-match.
func TestCapsuleNoMatchOnWeakTop(t *testing.T) {
	project, _ := indexedWidgets(t)
	weak := capsuleWith(t, project, map[string]any{"query": "Widget quantum flux reactor", "mode": "skeleton"})
	assert.Empty(t, weak.Results)
	require.NotNil(t, weak.NoMatch)
	assert.Less(t, weak.NoMatch.BestScore, 0.5)
	assert.NotEmpty(t, weak.NoMatch.Hint)
	require.NotNil(t, weak.Withheld, "the weak candidates are reported as withheld")
	assert.Equal(t, 2, weak.Withheld.Count)
	assert.Positive(t, weak.Withheld.Tokens)

	strong := capsuleWith(t, project, map[string]any{"query": "WidgetAlpha", "mode": "skeleton"})
	assert.Nil(t, strong.NoMatch)
	assert.Contains(t, resultNames(strong.Results), "WidgetAlpha")

	setFlag(t, flags.KeyRelevanceFloor, false)
	off := capsuleWith(t, project, map[string]any{"query": "Widget quantum flux reactor", "mode": "skeleton"})
	assert.Nil(t, off.NoMatch, "the floor is off with its flag")
	assert.Len(t, off.Results, 2)
}

// AC6: test-file hits fold into the collapsed list under the result they shadow, or into the
// tests group.
func TestCollapseTestHits(t *testing.T) {
	project, _ := indexedWidgets(t)
	writeAndIndex(t, filepath.Join(project, "widgets_test.go"), project, widgetsTestGo)
	got := capsuleWith(t, project, map[string]any{"query": "Widget", "mode": "skeleton", "min_relative_score": 0.01})
	assert.ElementsMatch(t, []string{"WidgetAlpha", "WidgetBeta"}, resultNames(got.Results))
	for _, r := range got.Results {
		assert.Equal(t, "widgets.go", r["file"])
	}
	assert.ElementsMatch(t, []render.Collapse{
		{Into: "WidgetAlpha", Paths: []string{"widgets_test.go"}, Count: 1},
		{Into: testsGroup, Paths: []string{"widgets_test.go"}, Count: 1},
	}, got.Collapsed)

	kept := capsuleWith(t, project, map[string]any{"query": "Widget", "mode": "skeleton", "min_relative_score": 0.01, "collapse": false})
	assert.Len(t, kept.Results, 4, "collapse=false keeps every hit")
	assert.Empty(t, kept.Collapsed)
}

// AC6: a query about tests keeps the test-file hits.
func TestCollapseSkippedForTestQueries(t *testing.T) {
	project, _ := indexedWidgets(t)
	writeAndIndex(t, filepath.Join(project, "widgets_test.go"), project, widgetsTestGo)
	got := capsuleWith(t, project, map[string]any{"query": "TestWidgetHelpers", "mode": "skeleton", "min_relative_score": 0.01})
	assert.Contains(t, resultNames(got.Results), "TestWidgetHelpers")
	assert.Empty(t, got.Collapsed)
}

// AC7: auto mode returns full source for the top three delivered hits and skeletons for the rest.
func TestEffectiveModeAutoTopThree(t *testing.T) {
	dbtest.Init(t)
	project := t.TempDir()
	src := "package widgets\n"
	for i := range 6 {
		src += fmt.Sprintf("\nfunc Widget%d() int {\n\treturn %d\n}\n", i, i)
	}
	writeAndIndex(t, filepath.Join(project, "widgets.go"), project, src)
	got := capsuleWith(t, project, map[string]any{"query": "Widget", "min_relative_score": 0.01})
	require.Len(t, got.Results, 6)
	for i, r := range got.Results {
		if i < autoFullCount {
			assert.Contains(t, r, "source", "rank %d", i)
			assert.NotContains(t, r, "skeleton", "rank %d", i)
		} else {
			assert.Contains(t, r, "skeleton", "rank %d", i)
			assert.NotContains(t, r, "source", "rank %d", i)
		}
	}
}

// AC8: capsule mode=edit returns the top hit's exact source and its callees' skeletons; the
// other hits are withheld.
func TestEditModeTargetAndCallees(t *testing.T) {
	dbtest.Init(t)
	project := t.TempDir()
	writeAndIndex(t, filepath.Join(project, "widgets.go"), project, editWidgetsGo)
	writeAndIndex(t, filepath.Join(project, "other.go"), project, editOtherGo)
	sid := t.Name()
	got := capsuleWith(t, project, map[string]any{"query": "Target", "mode": "edit", "session_id": sid, "min_relative_score": 0.01})
	require.NotEmpty(t, got.Results)
	target := got.Results[0]
	assert.Equal(t, "Target", target["name"])
	assert.Equal(t, "edit", target["mode"])
	assert.Contains(t, target["source"], "func Target(w *Widget) int")
	assert.Equal(t, []string{"helper", "Beta", "Target2"}, resultNames(got.Results[1:]))
	for _, c := range got.Results[1:] {
		assert.Equal(t, "skeleton", c["mode"])
		assert.NotContains(t, c, "source")
	}
	require.NotNil(t, got.Withheld, "the other hits are withheld")
	assert.Positive(t, got.Withheld.Count)
	modes := ReturnedModes(sid)
	assert.Equal(t, "edit", modes[SymbolDedupKey(filepath.Join(project, "widgets.go"), "Target", 9)])

	setFlag(t, flags.KeyModeV2, false)
	off := capsuleWith(t, project, map[string]any{"query": "Target", "mode": "edit", "min_relative_score": 0.01})
	require.NotEmpty(t, off.Results)
	assert.Contains(t, off.Results[0], "source", "edit falls back to full with the flag off")
	assert.NotContains(t, off.Results[0], "mode")
}

// PR-6: a hit over the budget is skipped, not the end of packing; five misses in a row end it.
func TestPackHitsSkipsOverBudget(t *testing.T) {
	dbtest.Init(t)
	project := t.TempDir()
	file := filepath.Join(project, "sizes.go")
	src := "package sizes\n\nfunc Big() string {\n\treturn \"" + strings.Repeat("x", 4000) + "\"\n}\n\nfunc Small() int { return 1 }\n"
	writeAndIndex(t, file, project, src)
	big := func() search.ScoredResult {
		return search.ScoredResult{Score: 1, Data: map[string]any{"name": "Big", "kind": "function", "file": file, "start_line": 3, "end_line": 5}}
	}
	tests := []struct {
		name         string
		bigs         int
		wantResults  []string
		wantWithheld int
	}{
		{name: "one miss then a fit", bigs: 1, wantResults: []string{"Small"}, wantWithheld: 1},
		{name: "four misses then a fit", bigs: 4, wantResults: []string{"Small"}, wantWithheld: 4},
		{name: "five misses stop packing", bigs: 5, wantWithheld: 6},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var scored []search.ScoredResult
			for range tt.bigs {
				scored = append(scored, big())
			}
			scored = append(scored, search.ScoredResult{Score: 1, Data: map[string]any{"name": "Small", "kind": "function", "file": file, "start_line": 7, "end_line": 7}})
			p := packHits(scored, project, "full", "", 200, map[string][]string{})
			assert.Equal(t, tt.wantResults, resultNames(p.Results))
			assert.Equal(t, tt.wantWithheld, p.Withheld.Count)
			assert.Positive(t, p.Withheld.Tokens)
		})
	}
}

// Withheld counts the hits the relevance floor and the token budget held back.
func TestCapsuleWithheldCounts(t *testing.T) {
	dbtest.Init(t)
	project := t.TempDir()
	src := "package widgets\n"
	for i := range 8 {
		src += fmt.Sprintf("\nfunc Widget%d() int {\n\treturn %d\n}\n", i, i)
	}
	writeAndIndex(t, filepath.Join(project, "widgets.go"), project, src)
	all := capsuleWith(t, project, map[string]any{"query": "Widget", "mode": "skeleton", "min_relative_score": 0.01})
	require.Len(t, all.Results, 8)
	assert.Nil(t, all.Withheld)
	budgeted := capsuleWith(t, project, map[string]any{"query": "Widget", "mode": "skeleton", "min_relative_score": 0.01, "token_budget": float64(60)})
	require.NotEmpty(t, budgeted.Results)
	require.NotNil(t, budgeted.Withheld)
	assert.Equal(t, 8, len(budgeted.Results)+budgeted.Withheld.Count)
	assert.Positive(t, budgeted.Withheld.Tokens)
}

// MO-4: a symbol returned as a skeleton is sent again when full source is asked for, and a
// full delivery covers a later skeleton request. With mode v2 off any delivery covers it.
func TestDedupIsModeAware(t *testing.T) {
	project, _ := indexedWidgets(t)
	sid := t.Name()
	args := func(mode, session string) map[string]any {
		return map[string]any{"query": "WidgetAlpha", "mode": mode, "session_id": session, "min_relative_score": 0.01}
	}
	skel := capsuleWith(t, project, args("skeleton", sid))
	require.Contains(t, resultNames(skel.Results), "WidgetAlpha")
	full := capsuleWith(t, project, args("full", sid))
	assert.Contains(t, resultNames(full.Results), "WidgetAlpha", "a skeleton does not cover a full request")
	assert.Zero(t, full.Deduped)
	again := capsuleWith(t, project, args("skeleton", sid))
	assert.Empty(t, again.Results, "full covers skeleton")
	assert.Positive(t, again.Deduped)
	for _, mode := range ReturnedModes(sid) {
		assert.Equal(t, "full", mode, "the richest delivered mode is kept")
	}

	setFlag(t, flags.KeyModeV2, false)
	legacy := t.Name() + "-legacy"
	capsuleWith(t, project, args("skeleton", legacy))
	assert.Empty(t, capsuleWith(t, project, args("full", legacy)).Results, "flag off: any delivery dedups")
}
