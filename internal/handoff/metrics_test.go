package handoff

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
)

// metricValue gathers the handoff collectors and returns name's value for the series whose
// labels include want: a counter or gauge value, or a histogram's sample count.
func metricValue(t *testing.T, name string, want map[string]string) float64 {
	t.Helper()
	reg := prometheus.NewRegistry()
	for _, c := range Collectors() {
		require.NoError(t, reg.Register(c))
	}
	families, err := reg.Gather()
	require.NoError(t, err)
	for _, mf := range families {
		if mf.GetName() != name {
			continue
		}
		for _, m := range mf.GetMetric() {
			labels := map[string]string{}
			for _, lp := range m.GetLabel() {
				labels[lp.GetName()] = lp.GetValue()
			}
			match := true
			for k, v := range want {
				match = match && labels[k] == v
			}
			if !match {
				continue
			}
			switch {
			case m.GetCounter() != nil:
				return m.GetCounter().GetValue()
			case m.GetGauge() != nil:
				return m.GetGauge().GetValue()
			case m.GetHistogram() != nil:
				return float64(m.GetHistogram().GetSampleCount())
			}
		}
	}
	return 0
}

// captureLogs points the service logger at a buffer for the test.
func captureLogs(t *testing.T, s *realService) *bytes.Buffer {
	t.Helper()
	var buf bytes.Buffer
	s.logger = slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug}))
	return &buf
}

// logLine returns the first captured line containing msg.
func logLine(t *testing.T, buf *bytes.Buffer, msg string) string {
	t.Helper()
	for _, line := range strings.Split(buf.String(), "\n") {
		if strings.Contains(line, `msg="`+msg+`"`) {
			return line
		}
	}
	t.Fatalf("no %q log line in:\n%s", msg, buf.String())
	return ""
}

func TestLifecycleMetricsAndLogs(t *testing.T) {
	s := newTestService(t)
	logs := captureLogs(t, s)
	before := map[string]float64{
		"created":   metricValue(t, "astcache_handoffs_created_total", nil),
		"opened":    metricValue(t, "astcache_handoff_children_opened_total", nil),
		"resumed":   metricValue(t, "astcache_handoff_children_resumed_total", nil),
		"done":      metricValue(t, "astcache_handoff_children_completed_total", map[string]string{"status": "done"}),
		"tokens":    metricValue(t, "astcache_handoff_tree_tokens", nil),
		"repeat":    metricValue(t, "astcache_handoff_child_searches_total", map[string]string{"repeat": "true"}),
		"nonRepeat": metricValue(t, "astcache_handoff_child_searches_total", map[string]string{"repeat": "false"}),
	}
	created := mustCreate(t, s, CreateRequest{SessionID: "parent-metrics", ProjectPath: "/p/metrics", Label: "metrics"})
	opened := mustOpen(t, s, created.Ref, "")
	_, err := s.Open(context.Background(), OpenRequest{Handoff: created.Ref, SessionID: opened.SessionID})
	require.NoError(t, err)
	_, err = s.Open(context.Background(), OpenRequest{Handoff: created.Ref, SessionID: opened.SessionID, Next: &PageCursor{Section: SectionTrail}})
	require.NoError(t, err)
	assert.Equal(t, 1.0, metricValue(t, "astcache_handoff_open_children", nil))
	assert.Equal(t, 1.0, metricValue(t, "astcache_handoff_open_trees", nil))
	s.countSearch(opened.SessionID, true)
	s.countSearch(opened.SessionID, false)
	complete(t, s, opened.SessionID, StatusDone, "")

	assert.Equal(t, before["created"]+1, metricValue(t, "astcache_handoffs_created_total", nil))
	assert.Equal(t, before["opened"]+1, metricValue(t, "astcache_handoff_children_opened_total", nil))
	assert.Equal(t, before["resumed"]+1, metricValue(t, "astcache_handoff_children_resumed_total", nil), "paging a digest is not a resume")
	assert.Equal(t, before["done"]+1, metricValue(t, "astcache_handoff_children_completed_total", map[string]string{"status": "done"}))
	assert.Equal(t, before["tokens"]+2, metricValue(t, "astcache_handoff_tree_tokens", nil), "observed on create and on complete")
	assert.Equal(t, before["repeat"]+1, metricValue(t, "astcache_handoff_child_searches_total", map[string]string{"repeat": "true"}))
	assert.Equal(t, before["nonRepeat"]+1, metricValue(t, "astcache_handoff_child_searches_total", map[string]string{"repeat": "false"}))
	assert.Zero(t, metricValue(t, "astcache_handoff_open_children", nil), "the completed child is no longer open")

	parent, child := "parent_session=parent-metrics", "child_session="+string(opened.SessionID)
	tree, ref, project := "tree="+string(created.TreeID), "handoff="+string(created.Ref), "project_path=/p/metrics"
	for _, tc := range []struct {
		msg  string
		want []string
	}{
		{msg: "Created handoff", want: []string{tree, ref, parent, project}},
		{msg: "Opened handoff", want: []string{tree, ref, parent, child, project}},
		{msg: "Resumed handoff child", want: []string{tree, ref, parent, child, project}},
		{msg: "Completed handoff child", want: []string{tree, ref, parent, child, project}},
	} {
		t.Run(tc.msg, func(t *testing.T) {
			line := logLine(t, logs, tc.msg)
			for _, w := range tc.want {
				assert.Contains(t, line, w)
			}
		})
	}
	_, err = s.Flush(context.Background(), FlushRequest{TreeID: created.TreeID})
	require.NoError(t, err)
	line := logLine(t, logs, "Flushed handoff tree")
	for _, w := range []string{tree, parent, project} {
		assert.Contains(t, line, w)
	}
}

func TestAbandonExpireMetrics(t *testing.T) {
	s := newTestService(t)
	logs := captureLogs(t, s)
	now := time.Date(2026, 10, 5, 12, 0, 0, 0, time.UTC)
	setClock(t, now)
	abandoned := metricValue(t, "astcache_handoff_children_abandoned_total", nil)
	expired := metricValue(t, "astcache_handoff_trees_expired_total", nil)
	stale := seedTree(t, "root-stale", 1, now.Add(-time.Hour))
	n, err := s.markAbandoned()
	require.NoError(t, err)
	require.Equal(t, 1, n)
	assert.Equal(t, abandoned+1, metricValue(t, "astcache_handoff_children_abandoned_total", nil))
	line := logLine(t, logs, "Marked handoff child abandoned")
	for _, w := range []string{
		"tree=" + string(stale.tree), "handoff=" + string(stale.ref), "parent_session=root-stale",
		"child_session=" + string(stale.children[0]), "project_path=/p", "released_claims=0",
	} {
		assert.Contains(t, line, w)
	}

	setClock(t, now.Add(8*24*time.Hour))
	n, err = s.sweep()
	require.NoError(t, err)
	require.Equal(t, 1, n)
	assert.Equal(t, expired+1, metricValue(t, "astcache_handoff_trees_expired_total", nil))
	line = logLine(t, logs, "Expired handoff tree")
	assert.Contains(t, line, "tree="+string(stale.tree))
	assert.Contains(t, line, "parent_session=root-stale")
}

func TestRepeatSearchRatio(t *testing.T) {
	dbtest.Init(t)
	now := time.Date(2026, 10, 5, 12, 0, 0, 0, time.UTC)
	setClock(t, now)
	assert.Zero(t, RepeatSearchRatio(), "no searches")
	recent := seedTree(t, "root-recent", 2, now.Add(-time.Hour))
	old := seedTree(t, "root-old", 1, now.Add(-25*time.Hour))
	exec(t, `UPDATE handoff_children SET search_calls = 4, repeat_calls = 1 WHERE child_session_id = ?`, recent.children[0])
	exec(t, `UPDATE handoff_children SET search_calls = 6, repeat_calls = 4 WHERE child_session_id = ?`, recent.children[1])
	exec(t, `UPDATE handoff_children SET search_calls = 10, repeat_calls = 10 WHERE child_session_id = ?`, old.children[0])
	assert.InDelta(t, 0.5, RepeatSearchRatio(), 1e-9, "only children active in the last 24h count")
	assert.InDelta(t, 0.5, metricValue(t, "astcache_handoff_repeat_search_ratio", nil), 1e-9)
}

func TestClaimWaitObservedByDefault(t *testing.T) {
	before := metricValue(t, "astcache_handoff_claim_wait_seconds", nil)
	now := time.Date(2026, 10, 5, 12, 0, 0, 0, time.UTC)
	observeClaimWait(now, sqlTime(now.Add(-90*time.Second)))
	assert.Equal(t, before+1, metricValue(t, "astcache_handoff_claim_wait_seconds", nil))
}
