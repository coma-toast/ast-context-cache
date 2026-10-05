package handoff

import (
	"path/filepath"
	"slices"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/trail"
)

// latencyIterations is how many timed calls each budget's p95 is taken over.
const latencyIterations = 200

// latencyBudget is one NFR-1/NFR-2 budget: an operation timed latencyIterations times and the
// p95 it must stay within.
type latencyBudget struct {
	name string
	p95  time.Duration
	// op runs iteration i and returns the time the measured call took, so untimed setup and
	// cleanup can sit around it.
	op func(t *testing.T, i int) time.Duration
}

// timed runs fn and returns how long it took.
func timed(fn func()) time.Duration {
	start := time.Now()
	fn()
	return time.Since(start)
}

// percentile95 is the nearest-rank 95th percentile of d.
func percentile95(d []time.Duration) time.Duration {
	s := slices.Clone(d)
	slices.Sort(s)
	return s[(len(s)*95+99)/100-1]
}

// TestLatencyBudgets checks NFR-1 (and NFR-2's annotation overhead) on a warm local database:
// each operation's p95 over 200 calls, with create and open at a snapshot just under the tree
// cap. Timing budgets are meaningless under -race or on a -short run, so both skip.
func TestLatencyBudgets(t *testing.T) {
	if testing.Short() {
		t.Skip("latency budgets skipped under -short")
	}
	if raceEnabled {
		t.Skip("latency budgets skipped under -race")
	}
	f := newPerfFixture(t)
	createReq, _ := f.capHandoff(t, "lat-create", ModeFresh)
	_, err := f.s.Flush(f.ctx, FlushRequest{SessionID: createReq.SessionID})
	require.NoError(t, err)
	_, freshCap := f.capHandoff(t, "lat-open", ModeFresh)
	_, forkCap := f.capHandoff(t, "lat-open-fork", ModeFork)
	treeChild := mustOpen(t, f.s, freshCap.Ref, f.project).SessionID
	_, padChildren := f.smallTree(t, "lat-pad", 8)
	_, claimChildren := f.smallTree(t, "lat-claim", 2)
	collected := f.completedTree(t, "lat-collect")

	parentEntry := trail.Entry{Tool: "get_context_capsule", Query: "query 7 about the retry backoff path", ProjectPath: f.project}
	parentEntry.QueryNorm = trail.NormalizeQuery(parentEntry.Query)
	ev := SearchEventFor(parentEntry)
	results := make([]map[string]any, 10)
	for i := range results {
		file := filepath.Join(f.project, "pkg", "f"+strconv.Itoa(i)+".go")
		results[i] = map[string]any{"file": file, "name": "Symbol" + strconv.Itoa(i), "start_line": i + 1}
		ev.CandidateKeys = append(ev.CandidateKeys, trail.HitRef("pkg/f"+strconv.Itoa(i)+".go", "Symbol"+strconv.Itoa(i), i+1))
	}

	budgets := []latencyBudget{
		{"create at cap", 250 * time.Millisecond, func(t *testing.T, _ int) time.Duration {
			d := timed(func() {
				_, err := f.s.Create(f.ctx, createReq)
				require.NoError(t, err)
			})
			_, err := f.s.Flush(f.ctx, FlushRequest{SessionID: createReq.SessionID})
			require.NoError(t, err)
			return d
		}},
		{"open fresh at cap", 100 * time.Millisecond, func(t *testing.T, _ int) time.Duration {
			return timed(func() {
				_, err := f.s.Open(f.ctx, OpenRequest{Handoff: freshCap.Ref, ProjectPath: f.project})
				require.NoError(t, err)
			})
		}},
		{"open fork at cap", 100 * time.Millisecond, func(t *testing.T, _ int) time.Duration {
			return timed(func() {
				_, err := f.s.Open(f.ctx, OpenRequest{Handoff: forkCap.Ref, ProjectPath: f.project})
				require.NoError(t, err)
			})
		}},
		{"scratchpad post", 50 * time.Millisecond, func(t *testing.T, i int) time.Duration {
			return timed(func() {
				_, err := f.s.Post(f.ctx, PostRequest{SessionID: padChildren[i%len(padChildren)], Type: EntryTypeFinding, Text: "finding " + strconv.Itoa(i), Refs: []string{"pkg/retry.go"}})
				require.NoError(t, err)
			})
		}},
		{"scratchpad read", 50 * time.Millisecond, func(t *testing.T, i int) time.Duration {
			return timed(func() {
				_, err := f.s.Read(f.ctx, ReadRequest{SessionID: padChildren[i%len(padChildren)]})
				require.NoError(t, err)
			})
		}},
		{"claim", 50 * time.Millisecond, func(t *testing.T, _ int) time.Duration {
			d := timed(func() {
				res, err := f.s.Claim(f.ctx, ClaimRequest{SessionID: claimChildren[0], Key: "pkg/retry.go"})
				require.NoError(t, err)
				require.Equal(t, ClaimGranted, res.Outcome)
			})
			_, err := f.s.Release(f.ctx, ReleaseRequest{SessionID: claimChildren[0], Key: "pkg/retry.go"})
			require.NoError(t, err)
			return d
		}},
		{"release", 50 * time.Millisecond, func(t *testing.T, _ int) time.Duration {
			_, err := f.s.Claim(f.ctx, ClaimRequest{SessionID: claimChildren[0], Key: "pkg/retry.go"})
			require.NoError(t, err)
			return timed(func() {
				res, err := f.s.Release(f.ctx, ReleaseRequest{SessionID: claimChildren[0], Key: "pkg/retry.go"})
				require.NoError(t, err)
				require.True(t, res.Released)
			})
		}},
		{"status at cap", 50 * time.Millisecond, func(t *testing.T, _ int) time.Duration {
			return timed(func() {
				_, err := f.s.Status(f.ctx, StatusRequest{Handoff: freshCap.Ref})
				require.NoError(t, err)
			})
		}},
		{"collect 16", 150 * time.Millisecond, func(t *testing.T, _ int) time.Duration {
			return timed(func() {
				res, err := f.s.Collect(f.ctx, CollectRequest{Handoff: collected.Ref})
				require.NoError(t, err)
				require.Len(t, res.Children, perfChildren)
			})
		}},
		// NFR-2: a session outside every tree pays one in-memory lookup; a tree session's
		// annotations stay within 10 ms.
		{"annotate non-tree", 10 * time.Microsecond, func(t *testing.T, _ int) time.Duration {
			return timed(func() {
				require.Nil(t, f.s.Annotate("lat-loner", ev, results))
			})
		}},
		{"annotate tree", 10 * time.Millisecond, func(t *testing.T, _ int) time.Duration {
			return timed(func() {
				require.NotNil(t, f.s.Annotate(treeChild, ev, results))
			})
		}},
	}
	f.s.Annotate("lat-loner", ev, results) // the first lookup reads the database once
	for _, b := range budgets {
		t.Run(b.name, func(t *testing.T) {
			samples := make([]time.Duration, latencyIterations)
			for i := range samples {
				samples[i] = b.op(t, i)
			}
			p95 := percentile95(samples)
			t.Logf("p95 %v (budget %v, max %v)", p95, b.p95, slices.Max(samples))
			assert.LessOrEqual(t, p95, b.p95)
		})
	}
}
