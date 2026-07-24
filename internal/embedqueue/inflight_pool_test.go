package embedqueue

import (
	"sync/atomic"
	"testing"
)

func TestSnapshotInFlightPools(t *testing.T) {
	prevIn, prevPri, prevAux := atomic.LoadInt64(&inFlight), atomic.LoadInt64(&inFlightPrimary), atomic.LoadInt64(&inFlightAux)
	atomic.StoreInt64(&inFlight, 3)
	atomic.StoreInt64(&inFlightPrimary, 1)
	atomic.StoreInt64(&inFlightAux, 2)
	defer func() {
		atomic.StoreInt64(&inFlight, prevIn)
		atomic.StoreInt64(&inFlightPrimary, prevPri)
		atomic.StoreInt64(&inFlightAux, prevAux)
	}()
	s := Snapshot()
	if s.InFlight != 3 || s.InFlightPrimary != 1 || s.InFlightAux != 2 {
		t.Fatalf("snapshot pools: total=%d primary=%d aux=%d", s.InFlight, s.InFlightPrimary, s.InFlightAux)
	}
}

func TestCurrentJobsIncludesPool(t *testing.T) {
	activeMu.Lock()
	prevJobs, prevProj, prevPools := activeJobs, activeProjects, activePools
	activeJobs = map[string]struct{}{}
	activeProjects = map[string]string{}
	activePools = map[string]string{}
	activeMu.Unlock()
	defer func() {
		activeMu.Lock()
		activeJobs, activeProjects, activePools = prevJobs, prevProj, prevPools
		activeMu.Unlock()
	}()

	trackJobStart("/a.go", "/proj", false)
	trackJobStart("/b.go", "/proj", true)
	jobs := CurrentJobs()
	if len(jobs) != 2 {
		t.Fatalf("jobs=%d want 2", len(jobs))
	}
	byFile := map[string]string{}
	for _, j := range jobs {
		byFile[j.File] = j.Pool
	}
	if byFile["/a.go"] != "primary" || byFile["/b.go"] != "aux" {
		t.Fatalf("pools=%v", byFile)
	}
	trackJobEnd("/a.go")
	trackJobEnd("/b.go")
	if len(CurrentJobs()) != 0 {
		t.Fatal("expected empty after end")
	}
}
