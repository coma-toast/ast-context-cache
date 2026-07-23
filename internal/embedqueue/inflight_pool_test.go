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
