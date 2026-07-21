package embedqueue

import "testing"

func TestMaybeQuietOnWorkersPausedNoopWhenNonZero(t *testing.T) {
	maybeQuietOnWorkersPaused(3) // must not block or panic
}

func TestRunQuietPeriodDoesNotPanic(t *testing.T) {
	runQuietPeriod("test")
}
