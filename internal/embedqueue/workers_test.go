package embedqueue

import "testing"

func TestSetWorkerCountValidation(t *testing.T) {
	if _, err := SetWorkerCount(-1); err == nil {
		t.Fatal("expected error for negative workers")
	}
	if _, err := SetWorkerCount(MaxWorkers + 1); err == nil {
		t.Fatal("expected error above max workers")
	}
}
