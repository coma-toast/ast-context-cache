package embedder

import "testing"

func TestEnsureONNXRuntimeIdempotent(t *testing.T) {
	err1 := ensureONNXRuntime()
	err2 := ensureONNXRuntime()
	if err1 != err2 {
		t.Fatalf("errs differ: %v vs %v", err1, err2)
	}
	if err1 != nil {
		t.Skipf("ONNX runtime unavailable in test env: %v", err1)
	}
}
