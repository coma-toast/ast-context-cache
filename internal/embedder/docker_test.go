package embedder

import "testing"

func TestNormalizeDMRBase(t *testing.T) {
	tests := []struct {
		in, want string
	}{
		{"", "http://127.0.0.1:12434/engines/v1"},
		{"localhost:12434", "http://localhost:12434/engines/v1"},
		{"http://127.0.0.1:12434", "http://127.0.0.1:12434/engines/v1"},
		{"http://127.0.0.1:12434/engines/v1", "http://127.0.0.1:12434/engines/v1"},
		{"http://127.0.0.1:12434/engines", "http://127.0.0.1:12434/engines/v1"},
	}
	for _, tc := range tests {
		if got := normalizeDMRBase(tc.in); got != tc.want {
			t.Errorf("normalizeDMRBase(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}
