package server

import "testing"

func TestRateLimiterAllowDrainsBucket(t *testing.T) {
	r := NewRateLimiter(1)
	if !r.Allow() {
		t.Fatal("first request should pass")
	}
	if r.Allow() {
		t.Fatal("second request should be limited")
	}
}
