package server

import (
	"net/http"
	"strings"
	"sync"
	"time"
)

// RateLimiter is a token bucket shared by all requests.
type RateLimiter struct {
	mu     sync.Mutex
	tokens float64
	rps    float64
	last   time.Time
}

// NewRateLimiter returns a limiter that allows rps requests per second.
func NewRateLimiter(rps int) *RateLimiter {
	return &RateLimiter{tokens: float64(rps), rps: float64(rps), last: time.Now()}
}

// Allow refills the bucket and takes one token if one is available.
func (r *RateLimiter) Allow() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	now := time.Now()
	r.tokens += now.Sub(r.last).Seconds() * r.rps
	if r.tokens > r.rps {
		r.tokens = r.rps
	}
	r.last = now
	if r.tokens < 1 {
		return false
	}
	r.tokens--
	return true
}

// Wrap rejects requests with 429 when the bucket is empty.
func (r *RateLimiter) Wrap(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if !r.Allow() {
			http.Error(w, "too many requests", http.StatusTooManyRequests)
			return
		}
		next.ServeHTTP(w, req)
	})
}

// authMiddleware requires a bearer token on every request except the health check.
func authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.URL.Path != "/healthz" && !strings.HasPrefix(req.Header.Get("Authorization"), "Bearer ") {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, req)
	})
}
