package embedder

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

const (
	probeIntervalOK  = 10 * time.Second
	probeIntervalErr = 5 * time.Second
)

var probeTimeout = 8 * time.Second

var (
	healthMu      sync.RWMutex
	healthState   = "idle" // idle, loading, ready, error
	healthLastErr string
	healthLastUse time.Time
	onRecovery    func()
	onError       func()
)

// SetOnRecovery registers a callback when embedder health recovers from error to ready.
func SetOnRecovery(fn func()) {
	onRecovery = fn
}

// SetOnError registers a callback when an embedding call fails (debounced work should be used).
func SetOnError(fn func()) {
	onError = fn
}

// MarkReady sets embedder state to ready (called when the process wires an embedder).
func MarkReady() {
	healthMu.Lock()
	prev := healthState
	healthState = "ready"
	healthLastUse = time.Now()
	healthLastErr = ""
	healthMu.Unlock()
	if prev != "ready" {
		realtime.Notify(realtime.HealthBar | realtime.IndexHealth)
	}
}

// MarkSuccess records a successful embedding call and clears any error state.
func MarkSuccess() {
	healthMu.Lock()
	prev := healthState
	healthState = "ready"
	healthLastUse = time.Now()
	healthLastErr = ""
	healthMu.Unlock()
	if prev != "ready" {
		realtime.Notify(realtime.HealthBar | realtime.IndexHealth)
	}
	if prev == "error" && onRecovery != nil {
		go onRecovery()
	}
}

// MarkError records an embedding failure and refreshes dashboard health panels.
func MarkError(err error) {
	if err == nil {
		return
	}
	healthMu.Lock()
	healthState = "error"
	healthLastErr = err.Error()
	healthMu.Unlock()
	realtime.Notify(realtime.HealthBar | realtime.IndexHealth)
	if onError != nil {
		go onError()
	}
}

// HealthState returns the current state and time since last successful use.
func HealthState() (state string, lastUse time.Duration) {
	healthMu.RLock()
	defer healthMu.RUnlock()
	return healthState, time.Since(healthLastUse)
}

// HealthError returns the most recent embedding error message, if any.
func HealthError() string {
	healthMu.RLock()
	defer healthMu.RUnlock()
	return healthLastErr
}

// HealthSnapshot returns state, time since last success, and last error text.
func HealthSnapshot() (state string, lastUse time.Duration, lastErr string) {
	healthMu.RLock()
	defer healthMu.RUnlock()
	return healthState, time.Since(healthLastUse), healthLastErr
}

var probeEpoch atomic.Uint64

// StartConnectivityProbe periodically probes network embed backends so a broken
// connection surfaces in the dashboard even when no indexing is in flight.
func StartConnectivityProbe(e Interface) {
	if !IsNetworkBackend(ActiveBackend) || e == nil {
		return
	}
	go func() {
		for {
			runConnectivityProbe(e)
			interval := probeIntervalOK
			if state, _, _ := HealthSnapshot(); state == "error" {
				interval = probeIntervalErr
			}
			time.Sleep(interval)
		}
	}()
}

func runConnectivityProbe(e Interface) {
	epoch := probeEpoch.Add(1)
	ch := make(chan error, 1)
	go func() {
		_, err := e.Embed([]string{"connectivity probe"})
		if probeEpoch.Load() != epoch {
			return
		}
		ch <- err
	}()
	select {
	case err := <-ch:
		if probeEpoch.Load() != epoch {
			return
		}
		if err != nil {
			MarkError(err)
		} else {
			MarkSuccess()
		}
	case <-time.After(probeTimeout):
		probeEpoch.Add(1)
		MarkError(fmt.Errorf("connectivity probe: timeout after %s", probeTimeout))
	}
}
