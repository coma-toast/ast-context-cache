package embedder

import (
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

var probeFailed atomic.Bool

const (
	probeIntervalOK   = 10 * time.Second
	probeIntervalErr  = 5 * time.Second
	recentErrorWindow = 5 * time.Minute
)

var probeTimeout = 8 * time.Second

var (
	healthMu         sync.RWMutex
	healthState      = "idle" // idle, loading, ready, degraded, error
	healthLastErr    string
	healthLastUse    time.Time
	healthRecentErrAt time.Time
	onRecovery       func()
	onError          func()
	onReady          func()
)

// SetOnRecovery registers a callback when embedder health recovers from error to ready.
func SetOnRecovery(fn func()) {
	onRecovery = fn
}

// SetOnError registers a callback when an embedding call fails (debounced work should be used).
func SetOnError(fn func()) {
	onError = fn
}

// SetOnReady registers a callback after a successful embed when not recovering from error.
func SetOnReady(fn func()) {
	onReady = fn
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

// MarkSuccess records a successful embedding call. Recent failures within
// recentErrorWindow leave state degraded so the dashboard does not show OK
// while embeds are still flaky.
func MarkSuccess() {
	healthMu.Lock()
	prevRaw := healthState
	prev := effectiveStateLocked()
	healthLastUse = time.Now()
	if recentErrorActiveLocked() {
		healthState = "degraded"
	} else {
		healthState = "ready"
		healthLastErr = ""
	}
	cur := effectiveStateLocked()
	healthMu.Unlock()
	if prev != cur || (prevRaw == "degraded" && cur == "ready") {
		realtime.Notify(realtime.HealthBar | realtime.IndexHealth)
	}
	if (prevRaw == "error" || prevRaw == "degraded") && cur == "ready" && onRecovery != nil {
		go onRecovery()
	} else if cur == "ready" && onReady != nil {
		go onReady()
	} else if cur == "degraded" && onReady != nil {
		go onReady()
	}
}

// MarkProbeResult records connectivity probe outcome without overriding worker-driven error state.
func MarkProbeResult(err error) {
	if err != nil {
		probeFailed.Store(true)
		MarkError(err)
		return
	}
	wasProbeFail := probeFailed.Swap(false)
	healthMu.RLock()
	wasError := effectiveStateLocked() == "error"
	lastErr := healthLastErr
	healthMu.RUnlock()
	if wasProbeFail || (wasError && IsNetworkBackend(ActiveBackend) && isConnectivityErr(lastErr)) {
		MarkSuccess()
		return
	}
	if onReady != nil {
		go onReady()
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
	healthRecentErrAt = time.Now()
	healthMu.Unlock()
	realtime.Notify(realtime.HealthBar | realtime.IndexHealth)
	if onError != nil {
		go onError()
	}
}

func recentErrorActiveLocked() bool {
	return !healthRecentErrAt.IsZero() && time.Since(healthRecentErrAt) < recentErrorWindow
}

func effectiveStateLocked() string {
	if healthState == "degraded" && !recentErrorActiveLocked() {
		return "ready"
	}
	return healthState
}

func isConnectivityErr(msg string) bool {
	m := strings.ToLower(msg)
	return strings.Contains(m, "connection refused") ||
		strings.Contains(m, "connection reset") ||
		strings.Contains(m, "dial tcp") ||
		strings.Contains(m, "dial http") ||
		strings.Contains(m, "no such host") ||
		strings.Contains(m, "connectivity probe") ||
		strings.Contains(m, "timeout") ||
		strings.Contains(m, "eof") ||
		strings.Contains(m, "503") ||
		strings.Contains(m, "502") ||
		strings.Contains(m, "504")
}

// HealthState returns the current state and time since last successful use.
func HealthState() (state string, lastUse time.Duration) {
	healthMu.RLock()
	defer healthMu.RUnlock()
	return effectiveStateLocked(), time.Since(healthLastUse)
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
	state = effectiveStateLocked()
	lastErr = healthLastErr
	if state == "ready" {
		lastErr = ""
	}
	return state, time.Since(healthLastUse), lastErr
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
			MarkProbeResult(err)
		} else {
			MarkProbeResult(nil)
		}
	case <-time.After(probeTimeout):
		probeEpoch.Add(1)
		MarkProbeResult(fmt.Errorf("connectivity probe: timeout after %s", probeTimeout))
	}
}
