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
	probeIntervalOK       = 10 * time.Second
	probeBaseTimeout      = 8 * time.Second
	probeMaxErrInterval   = 30 * time.Second
	probeBackoffStep      = 5 * time.Second
	recentErrorWindow     = 5 * time.Minute
	probeActivityGrace    = 20 * time.Second // skip probe failures while workers recently succeeded
)

type probeResult int

const (
	probeOK probeResult = iota
	probeFail
	probeSkipped
)

var (
	healthMu         sync.RWMutex
	healthState      = "idle" // idle, loading, ready, degraded, error
	healthLastErr    string
	healthLastUse       time.Time
	healthLastWorkerOK  time.Time
	healthRecentErrAt   time.Time
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
	healthLastUse = time.Time{}
	healthLastWorkerOK = time.Time{}
	healthLastErr = ""
	healthRecentErrAt = time.Time{}
	probeFailed.Store(false)
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
	now := time.Now()
	healthLastWorkerOK = now
	healthLastUse = now
	if isProbeErr(healthLastErr) {
		healthState = "ready"
		healthLastErr = ""
		healthRecentErrAt = time.Time{}
	} else if recentErrorActiveLocked() {
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
		healthMu.RLock()
		active := recentEmbedActivityLocked()
		healthMu.RUnlock()
		if active {
			return
		}
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

func recentEmbedActivityLocked() bool {
	return !healthLastWorkerOK.IsZero() && time.Since(healthLastWorkerOK) < probeActivityGrace
}

func isProbeErr(msg string) bool {
	return strings.Contains(strings.ToLower(msg), "connectivity probe")
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

var (
	probeEpoch  atomic.Uint64
	probeStopMu sync.Mutex
	probeStop   chan struct{}
)

func stopConnectivityProbe() {
	probeStopMu.Lock()
	defer probeStopMu.Unlock()
	if probeStop != nil {
		close(probeStop)
		probeStop = nil
	}
	probeEpoch.Add(1)
}

// probeErrorBackoff returns wait time before the next probe after the immediate 2x-timeout retry fails.
func probeErrorBackoff(streak uint32) time.Duration {
	if streak < 2 {
		return 0
	}
	d := time.Duration(streak-1) * probeBackoffStep
	if d > probeMaxErrInterval {
		return probeMaxErrInterval
	}
	return d
}

// StartConnectivityProbe periodically probes network embed backends so a broken
// connection surfaces in the dashboard even when no indexing is in flight.
func StartConnectivityProbe(e Interface) {
	stopConnectivityProbe()
	if !IsNetworkBackend(ActiveBackend) || e == nil {
		return
	}
	probeStopMu.Lock()
	stop := make(chan struct{})
	probeStop = stop
	probeStopMu.Unlock()
	go func() {
		var streak uint32
		for {
			select {
			case <-stop:
				return
			default:
			}
			timeout := probeBaseTimeout
			if streak == 1 {
				timeout = probeBaseTimeout * 2
			}
			switch runConnectivityProbe(e, timeout) {
			case probeOK:
				streak = 0
				time.Sleep(probeIntervalOK)
				continue
			case probeSkipped:
				if state, _, _ := HealthSnapshot(); state != "error" {
					streak = 0
				}
				time.Sleep(probeIntervalOK)
				continue
			case probeFail:
				streak++
				if streak == 1 {
					continue
				}
				sleep := probeErrorBackoff(streak)
				select {
				case <-stop:
					return
				case <-time.After(sleep):
				}
			}
		}
	}()
}

func runConnectivityProbe(e Interface, timeout time.Duration) probeResult {
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
			return probeSkipped
		}
		if err != nil {
			healthMu.RLock()
			active := recentEmbedActivityLocked()
			healthMu.RUnlock()
			if active {
				return probeSkipped
			}
			MarkProbeResult(err)
			return probeFail
		}
		MarkProbeResult(nil)
		return probeOK
	case <-time.After(timeout):
		probeEpoch.Add(1)
		healthMu.RLock()
		active := recentEmbedActivityLocked()
		healthMu.RUnlock()
		if active {
			return probeSkipped
		}
		MarkProbeResult(fmt.Errorf("connectivity probe: timeout after %s", timeout))
		return probeFail
	}
}
