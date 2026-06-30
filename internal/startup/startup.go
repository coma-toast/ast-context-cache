package startup

import (
	"sync"

	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

type Phase string

const (
	PhaseStarting Phase = "starting"
	PhaseReady    Phase = "ready"
	PhaseFailed   Phase = "failed"
)

var (
	mu      sync.RWMutex
	phase   = PhaseStarting
	message = "Starting up…"
	errText string
)

func CurrentPhase() Phase {
	mu.RLock()
	defer mu.RUnlock()
	return phase
}

func Message() string {
	mu.RLock()
	defer mu.RUnlock()
	return message
}

func Error() string {
	mu.RLock()
	defer mu.RUnlock()
	return errText
}

func Starting() bool {
	return CurrentPhase() == PhaseStarting
}

func Ready() bool {
	return CurrentPhase() == PhaseReady
}

func Failed() bool {
	return CurrentPhase() == PhaseFailed
}

func SetMessage(msg string) {
	mu.Lock()
	message = msg
	mu.Unlock()
	realtime.Notify(realtime.HealthBar | realtime.IndexHealth)
}

func MarkReady() {
	mu.Lock()
	phase = PhaseReady
	message = ""
	errText = ""
	mu.Unlock()
	realtime.Notify(realtime.HealthBar | realtime.IndexHealth | realtime.Stats)
}

func MarkFailed(err string) {
	mu.Lock()
	phase = PhaseFailed
	errText = err
	if message == "" {
		message = "Startup failed"
	}
	mu.Unlock()
	realtime.Notify(realtime.HealthBar | realtime.IndexHealth)
}
