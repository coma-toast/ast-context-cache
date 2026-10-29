package embedqueue

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

// Each embed channel is allocated once at its hard maximum; the operator-adjustable soft
// limit below it is what SubmitPriority, pushHigh and the pending retry path enforce, so a
// limit change applies live without recreating channels under load.
const (
	PendingCapHardMax = 1024
	HighCapHardMax    = 1024
	LowCapHardMax     = 16384

	DefaultPendingCap = 128
	DefaultHighCap    = 128
	DefaultLowCap     = 2048

	PendingCapSetting = "embed_queue_pending_cap"
	HighCapSetting    = "embed_queue_high_cap"
	LowCapSetting     = "embed_queue_low_cap"

	// spaceRecheck bounds how long a blocked sender sleeps between soft-limit checks when no
	// dequeue wakes it, e.g. for a receive path that does not signal.
	spaceRecheck = 100 * time.Millisecond
)

// QueueCap describes one embed channel's adjustable soft limit.
type QueueCap struct {
	Key     string `json:"key"`
	Env     string `json:"env"`
	Label   string `json:"label"`
	Default int    `json:"default"`
	HardMax int    `json:"hard_max"`
}

// QueueCapState is a QueueCap with its effective limit, current depth and env lock.
type QueueCapState struct {
	QueueCap
	Value     int  `json:"value"`
	Used      int  `json:"used"`
	EnvLocked bool `json:"env_locked"`
}

type gate struct {
	mu    sync.Mutex
	limit atomic.Int64
}

func newGate(n int) *gate {
	g := &gate{}
	g.limit.Store(int64(n))
	return g
}

var (
	pendingGate = newGate(DefaultPendingCap)
	highGate    = newGate(DefaultHighCap)
	lowGate     = newGate(DefaultLowCap)

	queueCaps = []QueueCap{
		{Key: PendingCapSetting, Env: "EMBED_QUEUE_PENDING_CAP", Label: "Retry (pending)", Default: DefaultPendingCap, HardMax: PendingCapHardMax},
		{Key: HighCapSetting, Env: "EMBED_QUEUE_HIGH_CAP", Label: "Priority", Default: DefaultHighCap, HardMax: HighCapHardMax},
		{Key: LowCapSetting, Env: "EMBED_QUEUE_LOW_CAP", Label: "Background", Default: DefaultLowCap, HardMax: LowCapHardMax},
	}

	spaceMu      sync.Mutex
	spaceCh      = make(chan struct{})
	spaceWaiters atomic.Int64
)

// QueueCaps lists the three adjustable embed queue limits (pending, high, low).
func QueueCaps() []QueueCap {
	return append([]QueueCap(nil), queueCaps...)
}

// IsQueueCapSetting reports whether key is an embed queue soft-limit setting.
func IsQueueCapSetting(key string) bool {
	_, ok := queueCapFor(key)
	return ok
}

func queueCapFor(key string) (QueueCap, bool) {
	for _, c := range queueCaps {
		if c.Key == key {
			return c, true
		}
	}
	return QueueCap{}, false
}

func gateFor(key string) *gate {
	switch key {
	case PendingCapSetting:
		return pendingGate
	case HighCapSetting:
		return highGate
	default:
		return lowGate
	}
}

func chanFor(key string) chan job {
	switch key {
	case PendingCapSetting:
		return pendingCh()
	case HighCapSetting:
		return highCh()
	default:
		return lowCh()
	}
}

// QueueCapEnvLocked reports whether the cap's environment variable is set, which makes it
// read-only from the dashboard.
func QueueCapEnvLocked(c QueueCap) bool {
	return strings.TrimSpace(os.Getenv(c.Env)) != ""
}

// resolveQueueCap is env > setting > default, clamped to 1..HardMax.
func resolveQueueCap(c QueueCap) int {
	n := db.SettingInt(c.Key, c.Env, c.Default)
	if n > c.HardMax {
		logger.Warn("Embed queue cap above hard max, clamping", "key", c.Key, "value", n, "max", c.HardMax)
		return c.HardMax
	}
	return n
}

// LoadQueueCaps re-reads every soft limit from env and settings and applies it live.
func LoadQueueCaps() {
	for _, c := range queueCaps {
		gateFor(c.Key).limit.Store(int64(resolveQueueCap(c)))
	}
	signalSpace()
}

// QueueCapStates returns every soft limit with its current depth and env lock.
func QueueCapStates() []QueueCapState {
	out := make([]QueueCapState, 0, len(queueCaps))
	for _, c := range queueCaps {
		used := 0
		if ch := chanFor(c.Key); ch != nil {
			used = len(ch)
		}
		out = append(out, QueueCapState{QueueCap: c, Value: int(gateFor(c.Key).limit.Load()), Used: used, EnvLocked: QueueCapEnvLocked(c)})
	}
	return out
}

// ValidateQueueCap parses raw for key and checks it against 1..HardMax.
func ValidateQueueCap(key, raw string) (int, error) {
	c, ok := queueCapFor(key)
	if !ok {
		return 0, errs.NewCode(errs.CodeNotFound, "unknown embed queue cap", "key", key)
	}
	n, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil || n < 1 || n > c.HardMax {
		return 0, errs.NewCode(errs.CodeInvalidInput, fmt.Sprintf("%s must be an integer 1–%d", key, c.HardMax), "key", key, "value", raw)
	}
	return n, nil
}

// SetQueueCap validates, persists and applies one soft limit without a restart. Raising a
// limit wakes senders blocked on it; lowering one leaves queued jobs in place and only
// throttles new submissions until the depth drops below it.
func SetQueueCap(key, raw string) (int, error) {
	n, err := ValidateQueueCap(key, raw)
	if err != nil {
		return 0, err
	}
	c, _ := queueCapFor(key)
	if QueueCapEnvLocked(c) {
		return 0, errs.NewCode(errs.CodeConflict, fmt.Sprintf("%s is set by %s", key, c.Env), "key", key)
	}
	if err := db.SetSetting(key, strconv.Itoa(n)); err != nil {
		return 0, errs.Wrap(err, "key", key)
	}
	gateFor(key).limit.Store(int64(n))
	signalSpace()
	logger.Info("Embed queue cap set", "key", key, "cap", n)
	realtime.Notify(realtime.EmbedFinished | realtime.IndexHealth | realtime.HealthBar)
	return n, nil
}

// trySend puts j on ch when ch is below g's soft limit (and has buffer room). The gate lock
// serializes senders, and workers only receive, so the depth cannot pass the limit.
func trySend(ch chan job, g *gate, j job) bool {
	if ch == nil {
		return false
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if int64(len(ch)) >= g.limit.Load() {
		return false
	}
	select {
	case ch <- j:
		return true
	default:
		return false
	}
}

type lane struct {
	ch chan job
	g  *gate
}

// sendWait puts j on the first lane below its soft limit, blocking until one has room: the
// backpressure a full buffered channel gave before the limits became adjustable.
func sendWait(j job, lanes ...lane) {
	if trySendAny(j, lanes) {
		return
	}
	spaceWaiters.Add(1)
	defer spaceWaiters.Add(-1)
	t := time.NewTimer(spaceRecheck)
	defer t.Stop()
	for {
		wake := spaceSignal()
		if trySendAny(j, lanes) {
			return
		}
		if !t.Stop() {
			select {
			case <-t.C:
			default:
			}
		}
		t.Reset(spaceRecheck)
		select {
		case <-wake:
		case <-t.C:
		}
	}
}

func trySendAny(j job, lanes []lane) bool {
	for _, l := range lanes {
		if trySend(l.ch, l.g, j) {
			return true
		}
	}
	return false
}

func spaceSignal() <-chan struct{} {
	spaceMu.Lock()
	defer spaceMu.Unlock()
	return spaceCh
}

// signalSpace wakes senders blocked in sendWait after a dequeue or a limit change.
func signalSpace() {
	if spaceWaiters.Load() == 0 {
		return
	}
	spaceMu.Lock()
	close(spaceCh)
	spaceCh = make(chan struct{})
	spaceMu.Unlock()
}
