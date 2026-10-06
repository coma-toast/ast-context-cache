// Package netlisten owns the optional extra listeners (a Tailscale IP, say) that the MCP
// and dashboard servers accept connections on beside their loopback/base listeners, and
// resolves the network access settings (extra addresses, trusted hosts, access token)
// that drive them and httpguard. Extra listeners are opened and closed live: Reconcile
// at startup and whenever the settings change. Base listeners are never touched here.
package netlisten

import (
	"errors"
	"net"
	"net/http"
	"slices"
	"strconv"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// Listener states reported in Status.
const (
	StatusListening = "listening"
	StatusError     = "error"
)

// defaultDropGrace lets a response already in flight on a removed listener finish (the
// settings POST that removed it, for one) before its connections are closed.
const defaultDropGrace = 2 * time.Second

// Status is one extra listener's state.
type Status struct {
	Server string `json:"server"`
	Addr   string `json:"addr"`
	Port   int    `json:"port"`
	Status string `json:"status"`
	Error  string `json:"error,omitempty"`
}

// Manager keeps one server's extra listeners in line with the addresses last passed to
// Reconcile. It is safe for concurrent use.
type Manager struct {
	name string
	srv  *http.Server
	port int
	// DropGrace is how long a removed listener's open connections live on; set it
	// before the first Reconcile.
	DropGrace time.Duration

	mu     sync.Mutex
	addrs  []string
	open   map[string]*trackedListener
	errs   map[string]string
	closed bool
}

// NewManager returns a manager serving srv on extra addresses at port. Port 0 picks an
// ephemeral port per listener, which Status reports.
func NewManager(name string, srv *http.Server, port int) *Manager {
	return &Manager{name: name, srv: srv, port: port, DropGrace: defaultDropGrace, open: map[string]*trackedListener{}, errs: map[string]string{}}
}

// Reconcile opens a listener for each address in addrs that has none (retrying ones
// that failed before) and closes those whose address is no longer listed. A bind
// failure is logged and reported by Status, never fatal. It does nothing after Close.
func (m *Manager) Reconcile(addrs []string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return
	}
	m.addrs = slices.Clone(addrs)
	for addr, l := range m.open {
		if !slices.Contains(addrs, addr) {
			delete(m.open, addr)
			logger.Info("Closing extra listener", "server", m.name, "addr", l.Addr().String())
			l.drop(m.DropGrace)
		}
	}
	for addr := range m.errs {
		if !slices.Contains(addrs, addr) {
			delete(m.errs, addr)
		}
	}
	for _, addr := range addrs {
		if _, ok := m.open[addr]; !ok {
			m.listen(addr)
		}
	}
}

// listen opens addr and serves it; m.mu must be held.
func (m *Manager) listen(addr string) {
	hostPort := net.JoinHostPort(addr, strconv.Itoa(m.port))
	ln, err := net.Listen("tcp", hostPort)
	if err != nil {
		err = errs.WrapMessage("failed to open extra listener", err, "server", m.name, "addr", hostPort)
		// The retry loop calls Reconcile again while a bind keeps failing; warn once per distinct error.
		if prev, seen := m.errs[addr]; !seen || prev != err.Error() {
			logger.Warn("Failed to open extra listener", "server", m.name, "addr", hostPort, "error", err)
		}
		m.errs[addr] = err.Error()
		return
	}
	delete(m.errs, addr)
	tl := newTrackedListener(ln)
	m.open[addr] = tl
	logger.Info("Listening on extra address", "server", m.name, "addr", ln.Addr().String())
	go func() {
		// Serve returns when the listener is dropped or the server shuts down; neither is a failure.
		if err := m.srv.Serve(tl); err != nil && !errors.Is(err, http.ErrServerClosed) && !tl.isDropped() {
			logger.Warn("Extra listener stopped", "server", m.name, "addr", ln.Addr().String(), "error", err)
		}
	}()
}

// Status lists each address from the last Reconcile, in that order.
func (m *Manager) Status() []Status {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]Status, 0, len(m.addrs))
	for _, addr := range m.addrs {
		st := Status{Server: m.name, Addr: addr, Port: m.port}
		if l, ok := m.open[addr]; ok {
			st.Status = StatusListening
			if ta, ok := l.Addr().(*net.TCPAddr); ok {
				st.Port = ta.Port
			}
		} else {
			st.Status, st.Error = StatusError, m.errs[addr]
		}
		out = append(out, st)
	}
	return out
}

// HasErrors reports whether any listed address failed to bind.
func (m *Manager) HasErrors() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.errs) > 0
}

// Close closes every extra listener and its connections, and makes later Reconcile
// calls no-ops so nothing reopens during shutdown.
func (m *Manager) Close() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closed = true
	for addr, l := range m.open {
		delete(m.open, addr)
		l.drop(0)
	}
	m.addrs, m.errs = nil, map[string]string{}
}

// trackedListener remembers the connections it accepted so that removing the address
// also cuts off clients still connected through it (keep-alive, SSE, WebSocket).
// Close only closes the listener, as http.Server.Shutdown expects; drop does both.
type trackedListener struct {
	net.Listener

	mu      sync.Mutex
	conns   map[*trackedConn]struct{}
	dropped bool
}

func newTrackedListener(ln net.Listener) *trackedListener {
	return &trackedListener{Listener: ln, conns: map[*trackedConn]struct{}{}}
}

func (l *trackedListener) Accept() (net.Conn, error) {
	c, err := l.Listener.Accept()
	if err != nil {
		return nil, err
	}
	tc := &trackedConn{Conn: c, l: l}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.dropped {
		c.Close()
		return nil, net.ErrClosed
	}
	l.conns[tc] = struct{}{}
	return tc, nil
}

// drop closes the listener now and its open connections after grace.
func (l *trackedListener) drop(grace time.Duration) {
	l.mu.Lock()
	l.dropped = true
	l.mu.Unlock()
	l.Listener.Close()
	closeConns := func() {
		l.mu.Lock()
		conns := make([]*trackedConn, 0, len(l.conns))
		for c := range l.conns {
			conns = append(conns, c)
		}
		l.mu.Unlock()
		for _, c := range conns {
			c.Close()
		}
	}
	if grace <= 0 {
		closeConns()
		return
	}
	time.AfterFunc(grace, closeConns)
}

func (l *trackedListener) isDropped() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.dropped
}

type trackedConn struct {
	net.Conn
	l    *trackedListener
	once sync.Once
}

func (c *trackedConn) Close() error {
	c.once.Do(func() {
		c.l.mu.Lock()
		delete(c.l.conns, c)
		c.l.mu.Unlock()
	})
	return c.Conn.Close()
}
