package netlisten

import (
	"io"
	"net"
	"net/http"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// testNetAddr is in TEST-NET-1 (RFC 5737), which no host has configured.
const testNetAddr = "192.0.2.1"

func newTestServer(t *testing.T) *http.Server {
	t.Helper()
	srv := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = io.WriteString(w, "ok") })}
	t.Cleanup(func() { srv.Close() })
	return srv
}

// fetch makes a request on a fresh connection.
func fetch(addr string, port int) (string, error) {
	c := &http.Client{Transport: &http.Transport{DisableKeepAlives: true}}
	resp, err := c.Get("http://" + net.JoinHostPort(addr, strconv.Itoa(port)) + "/")
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	return string(b), err
}

// serveBase stands in for the server's base listener: on [::1] when the host has IPv6
// loopback, else on a second 127.0.0.1 port, so extras on 127.0.0.1 never collide.
func serveBase(t *testing.T, srv *http.Server) (string, int) {
	t.Helper()
	ln, err := net.Listen("tcp", "[::1]:0")
	if err != nil {
		ln, err = net.Listen("tcp", "127.0.0.1:0")
		require.NoError(t, err)
	}
	go func() { _ = srv.Serve(ln) }()
	ta := ln.Addr().(*net.TCPAddr)
	return ta.IP.String(), ta.Port
}

func TestManagerReconcileOpensAndCloses(t *testing.T) {
	t.Parallel()
	srv := newTestServer(t)
	baseHost, basePort := serveBase(t, srv)
	m := NewManager("test", srv, 0)
	m.DropGrace = 0
	t.Cleanup(m.Close)

	m.Reconcile([]string{"127.0.0.1"})
	st := m.Status()
	require.Len(t, st, 1)
	require.Equal(t, StatusListening, st[0].Status, st[0].Error)
	assert.Equal(t, "test", st[0].Server)
	assert.Equal(t, "127.0.0.1", st[0].Addr)
	port := st[0].Port
	require.NotZero(t, port, "ephemeral port reported")
	body, err := fetch("127.0.0.1", port)
	require.NoError(t, err)
	assert.Equal(t, "ok", body)

	m.Reconcile([]string{"127.0.0.1"})
	assert.Equal(t, port, m.Status()[0].Port, "an open listener is kept, not reopened")

	// A keep-alive connection through the extra listener is cut off when it is removed.
	keepAlive := &http.Client{}
	resp, err := keepAlive.Get("http://" + net.JoinHostPort("127.0.0.1", strconv.Itoa(port)) + "/")
	require.NoError(t, err)
	_, _ = io.ReadAll(resp.Body)
	resp.Body.Close()

	m.Reconcile(nil)
	assert.Empty(t, m.Status())
	_, err = fetch("127.0.0.1", port)
	assert.Error(t, err, "removed listener refuses new connections")
	_, err = keepAlive.Get("http://" + net.JoinHostPort("127.0.0.1", strconv.Itoa(port)) + "/")
	assert.Error(t, err, "removed listener's open connections are closed")
	body, err = fetch(baseHost, basePort)
	require.NoError(t, err, "base listener untouched")
	assert.Equal(t, "ok", body)
}

func TestManagerBindFailureReported(t *testing.T) {
	t.Parallel()
	srv := newTestServer(t)
	m := NewManager("test", srv, 0)
	t.Cleanup(m.Close)
	m.Reconcile([]string{testNetAddr, "127.0.0.1"})
	st := m.Status()
	require.Len(t, st, 2)
	assert.Equal(t, testNetAddr, st[0].Addr, "listed in configured order")
	assert.Equal(t, StatusError, st[0].Status)
	assert.Contains(t, st[0].Error, "failed to open extra listener")
	assert.Contains(t, st[0].Error, testNetAddr)
	assert.Equal(t, StatusListening, st[1].Status, "one failure doesn't block the others")
	assert.True(t, m.HasErrors())

	m.Reconcile([]string{testNetAddr})
	assert.Equal(t, StatusError, m.Status()[0].Status, "retried and still failing")
	m.Reconcile(nil)
	assert.False(t, m.HasErrors(), "errors for removed addresses are forgotten")
}

func TestManagerPortInUse(t *testing.T) {
	t.Parallel()
	taken, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	t.Cleanup(func() { taken.Close() })
	m := NewManager("test", newTestServer(t), taken.Addr().(*net.TCPAddr).Port)
	t.Cleanup(m.Close)
	m.Reconcile([]string{"127.0.0.1"})
	st := m.Status()
	require.Len(t, st, 1)
	assert.Equal(t, StatusError, st[0].Status)
	assert.NotEmpty(t, st[0].Error)
}

func TestManagerCloseStopsReconcile(t *testing.T) {
	t.Parallel()
	m := NewManager("test", newTestServer(t), 0)
	m.Reconcile([]string{"127.0.0.1"})
	port := m.Status()[0].Port
	m.Close()
	assert.Empty(t, m.Status())
	_, err := fetch("127.0.0.1", port)
	assert.Error(t, err)
	m.Reconcile([]string{"127.0.0.1"})
	assert.Empty(t, m.Status(), "nothing reopens after Close")
}

func TestManagerServerShutdownClosesExtras(t *testing.T) {
	t.Parallel()
	srv := newTestServer(t)
	m := NewManager("test", srv, 0)
	t.Cleanup(m.Close)
	m.Reconcile([]string{"127.0.0.1"})
	port := m.Status()[0].Port
	_, err := fetch("127.0.0.1", port)
	require.NoError(t, err)
	require.NoError(t, srv.Shutdown(t.Context()))
	_, err = fetch("127.0.0.1", port)
	assert.Error(t, err)
}
