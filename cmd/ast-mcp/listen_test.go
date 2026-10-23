package main

import (
	"context"
	"errors"
	"net"
	"net/http"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// freePort finds a port that is currently free on IPv4 loopback.
func freePort(t *testing.T) int {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	port := ln.Addr().(*net.TCPAddr).Port
	require.NoError(t, ln.Close())
	return port
}

func setListenAddr(t *testing.T, addr string) {
	t.Helper()
	orig := listenAddr
	listenAddr = addr
	t.Cleanup(func() { listenAddr = orig })
}

func TestListenersAddIPv6LoopbackForDefault(t *testing.T) {
	setListenAddr(t, defaultListenAddr)
	port := freePort(t)
	lns, err := listeners(port)
	require.NoError(t, err)
	t.Cleanup(func() {
		for _, ln := range lns {
			ln.Close()
		}
	})
	require.NotEmpty(t, lns)
	assert.Equal(t, net.JoinHostPort(defaultListenAddr, strconv.Itoa(port)), lns[0].Addr().String())
	if len(lns) == 2 {
		assert.Equal(t, net.JoinHostPort("::1", strconv.Itoa(port)), lns[1].Addr().String())
	}
}

func TestListenersSingleForExplicitAddress(t *testing.T) {
	setListenAddr(t, "127.0.0.2")
	ln, err := net.Listen("tcp", "127.0.0.2:0")
	if err != nil {
		t.Skip("127.0.0.2 is not routable on this host")
	}
	port := ln.Addr().(*net.TCPAddr).Port
	ln.Close()
	lns, err := listeners(port)
	require.NoError(t, err)
	t.Cleanup(func() { lns[0].Close() })
	assert.Len(t, lns, 1)
}

func TestListenAndServeReturnsAfterShutdown(t *testing.T) {
	setListenAddr(t, defaultListenAddr)
	port := freePort(t)
	srv := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})}
	done := make(chan error, 1)
	go func() { done <- listenAndServe(srv, port) }()
	url := "http://" + net.JoinHostPort(defaultListenAddr, strconv.Itoa(port))
	require.Eventually(t, func() bool {
		resp, err := http.Get(url)
		if err != nil {
			return false
		}
		resp.Body.Close()
		return true
	}, 2*time.Second, 10*time.Millisecond)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	require.NoError(t, srv.Shutdown(ctx))
	select {
	case err := <-done:
		assert.NoError(t, err)
	case <-time.After(2 * time.Second):
		t.Fatal("listenAndServe did not return after Shutdown")
	}
}

func TestRestartReleasesPorts(t *testing.T) {
	// Verify that shutting down a server (simulating the degraded-mode restart/drain flow)
	// completely releases the port so a subsequent server instance can bind it immediately.
	port := freePort(t)
	addr := net.JoinHostPort(defaultListenAddr, strconv.Itoa(port))

	// First server instance
	srv1 := &http.Server{Addr: addr, Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})}
	ln, err := net.Listen("tcp", addr)
	require.NoError(t, err)

	done1 := make(chan error, 1)
	go func() { done1 <- srv1.Serve(ln) }()

	// Shut down first server
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	require.NoError(t, srv1.Shutdown(ctx))
	select {
	case <-done1:
	case <-time.After(2 * time.Second):
		t.Fatal("srv1 did not stop")
	}

	// Second server instance on the exact same port — must bind successfully (no "address already in use")
	srv2 := &http.Server{Addr: addr, Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})}
	ln2, err := net.Listen("tcp", addr)
	require.NoError(t, err)
	t.Cleanup(func() { ln2.Close() })

	done2 := make(chan error, 1)
	go func() { done2 <- srv2.Serve(ln2) }()

	ctx2, cancel2 := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel2()
	require.NoError(t, srv2.Shutdown(ctx2))
	select {
	case <-done2:
	case <-time.After(2 * time.Second):
		t.Fatal("srv2 did not stop")
	}
}

func TestServeWithRestartOutcomeExits1WhenServeFails(t *testing.T) {
	// When serve returns an error (not nil) with exiting=false,
	// serveWithRestartOutcome should call exit(1) and not block.
	var exitCalled, blockCalled bool
	fakeExit := func(code int) { exitCalled = true }
	fakeBlock := func() { blockCalled = true }

	serveWithRestartOutcome(
		func() error { return errors.New("boom") },
		func() bool { return false },
		fakeExit,
		fakeBlock,
	)

	require.True(t, exitCalled, "exit(1) should be called on serve error")
	require.False(t, blockCalled, "block should not be called on serve error")
}

func TestServeWithRestartOutcomeBlocksWhenExitingAndServeReturnsNil(t *testing.T) {
	// When serve returns nil and exiting=true (restart or signal),
	// serveWithRestartOutcome should call block() and not exit.
	// This is the regression test for the nil-return-after-Shutdown bug.
	var exitCalled, blockCalled bool
	fakeExit := func(code int) { exitCalled = true }
	fakeBlock := func() { blockCalled = true }

	serveWithRestartOutcome(
		func() error { return nil },
		func() bool { return true },
		fakeExit,
		fakeBlock,
	)

	require.True(t, blockCalled, "block() should be called during exit")
	require.False(t, exitCalled, "exit should not be called during exit")
}

func TestServeWithRestartOutcomeExits1WhenServeReturnsNilAndNotExiting(t *testing.T) {
	// When serve returns nil and exiting=false (unexpected shutdown),
	// serveWithRestartOutcome should call exit(1) so supervisors notice.
	var exitCalled, blockCalled bool
	fakeExit := func(code int) { exitCalled = true }
	fakeBlock := func() { blockCalled = true }

	serveWithRestartOutcome(
		func() error { return nil },
		func() bool { return false },
		fakeExit,
		fakeBlock,
	)

	require.True(t, exitCalled, "exit(1) should be called on unexpected shutdown")
	require.False(t, blockCalled, "block should not be called on unexpected shutdown")
}
