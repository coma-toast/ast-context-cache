package main

import (
	"context"
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

func TestListenAndServeShutsDownEveryListener(t *testing.T) {
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
