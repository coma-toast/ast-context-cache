package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/flags"
)

// AC28 allows one second from flags.Set to the notification.
const listChangedDeadline = time.Second

// testDebounce shortens the list_changed batch window so tests stay fast.
const testDebounce = 50 * time.Millisecond

// sseEvent is one parsed SSE event: a data message, or a comment such as a keep-alive.
type sseEvent struct {
	data    map[string]any
	comment string
}

// openStream sends req and returns its response plus a channel of parsed events. The
// request is cancelled when the test ends, which is how a client closes a stream.
func openStream(t *testing.T, req *http.Request) (*http.Response, <-chan sseEvent) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	resp, err := http.DefaultClient.Do(req.WithContext(ctx))
	require.NoError(t, err)
	events := make(chan sseEvent, 32)
	go func() {
		defer close(events)
		defer resp.Body.Close()
		sc := bufio.NewScanner(resp.Body)
		for sc.Scan() {
			line := sc.Text()
			if c, ok := strings.CutPrefix(line, ":"); ok {
				events <- sseEvent{comment: strings.TrimSpace(c)}
				continue
			}
			data, ok := strings.CutPrefix(line, "data: ")
			if !ok {
				continue
			}
			var msg map[string]any
			if json.Unmarshal([]byte(data), &msg) == nil {
				events <- sseEvent{data: msg}
			}
		}
	}()
	return resp, events
}

// nextMessage returns the next data event within timeout, skipping comments.
func nextMessage(t *testing.T, events <-chan sseEvent, timeout time.Duration) map[string]any {
	t.Helper()
	deadline := time.After(timeout)
	for {
		select {
		case ev, ok := <-events:
			require.True(t, ok, "stream closed before a message arrived")
			if ev.data != nil {
				return ev.data
			}
		case <-deadline:
			t.Fatalf("no message within %s", timeout)
		}
	}
}

// enableFlagChanges opens a database of the test's own (an earlier dbtest user may have
// closed TestMain's), wires flags to the hub, and makes feature_handoff settable and on, so
// turning it off is a change that must reach every stream.
func enableFlagChanges(t *testing.T) {
	t.Helper()
	t.Cleanup(flags.Reload)
	dbtest.Init(t)
	t.Setenv("AST_FEATURE_HANDOFF", "")
	Init()
	require.NoError(t, flags.Set(flags.KeyHandoff, true))
	setListChangedTiming(t, testDebounce, 0)
}

// setListChangedTiming shortens the coalescer's package vars for one test and clears its
// state, under its lock so a timer left by an earlier test never races the write.
func setListChangedTiming(t *testing.T, debounce, minInterval time.Duration) {
	t.Helper()
	listChanged.mu.Lock()
	origDebounce, origMin := listChangedDebounce, listChangedMinInterval
	listChangedDebounce, listChangedMinInterval = debounce, minInterval
	resetListChangedLocked()
	listChanged.mu.Unlock()
	t.Cleanup(func() {
		listChanged.mu.Lock()
		defer listChanged.mu.Unlock()
		listChangedDebounce, listChangedMinInterval = origDebounce, origMin
		resetListChangedLocked()
	})
}

func resetListChangedLocked() {
	if listChanged.timer != nil {
		listChanged.timer.Stop()
	}
	listChanged.timer, listChanged.toolsDirty, listChanged.lastFrame = nil, false, time.Time{}
}

// assertNoMessage fails if a data event arrives within window.
func assertNoMessage(t *testing.T, events <-chan sseEvent, window time.Duration) {
	t.Helper()
	deadline := time.After(window)
	for {
		select {
		case ev, ok := <-events:
			if !ok {
				return
			}
			assert.Nil(t, ev.data, "unexpected message")
		case <-deadline:
			return
		}
	}
}

func legacyStreamRequest(t *testing.T, srv *httptest.Server, session string) *http.Request {
	t.Helper()
	req, err := http.NewRequest(http.MethodGet, srv.URL, nil)
	require.NoError(t, err)
	req.Header.Set("Accept", sseContentType)
	req.Header.Set(headerSessionID, session)
	req.Header.Set(headerProtocolVersion, "2025-11-25")
	return req
}

func listenRequest(t *testing.T, srv *httptest.Server, id any, filter map[string]any) *http.Request {
	t.Helper()
	params := map[string]any{"notifications": filter, metaKey: map[string]any{metaProtocolVersion: ModernProtocolVersion, metaClientCapabilities: map[string]any{}}}
	req, err := http.NewRequest(http.MethodPost, srv.URL, bytes.NewReader(mustJSON(rpcRequest(id, "subscriptions/listen", params))))
	require.NoError(t, err)
	req.Header.Set("Accept", "application/json, text/event-stream")
	req.Header.Set(headerProtocolVersion, ModernProtocolVersion)
	req.Header.Set(headerMethod, "subscriptions/listen")
	return req
}

func TestLegacySessionToolListFrozen(t *testing.T) {
	srv := newMCPServer(t)
	enableFlagChanges(t)
	initResp, _ := initialize(t, srv, "2025-11-25")
	sid := initResp.Header.Get(headerSessionID)
	require.NotEmpty(t, sid)
	_, before := rpcCall(t, srv, map[string]string{headerSessionID: sid}, rpcRequest(2, "tools/list", nil))
	require.Contains(t, toolNames(t, resultOf(t, before)), "handoff")
	resp, events := openStream(t, legacyStreamRequest(t, srv, sid))
	require.Equal(t, http.StatusOK, resp.StatusCode)
	assert.Equal(t, sseContentType, resp.Header.Get("Content-Type"))
	require.NoError(t, flags.Set(flags.KeyHandoff, false))
	assertNoMessage(t, events, 4*testDebounce)

	// The session keeps the list it started with.
	_, after := rpcCall(t, srv, map[string]string{headerSessionID: sid}, rpcRequest(3, "tools/list", nil))
	assert.Equal(t, before["result"], after["result"], "a legacy session's tools/list is frozen")

	// A new session sees the change.
	fresh, _ := initialize(t, srv, "2025-11-25")
	_, body := rpcCall(t, srv, map[string]string{headerSessionID: fresh.Header.Get(headerSessionID)}, rpcRequest(4, "tools/list", nil))
	names := toolNames(t, resultOf(t, body))
	assert.Contains(t, names, "get_context_capsule")
	assert.NotContains(t, names, "handoff")
}

func TestListChangedCoalesced(t *testing.T) {
	srv := newMCPServer(t)
	enableFlagChanges(t)
	setListChangedTiming(t, testDebounce, time.Hour)
	resp, events := openStream(t, listenRequest(t, srv, "sub-1", map[string]any{"toolsListChanged": true}))
	require.Equal(t, http.StatusOK, resp.StatusCode)
	assert.Equal(t, "notifications/subscriptions/acknowledged", nextMessage(t, events, listChangedDeadline)["method"])
	require.NoError(t, flags.Set(flags.KeyHandoff, false))
	msg := nextMessage(t, events, listChangedDeadline)
	assert.Equal(t, listChangedMethod, msg["method"], "feature_handoff and its implied children make one frame")
	assertNoMessage(t, events, 4*testDebounce)
	require.NoError(t, flags.Set(flags.KeyHandoff, true))
	assertNoMessage(t, events, 4*testDebounce)
	listChanged.mu.Lock()
	assert.True(t, listChanged.toolsDirty, "a change inside the min interval waits for the next frame")
	listChanged.mu.Unlock()
}

func TestModernListenReceivesListChanged(t *testing.T) {
	srv := newMCPServer(t)
	enableFlagChanges(t)
	resp, events := openStream(t, listenRequest(t, srv, "sub-1", map[string]any{"toolsListChanged": true, "resourcesListChanged": true}))
	require.Equal(t, http.StatusOK, resp.StatusCode)
	assert.Equal(t, sseContentType, resp.Header.Get("Content-Type"))
	assert.Equal(t, "no", resp.Header.Get("X-Accel-Buffering"))
	assert.Empty(t, resp.Header.Get(headerSessionID))
	ack := nextMessage(t, events, listChangedDeadline)
	assert.Equal(t, "notifications/subscriptions/acknowledged", ack["method"])
	assert.Equal(t, map[string]any{
		metaKey:         map[string]any{metaSubscriptionID: "sub-1"},
		"notifications": map[string]any{"toolsListChanged": true},
	}, ack["params"], "unsupported types are left out of the acknowledgment")
	require.NoError(t, flags.Set(flags.KeyHandoff, false))
	msg := nextMessage(t, events, listChangedDeadline)
	assert.Equal(t, "notifications/tools/list_changed", msg["method"])
	assert.Equal(t, map[string]any{metaKey: map[string]any{metaSubscriptionID: "sub-1"}}, msg["params"])

	_, body := modernCall(t, srv, 3, "tools/list", nil, nil)
	names := toolNames(t, resultOf(t, body))
	assert.Contains(t, names, "get_context_capsule")
	assert.NotContains(t, names, "handoff")
}

func TestListenRejectsBadHeaders(t *testing.T) {
	srv := newMCPServer(t)
	req := listenRequest(t, srv, 1, map[string]any{"toolsListChanged": true})
	req.Header.Set(headerMethod, "tools/list")
	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()
	assert.Equal(t, http.StatusBadRequest, resp.StatusCode)
}

func TestDeleteEndsLegacyStream(t *testing.T) {
	srv := newMCPServer(t)
	initResp, _ := initialize(t, srv, "2025-06-18")
	sid := initResp.Header.Get(headerSessionID)
	_, events := openStream(t, legacyStreamRequest(t, srv, sid))
	req, err := http.NewRequest(http.MethodDelete, srv.URL, nil)
	require.NoError(t, err)
	req.Header.Set(headerSessionID, sid)
	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	resp.Body.Close()
	require.Equal(t, http.StatusNoContent, resp.StatusCode)
	select {
	case _, ok := <-events:
		assert.False(t, ok, "stream should end without further messages")
	case <-time.After(time.Second):
		t.Fatal("stream still open after DELETE")
	}
}

// streamServer serves sub-less streams from h, so heartbeat and shutdown can be tested on
// a hub of their own.
func streamServer(t *testing.T, h *streamHub, modern bool) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sub := &subscriber{ch: make(chan []byte, subscriberBuffer), done: make(chan struct{})}
		var last []byte
		if modern {
			sub.subscriptionID, sub.methods = 1, map[string]bool{}
			last = sseFrame(rpcResult(1, map[string]any{"resultType": "complete"}))
		}
		if !h.register(sub) {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		defer h.unregister(sub)
		h.stream(w, r, sub, nil, last)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestStreamKeepAliveAndGracefulClose(t *testing.T) {
	h := newStreamHub()
	h.heartbeat = 10 * time.Millisecond
	srv := streamServer(t, h, true)
	req, err := http.NewRequest(http.MethodPost, srv.URL, nil)
	require.NoError(t, err)
	_, events := openStream(t, req)
	select {
	case ev := <-events:
		assert.Equal(t, "keep-alive", ev.comment)
	case <-time.After(time.Second):
		t.Fatal("no keep-alive")
	}
	h.close()
	msg := nextMessage(t, events, time.Second)
	assert.Equal(t, map[string]any{"jsonrpc": "2.0", "id": float64(1), "result": map[string]any{"resultType": "complete"}}, msg)
	for range events {
	}
	resp, err := http.Post(srv.URL, "application/json", nil)
	require.NoError(t, err)
	resp.Body.Close()
	assert.Equal(t, http.StatusServiceUnavailable, resp.StatusCode, "a closed hub takes no new streams")
}

func TestBroadcastOncePerSession(t *testing.T) {
	h := newStreamHub()
	sid := h.newSession("2025-11-25")
	a := &subscriber{ch: make(chan []byte, 1), done: make(chan struct{}), sessionID: sid}
	b := &subscriber{ch: make(chan []byte, 1), done: make(chan struct{}), sessionID: sid}
	modern := &subscriber{ch: make(chan []byte, 1), done: make(chan struct{}), subscriptionID: 1, methods: map[string]bool{"notifications/tools/list_changed": true}}
	optedOut := &subscriber{ch: make(chan []byte, 1), done: make(chan struct{}), subscriptionID: 2, methods: map[string]bool{}}
	for _, sub := range []*subscriber{a, b, modern, optedOut} {
		require.True(t, h.register(sub))
	}
	h.broadcast("notifications/tools/list_changed", nil)
	assert.Equal(t, 1, len(a.ch)+len(b.ch), "one stream per session gets the message")
	assert.Len(t, modern.ch, 1)
	assert.Empty(t, optedOut.ch, "a subscription only gets the types it asked for")
	h.broadcast("notifications/tools/list_changed", nil)
	assert.Len(t, modern.ch, 1, "a full stream drops rather than blocks")
}

func TestSessionIdleEviction(t *testing.T) {
	h := newStreamHub()
	idle := h.newSession("2025-11-25")
	streaming := h.newSession("2025-11-25")
	sub := &subscriber{ch: make(chan []byte, 1), done: make(chan struct{}), sessionID: streaming}
	require.True(t, h.register(sub))
	h.mu.Lock()
	for _, s := range h.sessions {
		s.lastSeen = time.Now().Add(-2 * sessionIdleTTL)
	}
	h.mu.Unlock()
	h.newSession("2025-11-25")
	assert.False(t, h.touch(idle), "idle session evicted")
	assert.True(t, h.touch(streaming), "a session with an open stream stays")
	h.unregister(sub)
	assert.True(t, h.touch(streaming), "closing the stream counts as activity")
}
