package mcp

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/flags"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

// Server-to-client delivery for both eras. A legacy client opens GET /mcp with its
// Mcp-Session-Id and gets a standalone SSE stream; a modern client POSTs
// subscriptions/listen and gets an SSE response that stays open. Broadcast fans a
// notification out to every open stream of either kind. A legacy session's tools/list is
// frozen at first use, so tool-list changes only reach modern subscribers.

const (
	// sessionIdleTTL is how long a legacy session survives without requests or an open stream.
	sessionIdleTTL = time.Hour
	// subscriberBuffer bounds queued notifications per stream. list_changed is idempotent,
	// so dropping one for a stream that is already behind loses nothing.
	subscriberBuffer = 16
	sseContentType   = "text/event-stream"
	// heartbeatInterval spaces SSE keep-alive comments. Claude Code backs off for hours
	// after repeated stream drops, so idle proxies and clients must never see a quiet stream.
	heartbeatInterval = 20 * time.Second
	listChangedMethod = "notifications/tools/list_changed"
)

// Flag changes are coalesced: one batch sends one list_changed frame after
// listChangedDebounce, and never sooner than listChangedMinInterval after the last frame, so
// a burst of toggles costs connected agents one prompt-cache miss rather than several.
// Package vars so tests can shorten them (under listChanged.mu).
var (
	listChangedDebounce    = 400 * time.Millisecond
	listChangedMinInterval = 5 * time.Second
)

// mcpSession is one legacy Streamable HTTP session minted by initialize.
type mcpSession struct {
	id, version string
	lastSeen    time.Time
	streams     int
	// tools is the tools/list answer frozen at the session's first request, so a flag
	// change mid-session never invalidates the client's prompt cache.
	tools []Tool
}

// subscriber is one open SSE stream.
type subscriber struct {
	ch   chan []byte
	done chan struct{}
	once sync.Once
	// sessionID is set for a legacy GET stream.
	sessionID string
	// subscriptionID is the subscriptions/listen request id for a modern stream, and
	// methods the notifications it opted in to.
	subscriptionID any
	methods        map[string]bool
}

// streamHub tracks legacy sessions and every open stream.
type streamHub struct {
	mu        sync.Mutex
	sessions  map[string]*mcpSession
	subs      map[*subscriber]struct{}
	closed    chan struct{}
	once      sync.Once
	heartbeat time.Duration
}

var hub = newStreamHub()

func newStreamHub() *streamHub {
	return &streamHub{sessions: map[string]*mcpSession{}, subs: map[*subscriber]struct{}{}, closed: make(chan struct{}), heartbeat: heartbeatInterval}
}

// Broadcast sends a server notification to every open legacy stream (one per session, as
// the spec forbids repeating a message across a session's streams) and to every modern
// subscription that opted in to method, tagged with its subscriptionId.
func Broadcast(method string, params any) {
	hub.broadcast(method, params)
}

// CloseStreams ends every open stream, sending modern subscriptions their graceful
// completion. http.Server.Shutdown waits for idle connections, which an SSE stream never
// becomes, so main registers this with RegisterOnShutdown.
func CloseStreams() {
	hub.close()
}

// Init wires feature flags to MCP clients and the dashboard. Call it after db.Init: the
// Reload applies flags saved in the settings table, which lookups before the database
// opened couldn't see. A flag that changes tools/list sends one coalesced list_changed to
// every modern subscriber.
func Init() {
	initOnce.Do(func() { flags.OnChange(onFlagChange) })
	flags.Reload()
}

var initOnce sync.Once

func onFlagChange(key string, _ bool) {
	listChanged.mark(flags.AffectsTools(key))
}

var listChanged = &listChangedCoalescer{}

// listChangedCoalescer batches flag changes into one list_changed frame for modern
// subscribers and one dashboard SettingsChanged notification.
type listChangedCoalescer struct {
	mu         sync.Mutex
	timer      *time.Timer
	toolsDirty bool
	lastFrame  time.Time
}

// mark records a flag change, arming the batch timer if none is pending.
func (c *listChangedCoalescer) mark(tools bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.toolsDirty = c.toolsDirty || tools
	if c.timer == nil {
		c.timer = time.AfterFunc(listChangedDebounce, c.fire)
	}
}

// fire flushes the batch, re-arming instead when a tool change would follow the last frame
// sooner than listChangedMinInterval.
func (c *listChangedCoalescer) fire() {
	c.mu.Lock()
	if wait := listChangedMinInterval - time.Since(c.lastFrame); c.toolsDirty && !c.lastFrame.IsZero() && wait > 0 {
		c.timer = time.AfterFunc(wait, c.fire)
		c.mu.Unlock()
		return
	}
	tools := c.toolsDirty
	c.toolsDirty, c.timer = false, nil
	if tools {
		c.lastFrame = time.Now()
	}
	c.mu.Unlock()
	if tools {
		hub.send(listChangedMethod, nil, false)
	}
	realtime.Notify(realtime.SettingsChanged)
}

// toolsFor answers tools/list for sessionID: the session's frozen list, captured from the
// live config on first use, or the live list when there is no live session.
func (h *streamHub) toolsFor(sessionID string) []Tool {
	if sessionID == "" {
		return FilterTools(GetConfig())
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	s, ok := h.liveSessionLocked(sessionID)
	if !ok {
		return FilterTools(GetConfig())
	}
	if s.tools == nil {
		s.tools = FilterTools(GetConfig())
	}
	return s.tools
}

func (h *streamHub) newSession(version string) string {
	id := newSessionID()
	now := time.Now()
	h.mu.Lock()
	defer h.mu.Unlock()
	for sid, s := range h.sessions {
		if s.streams == 0 && now.Sub(s.lastSeen) > sessionIdleTTL {
			delete(h.sessions, sid)
		}
	}
	h.sessions[id] = &mcpSession{id: id, version: version, lastSeen: now}
	return id
}

// touch marks id as active and reports whether it is a live session.
func (h *streamHub) touch(id string) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	s, ok := h.liveSessionLocked(id)
	if ok {
		s.lastSeen = time.Now()
	}
	return ok
}

func (h *streamHub) liveSessionLocked(id string) (*mcpSession, bool) {
	s, ok := h.sessions[id]
	if !ok {
		return nil, false
	}
	if s.streams == 0 && time.Since(s.lastSeen) > sessionIdleTTL {
		delete(h.sessions, id)
		return nil, false
	}
	return s, true
}

// dropSession ends a legacy session and its streams, reporting whether it existed.
func (h *streamHub) dropSession(id string) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	if _, ok := h.liveSessionLocked(id); !ok {
		return false
	}
	delete(h.sessions, id)
	for sub := range h.subs {
		if sub.sessionID == id {
			sub.stop()
		}
	}
	return true
}

// register adds sub, failing for a closed hub or a legacy session that no longer exists.
func (h *streamHub) register(sub *subscriber) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	select {
	case <-h.closed:
		return false
	default:
	}
	if sub.sessionID != "" {
		s, ok := h.liveSessionLocked(sub.sessionID)
		if !ok {
			return false
		}
		s.streams++
	}
	h.subs[sub] = struct{}{}
	return true
}

func (h *streamHub) unregister(sub *subscriber) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if _, ok := h.subs[sub]; !ok {
		return
	}
	delete(h.subs, sub)
	if s, ok := h.sessions[sub.sessionID]; ok {
		s.streams--
		s.lastSeen = time.Now()
	}
}

func (h *streamHub) broadcast(method string, params any) {
	h.send(method, params, true)
}

// send delivers method to every modern subscriber that opted in and, when legacy is set, to
// one stream per legacy session.
func (h *streamHub) send(method string, params any, legacy bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	served := map[string]bool{}
	for sub := range h.subs {
		if sub.sessionID != "" && (!legacy || served[sub.sessionID]) {
			continue
		}
		frame, ok := sub.frame(method, params)
		if !ok {
			continue
		}
		select {
		case sub.ch <- frame:
			served[sub.sessionID] = sub.sessionID != ""
		default:
			logger.Warn("Dropped MCP notification for a slow stream", "method", method, "session", sub.sessionID, "subscription", sub.subscriptionID)
		}
	}
}

func (h *streamHub) close() {
	h.once.Do(func() { close(h.closed) })
}

// frame renders method as an SSE message for sub, or reports false when sub didn't opt in.
func (sub *subscriber) frame(method string, params any) ([]byte, bool) {
	if sub.methods != nil && !sub.methods[method] {
		return nil, false
	}
	msg := map[string]any{"jsonrpc": JSONRPCVersion, "method": method}
	if sub.methods != nil {
		p := map[string]any{}
		if m, ok := params.(map[string]any); ok {
			for k, v := range m {
				p[k] = v
			}
		}
		p[metaKey] = map[string]any{metaSubscriptionID: sub.subscriptionID}
		params = p
	}
	if params != nil {
		msg["params"] = params
	}
	return sseFrame(msg), true
}

func (sub *subscriber) stop() {
	sub.once.Do(func() { close(sub.done) })
}

// serveLegacyStream answers GET /mcp. A session is required: 2024-11-05 clients GET for
// an endpoint event this server doesn't send, so 405 tells them no stream is on offer,
// and 404 makes a client whose session expired re-initialize, as the spec prescribes.
func serveLegacyStream(w http.ResponseWriter, r *http.Request) {
	id := r.Header.Get(headerSessionID)
	if id == "" {
		methodNotAllowed(w)
		return
	}
	sub := &subscriber{ch: make(chan []byte, subscriberBuffer), done: make(chan struct{}), sessionID: id}
	if !hub.register(sub) {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}
	defer hub.unregister(sub)
	hub.stream(w, r, sub, nil, nil)
}

// serveListen answers a modern subscriptions/listen with a stream that opens with the
// acknowledgment of the notification types we will honor. Only tools change at runtime,
// so toolsListChanged is the only type ever acknowledged.
func serveListen(w http.ResponseWriter, r *http.Request, req JSONRPCRequest) {
	requested, _ := req.Params["notifications"].(map[string]any)
	agreed, methods := map[string]any{}, map[string]bool{}
	if on, _ := requested["toolsListChanged"].(bool); on {
		agreed["toolsListChanged"] = true
		methods[listChangedMethod] = true
	}
	sub := &subscriber{ch: make(chan []byte, subscriberBuffer), done: make(chan struct{}), subscriptionID: req.ID, methods: methods}
	if !hub.register(sub) {
		writeRPC(w, http.StatusServiceUnavailable, rpcError(req.ID, &JSONRPCError{Code: InternalError, Message: "Server is shutting down"}))
		return
	}
	defer hub.unregister(sub)
	tag := map[string]any{metaSubscriptionID: req.ID}
	ack := sseFrame(map[string]any{"jsonrpc": JSONRPCVersion, "method": "notifications/subscriptions/acknowledged", "params": map[string]any{metaKey: tag, "notifications": agreed}})
	end := sseFrame(rpcResult(req.ID, map[string]any{"resultType": "complete", metaKey: tag}))
	hub.stream(w, r, sub, ack, end)
}

// stream holds an SSE response open, writing first, then queued notifications and
// keep-alive comments, until the client disconnects (for a listen stream that is the
// cancellation), the session is dropped, or the hub closes, which writes last first.
func (h *streamHub) stream(w http.ResponseWriter, r *http.Request, sub *subscriber, first, last []byte) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", sseContentType)
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	if first != nil {
		w.Write(first)
	}
	flusher.Flush()
	ticker := time.NewTicker(h.heartbeat)
	defer ticker.Stop()
	for {
		var out []byte
		select {
		case <-r.Context().Done():
			return
		case <-sub.done:
			return
		case <-h.closed:
			if last != nil {
				w.Write(last)
				flusher.Flush()
			}
			return
		case out = <-sub.ch:
		case <-ticker.C:
			out = []byte(": keep-alive\n\n")
		}
		if _, err := w.Write(out); err != nil {
			logger.Debug("MCP stream closed", "session", sub.sessionID, "error", err)
			return
		}
		flusher.Flush()
	}
}

// sseFrame renders one JSON-RPC message as an SSE "message" event.
func sseFrame(msg any) []byte {
	b, _ := json.Marshal(msg)
	return []byte("event: message\ndata: " + string(b) + "\n\n")
}

// acceptsSSE reports whether the request's Accept header lists text/event-stream.
func acceptsSSE(r *http.Request) bool {
	for _, v := range r.Header.Values("Accept") {
		if strings.Contains(v, sseContentType) {
			return true
		}
	}
	return false
}

func methodNotAllowed(w http.ResponseWriter) {
	w.Header().Set("Allow", http.MethodPost)
	w.WriteHeader(http.StatusMethodNotAllowed)
}

// newSessionID returns 128 random bits as hex: unguessable and plain visible ASCII, as
// the spec requires of Mcp-Session-Id.
func newSessionID() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}
