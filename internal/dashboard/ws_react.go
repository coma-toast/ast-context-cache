package dashboard

import (
	"encoding/json"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

// ─── WebSocket hub ────────────────────────────────────────────────────

type wsMsg struct {
	Type      string      `json:"type"`
	Timestamp string      `json:"timestamp"`
	Data      interface{} `json:"data"`
}

type wsClient struct {
	hub  *wsHub
	conn *websocket.Conn
	send chan []byte
}

type wsHub struct {
	clients    map[*wsClient]bool
	register   chan *wsClient
	unregister chan *wsClient
	broadcast  chan wsMsg
	mu         sync.Mutex
}

func newWSHub() *wsHub {
	return &wsHub{
		clients:    make(map[*wsClient]bool),
		register:   make(chan *wsClient),
		unregister: make(chan *wsClient),
		broadcast:  make(chan wsMsg, 512),
	}
}

func (h *wsHub) run() {
	for {
		select {
		case c := <-h.register:
			h.clients[c] = true
		case c := <-h.unregister:
			if _, ok := h.clients[c]; ok {
				delete(h.clients, c)
				close(c.send)
			}
		case msg := <-h.broadcast:
			data, _ := json.Marshal(msg)
			var drop []*wsClient
			h.mu.Lock()
			for c := range h.clients {
				if !wsTrySend(c, data) {
					drop = append(drop, c)
				}
			}
			h.mu.Unlock()
			for _, c := range drop {
				select {
				case h.unregister <- c:
				default:
				}
			}
		}
	}
}

func (c *wsClient) readPump() {
	defer func() {
		c.hub.unregister <- c
		c.conn.Close()
	}()
	for {
		if _, _, err := c.conn.ReadMessage(); err != nil {
			break
		}
	}
}

func (c *wsClient) writePump() {
	ticker := time.NewTicker(54 * time.Second)
	defer func() {
		ticker.Stop()
		c.conn.Close()
	}()
	for {
		select {
		case msg, ok := <-c.send:
			if !ok {
				c.conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}
			if err := c.conn.WriteMessage(websocket.TextMessage, msg); err != nil {
				return
			}
		case <-ticker.C:
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

func handleWS(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		return
	}
	c := &wsClient{hub: hub, conn: conn, send: make(chan []byte, 256)}
	c.hub.register <- c
	go c.writePump()
	go c.readPump()
	pushInitialRefresh(c)
}

type dashboardPanel struct {
	name string
}

var overviewInitialPanels = []string{"health-bar", "stats", "index-health"}

var dashboardPanels = []dashboardPanel{
	{name: "index-health"},
	{name: "memory"},
	{name: "health-bar"},
	{name: "stats"},
	{name: "recent"},
	{name: "symbol-chart"},
	{name: "language-chart"},
	{name: "tool-chart"},
	{name: "import-chart"},
	{name: "settings"},
}

func wsTrySend(c *wsClient, data []byte) (ok bool) {
	defer func() {
		if recover() != nil {
			ok = false
		}
	}()
	select {
	case c.send <- data:
		return true
	default:
		return false
	}
}

func pushInitialRefresh(c *wsClient) {
	data, err := json.Marshal(wsMsg{
		Type:      "refresh",
		Timestamp: time.Now().Format(time.RFC3339),
		Data:      map[string]interface{}{"panels": overviewInitialPanels},
	})
	if err != nil {
		return
	}
	wsTrySend(c, data)
}

func broadcastRefresh(panels []string) {
	if hub == nil || len(panels) == 0 {
		return
	}
	hub.broadcast <- wsMsg{
		Type:      "refresh",
		Timestamp: time.Now().Format(time.RFC3339),
		Data:      map[string]interface{}{"panels": panels},
	}
}

var hub *wsHub

func init() {
	hub = newWSHub()
	go hub.run()
	initRealtimeBridge()
	initQueryLogBridge()
	initLogNotifyBridge()
}

func handleToastWS(w http.ResponseWriter, r *http.Request) {
	handleWS(w, r)
}

func broadcastToastWS(toolName, query, timeStr, savedText, durationMs, toolColor string) {
	if hub != nil {
		hub.broadcast <- wsMsg{
			Type:      "toast",
			Timestamp: timeStr,
			Data: map[string]string{
				"toolName":   toolName,
				"query":      query,
				"timeStr":    timeStr,
				"savedText":  savedText,
				"durationMs": durationMs,
				"toolColor":  toolColor,
			},
		}
	}
}
