package dashboard

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
)

// TestNewHandlerGuardsWrites checks the real dashboard handler is wrapped by
// httpguard: a cross-site POST must not reach the settings handler.
func TestNewHandlerGuardsWrites(t *testing.T) {
	dbtest.Init(t)
	h := NewHandler("127.0.0.1")
	tests := []struct {
		name   string
		host   string
		origin string
		value  string
		want   int
		stored bool
	}{
		{"foreign origin", "127.0.0.1:7830", "http://evil.example", "evil", http.StatusForbidden, false},
		{"rebinding host", "evil.example:7830", "", "rebind", http.StatusForbidden, false},
		{"loopback origin", "127.0.0.1:7830", "http://127.0.0.1:7830", "ui", http.StatusOK, true},
		{"no origin", "localhost:7830", "", "cli", http.StatusOK, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/api/settings", strings.NewReader(`{"key":"guard_test","value":"`+tt.value+`"}`))
			req.Host = tt.host
			req.Header.Set("Content-Type", "application/json")
			if tt.origin != "" {
				req.Header.Set("Origin", tt.origin)
			}
			rr := httptest.NewRecorder()
			h.ServeHTTP(rr, req)
			require.Equal(t, tt.want, rr.Code, rr.Body.String())
			assert.Equal(t, tt.stored, db.GetSetting("guard_test", "") == tt.value)
		})
	}
}
