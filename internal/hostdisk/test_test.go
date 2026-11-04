package hostdisk

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTestReportsDisks(t *testing.T) {
	body, err := os.ReadFile("testdata/smartctl_exporter.prom")
	require.NoError(t, err)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/metrics" {
			http.NotFound(w, r)
			return
		}
		w.Write(body)
	}))
	defer srv.Close()

	res := Test(context.Background(), srv.URL+"/metrics")
	require.True(t, res.OK, res.Error)
	assert.Positive(t, res.Disks)
	assert.Len(t, res.Devices, res.Disks)
	assert.Empty(t, res.Hint)
}

func TestTestHints(t *testing.T) {
	plain := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/metrics" {
			w.Write([]byte("# no smartctl metrics here\n"))
			return
		}
		http.NotFound(w, r)
	}))
	defer plain.Close()
	closed, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	closedAddr := closed.Addr().String()
	closed.Close()

	tests := []struct {
		name, url, hint string
	}{
		{"https to an http exporter", strings.Replace(plain.URL, "http://", "https://", 1) + "/metrics", "change https:// to http://"},
		{"nothing listening", "http://" + closedAddr + "/metrics", "Nothing is listening on that port"},
		{"missing /metrics", plain.URL, "Add the /metrics path: " + plain.URL + "/metrics"},
		{"wrong path", plain.URL + "/stats", "serves them at /metrics"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res := Test(context.Background(), tt.url)
			assert.False(t, res.OK)
			assert.NotEmpty(t, res.Error)
			assert.Contains(t, res.Hint, tt.hint)
		})
	}
}

func TestTestRejectsAnInvalidURL(t *testing.T) {
	res := Test(context.Background(), "192.168.1.10:9633/metrics")
	assert.False(t, res.OK)
	assert.Contains(t, res.Error, "must start with http://")
}
