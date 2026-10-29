package hostdisk

import (
	"bytes"
	"context"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

func TestValidateSetting(t *testing.T) {
	tests := []struct {
		in, want string
		wantErr  string
	}{
		{in: "", want: ""},
		{in: "   ", want: ""},
		{in: " http://192.168.1.10:9633/metrics ", want: "http://192.168.1.10:9633/metrics"},
		{in: "https://pve.tail1234.ts.net:9633/metrics#frag", want: "https://pve.tail1234.ts.net:9633/metrics"},
		{in: "http://[fd00::1]:9633/metrics", want: "http://[fd00::1]:9633/metrics"},
		{in: "192.168.1.10:9633/metrics", wantErr: "http:// or https://"},
		{in: "ftp://host/metrics", wantErr: "http:// or https://"},
		{in: "file:///etc/passwd", wantErr: "http:// or https://"},
		{in: "http:///metrics", wantErr: "needs a host"},
		{in: "http://admin:hunter2@host:9633/metrics", wantErr: "user name or password"},
		{in: "http://admin@host:9633/metrics", wantErr: "user name or password"},
		{in: "http://host:9633/%zz", wantErr: "not a valid URL"},
	}
	for _, tt := range tests {
		t.Run(tt.in, func(t *testing.T) {
			got, err := ValidateSetting(tt.in)
			if tt.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.wantErr)
				assert.True(t, errs.HasCode(err, errs.CodeInvalidInput))
				assert.NotContains(t, err.Error(), "hunter2")
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

// exporter is an httptest smartctl_exporter whose answer the test can change between polls.
type exporter struct {
	*httptest.Server
	mu     sync.Mutex
	status int
	body   string
	hits   atomic.Int32
}

func newExporter(t *testing.T, body string) *exporter {
	e := &exporter{status: http.StatusOK, body: body}
	e.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		e.hits.Add(1)
		e.mu.Lock()
		status, body := e.status, e.body
		e.mu.Unlock()
		w.WriteHeader(status)
		w.Write([]byte(body))
	}))
	t.Cleanup(e.Close)
	return e
}

func (e *exporter) set(status int, body string) {
	e.mu.Lock()
	e.status, e.body = status, body
	e.mu.Unlock()
}

// captureLogs routes the package logger into a buffer for the test.
func captureLogs(t *testing.T) *bytes.Buffer {
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})))
	t.Cleanup(func() { slog.SetDefault(prev) })
	return &buf
}

func TestSamplerKeepsLastGoodSampleAndWarnsOncePerStateChange(t *testing.T) {
	logs := captureLogs(t)
	ex := newExporter(t, fixture(t))
	u := ex.URL + "/metrics"
	changes := 0
	s := NewSampler(func() string { return u }, func() { changes++ })
	ctx := context.Background()

	s.Poll(ctx)
	snap := s.Snapshot()
	require.Empty(t, snap.Error)
	require.Len(t, snap.Disks, 5)
	good := snap.SampledAt
	assert.False(t, good.IsZero())
	assert.True(t, snap.URLSet)
	assert.Equal(t, 1, changes)

	ex.set(http.StatusBadGateway, "upstream down")
	s.Poll(ctx)
	s.Poll(ctx)
	snap = s.Snapshot()
	assert.Len(t, snap.Disks, 5, "last good sample kept")
	assert.Equal(t, good, snap.SampledAt)
	assert.Contains(t, snap.Error, "502")
	assert.Equal(t, 1, strings.Count(logs.String(), "Host disk health fetch failing"), "one WARN for the same failure")

	ex.set(http.StatusOK, "# TYPE node_load1 gauge\nnode_load1 1\n")
	s.Poll(ctx)
	assert.Contains(t, s.Snapshot().Error, "no smartctl_device metrics")
	assert.Equal(t, 2, strings.Count(logs.String(), "Host disk health fetch failing"), "a new failure kind warns again")

	ex.set(http.StatusOK, fixture(t))
	s.Poll(ctx)
	snap = s.Snapshot()
	assert.Empty(t, snap.Error)
	assert.True(t, snap.SampledAt.After(good) || snap.SampledAt.Equal(good))
	assert.Contains(t, logs.String(), "Host disk health fetch recovered")
	assert.Equal(t, 4, changes, "ok, failing, other failure, recovered")
}

func TestSamplerURLChanges(t *testing.T) {
	ex := newExporter(t, fixture(t))
	u := ex.URL + "/metrics"
	s := NewSampler(func() string { return u }, nil)
	ctx := context.Background()
	s.Poll(ctx)
	require.Len(t, s.Snapshot().Disks, 5)

	u = "http://127.0.0.1:1/metrics"
	s.Poll(ctx)
	snap := s.Snapshot()
	assert.Empty(t, snap.Disks, "a different URL does not inherit the old sample")
	assert.True(t, snap.SampledAt.IsZero())
	assert.NotEmpty(t, snap.Error)

	u = "http://user:secret@host/metrics"
	s.Poll(ctx)
	assert.Contains(t, s.Snapshot().Error, "user name or password")

	u = ""
	s.Poll(ctx)
	assert.Equal(t, Snapshot{}, s.Snapshot(), "clearing the URL clears the card")
}

func TestSamplerFetchLimits(t *testing.T) {
	other := newExporter(t, fixture(t))
	tests := []struct {
		name    string
		handler http.HandlerFunc
		wantErr string
	}{
		{"too large", func(w http.ResponseWriter, r *http.Request) {
			w.Write(bytes.Repeat([]byte("# padding\n"), MaxBodyBytes/10+1))
		}, "larger than the limit"},
		{"redirect to another host", func(w http.ResponseWriter, r *http.Request) {
			http.Redirect(w, r, strings.Replace(other.URL, "127.0.0.1", "localhost", 1)+"/metrics", http.StatusFound)
		}, "another host"},
		{"slow", func(w http.ResponseWriter, r *http.Request) {
			select {
			case <-r.Context().Done():
			case <-time.After(2 * time.Second):
			}
		}, "deadline exceeded"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(tt.handler)
			t.Cleanup(srv.Close)
			s := NewSampler(func() string { return srv.URL + "/metrics" }, nil)
			s.timeout = 200 * time.Millisecond
			s.Poll(context.Background())
			snap := s.Snapshot()
			assert.Contains(t, snap.Error, tt.wantErr)
			assert.Empty(t, snap.Disks)
		})
	}
	t.Run("redirect on the same host is followed", func(t *testing.T) {
		mux := http.NewServeMux()
		mux.HandleFunc("/old", func(w http.ResponseWriter, r *http.Request) {
			http.Redirect(w, r, "/metrics", http.StatusMovedPermanently)
		})
		mux.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(fixture(t))) })
		srv := httptest.NewServer(mux)
		t.Cleanup(srv.Close)
		s := NewSampler(func() string { return srv.URL + "/old" }, nil)
		s.Poll(context.Background())
		assert.Len(t, s.Snapshot().Disks, 5)
	})
}

func TestSamplerRunRefresh(t *testing.T) {
	ex := newExporter(t, fixture(t))
	s := NewSampler(func() string { return ex.URL }, nil)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		s.Run(ctx, time.Hour)
		close(done)
	}()
	require.Eventually(t, func() bool { return ex.hits.Load() == 1 }, 2*time.Second, 5*time.Millisecond)
	s.Refresh()
	require.Eventually(t, func() bool { return ex.hits.Load() == 2 }, 2*time.Second, 5*time.Millisecond)
	cancel()
	<-done
}
