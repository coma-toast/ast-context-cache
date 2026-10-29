// Package hostdisk reads SMART health for the disks of the machine this instance runs on as a
// guest (an LXC container or VM), from a prometheus-community smartctl_exporter on that host.
// A guest cannot read SMART data itself; the exporter runs on the host and this package polls
// its /metrics URL, set in the host_disk_health_url setting.
package hostdisk

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/url"
	"reflect"
	"strings"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/logging"
	"github.com/coma-toast/ast-context-cache/internal/sys"
)

// SettingKey stores the exporter's /metrics URL; empty turns host disks off.
const SettingKey = "host_disk_health_url"

const (
	// DefaultInterval matches smartctl_exporter's own default --smartctl.interval.
	DefaultInterval = 60 * time.Second
	// DefaultTimeout bounds one fetch, connect to last byte.
	DefaultTimeout = 5 * time.Second
	// MaxBodyBytes caps a response; a host with dozens of disks is well under 1 MiB.
	MaxBodyBytes = 4 << 20
	maxRedirects = 3
)

// Failure kinds; a WARN is logged when the kind changes, not on every failed poll.
const (
	kindInvalidURL = "invalid_url"
	kindRequest    = "unreachable"
	kindStatus     = "http_status"
	kindTooLarge   = "too_large"
	kindParse      = "parse"
	kindNoDevices  = "no_devices"
)

var logger = logging.Tagged("hostdisk")

// ParseURL checks a host disk health URL: absolute http or https with a host, and no
// user:password, which would be stored in plain text and shown in Settings.
func ParseURL(raw string) (*url.URL, error) {
	raw = strings.TrimSpace(raw)
	u, err := url.Parse(raw)
	switch {
	case !strings.Contains(raw, "://"):
		return nil, errs.NewCode(errs.CodeInvalidInput, "host disk health URL must start with http:// or https://")
	case err != nil:
		return nil, errs.NewCode(errs.CodeInvalidInput, "host disk health URL is not a valid URL")
	case u.Scheme != "http" && u.Scheme != "https":
		return nil, errs.NewCode(errs.CodeInvalidInput, "host disk health URL must start with http:// or https://")
	case u.Host == "" || u.Hostname() == "":
		return nil, errs.NewCode(errs.CodeInvalidInput, "host disk health URL needs a host, e.g. http://192.168.1.10:9633/metrics")
	case u.User != nil:
		return nil, errs.NewCode(errs.CodeInvalidInput, "host disk health URL must not contain a user name or password")
	}
	u.Fragment, u.RawFragment = "", ""
	return u, nil
}

// ValidateSetting returns the value to store for SettingKey: empty, or a normalized URL.
func ValidateSetting(raw string) (string, error) {
	if strings.TrimSpace(raw) == "" {
		return "", nil
	}
	u, err := ParseURL(raw)
	if err != nil {
		return "", err
	}
	return u.String(), nil
}

// Snapshot is what the disk card shows: the last good sample, plus the current error if the
// latest poll failed.
type Snapshot struct {
	URLSet    bool
	Disks     []sys.DiskInfo
	Error     string
	SampledAt time.Time
}

// Sampler polls the configured URL and keeps the last good sample.
type Sampler struct {
	urlFn    func() string
	onChange func()
	client   *http.Client
	timeout  time.Duration
	kick     chan struct{}

	mu       sync.Mutex
	snap     Snapshot
	goodURL  string
	failKind string
}

// NewSampler polls whatever urlFn returns at each poll; onChange (may be nil) runs after a poll
// that changed the snapshot.
func NewSampler(urlFn func() string, onChange func()) *Sampler {
	s := &Sampler{urlFn: urlFn, onChange: onChange, timeout: DefaultTimeout, kick: make(chan struct{}, 1)}
	s.client = &http.Client{CheckRedirect: sameHostRedirect}
	return s
}

// Snapshot returns a copy of the current state.
func (s *Sampler) Snapshot() Snapshot {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := s.snap
	out.Disks = append([]sys.DiskInfo(nil), s.snap.Disks...)
	return out
}

// Refresh asks a running Run loop to poll now, e.g. after the URL setting changed.
func (s *Sampler) Refresh() {
	select {
	case s.kick <- struct{}{}:
	default:
	}
}

// Run polls immediately, then every interval and on Refresh, until ctx is done.
func (s *Sampler) Run(ctx context.Context, interval time.Duration) {
	s.Poll(ctx)
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		case <-s.kick:
		}
		s.Poll(ctx)
	}
}

// Poll fetches the URL once and updates the snapshot. On failure the last good sample from the
// same URL is kept and Error says why; a new or cleared URL drops the old sample.
func (s *Sampler) Poll(ctx context.Context) {
	raw := strings.TrimSpace(s.urlFn())
	if raw == "" {
		s.apply(nil, "", "", "", nil)
		return
	}
	u, err := ParseURL(raw)
	if err != nil {
		s.apply(nil, raw, "", kindInvalidURL, err)
		return
	}
	disks, kind, err := s.fetch(ctx, u)
	s.apply(disks, u.String(), u.Redacted(), kind, err)
}

// apply stores a poll's result, logs a failure kind change once (WARN) and a recovery (INFO),
// and calls onChange when the visible state changed. Only the redacted URL is logged.
func (s *Sampler) apply(disks []sys.DiskInfo, key, redacted, kind string, err error) {
	s.mu.Lock()
	before, prevKind := s.snap, s.failKind
	next := Snapshot{URLSet: key != "", Disks: disks}
	switch {
	case err == nil && key != "":
		next.SampledAt, s.goodURL = time.Now(), key
	case err != nil && key == s.goodURL:
		next.Disks, next.SampledAt, next.Error = before.Disks, before.SampledAt, err.Error()
	case err != nil:
		next.Error, s.goodURL = err.Error(), ""
	default:
		s.goodURL = ""
	}
	s.snap, s.failKind = next, kind
	changed := !reflect.DeepEqual(before.Disks, next.Disks) || before.Error != next.Error || before.URLSet != next.URLSet
	s.mu.Unlock()
	switch {
	case kind != "" && kind != prevKind:
		logger.Warn("Host disk health fetch failing; keeping the last good sample", "url", redacted, "reason", kind, "error", err)
	case kind == "" && prevKind != "" && key != "":
		logger.Info("Host disk health fetch recovered", "url", redacted, "disks", len(disks))
	}
	if changed && s.onChange != nil {
		s.onChange()
	}
}

// fetch GETs the exporter page within the timeout and size cap and parses it.
func (s *Sampler) fetch(ctx context.Context, u *url.URL) ([]sys.DiskInfo, string, error) {
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, kindInvalidURL, errs.WrapCodeMessage(errs.CodeInvalidInput, "build request", err)
	}
	req.Header.Set("Accept", "text/plain;version=0.0.4;q=1,*/*;q=0.1")
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, kindRequest, errs.WrapMessage("fetch host disk health", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, kindStatus, errs.New("host disk health URL answered "+resp.Status, "status", resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, MaxBodyBytes+1))
	if err != nil {
		return nil, kindRequest, errs.WrapMessage("read host disk health response", err)
	}
	if len(body) > MaxBodyBytes {
		return nil, kindTooLarge, errs.New("host disk health response is larger than the limit", "limit_bytes", MaxBodyBytes)
	}
	disks, err := Parse(bytes.NewReader(body))
	switch {
	case errs.HasCode(err, CodeNoDevices):
		return nil, kindNoDevices, err
	case err != nil:
		return nil, kindParse, err
	}
	return disks, "", nil
}

// sameHostRedirect follows at most maxRedirects redirects, and only to the same host over
// http or https.
func sameHostRedirect(req *http.Request, via []*http.Request) error {
	if len(via) >= maxRedirects {
		return errs.New("too many redirects", "max", maxRedirects)
	}
	if req.URL.Host != via[0].URL.Host || (req.URL.Scheme != "http" && req.URL.Scheme != "https") {
		return errs.New("refused a redirect to another host or scheme")
	}
	return nil
}
