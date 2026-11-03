package hostdisk

import (
	"context"
	"errors"
	"net/url"
	"strings"
	"time"
)

// TestResult is the outcome of a one-off check of a host disk health URL from Settings.
type TestResult struct {
	OK        bool     `json:"ok"`
	Disks     int      `json:"disks"`
	Devices   []string `json:"devices,omitempty"`
	LatencyMs int64    `json:"latency_ms"`
	Error     string   `json:"error,omitempty"`
	Hint      string   `json:"hint,omitempty"`
}

// Test fetches and parses raw once, with the same limits as the poller, without touching the
// saved setting or the disk card. A failure carries a hint for the usual setup mistakes.
func Test(ctx context.Context, raw string) TestResult {
	u, err := ParseURL(raw)
	if err != nil {
		return TestResult{Error: err.Error()}
	}
	start := time.Now()
	disks, kind, err := NewSampler(nil, nil).fetch(ctx, u)
	res := TestResult{LatencyMs: time.Since(start).Milliseconds()}
	if err != nil {
		res.Error, res.Hint = err.Error(), testHint(u, kind, err)
		return res
	}
	res.OK, res.Disks = true, len(disks)
	for _, d := range disks {
		res.Devices = append(res.Devices, d.Device)
	}
	return res
}

func testHint(u *url.URL, kind string, err error) string {
	msg := err.Error()
	noPath := u.Path == "" || u.Path == "/"
	switch {
	case strings.Contains(msg, "server gave HTTP response to HTTPS client"):
		return "The exporter speaks plain HTTP: change https:// to http://."
	case strings.Contains(msg, "connection refused"):
		return "Nothing is listening on that port. Check that smartctl_exporter runs on the host and listens on this address (default port 9633)."
	case strings.Contains(msg, "no such host"):
		return "This machine can't resolve that host name. Try the host's IP address."
	case errors.Is(err, context.DeadlineExceeded) || strings.Contains(msg, "Client.Timeout"):
		return "No answer within 5 s. Check the address, and that a firewall allows port 9633 from this machine."
	case (kind == kindStatus || kind == kindParse) && noPath:
		return "Add the /metrics path: " + withMetricsPath(u)
	case kind == kindStatus || kind == kindParse:
		return "That URL doesn't serve smartctl_exporter metrics. The exporter serves them at /metrics."
	case kind == kindNoDevices:
		return "The exporter answers but reports no disks. It needs root and smartmontools 7.0 or newer."
	}
	return ""
}

func withMetricsPath(u *url.URL) string {
	c := *u
	c.Path = "/metrics"
	return c.String()
}
