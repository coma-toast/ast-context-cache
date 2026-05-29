package version

import (
	_ "embed"
	"strings"
)

// Version is the release version (overridden at link time via -ldflags).
var Version = "dev"

//go:embed VERSION
var embedded string

func init() {
	if Version != "dev" {
		return
	}
	if v := strings.TrimSpace(embedded); v != "" {
		Version = v
	}
}
