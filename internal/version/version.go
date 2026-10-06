package version

import (
	_ "embed"
	"strings"
)

// Version is the release version (overridden at link time via -ldflags).
var Version = "dev"

// Build is "release" for binaries GoReleaser publishes (set via -ldflags) and
// "source" for local builds, which can carry a release's version number while
// running unreleased code.
var Build = "source"

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
