package installer

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
)

// macOS: the app bundle's main executable, wherever the bundle lives (/Applications or
// ~/Applications). Its helpers live under Contents/Frameworks, so they don't match.
const darwinClaudeDesktopSuffix = "/Claude.app/Contents/MacOS/Claude"

// Linux has no official build; the community packages run Electron from a claude-desktop
// directory (/usr/lib/claude-desktop/...) or a claude-desktop launcher.
const linuxClaudeDesktopDir = "claude-desktop"

// claudeDesktopRunning reports whether Claude Desktop's main process is running. Detection
// failures report false: the plan then shows the generic warning instead of the running one.
func claudeDesktopRunning(goos string) bool {
	switch goos {
	case "darwin":
		// On macOS, comm is the full executable path.
		out, err := exec.Command("ps", "-axo", "comm=").Output()
		if err != nil {
			return false
		}
		return slices.ContainsFunc(strings.Split(string(out), "\n"), func(line string) bool {
			return isClaudeDesktopProcess(goos, strings.TrimSpace(line))
		})
	case "linux":
		cmdlines, _ := filepath.Glob("/proc/[0-9]*/cmdline")
		return slices.ContainsFunc(cmdlines, func(p string) bool {
			b, err := os.ReadFile(p)
			if err != nil {
				return false
			}
			argv0, _, _ := bytes.Cut(b, []byte{0})
			return isClaudeDesktopProcess(goos, string(argv0))
		})
	}
	return false
}

// isClaudeDesktopProcess matches one process's executable (macOS) or argv[0] (Linux).
func isClaudeDesktopProcess(goos, exe string) bool {
	switch goos {
	case "darwin":
		return strings.HasSuffix(exe, darwinClaudeDesktopSuffix)
	case "linux":
		return slices.Contains(strings.Split(exe, "/"), linuxClaudeDesktopDir)
	}
	return false
}
