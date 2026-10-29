package sys

import (
	"bufio"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
)

var guestOnce = sync.OnceValue(func() bool { return runtime.GOOS == "linux" && guestAt("/") })

// RunsAsGuest reports whether this Linux process runs in a container or VM, where SMART data of
// the physical disks is not visible. It is a best-effort check: container markers left by
// systemd, Docker and Podman, then the CPU's hypervisor flag.
func RunsAsGuest() bool {
	return guestOnce()
}

// guestAt runs the RunsAsGuest checks against a filesystem rooted at root.
func guestAt(root string) bool {
	for _, marker := range []string{"run/systemd/container", ".dockerenv", "run/.containerenv"} {
		if _, err := os.Stat(filepath.Join(root, marker)); err == nil {
			return true
		}
	}
	f, err := os.Open(filepath.Join(root, "proc/cpuinfo"))
	if err != nil {
		return false
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		key, val, ok := strings.Cut(sc.Text(), ":")
		if ok && strings.TrimSpace(key) == "flags" {
			return strings.Contains(" "+val+" ", " hypervisor ")
		}
	}
	return false
}
