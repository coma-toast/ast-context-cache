package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// mcp-local (github.com/coma-toast/mcp-local) records each service it starts in
// ~/.mcp-local/pids/<service>.pid, as JSON {"pid","start","exe"} or a legacy plain PID,
// and treats the service as its own only while that PID is alive with the recorded start
// time. A spawn restart changes our PID, so the file must follow it or mcp-local reports
// the service stale and can no longer stop it.

// mcpLocalIdentity mirrors mcp-local's process.Identity.
type mcpLocalIdentity struct {
	PID   int    `json:"pid"`
	Start string `json:"start,omitempty"`
	Exe   string `json:"exe,omitempty"`
}

func mcpLocalPIDDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".mcp-local", "pids")
}

// adoptMCPLocalPIDFile rewrites every PID file in dir that names oldPID to name newPID,
// with the start time and executable info reports for it. Files naming other processes
// are left alone, so this does nothing unless mcp-local started us. It returns the paths
// it rewrote.
func adoptMCPLocalPIDFile(dir string, oldPID, newPID int, info func(pid int) (start, exe string, err error)) []string {
	if dir == "" {
		return nil
	}
	paths, _ := filepath.Glob(filepath.Join(dir, "*.pid"))
	var updated []string
	for _, p := range paths {
		data, err := os.ReadFile(p)
		if err != nil || pidFilePID(data) != oldPID {
			continue
		}
		id := mcpLocalIdentity{PID: newPID}
		if start, exe, err := info(newPID); err == nil {
			id.Start, id.Exe = start, exe
		} else {
			logger.Warn("Recording replacement PID without its start time", "path", p, "pid", newPID, "error", err)
		}
		if err := writePIDFile(p, id); err != nil {
			logger.Warn("Failed to update mcp-local PID file", "path", p, "pid", newPID, "error", err)
			continue
		}
		logger.Info("Updated mcp-local PID file", "path", p, "old_pid", oldPID, "pid", newPID)
		updated = append(updated, p)
	}
	return updated
}

// pidFilePID reads either PID file format; 0 when it names no PID.
func pidFilePID(data []byte) int {
	s := strings.TrimSpace(string(data))
	if strings.HasPrefix(s, "{") {
		var id mcpLocalIdentity
		if json.Unmarshal([]byte(s), &id) != nil {
			return 0
		}
		return id.PID
	}
	n, _ := strconv.Atoi(s)
	return n
}

// writePIDFile replaces path atomically, keeping mcp-local's 0600 mode.
func writePIDFile(path string, id mcpLocalIdentity) error {
	data, err := json.Marshal(id)
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// procIdentity returns pid's start time and executable exactly as mcp-local's
// process.ProcInfo computes them, since mcp-local compares the start string verbatim.
func procIdentity(pid int) (start, exe string, err error) {
	if runtime.GOOS == "linux" {
		return procIdentityLinux(pid)
	}
	cmd := exec.Command("ps", "-o", "lstart=,comm=", "-p", strconv.Itoa(pid))
	cmd.Env = append(os.Environ(), "LC_ALL=C")
	out, err := cmd.Output()
	if err != nil {
		return "", "", errs.New("no such process", "pid", pid)
	}
	f := strings.Fields(string(out))
	if len(f) < 6 {
		return "", "", errs.New("unexpected ps output", "pid", pid, "output", string(out))
	}
	return strings.Join(f[:5], " "), strings.Join(f[5:], " "), nil
}

func procIdentityLinux(pid int) (string, string, error) {
	dir := "/proc/" + strconv.Itoa(pid)
	stat, err := os.ReadFile(dir + "/stat")
	if err != nil {
		return "", "", errs.New("no such process", "pid", pid)
	}
	// comm (field 2) may contain spaces; starttime is field 22, the 20th after the ")".
	i := bytes.LastIndexByte(stat, ')')
	if i < 0 {
		return "", "", errs.New("unexpected stat", "pid", pid, "stat", string(stat))
	}
	f := strings.Fields(string(stat[i+1:]))
	if len(f) < 20 {
		return "", "", errs.New("unexpected stat", "pid", pid, "stat", string(stat))
	}
	boot, _ := os.ReadFile("/proc/sys/kernel/random/boot_id")
	start := strings.TrimSpace(string(boot)) + ":" + f[19]
	exe, err := os.Readlink(dir + "/exe")
	if err != nil {
		comm, _ := os.ReadFile(dir + "/comm")
		exe = strings.TrimSpace(string(comm))
	}
	return start, strings.TrimSuffix(exe, " (deleted)"), nil
}
