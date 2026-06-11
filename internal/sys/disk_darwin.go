//go:build darwin

package sys

import (
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

var (
	diskIOMu      sync.Mutex
	diskIOReady   bool
	diskIOAt      time.Time
	diskIORead    uint64
	diskIOWrite   uint64
	diskHealthMu  sync.Mutex
	diskHealth    SSDHealth
	diskHealthAt  time.Time
	diskHealthTTL = 5 * time.Minute
)

var (
	reIORead  = regexp.MustCompile(`"Bytes \(Read\)"=(\d+)`)
	reIOWrite = regexp.MustCompile(`"Bytes \(Write\)"=(\d+)`)
	reKV      = regexp.MustCompile(`(?m)^\s+([^:]+):\s+(.*)$`)
)

// DiskIORates returns read/write MB/s since the previous dashboard sample.
func DiskIORates() DiskIO {
	diskIOMu.Lock()
	defer diskIOMu.Unlock()
	now := time.Now()
	read, write, ok := blockStorageBytes()
	if !ok {
		return DiskIO{}
	}
	if !diskIOReady {
		diskIORead, diskIOWrite = read, write
		diskIOAt = now
		diskIOReady = true
		return DiskIO{}
	}
	elapsed := now.Sub(diskIOAt).Seconds()
	if elapsed < 0.5 {
		return DiskIO{}
	}
	dRead := float64(read-diskIORead) / (1024 * 1024) / elapsed
	dWrite := float64(write-diskIOWrite) / (1024 * 1024) / elapsed
	if dRead < 0 {
		dRead = 0
	}
	if dWrite < 0 {
		dWrite = 0
	}
	diskIORead, diskIOWrite = read, write
	diskIOAt = now
	return DiskIO{ReadMBps: dRead, WriteMBps: dWrite}
}

func blockStorageBytes() (read, write uint64, ok bool) {
	out, err := exec.Command("ioreg", "-r", "-c", "IOBlockStorageDriver").Output()
	if err != nil {
		return 0, 0, false
	}
	text := string(out)
	if m := reIORead.FindStringSubmatch(text); len(m) == 2 {
		read, _ = strconv.ParseUint(m[1], 10, 64)
	}
	if m := reIOWrite.FindStringSubmatch(text); len(m) == 2 {
		write, _ = strconv.ParseUint(m[1], 10, 64)
	}
	return read, write, read > 0 || write > 0
}

// SSDHealthInfo returns cached SSD health for the boot disk (diskutil + NVMe profile).
func SSDHealthInfo() SSDHealth {
	diskHealthMu.Lock()
	defer diskHealthMu.Unlock()
	if diskHealth.Available && time.Since(diskHealthAt) < diskHealthTTL {
		return diskHealth
	}
	h := probeSSDHealth()
	diskHealth = h
	diskHealthAt = time.Now()
	return h
}

func probeSSDHealth() SSDHealth {
	h := SSDHealth{Device: "disk0"}
	out, err := exec.Command("diskutil", "info", "disk0").Output()
	if err != nil {
		return h
	}
	kv := parseDiskutilKV(string(out))
	h.Model = kv["device / media name"]
	if h.Model == "" {
		h.Model = kv["media name"]
	}
	h.SmartStatus = kv["smart status"]
	h.Protocol = kv["protocol"]
	h.SolidState = strings.EqualFold(kv["solid state"], "yes")
	if h.Model != "" || h.SmartStatus != "" {
		h.Available = true
	}
	if prof := nvmeProfile(); prof != nil {
		if h.Model == "" {
			h.Model = prof["model"]
		}
		if h.SmartStatus == "" {
			h.SmartStatus = prof["smart"]
		}
		if h.Capacity == "" {
			h.Capacity = prof["capacity"]
		}
		h.TrimSupport = strings.EqualFold(prof["trim"], "yes")
		h.Available = true
	}
	return h
}

func parseDiskutilKV(text string) map[string]string {
	out := map[string]string{}
	for _, m := range reKV.FindAllStringSubmatch(text, -1) {
		if len(m) != 3 {
			continue
		}
		out[strings.ToLower(strings.TrimSpace(m[1]))] = strings.TrimSpace(m[2])
	}
	return out
}

func nvmeProfile() map[string]string {
	out, err := exec.Command("system_profiler", "SPNVMeDataType").Output()
	if err != nil {
		return nil
	}
	lines := strings.Split(string(out), "\n")
	result := map[string]string{}
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if !strings.Contains(line, ":") {
			continue
		}
		parts := strings.SplitN(line, ":", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.ToLower(strings.TrimSpace(parts[0]))
		val := strings.TrimSpace(parts[1])
		switch key {
		case "model":
			result["model"] = val
		case "s.m.a.r.t. status":
			result["smart"] = val
		case "capacity":
			if result["capacity"] == "" {
				result["capacity"] = val
			}
		case "trim support":
			result["trim"] = val
		}
	}
	if len(result) == 0 {
		return nil
	}
	return result
}
