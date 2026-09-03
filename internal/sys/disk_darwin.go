//go:build darwin

package sys

import (
	"encoding/json"
	"math"
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
	diskHealthFor string
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

// SSDHealthInfo returns cached health for the disk holding path — the boot disk by
// default (diskutil + NVMe profile, as before), or whatever disk path resolves to once
// the data directory has been moved (e.g. to a USB drive).
func SSDHealthInfo(path string) SSDHealth {
	diskHealthMu.Lock()
	defer diskHealthMu.Unlock()
	if diskHealth.Available && diskHealthFor == path && time.Since(diskHealthAt) < diskHealthTTL {
		return diskHealth
	}
	h := probeDiskHealth(path)
	diskHealth = h
	diskHealthFor = path
	diskHealthAt = time.Now()
	return h
}

// probeDiskHealth resolves path to its containing disk via `diskutil info`, which
// accepts an arbitrary file/mount path and not just a disk or volume identifier. The
// internal boot disk keeps the exact behavior this had before (Apple NVMe controllers
// expose richer SMART data through system_profiler and diskutil's raw SMART key dump
// than the generic path below asks for); any other disk — typically a moved external/
// USB data directory — is probed generically, with smartmontools tried on top when
// installed, since it sees through some USB bridges that diskutil's SMART Status can't.
func probeDiskHealth(path string) SSDHealth {
	empty := SSDHealth{WearUsedPct: -1, SparePct: -1, DataWrittenTB: -1, TemperatureC: -1}
	device := deviceForPath(path)
	if device == "" {
		return empty
	}
	out, err := exec.Command("diskutil", "info", device).Output()
	if err != nil {
		return empty
	}
	kv := parseDiskutilKV(string(out))
	wholeDisk := kv["part of whole"]
	if wholeDisk == "" {
		wholeDisk = kv["device identifier"]
	}
	// There's exactly one internal disk controller worth profiling this way regardless
	// of which specific APFS volume/container path resolves to (the boot volume is a
	// synthesized container like disk3, not the physical disk0) — "not external" is the
	// actual condition that matters, not a specific disk identifier.
	if !strings.EqualFold(kv["device location"], "external") {
		return probeSSDHealth()
	}

	h := SSDHealth{
		Device:        wholeDisk,
		Model:         kv["device / media name"],
		SmartStatus:   kv["smart status"],
		Protocol:      kv["protocol"],
		SolidState:    strings.EqualFold(kv["solid state"], "yes"),
		TrimSupport:   strings.EqualFold(kv["trim support"], "yes"),
		IsExternal:    strings.EqualFold(kv["device location"], "external"),
		Capacity:      humanBeforeParen(kv["disk size"]),
		FreeSpace:     humanBeforeParen(kv["volume free space"]),
		WearUsedPct:   -1,
		SparePct:      -1,
		DataWrittenTB: -1,
		TemperatureC:  -1,
	}
	if h.Model == "" {
		h.Model = kv["media name"]
	}
	if h.SmartStatus != "" {
		h.SmartSource = "diskutil"
	}
	if h.Model != "" || h.SmartStatus != "" || h.Capacity != "" {
		h.Available = true
	}
	if applySmartctl(&h, wholeDisk) {
		h.Available = true
	}
	return h
}

func probeSSDHealth() SSDHealth {
	h := SSDHealth{Device: "disk0", WearUsedPct: -1, SparePct: -1, DataWrittenTB: -1, TemperatureC: -1}
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
	h.FreeSpace = humanBeforeParen(kv["volume free space"])
	if h.SmartStatus != "" {
		h.SmartSource = "diskutil"
	}
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
	applySmartWear(&h, smartKeysFromDiskutil())
	applySmartctl(&h, "disk0")
	return h
}

// humanBeforeParen extracts the human-readable prefix of a diskutil size field, e.g.
// "263.1 MB" from "263.1 MB (263090176 Bytes) (exactly 513848 512-Byte-Units)".
func humanBeforeParen(v string) string {
	if i := strings.Index(v, " ("); i > 0 {
		return v[:i]
	}
	return v
}

// deviceForPath resolves an arbitrary file/directory path to the device identifier
// (e.g. "disk3s5") of the volume containing it. `diskutil info` only accepts a disk,
// volume, or mount-point identifier — not an arbitrary subdirectory — so an ordinary
// path like ~/.astcache has to be resolved to its containing filesystem via `df` first.
func deviceForPath(path string) string {
	out, err := exec.Command("df", path).Output()
	if err != nil {
		return ""
	}
	lines := strings.Split(strings.TrimRight(string(out), "\n"), "\n")
	if len(lines) < 2 {
		return ""
	}
	fields := strings.Fields(lines[len(lines)-1])
	if len(fields) == 0 {
		return ""
	}
	return strings.TrimPrefix(fields[0], "/dev/")
}

const nvmeDataUnitBytes = 512_000 // NVMe SMART log data unit size

func applySmartWear(h *SSDHealth, smart map[string]uint64) {
	if len(smart) == 0 {
		return
	}
	if v, ok := smart["PERCENTAGE_USED"]; ok {
		h.WearUsedPct = int(v)
		h.Available = true
	}
	if v, ok := smart["AVAILABLE_SPARE"]; ok {
		h.SparePct = int(v)
		h.Available = true
	}
	low := smart["DATA_UNITS_WRITTEN_0"]
	high := smart["DATA_UNITS_WRITTEN_1"]
	if low > 0 || high > 0 {
		units := high<<32 | low
		tb := float64(units*nvmeDataUnitBytes) / 1e12
		if !math.IsNaN(tb) && tb >= 0 {
			h.DataWrittenTB = math.Round(tb*10) / 10
			h.Available = true
		}
	}
	if v, ok := smart["TEMPERATURE"]; ok && v > 0 {
		h.TemperatureC = float64(v) / 10
	}
}

type smartctlResult struct {
	SmartStatus *struct {
		Passed bool `json:"passed"`
	} `json:"smart_status"`
	Temperature *struct {
		Current int `json:"current"`
	} `json:"temperature"`
	ModelName                     string `json:"model_name"`
	ModelFamily                   string `json:"model_family"`
	NVMeSmartHealthInformationLog *struct {
		PercentageUsed   int    `json:"percentage_used"`
		AvailableSpare   int    `json:"available_spare"`
		DataUnitsWritten uint64 `json:"data_units_written"`
	} `json:"nvme_smart_health_information_log"`
}

// applySmartctl tries smartmontools, when installed, on top of whatever diskutil
// already found — it sees through some USB-SATA/USB-NVMe bridge chips that diskutil's
// own SMART Status reports as "Not Supported". Only fills in fields diskutil (or, for
// the internal disk, the NVMe profile) left unknown, so a working diskutil/NVMe read
// is never overwritten by a weaker smartctl one. Returns whether anything was added.
func applySmartctl(h *SSDHealth, wholeDisk string) bool {
	if wholeDisk == "" {
		return false
	}
	smartctlPath, err := exec.LookPath("smartctl")
	if err != nil {
		return false
	}
	// smartctl exits non-zero for various informational bit flags (e.g. "SMART usage
	// attribute exceeded") even when it produced perfectly good JSON, so parse
	// whatever came back on stdout regardless of the exit status.
	out, _ := exec.Command(smartctlPath, "-a", "-j", "/dev/"+wholeDisk).Output()
	return applySmartctlJSON(h, out)
}

// applySmartctlJSON fills in whatever smartctl -j reported that diskutil (or the NVMe
// profile) hadn't already found. Split out from applySmartctl so the parsing/merge
// logic is testable without shelling out to a real smartctl binary.
func applySmartctlJSON(h *SSDHealth, out []byte) bool {
	if len(out) == 0 {
		return false
	}
	var r smartctlResult
	if json.Unmarshal(out, &r) != nil {
		return false
	}
	if r.SmartStatus == nil && r.Temperature == nil && r.NVMeSmartHealthInformationLog == nil {
		// Most commonly a permission error (raw device access needs root on macOS) or
		// the bridge chip didn't answer — nothing usable came back either way.
		return false
	}

	found := false
	if r.SmartStatus != nil && (h.SmartStatus == "" || strings.EqualFold(h.SmartStatus, "not supported")) {
		if r.SmartStatus.Passed {
			h.SmartStatus = "Verified"
		} else {
			h.SmartStatus = "Failing"
		}
		h.SmartSource = "smartctl"
		found = true
	}
	if h.Model == "" {
		if r.ModelName != "" {
			h.Model = r.ModelName
		} else if r.ModelFamily != "" {
			h.Model = r.ModelFamily
		}
	}
	if r.Temperature != nil && r.Temperature.Current > 0 && h.TemperatureC < 0 {
		h.TemperatureC = float64(r.Temperature.Current)
		found = true
	}
	if log := r.NVMeSmartHealthInformationLog; log != nil {
		if h.WearUsedPct < 0 {
			h.WearUsedPct = log.PercentageUsed
			found = true
		}
		if h.SparePct < 0 {
			h.SparePct = log.AvailableSpare
			found = true
		}
		if h.DataWrittenTB < 0 && log.DataUnitsWritten > 0 {
			tb := float64(log.DataUnitsWritten*nvmeDataUnitBytes) / 1e12
			if !math.IsNaN(tb) && tb >= 0 {
				h.DataWrittenTB = math.Round(tb*10) / 10
				found = true
			}
		}
	}
	return found
}

func smartKeysFromDiskutil() map[string]uint64 {
	out, err := exec.Command("bash", "-c", "diskutil info -plist disk0 2>/dev/null | plutil -convert json -r -o - -").Output()
	if err != nil {
		return nil
	}
	var parsed struct {
		Smart map[string]interface{} `json:"SMARTDeviceSpecificKeysMayVaryNotGuaranteed"`
	}
	if json.Unmarshal(out, &parsed) != nil || len(parsed.Smart) == 0 {
		return nil
	}
	result := map[string]uint64{}
	for k, v := range parsed.Smart {
		if n, ok := smartUint(v); ok {
			result[k] = n
		}
	}
	return result
}

func smartUint(v interface{}) (uint64, bool) {
	switch x := v.(type) {
	case float64:
		if x < 0 {
			return 0, false
		}
		return uint64(x), true
	case json.Number:
		n, err := x.Int64()
		if err != nil || n < 0 {
			return 0, false
		}
		return uint64(n), true
	case int:
		if x < 0 {
			return 0, false
		}
		return uint64(x), true
	case int64:
		if x < 0 {
			return 0, false
		}
		return uint64(x), true
	default:
		return 0, false
	}
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
