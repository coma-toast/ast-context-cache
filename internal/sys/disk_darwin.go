//go:build darwin

package sys

import (
	"encoding/json"
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
	diskHealths   []DiskInfo
	diskHealthFor string
	diskHealthAt  time.Time
	diskHealthTTL = 5 * time.Minute
)

var (
	reIORead  = regexp.MustCompile(`"Bytes \(Read\)"=(\d+)`)
	reIOWrite = regexp.MustCompile(`"Bytes \(Write\)"=(\d+)`)
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

// DiskHealthInfo returns cached SMART/identity info for every disk on this Mac,
// with the disk holding path marked as DataDirDisk. The data-dir disk is sorted
// first, then the rest by disk id.
func DiskHealthInfo(path string) []DiskInfo {
	diskHealthMu.Lock()
	defer diskHealthMu.Unlock()
	if len(diskHealths) > 0 && diskHealthFor == path && time.Since(diskHealthAt) < diskHealthTTL {
		return diskHealths
	}
	diskHealths = probeAllDisks(path)
	diskHealthFor = path
	diskHealthAt = time.Now()
	return diskHealths
}

// SSDHealthInfo returns cached health for the disk holding path — a backwards-compatible
// projection of DiskHealthInfo. The data-dir disk (or first disk) is projected into SSDHealth.
func SSDHealthInfo(path string) SSDHealth {
	disks := DiskHealthInfo(path)
	if len(disks) == 0 {
		return SSDHealth{WearUsedPct: -1, SparePct: -1, DataWrittenTB: -1, TemperatureC: -1}
	}
	for _, d := range disks {
		if d.DataDirDisk {
			return SSDHealthFromDisk(d)
		}
	}
	return SSDHealthFromDisk(disks[0])
}

func probeAllDisks(path string) []DiskInfo {
	out, err := exec.Command("diskutil", "list").Output()
	if err != nil {
		return nil
	}
	ids := parseDiskutilList(string(out))
	if len(ids) == 0 {
		return nil
	}

	dataDirDisk := dataDirWholeDisk(path)
	prof := nvmeProfile()

	var disks []DiskInfo
	for i, id := range ids {
		d := probeOneDisk(id, prof, i == 0)
		if id == dataDirDisk {
			d.DataDirDisk = true
		}
		applyDiskWarnings(&d)
		disks = append(disks, d)
	}

	// Sort: data-dir disk first, then by original order (disk id).
	for i := range disks {
		if disks[i].DataDirDisk {
			disks[0], disks[i] = disks[i], disks[0]
			break
		}
	}
	return disks
}

func probeOneDisk(id string, prof map[string]string, firstInternal bool) DiskInfo {
	d := DiskInfo{Device: id}
	out, err := exec.Command("diskutil", "info", id).Output()
	if err != nil {
		return d
	}
	kv := parseDiskutilKV(string(out))
	d.Model = kv["device / media name"]
	if d.Model == "" {
		d.Model = kv["media name"]
	}
	d.SmartStatus = kv["smart status"]
	d.Protocol = kv["protocol"]
	d.Capacity = humanBeforeParen(kv["disk size"])
	d.FreeSpace = humanBeforeParen(kv["volume free space"])
	d.IsExternal = strings.EqualFold(kv["device location"], "external")
	d.TrimSupport = strings.EqualFold(kv["trim support"], "yes")
	ss := strings.EqualFold(kv["solid state"], "yes")
	d.SolidState = &ss
	d.Serial = kv["serial number (physical)"]
	if d.Serial == "" {
		d.Serial = kv["serial number"]
	}
	if d.SmartStatus != "" {
		d.SmartSource = "diskutil"
	}

	if !d.IsExternal {
		if prof != nil {
			if d.Model == "" {
				d.Model = prof["model"]
			}
			if d.SmartStatus == "" {
				d.SmartStatus = prof["smart"]
			}
			if d.Capacity == "" {
				d.Capacity = prof["capacity"]
			}
			if strings.EqualFold(prof["trim"], "yes") {
				d.TrimSupport = true
			}
			if firstInternal {
				applySmartWear(&d, smartKeysForDisk(id))
			}
		}
	}

	if applySmartctl(&d, id) {
		d.Available = true
	}
	if d.Model != "" || d.SmartStatus != "" || d.Capacity != "" {
		d.Available = true
	}
	return d
}

// dataDirWholeDisk resolves an arbitrary file/directory path to the whole-disk
// identifier (e.g. "disk0") of the volume containing it.
func dataDirWholeDisk(path string) string {
	device := deviceForPath(path)
	if device == "" {
		return ""
	}
	out, err := exec.Command("diskutil", "info", device).Output()
	if err != nil {
		return ""
	}
	kv := parseDiskutilKV(string(out))
	if w := kv["part of whole"]; w != "" {
		return w
	}
	return kv["device identifier"]
}

// deviceForPath resolves an arbitrary file/directory path to the device identifier
// (e.g. "disk3s5") of the volume containing it.
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

// applySmartctl tries smartmontools, when installed, on top of whatever diskutil
// already found. Only fills nil fields; a working diskutil/NVMe read is never
// overwritten by a weaker smartctl one. Returns whether anything was added.
func applySmartctl(d *DiskInfo, wholeDisk string) bool {
	if wholeDisk == "" {
		return false
	}
	smartctlPath, err := exec.LookPath("smartctl")
	if err != nil {
		return false
	}
	out, _ := exec.Command(smartctlPath, "-a", "-j", "/dev/"+wholeDisk).Output()
	return applySmartctlJSON(d, out)
}

func smartKeysForDisk(diskID string) map[string]uint64 {
	out, err := exec.Command("bash", "-c",
		"diskutil info -plist "+diskID+" 2>/dev/null | plutil -convert json -r -o - -").Output()
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
