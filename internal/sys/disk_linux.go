//go:build linux

package sys

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	// smartctl --scan-open opens every candidate device, so a hung USB bridge
	// can stall it well past a normal device probe.
	smartctlScanTimeout   = 60 * time.Second
	smartctlDeviceTimeout = 30 * time.Second
	// Probe a few disks at once: USB SAT round-trips are slow and independent.
	smartctlMaxParallel = 4
)

// nvmeDataUnitBytes is the NVMe SMART log data unit size (512 KiB).
const nvmeDataUnitBytes = 512_000

// DiskHealthInfo returns SMART health for every locally accessible disk,
// with the disk holding path (typically the data dir) first. The second
// return value is an environment-level note when SMART data is unavailable
// (smartctl missing, no permission, no local disks); it is empty when the
// returned list is trustworthy.
func DiskHealthInfo(path string) ([]DiskInfo, string) {
	smartctlPath, err := exec.LookPath("smartctl")
	if err != nil {
		return nil, "smartctl not found — install smartmontools (e.g. apt install smartmontools)"
	}

	blk := lsblkDisks()
	dataDev, freeSpace := dataDirDevice(path)

	ctx, cancel := context.WithTimeout(context.Background(), smartctlScanTimeout)
	scanOut, scanErr := exec.CommandContext(ctx, smartctlPath, "--scan-open", "-j").Output()
	scanTimedOut := errors.Is(ctx.Err(), context.DeadlineExceeded)
	cancel()

	devs := parseSmartctlScan(scanOut)
	if devs == nil {
		if scanTimedOut {
			return nil, "smartctl --scan-open timed out after 60s (a USB bridge may be hanging)"
		}
		return nil, "smartctl --scan-open failed: " + firstLine(stderrOf(scanErr, scanOut))
	}
	if len(devs) == 0 {
		return nil, "smartctl found no locally accessible disks (container?); use the Proxmox source for host disks"
	}

	// Probe each device in parallel (bounded) so a slow USB disk does not
	// serialize the whole sweep.
	results := make([]DiskInfo, len(devs))
	sem := make(chan struct{}, smartctlMaxParallel)
	var wg sync.WaitGroup
	for i := range devs {
		i, d := i, devs[i]
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			results[i] = probeSmartctlDevice(smartctlPath, d, blk, dataDev, freeSpace)
		}()
	}
	wg.Wait()

	permDenied := 0
	for i := range results {
		if strings.Contains(results[i].WarningMsg, "permission denied") {
			permDenied++
		}
	}
	sort.SliceStable(results, func(i, j int) bool {
		if results[i].DataDirDisk != results[j].DataDirDisk {
			return results[i].DataDirDisk
		}
		return results[i].Device < results[j].Device
	})
	if permDenied == len(results) {
		return results, "SMART unreadable on all disks: permission denied — run as root or grant CAP_SYS_RAWIO"
	}
	if permDenied > 0 {
		return results, fmt.Sprintf("SMART unreadable on %d of %d disk(s): permission denied — run as root or grant CAP_SYS_RAWIO", permDenied, len(results))
	}
	return results, ""
}

// smartctlScannedDev is one entry of smartctl --scan-open -j ("devices" array).
type smartctlScannedDev struct {
	Name      string `json:"name"`
	InfoName  string `json:"info_name"`
	Type      string `json:"type"`
	Protocol  string `json:"protocol"`
	OpenError string `json:"open_error"`
}

// parseSmartctlScan decodes the --scan-open -j output. It returns nil when no
// parseable scan came back; a parseable-but-empty device list is returned as
// an empty (non-nil) slice so callers can tell the two apart.
func parseSmartctlScan(out []byte) []smartctlScannedDev {
	var r struct {
		Devices []smartctlScannedDev `json:"devices"`
	}
	if json.Unmarshal(out, &r) != nil {
		return nil
	}
	devs := r.Devices
	return devs
}

// newDiskInfo starts a DiskInfo with the identity smartctl already knows and
// every numeric attribute in the "unknown" (-1) state.
func newDiskInfo(device, smartType, protocol string) DiskInfo {
	return DiskInfo{
		Device:               device,
		SmartType:            smartType,
		Protocol:             protocol,
		SmartStatus:          "Unknown",
		WearUsedPct:          -1,
		SparePct:             -1,
		DataWrittenTB:        -1,
		TemperatureC:         -1,
		PowerOnHours:         -1,
		PowerCycles:          -1,
		UnsafeShutdowns:      -1,
		MediaErrors:          -1,
		CriticalWarning:      -1,
		ReallocatedSectors:   -1,
		PendingSectors:       -1,
		OfflineUncorrectable: -1,
		UDMACRCErrors:        -1,
		SSDLifeLeft:          -1,
		SASGrownDefects:      -1,
		SASUncorrectedErrors: -1,
	}
}

// probeSmartctlDevice runs `smartctl -a -j -d <type> <dev>` for one scanned
// device and fills a DiskInfo from the response plus lsblk/df context.
func probeSmartctlDevice(smartctlPath string, d smartctlScannedDev, blk map[string]lsblkDisk, dataDev, freeSpace string) DiskInfo {
	di := newDiskInfo(d.Name, d.Type, d.Protocol)

	if d.OpenError != "" {
		di.WarningMsg = smartctlErrorNote(nil, []byte(d.OpenError))
		enrichFromLsblk(&di, blk)
		return di
	}

	ctx, cancel := context.WithTimeout(context.Background(), smartctlDeviceTimeout)
	out, err := exec.CommandContext(ctx, smartctlPath, "-a", "-j", "-d", d.Type, d.Name).Output()
	timedOut := errors.Is(ctx.Err(), context.DeadlineExceeded)
	cancel()

	if !applySmartctlFull(&di, out) {
		if timedOut {
			di.WarningMsg = "SMART probe timed out after 30s"
		} else {
			di.WarningMsg = smartctlErrorNote(err, out)
		}
	}

	enrichFromLsblk(&di, blk)
	if dataDev != "" && dataDevMatches(di.Device, dataDev, blk) {
		di.DataDirDisk = true
		di.FreeSpace = freeSpace
	}
	return di
}

// jsonBytes accepts smartctl's user_capacity as either a bare byte count
// (smartmontools < 7) or a {"blocks":…, "bytes":…} object (7+).
type jsonBytes struct {
	Bytes uint64
}

func (b *jsonBytes) UnmarshalJSON(data []byte) error {
	var n json.Number
	if err := json.Unmarshal(data, &n); err == nil {
		if v, err := n.Int64(); err == nil && v > 0 {
			b.Bytes = uint64(v)
		}
		return nil
	}
	var o struct {
		Bytes json.Number `json:"bytes"`
	}
	if err := json.Unmarshal(data, &o); err != nil {
		return nil
	}
	if v, err := o.Bytes.Int64(); err == nil && v > 0 {
		b.Bytes = uint64(v)
	}
	return nil
}

type smartctlAttr struct {
	ID    int    `json:"id"`
	Name  string `json:"name"`
	Value int    `json:"value"`
	Raw   *struct {
		Value  json.Number `json:"value"`
		String string      `json:"string"`
	} `json:"raw"`
}

// smartctlFull mirrors the device sections of smartctl -a -j. Field names
// follow the smartmontools 7.4 JSON schema; the 6.x attribute-table key is
// kept as a fallback for older distro packages.
type smartctlFull struct {
	Device *struct {
		Name     string `json:"name"`
		InfoName string `json:"info_name"`
		Type     string `json:"type"`
		Protocol string `json:"protocol"`
	} `json:"device"`
	SmartStatus *struct {
		Passed bool `json:"passed"`
	} `json:"smart_status"`
	ModelName       string     `json:"model_name"`
	ModelFamily     string     `json:"model_family"`
	SerialNumber    string     `json:"serial_number"`
	FirmwareVersion string     `json:"firmware_version"`
	UserCapacity    *jsonBytes `json:"user_capacity"`
	RotationRate    *int       `json:"rotation_rate"`
	FormFactor      *struct {
		Name string `json:"name"`
	} `json:"form_factor"`
	Trim *struct {
		Supported bool `json:"supported"`
	} `json:"trim"`
	Temperature *struct {
		Current float64 `json:"current"`
	} `json:"temperature"`
	PowerOnTime *struct {
		Hours json.Number `json:"hours"`
	} `json:"power_on_time"`
	PowerCycleCount *json.Number `json:"power_cycle_count"`
	NVMeSmart       *struct {
		CriticalWarning  int         `json:"critical_warning"`
		AvailableSpare   int         `json:"available_spare"`
		PercentageUsed   int         `json:"percentage_used"`
		DataUnitsRead    json.Number `json:"data_units_read"`
		DataUnitsWritten json.Number `json:"data_units_written"`
		MediaErrors      json.Number `json:"media_errors"`
		UnsafeShutdowns  json.Number `json:"unsafe_shutdowns"`
		PowerOnHours     json.Number `json:"power_on_hours"`
		PowerCycles      json.Number `json:"power_cycles"`
	} `json:"nvme_smart_health_information_log"`
	ATAAttributes *struct {
		Table []smartctlAttr `json:"table"`
	} `json:"ata_smart_attributes"`
	ATAAttributes6x *struct {
		Table []smartctlAttr `json:"table"`
	} `json:"ata_smart_read_attributes"`
	SCSIModelName       string       `json:"scsi_model_name"`
	SCSIVendor          string       `json:"scsi_vendor"`
	SCSIProduct         string       `json:"scsi_product"`
	ScsiGrownDefectList *json.Number `json:"scsi_grown_defect_list"`
	ScsiErrorCounterLog map[string]*struct {
		TotalUncorrectedErrors json.Number `json:"total_uncorrected_errors"`
	} `json:"scsi_error_counter_log"`
	Smartctl *struct {
		ExitStatus int `json:"exit_status"`
		Messages   []struct {
			String string `json:"string"`
		} `json:"messages"`
	} `json:"smartctl"`
}

// applySmartctlFull fills d from one `smartctl -a -j` response. It returns
// false when no usable device data came back (open failed, permission
// denied, empty or unparseable output).
func applySmartctlFull(d *DiskInfo, out []byte) bool {
	dec := json.NewDecoder(strings.NewReader(string(out)))
	dec.UseNumber()
	var f smartctlFull
	if dec.Decode(&f) != nil {
		return false
	}
	if f.Device != nil {
		if f.Device.Name != "" {
			d.Device = f.Device.Name
		}
		if f.Device.Protocol != "" {
			d.Protocol = f.Device.Protocol
		}
		if f.Device.Type != "" {
			d.SmartType = f.Device.Type
		}
	}
	if f.SmartStatus == nil && f.ModelName == "" && f.ModelFamily == "" &&
		f.SCSIModelName == "" && f.NVMeSmart == nil && f.ATAAttributes == nil &&
		f.ATAAttributes6x == nil && f.Temperature == nil && f.PowerOnTime == nil {
		// No device data at all — most commonly a permission failure whose
		// message only lives in smartctl.messages.
		return false
	}

	d.SmartSource = "smartctl"
	if f.SmartStatus != nil {
		if f.SmartStatus.Passed {
			d.SmartStatus = "Verified"
		} else {
			d.SmartStatus = "Failing"
		}
	}
	if d.Model == "" {
		d.Model = f.ModelName
	}
	if d.Model == "" {
		d.Model = f.ModelFamily
	}
	if d.Model == "" {
		d.Model = f.SCSIModelName
	}
	d.Serial = f.SerialNumber
	if f.UserCapacity != nil && f.UserCapacity.Bytes > 0 {
		d.Capacity = humanBytes(f.UserCapacity.Bytes)
	}
	if f.RotationRate != nil {
		d.SolidState = *f.RotationRate == 0
	}
	if f.Trim != nil {
		d.TrimSupport = f.Trim.Supported
	}
	if f.Temperature != nil && f.Temperature.Current > 0 {
		d.TemperatureC = f.Temperature.Current
	}
	if f.PowerOnTime != nil {
		if h, ok := intFromNumber(f.PowerOnTime.Hours); ok {
			d.PowerOnHours = h
		}
	}
	if f.PowerCycleCount != nil {
		if c, ok := intFromNumber(*f.PowerCycleCount); ok {
			d.PowerCycles = c
		}
	}

	if log := f.NVMeSmart; log != nil {
		d.CriticalWarning = log.CriticalWarning
		if d.WearUsedPct < 0 {
			d.WearUsedPct = log.PercentageUsed
		}
		if d.SparePct < 0 {
			d.SparePct = log.AvailableSpare
		}
		if d.DataWrittenTB < 0 {
			if units, ok := uintFromNumber(log.DataUnitsWritten); ok {
				d.DataWrittenTB = round1(float64(units*nvmeDataUnitBytes) / 1e12)
			}
		}
		if h, ok := intFromNumber(log.PowerOnHours); ok {
			if d.PowerOnHours < 0 {
				d.PowerOnHours = h
			}
		}
		if c, ok := intFromNumber(log.PowerCycles); ok {
			if d.PowerCycles < 0 {
				d.PowerCycles = c
			}
		}
		if u, ok := intFromNumber(log.UnsafeShutdowns); ok {
			if d.UnsafeShutdowns < 0 {
				d.UnsafeShutdowns = u
			}
		}
		if m, ok := intFromNumber(log.MediaErrors); ok {
			if d.MediaErrors < 0 {
				d.MediaErrors = m
			}
		}
	}

	attrs := f.ATAAttributes
	if attrs == nil || attrs.Table == nil {
		attrs = f.ATAAttributes6x
	}
	if attrs != nil {
		applyATAAttributes(d, attrs.Table)
	}

	if f.ScsiGrownDefectList != nil {
		if n, ok := intFromNumber(*f.ScsiGrownDefectList); ok {
			d.SASGrownDefects = n
		}
	}
	if len(f.ScsiErrorCounterLog) > 0 {
		total := 0
		for _, page := range f.ScsiErrorCounterLog {
			if page == nil {
				continue
			}
			if n, ok := intFromNumber(page.TotalUncorrectedErrors); ok {
				total += n
			}
		}
		d.SASUncorrectedErrors = total
	}
	return true
}

// applyATAAttributes maps the ATA attribute list onto the vendor-specific
// fields. Raw values are used (normalized "value" is not comparable across
// vendors); attributes the drive does not report leave the -1 default.
func applyATAAttributes(d *DiskInfo, attrs []smartctlAttr) {
	for _, a := range attrs {
		name := strings.ToLower(a.Name)
		// Vendor-specific names with no stable ID: unsafe-shutdown counters.
		if strings.Contains(name, "unsafe") {
			if raw, ok := attrRawInt(a); ok && d.UnsafeShutdowns < 0 {
				d.UnsafeShutdowns = raw
			}
			continue
		}
		raw, ok := attrRawInt(a)
		if !ok {
			continue
		}
		switch a.ID {
		case 5:
			d.ReallocatedSectors = raw
		case 12: // Power_Cycle_Count (ID 8 is Start_Stop_Count)
			if d.PowerCycles < 0 {
				d.PowerCycles = raw
			}
		case 9:
			if d.PowerOnHours < 0 {
				d.PowerOnHours = raw
			}
		case 197:
			d.PendingSectors = raw
		case 198:
			d.OfflineUncorrectable = raw
		case 199:
			d.UDMACRCErrors = raw
		case 231, 233: // SSD life left: Intel / Samsung
			if d.SSDLifeLeft < 0 {
				d.SSDLifeLeft = raw
			}
		case 202:
			// Vendor dependent: most drives report temperature here, some
			// (e.g. Micron) report a life indicator.
			if strings.Contains(name, "temp") {
				if d.TemperatureC < 0 {
					d.TemperatureC = float64(raw)
				}
			} else if d.SSDLifeLeft < 0 {
				d.SSDLifeLeft = raw
			}
		}
	}
}

func attrRawInt(a smartctlAttr) (int, bool) {
	if a.Raw == nil {
		return 0, false
	}
	n, ok := intFromNumber(a.Raw.Value)
	return n, ok
}

// lsblkDisk is the top-level entry for one whole disk from lsblk -J -b.
type lsblkDisk struct {
	Name       string
	Size       uint64
	Rota       int
	Transport  string
	Model      string
	Removable  int
	ChildNames []string
}

// lsblkDisks returns one entry per whole disk (NAME without the /dev/
// prefix). Missing or failing lsblk yields an empty map — the SMART data
// stands on its own without it.
func lsblkDisks() map[string]lsblkDisk {
	out, err := exec.Command("lsblk", "-J", "-b", "-n",
		"-o", "NAME,SIZE,ROTA,TRAN,MODEL,REMovable,MOUNTPOINT,TYPE").Output()
	if err != nil {
		return nil
	}
	return parseLsblkJSON(out)
}

func parseLsblkJSON(out []byte) map[string]lsblkDisk {
	var r struct {
		BlockDevices []struct {
			Name      string `json:"name"`
			Size      uint64 `json:"size"`
			Rota      int    `json:"rota"`
			Tran      string `json:"tran"`
			Model     string `json:"model"`
			Removable int    `json:"removable"`
			Type      string `json:"type"`
			Children  []struct {
				Name string `json:"name"`
			} `json:"children"`
		} `json:"blockdevices"`
	}
	if json.Unmarshal(out, &r) != nil {
		return nil
	}
	m := map[string]lsblkDisk{}
	for _, b := range r.BlockDevices {
		if b.Type != "disk" {
			continue
		}
		d := lsblkDisk{Name: b.Name, Size: b.Size, Rota: b.Rota, Transport: b.Tran, Model: b.Model, Removable: b.Removable}
		for _, c := range b.Children {
			d.ChildNames = append(d.ChildNames, c.Name)
		}
		m[b.Name] = d
	}
	return m
}

// enrichFromLsblk fills identity fields smartctl did not report (capacity,
// model) and derives IsExternal from the transport. It never overwrites
// fields smartctl already provided.
func enrichFromLsblk(d *DiskInfo, blk map[string]lsblkDisk) {
	if len(blk) == 0 {
		return
	}
	b, ok := blk[strings.TrimPrefix(d.Device, "/dev/")]
	if !ok {
		return
	}
	if d.Capacity == "" && b.Size > 0 {
		d.Capacity = humanBytes(b.Size)
	}
	if d.Model == "" && b.Model != "" {
		d.Model = b.Model
	}
	if b.Transport == "usb" {
		d.IsExternal = true
	}
	if !d.SolidState && b.Rota == 0 {
		d.SolidState = true
	}
}

// dataDirDevice resolves path to its backing device (via df) and the
// filesystem's free space. Empty results mean "could not determine".
func dataDirDevice(path string) (dev, freeSpace string) {
	out, err := exec.Command("df", "-k", path).Output()
	if err != nil {
		return "", ""
	}
	return parseDfOutput(out)
}

// parseDfOutput extracts (device, free space) from `df -k <path>` output.
func parseDfOutput(out []byte) (dev, freeSpace string) {
	lines := strings.Split(strings.TrimRight(string(out), "\n"), "\n")
	if len(lines) < 2 {
		return "", ""
	}
	fields := strings.Fields(lines[len(lines)-1])
	if len(fields) < 4 {
		return "", ""
	}
	availKiB, err := strconv.ParseUint(fields[3], 10, 64)
	if err != nil {
		return fields[0], ""
	}
	return fields[0], humanBytes(availKiB * 1024)
}

// dataDevMatches reports whether wholeDisk (a smartctl device name like
// /dev/nvme0n1) contains dev (a df source like /dev/nvme0n1p2) — either
// directly or via a partition from the lsblk tree, with a numeric-suffix
// strip as fallback when lsblk has no record of the disk.
func dataDevMatches(wholeDisk, dev string, blk map[string]lsblkDisk) bool {
	if wholeDisk == dev {
		return true
	}
	base := strings.TrimPrefix(dev, "/dev/")
	if b, ok := blk[strings.TrimPrefix(wholeDisk, "/dev/")]; ok {
		for _, c := range b.ChildNames {
			if c == base {
				return true
			}
		}
		return false
	}
	// No lsblk record for this disk: fall back to stripping the partition
	// suffix from the df source.
	return strings.TrimPrefix(stripPartitionSuffix(dev), "/dev/") ==
		strings.TrimPrefix(wholeDisk, "/dev/")
}

// stripPartitionSuffix reduces a partition name to its whole-disk name:
// /dev/nvme0n1p2 → /dev/nvme0n1, /dev/sda1 → /dev/sda, /dev/mmcblk0p1 →
// /dev/mmcblk0. Whole-disk names (no partition suffix) come back unchanged;
// unrecognized shapes degrade to "no match" in dataDevMatches rather than
// guessing.
func stripPartitionSuffix(dev string) string {
	i := strings.LastIndexByte(dev, '/')
	base := dev[i+1:]
	j := len(base)
	for j > 0 && base[j-1] >= '0' && base[j-1] <= '9' {
		j--
	}
	if j == len(base) {
		return dev
	}
	// <disk>p<n>: nvme0n1p2, mmcblk0p1 — the 'p' sits right after a digit.
	if j > 1 && base[j-1] == 'p' && base[j-2] >= '0' && base[j-2] <= '9' {
		return dev[:i+1+j-1]
	}
	// <disk><n>: sda1 — but not nvme0n1, whose trailing digit belongs to
	// the controller name (ends in 'n'+digit and is already a whole disk).
	if j >= 3 && base[j-1] != 'n' {
		return dev[:i+1+j]
	}
	return dev
}

// smartctlErrorNote turns a smartctl failure (and the JSON messages that
// carry the real error text) into a short, actionable note.
func smartctlErrorNote(err error, out []byte) string {
	text := string(out)
	if err != nil {
		var ee *exec.ExitError
		if errors.As(err, &ee) && len(ee.Stderr) > 0 {
			text = string(ee.Stderr) + "\n" + text
		} else {
			text = err.Error() + "\n" + text
		}
	}
	low := strings.ToLower(text)
	switch {
	case strings.Contains(low, "permission denied"),
		strings.Contains(low, "operation not permitted"),
		strings.Contains(low, "eacces"),
		strings.Contains(low, "eperm"):
		return "permission denied — smartctl needs root or CAP_SYS_RAWIO"
	case strings.Contains(low, "not found") && strings.Contains(low, "smartctl"):
		return "smartctl not found"
	}
	msg := firstLine(text)
	if msg == "" {
		msg = "no SMART data returned"
	}
	if len(msg) > 120 {
		msg = msg[:120]
	}
	return "SMART unreadable: " + msg
}

// humanBytes renders a byte count the way the UI already shows disk sizes
// (1024-based, one decimal: 931.5 GB).
func humanBytes(n uint64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := uint64(unit), 0
	for m := n / unit; m >= unit; m /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(n)/float64(div), "KMGTPE"[exp])
}

func round1(v float64) float64 {
	return math.Round(v*10) / 10
}

// intFromNumber converts a smartctl JSON number, tolerating values that do
// not fit an int64 (huge 128-bit counters) by returning ok=false for them.
func intFromNumber(n json.Number) (int, bool) {
	v, err := n.Int64()
	if err != nil || v < -1 {
		return 0, false
	}
	return int(v), true
}

func uintFromNumber(n json.Number) (uint64, bool) {
	v, err := n.Int64()
	if err != nil || v < 0 {
		return 0, false
	}
	return uint64(v), true
}

func firstLine(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	return strings.TrimSpace(s)
}

// stderrOf pulls the most useful text from a command failure: stderr first,
// then the (JSON) stdout that smartctl puts its messages in.
func stderrOf(err error, out []byte) string {
	var ee *exec.ExitError
	if errors.As(err, &ee) && len(ee.Stderr) > 0 {
		return firstLine(string(ee.Stderr))
	}
	if s := firstLine(string(out)); s != "" {
		return s
	}
	if err != nil {
		return err.Error()
	}
	return "unknown error"
}
