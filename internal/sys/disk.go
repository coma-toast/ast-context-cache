package sys

import (
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"strings"
)

// DiskIO holds sampled host disk throughput (primary internal block device).
type DiskIO struct {
	ReadMBps  float64
	WriteMBps float64
}

// SSDHealth summarizes the health and identity of the disk holding a given path
// (typically the data directory — the internal boot SSD by default, or a moved
// external/USB drive), best-effort per OS and per what the drive/enclosure exposes.
type SSDHealth struct {
	Available     bool
	Device        string
	Model         string
	SmartStatus   string
	SmartSource   string // "diskutil", "smartctl", or "" if no SMART data was available
	Protocol      string
	Capacity      string
	FreeSpace     string
	IsExternal    bool
	SolidState    bool
	TrimSupport   bool
	WearUsedPct   int     // NVMe PERCENTAGE_USED; -1 if unknown
	SparePct      int     // NVMe AVAILABLE_SPARE; -1 if unknown
	DataWrittenTB float64 // lifetime host writes in TB; -1 if unknown
	TemperatureC  float64 // Celsius; -1 if unknown
}

// DiskInfo is one disk on the dashboard's disk health card; the dashboard's DiskInfo type reads
// the same field names. Pointer fields are omitted when the source does not report them, so
// the card shows nothing rather than a zero. Host marks a disk of the machine this instance runs
// on as a guest (an LXC container or VM), read from that host's smartctl_exporter.
type DiskInfo struct {
	Device               string   `json:"Device"`
	Model                string   `json:"Model,omitempty"`
	Serial               string   `json:"Serial,omitempty"`
	Protocol             string   `json:"Protocol,omitempty"`
	SmartType            string   `json:"SmartType,omitempty"`
	SmartStatus          string   `json:"SmartStatus,omitempty"`
	SmartSource          string   `json:"SmartSource,omitempty"`
	Capacity             string   `json:"Capacity,omitempty"`
	FreeSpace            string   `json:"FreeSpace,omitempty"`
	SolidState           *bool    `json:"SolidState,omitempty"`
	IsExternal           bool     `json:"IsExternal,omitempty"`
	TrimSupport          bool     `json:"TrimSupport,omitempty"`
	DataDirDisk          bool     `json:"DataDirDisk,omitempty"`
	Host                 bool     `json:"Host,omitempty"`
	Available            bool     `json:"Available,omitempty"`
	WarningLevel         string   `json:"WarningLevel,omitempty"`
	WarningMsg           string   `json:"WarningMsg,omitempty"`
	WearUsedPct          *int     `json:"WearUsedPct,omitempty"`
	SparePct             *int     `json:"SparePct,omitempty"`
	DataWrittenTB        *float64 `json:"DataWrittenTB,omitempty"`
	TemperatureC         *float64 `json:"TemperatureC,omitempty"`
	PowerOnHours         *int     `json:"PowerOnHours,omitempty"`
	PowerCycles          *int     `json:"PowerCycles,omitempty"`
	UnsafeShutdowns      *int     `json:"UnsafeShutdowns,omitempty"`
	MediaErrors          *int     `json:"MediaErrors,omitempty"`
	CriticalWarning      *int     `json:"CriticalWarning,omitempty"`
	ReallocatedSectors   *int     `json:"ReallocatedSectors,omitempty"`
	PendingSectors       *int     `json:"PendingSectors,omitempty"`
	OfflineUncorrectable *int     `json:"OfflineUncorrectable,omitempty"`
	UDMACRCErrors        *int     `json:"UDMACRCErrors,omitempty"`
	SASGrownDefects      *int     `json:"SASGrownDefects,omitempty"`
	SASUncorrectedErrors *int     `json:"SASUncorrectedErrors,omitempty"`
	SSDLifeLeft          *int     `json:"SSDLifeLeft,omitempty"`
}

// SSDHealthFromDisk projects one DiskInfo into the legacy SSDHealth shape the dashboard
// API and partials still read. Pointer fields become -1 when nil.
func SSDHealthFromDisk(d DiskInfo) SSDHealth {
	return SSDHealth{
		Available:     d.Available,
		Device:        d.Device,
		Model:         d.Model,
		SmartStatus:   d.SmartStatus,
		SmartSource:   d.SmartSource,
		Protocol:      d.Protocol,
		Capacity:      d.Capacity,
		FreeSpace:     d.FreeSpace,
		IsExternal:    d.IsExternal,
		SolidState:    derefBool(d.SolidState),
		TrimSupport:   d.TrimSupport,
		WearUsedPct:   derefNeg1(d.WearUsedPct),
		SparePct:      derefNeg1(d.SparePct),
		DataWrittenTB: derefNeg1F(d.DataWrittenTB),
		TemperatureC:  derefNeg1F(d.TemperatureC),
	}
}

func derefBool(p *bool) bool {
	if p == nil {
		return false
	}
	return *p
}

func derefNeg1(p *int) int {
	if p == nil {
		return -1
	}
	return *p
}

func derefNeg1F(p *float64) float64 {
	if p == nil {
		return -1
	}
	return *p
}

// --- shared disk helpers (no build tag; testable on any OS) ---

var reKV = regexp.MustCompile(`(?m)^\s+([^:]+):\s+(.*)$`)

const nvmeDataUnitBytes = 512_000 // NVMe SMART log data unit size

// parseDiskutilKV parses `diskutil info` key: value output into a lowercase-keyed map.
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

var reWholeDisk = regexp.MustCompile(`^disk\d+$`)
var reDiskHeader = regexp.MustCompile(`(?m)^\s*Disk:\s+(disk\d+)`)

// parseDiskutilList extracts whole-disk identifiers (disk0, disk1, …) from `diskutil list`
// plain-text output. Handles both APFS container headers ("Disk:  disk0 (APFS)") and
// MBR/FD partition tables (where the whole-disk row's last field is "disk1").
func parseDiskutilList(text string) []string {
	seen := map[string]bool{}
	var out []string
	add := func(id string) {
		if !seen[id] {
			seen[id] = true
			out = append(out, id)
		}
	}
	for _, m := range reDiskHeader.FindAllStringSubmatch(text, -1) {
		if len(m) == 2 {
			add(m[1])
		}
	}
	for _, line := range strings.Split(text, "\n") {
		fields := strings.Fields(line)
		if len(fields) == 0 {
			continue
		}
		last := fields[len(fields)-1]
		if reWholeDisk.MatchString(last) {
			add(last)
		}
	}
	return out
}

// humanBeforeParen extracts the human-readable prefix of a diskutil size field, e.g.
// "263.1 MB" from "263.1 MB (263090176 Bytes) (exactly 513848 512-Byte-Units)".
func humanBeforeParen(v string) string {
	if i := strings.Index(v, " ("); i > 0 {
		return v[:i]
	}
	return v
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

type smartctlResult struct {
	SmartStatus *struct {
		Passed bool `json:"passed"`
	} `json:"smart_status"`
	Temperature *struct {
		Current int `json:"current"`
	} `json:"temperature"`
	ModelName   string `json:"model_name"`
	ModelFamily string `json:"model_family"`
	Serial      string `json:"serial_number"`
	PowerOnTime *struct {
		Hours int `json:"hours"`
	} `json:"power_on_time"`
	PowerCycleCount int `json:"power_cycle_count"`
	NVMeSmartHealthInformationLog *struct {
		PercentageUsed   int    `json:"percentage_used"`
		AvailableSpare   int    `json:"available_spare"`
		DataUnitsWritten uint64 `json:"data_units_written"`
		CriticalWarning  *int   `json:"critical_warning"`
		MediaErrors      *int   `json:"media_errors"`
		UnsafeShutdowns  *int   `json:"unsafe_shutdowns"`
	} `json:"nvme_smart_health_information_log"`
	SmartAttributes *struct {
		Table []struct {
			ID   int    `json:"id"`
			Name string `json:"name"`
			Raw  *struct {
				Value uint64 `json:"value"`
			} `json:"raw"`
		} `json:"table"`
	} `json:"smart_attributes"`
}

// applySmartctlJSON fills in whatever smartctl -j reported that diskutil (or the NVMe
// profile) hadn't already found. Only nil pointer fields are filled; a non-nil value
// from a prior source is never overwritten.
func applySmartctlJSON(d *DiskInfo, out []byte) bool {
	if len(out) == 0 {
		return false
	}
	var r smartctlResult
	if json.Unmarshal(out, &r) != nil {
		return false
	}
	if r.SmartStatus == nil && r.Temperature == nil && r.NVMeSmartHealthInformationLog == nil &&
		r.SmartAttributes == nil && r.PowerOnTime == nil && r.ModelName == "" {
		return false
	}

	found := false
	if r.SmartStatus != nil && (d.SmartStatus == "" || strings.EqualFold(d.SmartStatus, "not supported")) {
		if r.SmartStatus.Passed {
			d.SmartStatus = "Verified"
		} else {
			d.SmartStatus = "Failing"
		}
		d.SmartSource = "smartctl"
		found = true
	}
	if d.Model == "" {
		if r.ModelName != "" {
			d.Model = r.ModelName
		} else if r.ModelFamily != "" {
			d.Model = r.ModelFamily
		}
	}
	if d.Serial == "" && r.Serial != "" {
		d.Serial = r.Serial
	}
	if r.Temperature != nil && r.Temperature.Current > 0 && d.TemperatureC == nil {
		t := float64(r.Temperature.Current)
		d.TemperatureC = &t
		found = true
	}
	if r.PowerOnTime != nil && d.PowerOnHours == nil {
		h := r.PowerOnTime.Hours
		d.PowerOnHours = &h
		found = true
	}
	if r.PowerCycleCount > 0 && d.PowerCycles == nil {
		p := r.PowerCycleCount
		d.PowerCycles = &p
		found = true
	}
	if log := r.NVMeSmartHealthInformationLog; log != nil {
		if d.WearUsedPct == nil {
			v := log.PercentageUsed
			d.WearUsedPct = &v
			found = true
		}
		if d.SparePct == nil {
			v := log.AvailableSpare
			d.SparePct = &v
			found = true
		}
		if d.DataWrittenTB == nil && log.DataUnitsWritten > 0 {
			tb := float64(log.DataUnitsWritten*nvmeDataUnitBytes) / 1e12
			if !math.IsNaN(tb) && tb >= 0 {
				tb = math.Round(tb*10) / 10
				d.DataWrittenTB = &tb
				found = true
			}
		}
		if d.CriticalWarning == nil && log.CriticalWarning != nil {
			d.CriticalWarning = log.CriticalWarning
			found = true
		}
		if d.MediaErrors == nil && log.MediaErrors != nil {
			d.MediaErrors = log.MediaErrors
			found = true
		}
		if d.UnsafeShutdowns == nil && log.UnsafeShutdowns != nil {
			d.UnsafeShutdowns = log.UnsafeShutdowns
			found = true
		}
	}
	if attrs := r.SmartAttributes; attrs != nil {
		for _, a := range attrs.Table {
			if a.Raw == nil {
				continue
			}
			v := int(a.Raw.Value)
			switch {
			case a.ID == 5 || a.Name == "Reallocated_Sector_Ct":
				if d.ReallocatedSectors == nil {
					d.ReallocatedSectors = &v
					found = true
				}
			case a.ID == 197 || a.Name == "Current_Pending_Sector":
				if d.PendingSectors == nil {
					d.PendingSectors = &v
					found = true
				}
			case a.ID == 198 || a.Name == "Offline_Uncorrectable":
				if d.OfflineUncorrectable == nil {
					d.OfflineUncorrectable = &v
					found = true
				}
			case a.ID == 199 || a.Name == "UDMA_CRC_Error_Count":
				if d.UDMACRCErrors == nil {
					d.UDMACRCErrors = &v
					found = true
				}
			case a.ID == 233 || a.ID == 231 || a.ID == 202 ||
				a.Name == "Percentage_Used" || a.Name == "Device_Wear_Leveling_Count":
				if d.SSDLifeLeft == nil {
					d.SSDLifeLeft = &v
					found = true
				}
			case a.Name == "SAS_Grown_Defect":
				if d.SASGrownDefects == nil {
					d.SASGrownDefects = &v
					found = true
				}
			case a.Name == "SAS_Uncorrected":
				if d.SASUncorrectedErrors == nil {
					d.SASUncorrectedErrors = &v
					found = true
				}
			}
		}
	}
	d.Available = true
	return found
}

// applySmartWear fills NVMe SMART fields from a diskutil SMART key dump (per-disk).
func applySmartWear(d *DiskInfo, smart map[string]uint64) {
	if len(smart) == 0 {
		return
	}
	if v, ok := smart["PERCENTAGE_USED"]; ok && d.WearUsedPct == nil {
		p := int(v)
		d.WearUsedPct = &p
		d.Available = true
	}
	if v, ok := smart["AVAILABLE_SPARE"]; ok && d.SparePct == nil {
		p := int(v)
		d.SparePct = &p
		d.Available = true
	}
	low := smart["DATA_UNITS_WRITTEN_0"]
	high := smart["DATA_UNITS_WRITTEN_1"]
	if (low > 0 || high > 0) && d.DataWrittenTB == nil {
		units := high<<32 | low
		tb := float64(units*nvmeDataUnitBytes) / 1e12
		if !math.IsNaN(tb) && tb >= 0 {
			tb = math.Round(tb*10) / 10
			d.DataWrittenTB = &tb
			d.Available = true
		}
	}
	if v, ok := smart["TEMPERATURE"]; ok && v > 0 && d.TemperatureC == nil {
		t := float64(v) / 10
		d.TemperatureC = &t
	}
}

// applyDiskWarnings sets WarningLevel and WarningMsg from the SMART signals present
// on this disk. Follows the same grading as hostdisk/warnings: SMART FAILED or a
// non-zero critical_warning is critical; positive counts of media errors, reallocated
// sectors, pending sectors, offline uncorrectable, or SAS grown defects is warning.
func applyDiskWarnings(d *DiskInfo) {
	var crit, warn []string
	if strings.EqualFold(d.SmartStatus, "Failing") || strings.EqualFold(d.SmartStatus, "FAILED") {
		crit = append(crit, "SMART overall health check failed")
	}
	if n := derefInt(d.CriticalWarning); n != 0 {
		crit = append(crit, fmt.Sprintf("NVMe critical warning 0x%02x", n))
	}
	for _, c := range []struct {
		n    *int
		what string
	}{
		{d.MediaErrors, "media errors"},
		{d.ReallocatedSectors, "reallocated sectors"},
		{d.PendingSectors, "pending sectors"},
		{d.OfflineUncorrectable, "offline uncorrectable sectors"},
		{d.SASGrownDefects, "grown defects"},
	} {
		if n := derefInt(c.n); n > 0 {
			warn = append(warn, fmt.Sprintf("%d %s", n, c.what))
		}
	}
	switch {
	case len(crit) > 0:
		d.WarningLevel = "critical"
		d.WarningMsg = "critical: " + strings.Join(append(crit, warn...), "; ")
	case len(warn) > 0:
		d.WarningLevel = "warning"
		d.WarningMsg = "warning: " + strings.Join(warn, "; ")
	}
}

func derefInt(p *int) int {
	if p == nil {
		return 0
	}
	return *p
}
