package sys

import (
	"fmt"
	"strings"
)

// Disk warning levels stored in DiskInfo.WarningLevel.
const (
	WarningOK       = "ok"
	WarningWarning  = "warning"
	WarningCritical = "critical"
)

// Thresholds for the disk warning engine. Integer attributes use -1 to mean
// "not reported by this device" and are skipped by the engine.
const (
	wearWarnPct     = 75 // NVMe PERCENTAGE_USED: lifetime wear consumed
	wearCritPct     = 90
	lifeLeftWarnPct = 20 // vendor "life left" attribute (231/233/202)
	lifeLeftCritPct = 10
	spareWarnPct    = 20 // NVMe AVAILABLE_SPARE
	spareCritPct    = 10
	hddTempWarnC    = 55.0
	hddTempCritC    = 65.0
	ssdTempWarnC    = 70.0
	ssdTempCritC    = 80.0
	reallocWarnN    = 1
	reallocCritN    = 50
	zfsCapWarnPct   = 85
	zfsCapCritPct   = 90
	zfsFragWarnPct  = 50
)

func warningRank(level string) int {
	switch level {
	case WarningCritical:
		return 2
	case WarningWarning:
		return 1
	default:
		return 0
	}
}

// EvaluateDiskWarnings applies the threshold engine to one disk and returns
// the overall level (WarningOK / WarningWarning / WarningCritical) plus a
// human-readable message listing every breached threshold (empty when ok).
func EvaluateDiskWarnings(d DiskInfo) (string, string) {
	// The collector already reported an environment problem (permission
	// denied, smartctl missing, probe timeout): surface it as a warning and
	// keep the note instead of re-evaluating unreadable attributes.
	if d.WarningMsg != "" && d.WarningLevel == "" {
		return WarningWarning, d.WarningMsg
	}

	level := WarningOK
	var msgs []string
	add := func(l, m string) {
		if warningRank(l) > warningRank(level) {
			level = l
		}
		msgs = append(msgs, m)
	}

	switch d.SmartStatus {
	case "Failing":
		add(WarningCritical, "SMART self-assessment failing")
	}

	if d.CriticalWarning != 0 {
		if d.CriticalWarning&0x0C != 0 { // bit 2: read-only, bit 3: backup failed
			add(WarningCritical, fmt.Sprintf("NVMe critical warning set (0x%02x)", d.CriticalWarning))
		} else {
			add(WarningWarning, fmt.Sprintf("NVMe critical warning set (0x%02x)", d.CriticalWarning))
		}
	}

	if d.WearUsedPct >= 0 {
		switch {
		case d.WearUsedPct >= wearCritPct:
			add(WarningCritical, fmt.Sprintf("lifetime wear used %d%%", d.WearUsedPct))
		case d.WearUsedPct >= wearWarnPct:
			add(WarningWarning, fmt.Sprintf("lifetime wear used %d%%", d.WearUsedPct))
		}
	}

	if d.SSDLifeLeft >= 0 {
		switch {
		case d.SSDLifeLeft <= lifeLeftCritPct:
			add(WarningCritical, fmt.Sprintf("SSD life left %d%%", d.SSDLifeLeft))
		case d.SSDLifeLeft <= lifeLeftWarnPct:
			add(WarningWarning, fmt.Sprintf("SSD life left %d%%", d.SSDLifeLeft))
		}
	}

	if d.SparePct >= 0 {
		switch {
		case d.SparePct <= spareCritPct:
			add(WarningCritical, fmt.Sprintf("available spare %d%%", d.SparePct))
		case d.SparePct <= spareWarnPct:
			add(WarningWarning, fmt.Sprintf("available spare %d%%", d.SparePct))
		}
	}

	if d.PendingSectors > 0 {
		add(WarningCritical, fmt.Sprintf("%d pending sector(s) awaiting reallocation", d.PendingSectors))
	}
	if d.OfflineUncorrectable > 0 {
		add(WarningCritical, fmt.Sprintf("%d offline uncorrectable sector(s)", d.OfflineUncorrectable))
	}
	if d.ReallocatedSectors >= reallocWarnN {
		switch {
		case d.ReallocatedSectors >= reallocCritN:
			add(WarningCritical, fmt.Sprintf("%d reallocated sector(s)", d.ReallocatedSectors))
		default:
			add(WarningWarning, fmt.Sprintf("%d reallocated sector(s)", d.ReallocatedSectors))
		}
	}
	if d.UDMACRCErrors > 0 {
		add(WarningWarning, fmt.Sprintf("%d UDMA CRC error(s)", d.UDMACRCErrors))
	}
	if d.MediaErrors > 0 {
		add(WarningWarning, fmt.Sprintf("%d media error(s)", d.MediaErrors))
	}
	if d.SASGrownDefects > 0 {
		add(WarningWarning, fmt.Sprintf("%d SAS grown defect(s)", d.SASGrownDefects))
	}
	if d.SASUncorrectedErrors > 0 {
		add(WarningWarning, fmt.Sprintf("%d SAS uncorrected error(s)", d.SASUncorrectedErrors))
	}

	if d.TemperatureC >= 0 {
		warn, crit := hddTempWarnC, hddTempCritC
		if d.SolidState {
			warn, crit = ssdTempWarnC, ssdTempCritC
		}
		switch {
		case d.TemperatureC >= crit:
			add(WarningCritical, fmt.Sprintf("temperature %.0f°C", d.TemperatureC))
		case d.TemperatureC >= warn:
			add(WarningWarning, fmt.Sprintf("temperature %.0f°C", d.TemperatureC))
		}
	}

	// ZFS attributes only exist when a pool is attached to the device
	// (Proxmox source); a plain block disk has no pool name.
	if d.ZFSPool != "" {
		switch strings.ToLower(d.ZFSHealth) {
		case "online":
			// healthy
		case "degraded", "resilvering", "repair":
			add(WarningWarning, fmt.Sprintf("ZFS pool %q health: %s", d.ZFSPool, d.ZFSHealth))
		default:
			add(WarningCritical, fmt.Sprintf("ZFS pool %q health: %s", d.ZFSPool, d.ZFSHealth))
		}
		if d.ZFSCapacityPct >= 0 {
			switch {
			case d.ZFSCapacityPct >= zfsCapCritPct:
				add(WarningCritical, fmt.Sprintf("ZFS pool %q at %d%% capacity", d.ZFSPool, d.ZFSCapacityPct))
			case d.ZFSCapacityPct >= zfsCapWarnPct:
				add(WarningWarning, fmt.Sprintf("ZFS pool %q at %d%% capacity", d.ZFSPool, d.ZFSCapacityPct))
			}
		}
		if d.ZFSFragPct >= zfsFragWarnPct {
			add(WarningWarning, fmt.Sprintf("ZFS pool %q at %d%% fragmentation", d.ZFSPool, d.ZFSFragPct))
		}
	}

	if len(msgs) == 0 {
		return WarningOK, ""
	}
	return level, strings.Join(msgs, "; ")
}

// ApplyDiskWarnings evaluates the threshold engine for each disk and sets
// WarningLevel/WarningMsg. Disks the collector already flagged with both a
// level and a message are left untouched.
func ApplyDiskWarnings(disks []DiskInfo) {
	for i := range disks {
		if disks[i].WarningLevel != "" && disks[i].WarningMsg != "" {
			continue
		}
		level, msg := EvaluateDiskWarnings(disks[i])
		disks[i].WarningLevel = level
		disks[i].WarningMsg = msg
	}
}
