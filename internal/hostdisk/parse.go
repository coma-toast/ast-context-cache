package hostdisk

import (
	"fmt"
	"io"
	"maps"
	"math"
	"slices"
	"strings"

	dto "github.com/prometheus/client_model/go"
	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"

	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/sys"
)

// Source is the SmartSource of every disk read from the exporter.
const Source = "smartctl_exporter"

// CodeNoDevices marks a response that parsed but held no smartctl_device* series, which
// usually means the URL points at some other exporter.
const CodeNoDevices errs.Code = "no_devices"

// rawAttributes maps the ATA SMART attributes the disk card shows to their DiskInfo field.
var rawAttributes = map[string]func(d *sys.DiskInfo) **int{
	"Reallocated_Sector_Ct":  func(d *sys.DiskInfo) **int { return &d.ReallocatedSectors },
	"Current_Pending_Sector": func(d *sys.DiskInfo) **int { return &d.PendingSectors },
	"Offline_Uncorrectable":  func(d *sys.DiskInfo) **int { return &d.OfflineUncorrectable },
	"UDMA_CRC_Error_Count":   func(d *sys.DiskInfo) **int { return &d.UDMACRCErrors },
}

// Parse reads a smartctl_exporter /metrics response (Prometheus text format) into one DiskInfo
// per device label, sorted by device and marked as host disks. Series it does not use, and
// every non-smartctl family, are ignored.
func Parse(r io.Reader) ([]sys.DiskInfo, error) {
	p := expfmt.NewTextParser(model.UTF8Validation)
	fams, err := p.TextToMetricFamilies(r)
	if err != nil {
		return nil, errs.WrapCodeMessage(errs.CodeInvalidInput, "parse metrics", err)
	}
	disks := map[string]*sys.DiskInfo{}
	nvme := map[string]bool{}
	disk := func(m *dto.Metric) *sys.DiskInfo {
		dev := label(m, "device")
		if dev == "" {
			return nil
		}
		if d, ok := disks[dev]; ok {
			return d
		}
		d := &sys.DiskInfo{Device: dev, Host: true, SmartSource: Source}
		disks[dev] = d
		return d
	}
	for _, m := range family(fams, "smartctl_device") {
		if d := disk(m); d != nil {
			d.Model, d.Serial, d.Protocol, d.SmartType = label(m, "model_name"), label(m, "serial_number"), label(m, "protocol"), label(m, "interface")
			nvme[d.Device] = strings.EqualFold(d.Protocol, "NVMe") || d.SmartType == "nvme"
		}
	}
	each := func(name string, fn func(d *sys.DiskInfo, m *dto.Metric, v float64)) {
		for _, m := range family(fams, name) {
			if v, ok := value(m); ok {
				if d := disk(m); d != nil {
					fn(d, m, v)
				}
			}
		}
	}
	each("smartctl_device_smart_status", func(d *sys.DiskInfo, _ *dto.Metric, v float64) {
		d.SmartStatus = map[bool]string{true: "PASSED", false: "FAILED"}[v == 1]
	})
	each("smartctl_device_temperature", func(d *sys.DiskInfo, m *dto.Metric, v float64) {
		if t := label(m, "temperature_type"); t == "" || t == "current" {
			d.TemperatureC = ptr(v)
		}
	})
	each("smartctl_device_capacity_bytes", func(d *sys.DiskInfo, _ *dto.Metric, v float64) { d.Capacity = formatCapacity(v) })
	each("smartctl_device_nvme_capacity_bytes", func(d *sys.DiskInfo, _ *dto.Metric, v float64) {
		if d.Capacity == "" {
			d.Capacity = formatCapacity(v)
		}
	})
	each("smartctl_device_power_on_seconds", func(d *sys.DiskInfo, _ *dto.Metric, v float64) { d.PowerOnHours = intPtr(v / 3600) })
	each("smartctl_device_power_cycle_count", func(d *sys.DiskInfo, _ *dto.Metric, v float64) { d.PowerCycles = intPtr(v) })
	each("smartctl_device_rotation_rate", func(d *sys.DiskInfo, _ *dto.Metric, v float64) { d.SolidState = ptr(v == 0) })
	each("smartctl_device_bytes_written", func(d *sys.DiskInfo, _ *dto.Metric, v float64) {
		if v > 0 {
			d.DataWrittenTB = ptr(math.Round(v/1e10) / 100)
		}
	})
	each("smartctl_scsi_grown_defect_list", func(d *sys.DiskInfo, _ *dto.Metric, v float64) { d.SASGrownDefects = intPtr(v) })
	each("smartctl_scsi_percentage_used_endurance_indicator", func(d *sys.DiskInfo, _ *dto.Metric, v float64) { d.WearUsedPct = intPtr(v) })
	each("smartctl_device_attribute", func(d *sys.DiskInfo, m *dto.Metric, v float64) {
		if field, ok := rawAttributes[label(m, "attribute_name")]; ok && label(m, "attribute_value_type") == "raw" {
			*field(d) = intPtr(v)
		}
	})
	// Older exporters report these NVMe health-log fields as 0 for every ATA disk too.
	nvmeOnly := func(name string, fn func(d *sys.DiskInfo, v float64)) {
		each(name, func(d *sys.DiskInfo, _ *dto.Metric, v float64) {
			if nvme[d.Device] {
				fn(d, v)
			}
		})
	}
	nvmeOnly("smartctl_device_percentage_used", func(d *sys.DiskInfo, v float64) { d.WearUsedPct = intPtr(v) })
	nvmeOnly("smartctl_device_available_spare", func(d *sys.DiskInfo, v float64) { d.SparePct = intPtr(v) })
	nvmeOnly("smartctl_device_media_errors", func(d *sys.DiskInfo, v float64) { d.MediaErrors = intPtr(v) })
	nvmeOnly("smartctl_device_critical_warning", func(d *sys.DiskInfo, v float64) { d.CriticalWarning = intPtr(v) })
	if len(disks) == 0 {
		return nil, errs.NewCode(CodeNoDevices, "no smartctl_device metrics in the response; is this a smartctl_exporter /metrics URL?")
	}
	out := make([]sys.DiskInfo, 0, len(disks))
	for _, dev := range slices.Sorted(maps.Keys(disks)) {
		d := disks[dev]
		if nvme[dev] && d.SolidState == nil {
			d.SolidState = ptr(true)
		}
		d.WarningLevel, d.WarningMsg = warnings(d)
		out = append(out, *d)
	}
	return out, nil
}

// warnings is the level ("critical" or "warning") and message for the SMART signals that the
// card does not grade itself; wear and temperature have their own thresholds in the UI.
func warnings(d *sys.DiskInfo) (string, string) {
	var crit, warn []string
	if d.SmartStatus == "FAILED" {
		crit = append(crit, "SMART overall health check failed")
	}
	if n := deref(d.CriticalWarning); n != 0 {
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
		if n := deref(c.n); n > 0 {
			warn = append(warn, fmt.Sprintf("%d %s", n, c.what))
		}
	}
	switch {
	case len(crit) > 0:
		return "critical", "critical: " + strings.Join(append(crit, warn...), "; ")
	case len(warn) > 0:
		return "warning", "warning: " + strings.Join(warn, "; ")
	}
	return "", ""
}

// family returns the samples of one metric family, or nil when the response lacks it.
func family(fams map[string]*dto.MetricFamily, name string) []*dto.Metric {
	if f, ok := fams[name]; ok {
		return f.GetMetric()
	}
	return nil
}

// value reads a sample whatever type the exporter declared, skipping NaN and infinities.
func value(m *dto.Metric) (float64, bool) {
	var v float64
	switch {
	case m.Gauge != nil:
		v = m.GetGauge().GetValue()
	case m.Untyped != nil:
		v = m.GetUntyped().GetValue()
	case m.Counter != nil:
		v = m.GetCounter().GetValue()
	default:
		return 0, false
	}
	return v, !math.IsNaN(v) && !math.IsInf(v, 0)
}

func label(m *dto.Metric, name string) string {
	for _, lp := range m.GetLabel() {
		if lp.GetName() == name {
			return lp.GetValue()
		}
	}
	return ""
}

// formatCapacity renders bytes in decimal units, the way drives are sold (12 TB, not 10.9 TiB).
func formatCapacity(b float64) string {
	units := []string{"B", "KB", "MB", "GB", "TB", "PB"}
	i := 0
	for b >= 1000 && i < len(units)-1 {
		b /= 1000
		i++
	}
	if b >= 100 || i == 0 {
		return fmt.Sprintf("%.0f %s", b, units[i])
	}
	return fmt.Sprintf("%.1f %s", b, units[i])
}

func ptr[T any](v T) *T { return &v }

func intPtr(v float64) *int { return ptr(int(math.Round(v))) }

func deref(p *int) int {
	if p == nil {
		return 0
	}
	return *p
}
