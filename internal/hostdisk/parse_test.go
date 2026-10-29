package hostdisk

import (
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/sys"
)

// fixture is smartctl_exporter output; see the header of testdata/smartctl_exporter.prom.
func fixture(t *testing.T) string {
	t.Helper()
	b, err := os.ReadFile("testdata/smartctl_exporter.prom")
	require.NoError(t, err)
	return string(b)
}

func TestParseFixture(t *testing.T) {
	disks, err := Parse(strings.NewReader(fixture(t)))
	require.NoError(t, err)
	byDev := map[string]sys.DiskInfo{}
	devs := []string{}
	for _, d := range disks {
		byDev[d.Device] = d
		devs = append(devs, d.Device)
	}
	assert.Equal(t, []string{"nvme0", "sda", "sdb", "sdc", "sdd"}, devs)
	tests := []struct {
		dev  string
		want sys.DiskInfo
	}{
		{"nvme0", sys.DiskInfo{
			Device: "nvme0", Model: "PLEXTOR PX-256M9PY +", Serial: "P02050305257", Protocol: "NVMe", SmartType: "nvme",
			SmartStatus: "FAILED", SmartSource: Source, Capacity: "256 GB", SolidState: ptr(true), Host: true,
			WarningLevel: "critical", WarningMsg: "critical: SMART overall health check failed; NVMe critical warning 0x04",
			WearUsedPct: ptr(161), SparePct: ptr(100), DataWrittenTB: ptr(19.63), TemperatureC: ptr(37.0), PowerOnHours: ptr(14259),
			PowerCycles: ptr(11), MediaErrors: ptr(0), CriticalWarning: ptr(4),
		}},
		{"sda", sys.DiskInfo{
			Device: "sda", Model: "TOSHIBA MG07ACA12TE", Serial: "X010A0GTF96G", Protocol: "ATA", SmartType: "sat",
			SmartStatus: "PASSED", SmartSource: Source, Capacity: "12.0 TB", SolidState: ptr(false), Host: true,
			TemperatureC: ptr(40.0), PowerOnHours: ptr(13359), PowerCycles: ptr(11),
			ReallocatedSectors: ptr(0), PendingSectors: ptr(0), OfflineUncorrectable: ptr(0), UDMACRCErrors: ptr(0),
		}},
		{"sdd", sys.DiskInfo{
			Device: "sdd", Model: "ADATA SP920SS", Serial: "8H0920011311", Protocol: "ATA", SmartType: "sat",
			SmartStatus: "PASSED", SmartSource: Source, Capacity: "128 GB", Host: true,
			TemperatureC: ptr(39.0), PowerOnHours: ptr(37412), PowerCycles: ptr(44), ReallocatedSectors: ptr(0),
		}},
	}
	for _, tt := range tests {
		t.Run(tt.dev, func(t *testing.T) {
			assert.Equal(t, tt.want, byDev[tt.dev])
		})
	}
}

func TestParse(t *testing.T) {
	tests := []struct {
		name    string
		in      string
		want    []sys.DiskInfo
		errCode errs.Code
	}{
		{
			name: "untyped samples without HELP or TYPE, failing ATA disk with bad sectors",
			in: `smartctl_device{device="sdb",interface="sat",protocol="ATA",model_name="WDC WD40EFRX",serial_number="WD-1"} 1
smartctl_device_smart_status{device="sdb"} 0
smartctl_device_percentage_used{device="sdb"} 0
smartctl_device_attribute{attribute_id="5",attribute_name="Reallocated_Sector_Ct",attribute_value_type="raw",device="sdb"} 24
smartctl_device_attribute{attribute_id="5",attribute_name="Reallocated_Sector_Ct",attribute_value_type="value",device="sdb"} 100
smartctl_device_attribute{attribute_id="197",attribute_name="Current_Pending_Sector",attribute_value_type="raw",device="sdb"} 8
`,
			want: []sys.DiskInfo{{
				Device: "sdb", Model: "WDC WD40EFRX", Serial: "WD-1", Protocol: "ATA", SmartType: "sat", SmartStatus: "FAILED",
				SmartSource: Source, Host: true, ReallocatedSectors: ptr(24), PendingSectors: ptr(8), WarningLevel: "critical",
				WarningMsg: "critical: SMART overall health check failed; 24 reallocated sectors; 8 pending sectors",
			}},
		},
		{
			name: "SCSI disk with grown defects and endurance, other temperature types ignored",
			in: `# HELP smartctl_device_temperature Device temperature celsius
# TYPE smartctl_device_temperature gauge
smartctl_device_temperature{device="sg1",temperature_type="current"} 33
smartctl_device_temperature{device="sg1",temperature_type="drive_trip"} 65
smartctl_scsi_grown_defect_list{device="sg1"} 3
smartctl_scsi_percentage_used_endurance_indicator{device="sg1"} 7
smartctl_device_rotation_rate{device="sg1"} 0
smartctl_device_smart_status{device="sg1"} 1
`,
			want: []sys.DiskInfo{{
				Device: "sg1", SmartStatus: "PASSED", SmartSource: Source, Host: true, TemperatureC: ptr(33.0), SASGrownDefects: ptr(3),
				WearUsedPct: ptr(7), SolidState: ptr(true), WarningLevel: "warning", WarningMsg: "warning: 3 grown defects",
			}},
		},
		{
			name: "NVMe capacity fallback, other exporters' families ignored",
			in: `go_goroutines 12
node_load1 0.5
smartctl_device{device="nvme1",interface="nvme",protocol="NVMe",model_name="Samsung 990"} 1
smartctl_device_nvme_capacity_bytes{device="nvme1"} 2e+12
smartctl_device_media_errors{device="nvme1"} 2
`,
			want: []sys.DiskInfo{{
				Device: "nvme1", Model: "Samsung 990", Protocol: "NVMe", SmartType: "nvme", SmartSource: Source, Host: true,
				Capacity: "2.0 TB", SolidState: ptr(true), MediaErrors: ptr(2), WarningLevel: "warning", WarningMsg: "warning: 2 media errors",
			}},
		},
		{name: "node_exporter page", in: "# TYPE node_load1 gauge\nnode_load1 0.5\n", errCode: CodeNoDevices},
		{name: "samples without a device label", in: "smartctl_version{smartctl_version=\"7.4\"} 1\nsmartctl_devices 2\n", errCode: CodeNoDevices},
		{name: "empty body", in: "", errCode: CodeNoDevices},
		{name: "HTML error page", in: "<html><body>Bad gateway</body></html>\n", errCode: errs.CodeInvalidInput},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Parse(strings.NewReader(tt.in))
			if tt.errCode != "" {
				require.Error(t, err)
				assert.True(t, errs.HasCode(err, tt.errCode), "error %v should carry %s", err, tt.errCode)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestFormatCapacity(t *testing.T) {
	tests := []struct {
		in   float64
		want string
	}{
		{512, "512 B"}, {256060514304, "256 GB"}, {12000138625024, "12.0 TB"}, {1.92e12, "1.9 TB"}, {4e15, "4.0 PB"},
	}
	for _, tt := range tests {
		assert.Equal(t, tt.want, formatCapacity(tt.in), "%v", tt.in)
	}
}
