package dashboard

import (
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/sys"
)

func gaugeValue(t *testing.T, mfs []*dto.MetricFamily, name string, labelValues ...string) float64 {
	t.Helper()
	for _, mf := range mfs {
		if mf.GetName() != name {
			continue
		}
		if len(labelValues) > 0 {
			for _, m := range mf.GetMetric() {
				got := map[string]string{}
				for _, lp := range m.GetLabel() {
					got[lp.GetName()] = lp.GetValue()
				}
				match := true
				for i := 0; i < len(labelValues); i += 2 {
					if got[labelValues[i]] != labelValues[i+1] {
						match = false
						break
					}
				}
				if match {
					return m.GetGauge().GetValue()
				}
			}
			t.Fatalf("metric %s with labels %v not found", name, labelValues)
		}
		require.Equal(t, 1, len(mf.GetMetric()), "expected exactly one sample for %s", name)
		return mf.GetMetric()[0].GetGauge().GetValue()
	}
	t.Fatalf("metric %s not found", name)
	return 0
}

func TestDiskMetricsCollect(t *testing.T) {
	old := diskSnapshotFunc
	t.Cleanup(func() { diskSnapshotFunc = old })
	diskSnapshotFunc = func() sys.DiskSamplerSnapshot {
		return sys.DiskSamplerSnapshot{
			Running:      true,
			LastSampleAt: time.Now().Add(-30 * time.Second),
			LastOK:       true,
			Disks: []sys.DiskSampleSummary{
				{Device: "/dev/nvme0n1", TemperatureC: 42.5, WearUsedPct: 88, SparePct: 12, SSDLifeLeft: 12, WarningLevel: sys.WarningWarning},
				{Device: "/dev/sda", TemperatureC: 30, WarningLevel: sys.WarningOK, ReallocatedSectors: 3, PendingSectors: 1, OfflineUncorrectable: 2},
			},
			WarningCount: 1,
		}
	}

	reg := prometheus.NewRegistry()
	require.NoError(t, reg.Register(diskMetricsCollector{}))

	mfs, err := reg.Gather()
	require.NoError(t, err)

	require.EqualValues(t, 2, gaugeValue(t, mfs, "astcache_disk_count"))
	require.EqualValues(t, 1, gaugeValue(t, mfs, "astcache_disk_sampler_ok"))

	age := gaugeValue(t, mfs, "astcache_disk_sample_age_seconds")
	require.GreaterOrEqual(t, age, 25.0)
	require.LessOrEqual(t, age, 60.0)

	require.EqualValues(t, 1, gaugeValue(t, mfs, "astcache_disk_warning", "device", "/dev/nvme0n1"))
	require.EqualValues(t, 0, gaugeValue(t, mfs, "astcache_disk_warning", "device", "/dev/sda"))
	require.InDelta(t, 42.5, gaugeValue(t, mfs, "astcache_disk_temperature_c", "device", "/dev/nvme0n1"), 0.01)
	require.EqualValues(t, 88, gaugeValue(t, mfs, "astcache_disk_wear_pct", "device", "/dev/nvme0n1"))
	require.EqualValues(t, 12, gaugeValue(t, mfs, "astcache_disk_spare_pct", "device", "/dev/nvme0n1"))
	require.EqualValues(t, 12, gaugeValue(t, mfs, "astcache_disk_life_left_pct", "device", "/dev/nvme0n1"))
	require.EqualValues(t, 3, gaugeValue(t, mfs, "astcache_disk_reallocated_sectors", "device", "/dev/sda"))
	require.EqualValues(t, 1, gaugeValue(t, mfs, "astcache_disk_pending_sectors", "device", "/dev/sda"))
	require.EqualValues(t, 2, gaugeValue(t, mfs, "astcache_disk_offline_uncorrectable", "device", "/dev/sda"))
}

func TestDiskMetricsCollectZeroState(t *testing.T) {
	old := diskSnapshotFunc
	t.Cleanup(func() { diskSnapshotFunc = old })
	diskSnapshotFunc = func() sys.DiskSamplerSnapshot { return sys.DiskSamplerSnapshot{} }

	reg := prometheus.NewRegistry()
	require.NoError(t, reg.Register(diskMetricsCollector{}))

	mfs, err := reg.Gather()
	require.NoError(t, err)

	require.EqualValues(t, 0, gaugeValue(t, mfs, "astcache_disk_count"))
	require.EqualValues(t, 0, gaugeValue(t, mfs, "astcache_disk_sampler_ok"))
	require.EqualValues(t, 0, gaugeValue(t, mfs, "astcache_disk_sample_age_seconds"))
	for _, mf := range mfs {
		if mf.GetName() == "astcache_disk_warning" {
			t.Fatal("per-device metrics should be absent before the first sample")
		}
	}
}
