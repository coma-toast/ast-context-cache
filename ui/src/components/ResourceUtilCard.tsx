import { useCallback, useEffect, useMemo, useRef } from 'react'
import { Alert, Box, Card, CardContent, Chip, Grid, Stack, Tooltip, Typography } from '@mui/material'
import { Gauge, gaugeClasses } from '@mui/x-charts/Gauge'
import type { IndexHealth, DiskInfo } from '../api/types'
import { chartColors } from '../lib/chartColors'
import {
  cpuGaugePct,
  diskFootprintPct,
  diskIoPct,
  diskSizeLabel,
  diskHealthLevel,
  diskLabel,
  diskSubtitle,
  diskTempColor,
  diskTone,
  fdColor,
  fdLabel,
  fdPct,
  formatDiskIo,
  formatPowerOnHours,
  loadAvgBarWidth,
  loadAvgLabel,
  memoryPct,
  sampleAge,
  ssdSmartTone,
  ssdWearColor,
} from '../lib/resourceGauges'
import { useToast } from '../context/ToastContext'

const CPUS = typeof navigator !== 'undefined' ? navigator.hardwareConcurrency || 1 : 1

type Row = { label: string; pct: number; display: string; color: string; hint?: string }

function buildRows(data: IndexHealth): Row[] {
  const diskLabel$ = diskSizeLabel(
    data.DiskSize || '-',
    data.WalSize || '',
    data.WALMaintenanceActive,
    data.WALWalStartBytes,
    data.WALWalCurrentBytes,
  )
  const rows: Row[] = [
    {
      label: 'CPU',
      pct: cpuGaugePct(data.CPUPercent ?? 0),
      display: `${data.CPUPercent?.toFixed(0) ?? 0}%`,
      color: chartColors.cpu,
      hint: 'Process CPU — can exceed 100% on multi-core',
    },
  ]
  if (data.FDAvailable) {
    rows.push({
      label: 'Open files',
      pct: fdPct(data.OpenFDs ?? 0, data.FDSoftLimit ?? 0),
      display: fdLabel(data.OpenFDs ?? 0, data.FDSoftLimit ?? 0),
      color: fdColor(data.FDLevel),
      hint: `File descriptors held vs the process limit. Watchers use ${data.WatchBackend || 'fsnotify'}${
        data.WatchBackend === 'kqueue' ? ' (one descriptor per watched file and directory)' : ''
      }; calls fail with "too many open files" at the limit.`,
    })
  }
  if (data.LoadAvgAvailable) {
    rows.push({
      label: 'Load avg',
      pct: loadAvgBarWidth(data.LoadAvg1 ?? 0, CPUS),
      display: loadAvgLabel(data.LoadAvg1 ?? 0, data.LoadAvg5 ?? 0, data.LoadAvg15 ?? 0, CPUS),
      color: chartColors.load,
      hint: `Host load average (1 / 5 / 15 min) ÷ ${CPUS} cores`,
    })
  }
  rows.push(
    { label: 'Heap', pct: memoryPct(data.MemoryMB ?? data.HeapMB ?? 0), display: `${(data.MemoryMB ?? data.HeapMB ?? 0).toFixed(1)} MB`, color: chartColors.heap },
    { label: 'Vector cache', pct: memoryPct(data.VectorMemMB ?? 0), display: `${(data.VectorMemMB ?? 0).toFixed(2)} MB`, color: chartColors.vector },
    { label: 'Database', pct: diskFootprintPct(data.DiskMB ?? 0, data.WalMB ?? 0), display: diskLabel$, color: chartColors.disk, hint: 'SQLite usage.db + WAL journal on disk' },
    { label: 'Disk read', pct: diskIoPct(data.DiskReadMBps ?? 0), display: formatDiskIo(data.DiskReadMBps ?? 0), color: chartColors.read, hint: 'Host block device read throughput' },
    { label: 'Disk write', pct: diskIoPct(data.DiskWriteMBps ?? 0), display: formatDiskIo(data.DiskWriteMBps ?? 0), color: chartColors.write, hint: 'Host block device write throughput' },
  )
  return rows
}

function ResourceBarRow({ row }: { row: Row }) {
  return (
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '72px 1fr', sm: '88px 1fr auto' }, gap: 1, alignItems: 'center' }}>
      {row.hint ? (
        <Tooltip title={row.hint}>
          <Typography variant="caption" color="text.secondary" sx={{ cursor: 'help' }}>
            {row.label}
          </Typography>
        </Tooltip>
      ) : (
        <Typography variant="caption" color="text.secondary">
          {row.label}
        </Typography>
      )}
      <Box sx={{ bgcolor: chartColors.track, borderRadius: 1, overflow: 'hidden', height: 8 }}>
        <Box sx={{ height: '100%', width: `${row.pct}%`, bgcolor: row.color, borderRadius: 1, transition: 'width 0.35s ease' }} />
      </Box>
      <Typography variant="caption" sx={{ fontFamily: 'ui-monospace, monospace', display: { xs: 'none', sm: 'block' }, whiteSpace: 'nowrap' }}>
        {row.display}
      </Typography>
    </Box>
  )
}

function DiskChip({ label, tone }: { label: string; tone: 'success' | 'warning' | 'error' | 'default' }) {
  const colorMap: Record<string, 'success' | 'warning' | 'error' | 'default'> = { success: 'success', warning: 'warning', error: 'error', default: 'default' }
  return (
    <Chip
      size="small"
      label={label}
      color={colorMap[tone] || 'default'}
      variant="outlined"
      sx={{
        borderColor: tone === 'default' ? 'divider' : undefined,
        color: tone === 'default' ? 'text.secondary' : undefined,
        fontWeight: 600,
      }}
    />
  )
}

function DiskGauge({ value }: { value: number }) {
  const color = ssdWearColor(value)
  return (
    <Gauge
      width={56}
      height={56}
      value={value}
      valueMax={100}
      startAngle={-110}
      endAngle={110}
      innerRadius="72%"
      outerRadius="100%"
      text={`${value}%`}
      sx={{
        [`& .${gaugeClasses.referenceArc}`]: { fill: chartColors.track },
        [`& .${gaugeClasses.valueArc}`]: { fill: color },
        [`& .${gaugeClasses.valueText}`]: { fill: '#e6edf3', fontSize: 11 },
      }}
    />
  )
}

function DiskHealthSection({ disks }: { disks: DiskInfo[] }) {
  return (
    <Box sx={{ mt: 2.5, pt: 2, borderTop: '1px solid', borderColor: 'divider' }}>
      <Typography variant="subtitle2" gutterBottom>
        Disk health
      </Typography>
      {disks.map((disk, idx) => {
        const tone = diskTone(disk.SmartStatus)
        const level = diskHealthLevel(disk)
        const tempColor = diskTempColor(disk.TemperatureC)
        const label$ = diskLabel(disk)
        const subtitle$ = diskSubtitle(disk)
        return (
          <Box key={disk.Serial || disk.Device || idx} sx={{ mb: 2 }}>
            <Stack direction="row" spacing={1} sx={{ alignItems: 'center', mb: 1 }}>
              <Typography variant="subtitle2" noWrap sx={{ maxWidth: 300 }}>
                {label$}
              </Typography>
              <DiskChip label={disk.SmartStatus || 'Unknown'} tone={tone} />
              {disk.Host && (
                <Tooltip title={`Host disk, read from the host's ${disk.SmartSource || 'smartctl_exporter'}`}>
                  <Chip size="small" label="Host" color="info" variant="outlined" />
                </Tooltip>
              )}
              {disk.DataDirDisk && <Chip size="small" label={disk.DataDirDisk} variant="outlined" />}
              {disk.ZFSPool && <Chip size="small" label={disk.ZFSPool} color="secondary" variant="outlined" />}
            </Stack>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
              {subtitle$}
            </Typography>
            <Grid container spacing={1.5}>
              {(disk.WearUsedPct ?? -1) >= 0 && (
                <Grid size={{ xs: 12, sm: 8 }}>
                  <Stack direction="row" spacing={1.5} sx={{ alignItems: 'center' }}>
                    <DiskGauge value={disk.WearUsedPct!} />
                    <Box>
                      <Typography variant="body2">Wear Used</Typography>
                      <Typography variant="caption" color="text.secondary">
                        {disk.SSDLifeLeft != null ? `Life left: ${disk.SSDLifeLeft}%` : 'NVMe percentage used'}
                      </Typography>
                    </Box>
                  </Stack>
                </Grid>
              )}
              {(disk.TemperatureC ?? -1) >= 0 && (
                <Grid size={{ xs: 12, sm: 4 }}>
                  <Stack direction="row" spacing={1.5} sx={{ alignItems: 'center' }}>
                    <Typography variant="h6" sx={{ fontSize: '1.25rem', lineHeight: 1, color: tempColor, minWidth: 48 }}>
                      {disk.TemperatureC}°
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Temperature
                    </Typography>
                  </Stack>
                </Grid>
              )}
              <Grid size={{ xs: 6, sm: 3 }}>
                <DiskItem label="Capacity" value={disk.Capacity || '-'} />
              </Grid>
              {disk.FreeSpace && (
                <Grid size={{ xs: 6, sm: 3 }}>
                  <DiskItem label="Free" value={disk.FreeSpace} />
                </Grid>
              )}
              {disk.SparePct != null && (
                <Grid size={{ xs: 6, sm: 3 }}>
                  <DiskItem label="Spare" value={`${disk.SparePct}%`} />
                </Grid>
              )}
              {disk.DataWrittenTB != null && (
                <Grid size={{ xs: 6, sm: 3 }}>
                  <DiskItem label="Written" value={disk.DataWrittenTB >= 100 ? `${Math.round(disk.DataWrittenTB)} TB` : `${disk.DataWrittenTB.toFixed(1)} TB`} />
                </Grid>
              )}
              {disk.PowerOnHours != null && (
                <Grid size={{ xs: 6, sm: 3 }}>
                  <DiskItem label="Power on" value={formatPowerOnHours(disk.PowerOnHours)} />
                </Grid>
              )}
              {disk.ReallocatedSectors != null && (
                <Grid size={{ xs: 6, sm: 3 }}>
                  <DiskItem label="Realloc" value={disk.ReallocatedSectors.toString()} />
                </Grid>
              )}
              {disk.PendingSectors != null && (
                <Grid size={{ xs: 6, sm: 3 }}>
                  <DiskItem label="Pending" value={disk.PendingSectors.toString()} />
                </Grid>
              )}
              {disk.SmartType && (
                <Grid size={{ xs: 6, sm: 3 }}>
                  <DiskItem label="SMART" value={disk.SmartType} />
                </Grid>
              )}
              {disk.SolidState != null && (
                <Grid size={{ xs: 6, sm: 3 }}>
                  <DiskItem label="Type" value={disk.SolidState ? 'SSD/NVMe' : 'HDD'} />
                </Grid>
              )}
              {disk.TrimSupport != null && (
                <Grid size={{ xs: 6, sm: 3 }}>
                  <DiskItem label="TRIM" value={disk.TrimSupport ? 'Yes' : 'No'} />
                </Grid>
              )}
              {disk.ZFSHealth && (
                <Grid size={{ xs: 6, sm: 3 }}>
                  <DiskItem label="ZFS" value={disk.ZFSHealth} />
                </Grid>
              )}
              {disk.ZFSCapacityPct != null && (
                <Grid size={{ xs: 6, sm: 3 }}>
                  <DiskItem label="ZFS cap" value={`${disk.ZFSCapacityPct}%`} />
                </Grid>
              )}
            </Grid>
            {disk.WarningMsg && (
              <Typography variant="caption" color={level === 'critical' ? 'error' : 'warning.main'} sx={{ display: 'block', mt: 1 }}>
                ⚠ {disk.WarningMsg}
              </Typography>
            )}
          </Box>
        )
      })}
    </Box>
  )
}

/**
 * Host disk state the rows can't show: the latest poll failed (rows are the last good sample),
 * or this guest has no host disk URL set and so shows no SMART data at all.
 */
export function HostDiskNotice({ data }: { data: IndexHealth }) {
  const hostRows = (data.Disks ?? []).filter((d) => d.Host).length
  if (data.HostDiskError) {
    const age = sampleAge(data.HostDiskSampledUnix)
    return (
      <Alert severity="warning" variant="outlined" sx={{ mt: 2, py: 0 }}>
        Can't read host disks: {data.HostDiskError}
        {hostRows > 0 && age ? `. Showing the last good sample from ${age}.` : ''}
      </Alert>
    )
  }
  if (data.HostDiskGuest && !data.HostDiskURLSet && hostRows === 0) {
    return (
      <Alert severity="info" variant="outlined" sx={{ mt: 2, py: 0 }}>
        Disks belong to the host; SMART isn't visible from here. Set Host disk health URL in Settings → Host disks to
        show them.
      </Alert>
    )
  }
  return null
}

function DiskItem({ label, value }: { label: string; value: string }) {
  return (
    <Box sx={{ minWidth: 0 }}>
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
        {label}
      </Typography>
      <Typography variant="body2" noWrap>
        {value}
      </Typography>
    </Box>
  )
}

export function ResourceUtilCard({ data }: { data: IndexHealth }) {
  const { showToast } = useToast()
  const rows = buildRows(data)
  const prevLevelRef = useRef<string>('')

  const prevLevel = useMemo(() => {
    if (!data.Disks || data.Disks.length === 0) return ''
    const level = data.Disks.map((d) => diskHealthLevel(d)).sort().reverse().join('')
    return level
  }, [data.Disks])

  useEffect(() => {
    if (!data.Disks || data.Disks.length === 0) return
    const prev = prevLevelRef.current
    if (prev && prev === prevLevel) return
    prevLevelRef.current = prevLevel

    for (const disk of data.Disks) {
      const level = diskHealthLevel(disk)
      if (level === 'critical') {
        const key = disk.Serial || disk.Device || 'disk'
        showToast(
          `${disk.Device || key}: ${disk.WarningMsg || 'Critical health'}`,
          'error',
        )
      } else if (level === 'warn') {
        const key = disk.Serial || disk.Device || 'disk'
        showToast(
          `${disk.Device || key}: ${disk.WarningMsg || 'Warning threshold reached'}`,
          'warning',
        )
      }
    }
  }, [data.Disks, showToast, prevLevel])

  return (
    <Card variant="outlined" sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="subtitle2" gutterBottom>
          Server utilization
        </Typography>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.25, mt: 0.5 }}>
          {rows.map((r) => (
            <ResourceBarRow key={r.label} row={r} />
          ))}
        </Box>

        {data.Disks && data.Disks.length > 0 && <DiskHealthSection disks={data.Disks} />}
        <HostDiskNotice data={data} />
        {!(data.Disks ?? []).some((d) => !d.Host) && data.SSDAvailable ? (
          <Box sx={{ mt: 2.5, pt: 2, borderTop: '1px solid', borderColor: 'divider' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1.5 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography variant="subtitle2">{data.SSDExternal ? 'External drive health' : 'SSD health'}</Typography>
                {data.SSDExternal && <Chip size="small" label="USB" variant="outlined" />}
              </Box>
              <Tooltip title={data.SSDSmartSource ? `via ${data.SSDSmartSource}` : ''}>
                <Chip size="small" label={data.SSDSmartStatus || 'Unknown'} color={ssdSmartTone(data.SSDSmartStatus || '')} variant="outlined" />
              </Tooltip>
            </Box>
            {(data.SSDWearUsedPct ?? -1) >= 0 && (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 1.5 }}>
                <Gauge
                  width={72}
                  height={72}
                  value={data.SSDWearUsedPct ?? 0}
                  valueMax={100}
                  startAngle={-110}
                  endAngle={110}
                  innerRadius="72%"
                  outerRadius="100%"
                  text={`${data.SSDWearUsedPct}%`}
                  sx={{
                    [`& .${gaugeClasses.referenceArc}`]: { fill: chartColors.track },
                    [`& .${gaugeClasses.valueArc}`]: { fill: ssdWearColor(data.SSDWearUsedPct ?? 0) },
                    [`& .${gaugeClasses.valueText}`]: { fill: '#e6edf3', fontSize: 12 },
                  }}
                />
                <Typography variant="caption" color="text.secondary">
                  NVMe wear (PERCENTAGE_USED)
                </Typography>
              </Box>
            )}
            <Grid container spacing={1.5}>
              <Grid size={{ xs: 12, sm: 6 }}>
                <SsdItem label="Model" value={shortModel(data.SSDModel)} title={data.SSDModel} />
              </Grid>
              <Grid size={{ xs: 6, sm: 3 }}>
                <SsdItem label="Capacity" value={data.SSDCapacity || '-'} />
              </Grid>
              {data.SSDFreeSpace && (
                <Grid size={{ xs: 6, sm: 3 }}>
                  <SsdItem label="Free" value={data.SSDFreeSpace} />
                </Grid>
              )}
              <Grid size={{ xs: 6, sm: 3 }}>
                <SsdItem label="Protocol" value={data.SSDProtocol || '-'} />
              </Grid>
              {(data.SSDSparePct ?? -1) >= 0 && (
                <Grid size={{ xs: 6, sm: 3 }}>
                  <SsdItem label="Spare" value={`${data.SSDSparePct}%`} />
                </Grid>
              )}
              {(data.SSDDataWrittenTB ?? -1) >= 0 && (
                <Grid size={{ xs: 6, sm: 3 }}>
                  <SsdItem label="Written" value={formatWrittenTb(data.SSDDataWrittenTB ?? 0)} />
                </Grid>
              )}
              {(data.SSDTemperatureC ?? -1) >= 0 && (
                <Grid size={{ xs: 6, sm: 3 }}>
                  <SsdItem label="Temp" value={`${data.SSDTemperatureC?.toFixed(1)} °C`} />
                </Grid>
              )}
              <Grid size={{ xs: 6, sm: 3 }}>
                <SsdItem label="TRIM" value={data.SSDTrim ? 'Yes' : 'No'} />
              </Grid>
            </Grid>
          </Box>
        ) : null}

        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 2 }}>
          Server-wide · relative load (soft caps for gauges)
        </Typography>
      </CardContent>
    </Card>
  )
}

function SsdItem({ label, value, title }: { label: string; value: string; title?: string }) {
  return (
    <Box sx={{ minWidth: 0 }}>
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
        {label}
      </Typography>
      <Typography variant="body2" noWrap title={title || value} sx={{ fontFamily: label === 'Model' ? 'ui-monospace, monospace' : undefined }}>
        {value}
      </Typography>
    </Box>
  )
}

function shortModel(m?: string) {
  if (!m) return '-'
  return m.length <= 28 ? m : `${m.slice(0, 25)}…`
}

function formatWrittenTb(tb: number) {
  if (tb >= 100) return `${Math.round(tb)} TB`
  return `${tb.toFixed(1)} TB`
}
