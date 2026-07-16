import { Box, Card, CardContent, Chip, Grid, Tooltip, Typography } from '@mui/material'
import { Gauge, gaugeClasses } from '@mui/x-charts/Gauge'
import type { IndexHealth } from '../api/types'
import { chartColors } from '../lib/chartColors'
import {
  cpuGaugePct,
  diskFootprintPct,
  diskIoPct,
  diskSizeLabel,
  formatDiskIo,
  loadAvgBarWidth,
  loadAvgLabel,
  memoryPct,
  ssdSmartTone,
  ssdWearColor,
} from '../lib/resourceGauges'

const CPUS = typeof navigator !== 'undefined' ? navigator.hardwareConcurrency || 1 : 1

type Row = { label: string; pct: number; display: string; color: string; hint?: string }

function buildRows(data: IndexHealth): Row[] {
  const diskLabel = diskSizeLabel(
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
    { label: 'Database', pct: diskFootprintPct(data.DiskMB ?? 0, data.WalMB ?? 0), display: diskLabel, color: chartColors.disk, hint: 'SQLite usage.db + WAL journal on disk' },
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

export function ResourceUtilCard({ data }: { data: IndexHealth }) {
  const rows = buildRows(data)
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

        {data.SSDAvailable && (
          <Box sx={{ mt: 2.5, pt: 2, borderTop: '1px solid', borderColor: 'divider' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1.5 }}>
              <Typography variant="subtitle2">SSD health</Typography>
              <Chip size="small" label={data.SSDSmartStatus || 'Unknown'} color={ssdSmartTone(data.SSDSmartStatus || '')} variant="outlined" />
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
        )}

        <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 2 }}>
          Server-wide · relative load (soft caps for gauges)
        </Typography>
      </CardContent>
    </Card>
  )
}

function SsdItem({ label, value, title }: { label: string; value: string; title?: string }) {
  return (
    <Box sx={{ minWidth: 0 }}>
      <Typography variant="caption" color="text.secondary" display="block">
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
