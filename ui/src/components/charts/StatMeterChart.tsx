import { Box } from '@mui/material'
import type { MeterFill } from '../../lib/statMeters'
import { chartColors } from '../../lib/chartColors'

function clampPct(n: number) {
  if (!Number.isFinite(n) || n < 0) return 0
  return n > 100 ? 100 : n
}

export function StatMeterChart({ fill }: { fill: MeterFill }) {
  const overlap = clampPct(fill.overlapPct)
  const day = clampPct(fill.dayOnlyPct)
  const avg = clampPct(fill.avgOnlyPct)
  return (
    <Box sx={{ display: 'flex', height: 10, borderRadius: 1, overflow: 'hidden', bgcolor: chartColors.track, width: '100%' }}>
      <Box sx={{ width: `${overlap}%`, bgcolor: chartColors.overlap, transition: 'width 0.35s ease' }} />
      <Box sx={{ width: `${day}%`, bgcolor: chartColors.day, transition: 'width 0.35s ease' }} />
      <Box sx={{ width: `${avg}%`, bgcolor: chartColors.avg, transition: 'width 0.35s ease' }} />
    </Box>
  )
}
