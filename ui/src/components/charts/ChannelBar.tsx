import { Box, Typography } from '@mui/material'
import { chartColors } from '../../lib/chartColors'
import { gaugeLevelFromPct } from '../../lib/statMeters'

const levelColor = { ok: chartColors.green, warn: chartColors.orange, critical: '#f85149' } as const

export function ChannelBar({
  label,
  used,
  cap,
  color,
}: {
  label: string
  used: number
  cap: number
  color: string
}) {
  const pct = cap > 0 ? Math.min(100, (used / cap) * 100) : 0
  const level = gaugeLevelFromPct(pct)
  const fill = color || levelColor[level]
  return (
    <Box sx={{ mb: 1 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
        <Typography variant="caption" color="text.secondary">
          {label}
        </Typography>
        <Typography variant="caption" sx={{ fontFamily: 'ui-monospace, monospace' }}>
          {used.toLocaleString()} / {cap.toLocaleString()}
        </Typography>
      </Box>
      <Box sx={{ height: 8, bgcolor: chartColors.track, borderRadius: 1, overflow: 'hidden' }}>
        <Box sx={{ height: '100%', width: `${pct}%`, bgcolor: fill, borderRadius: 1, transition: 'width 0.35s ease' }} />
      </Box>
    </Box>
  )
}
