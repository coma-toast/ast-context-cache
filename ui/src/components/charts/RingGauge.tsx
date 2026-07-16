import { Box, Typography } from '@mui/material'
import { Gauge, gaugeClasses } from '@mui/x-charts/Gauge'
import { gaugeLevelFromPct } from '../../lib/statMeters'
import { chartColors } from '../../lib/chartColors'

const levelColor = {
  ok: chartColors.green,
  warn: chartColors.orange,
  critical: '#f85149',
} as const

export function RingGauge({
  value,
  max,
  caption,
  size = 88,
}: {
  value: number
  max: number
  caption: string
  size?: number
}) {
  const safeValue = Number.isFinite(value) ? value : 0
  const safeMax = Number.isFinite(max) && max > 0 ? max : 1
  const pct = Math.min(100, (safeValue / safeMax) * 100)
  const level = gaugeLevelFromPct(pct)
  const color = levelColor[level]
  return (
    <Box sx={{ position: 'relative', width: size, height: size, flexShrink: 0 }}>
      <Gauge
        width={size}
        height={size}
        value={pct}
        valueMax={100}
        startAngle={-110}
        endAngle={110}
        innerRadius="72%"
        outerRadius="100%"
        sx={{
          [`& .${gaugeClasses.referenceArc}`]: { fill: chartColors.track },
          [`& .${gaugeClasses.valueArc}`]: { fill: color },
          [`& .${gaugeClasses.valueText}`]: { display: 'none' },
        }}
      />
      <Box
        sx={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          pointerEvents: 'none',
          lineHeight: 1.1,
        }}
      >
        <Typography variant="body2" fontWeight={700} sx={{ fontFamily: 'ui-monospace, monospace', fontSize: 15 }}>
          {safeValue}
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ fontSize: 10 }}>
          {caption}
        </Typography>
      </Box>
    </Box>
  )
}
