import { Box, Card, CardContent, Typography } from '@mui/material'
import type { MeterFill } from '../../lib/statMeters'
import { StatMeterChart } from './StatMeterChart'

export function MetricStatCard({
  title,
  value,
  sub,
  fill,
  accent = '#58a6ff',
}: {
  title: string
  value: string
  sub?: string
  fill: MeterFill
  accent?: string
}) {
  return (
    <Card
      variant="outlined"
      sx={{
        height: '100%',
        transition: 'border-color 0.15s',
        '&:hover': { borderColor: 'primary.dark' },
      }}
    >
      <CardContent sx={{ pb: '16px !important' }}>
        <Typography variant="overline" color="text.secondary" display="block">
          {title}
        </Typography>
        <Typography
          variant="h4"
          fontWeight={700}
          sx={{ color: accent, fontFamily: '"JetBrains Mono", ui-monospace, monospace', fontSize: 28, lineHeight: 1.2, my: 0.5 }}
        >
          {value}
        </Typography>
        <Box sx={{ width: '100%', overflow: 'hidden', mb: 1 }}>
          <StatMeterChart fill={fill} />
        </Box>
        {sub && (
          <Typography variant="caption" color="text.secondary" sx={{ lineHeight: 1.4 }}>
            {sub}
          </Typography>
        )}
      </CardContent>
    </Card>
  )
}
