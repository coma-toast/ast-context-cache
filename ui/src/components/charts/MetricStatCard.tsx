import { Box, Card, CardContent, Stack, Tooltip, Typography } from '@mui/material'
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined'
import type { MeterFill } from '../../lib/statMeters'
import { StatMeterChart } from './StatMeterChart'

export function MetricStatCard({
  title,
  value,
  sub,
  detail,
  fill,
  accent = '#58a6ff',
}: {
  title: string
  value: string
  sub?: string
  /** Secondary figures shown on hover, to keep the card itself to one short sub-line. */
  detail?: string
  fill: MeterFill
  accent?: string
}) {
  const card = (
    <Card
      variant="outlined"
      sx={{
        height: '100%',
        transition: 'border-color 0.15s',
        '&:hover': { borderColor: 'primary.dark' },
      }}
    >
      <CardContent sx={{ pb: '16px !important' }}>
        <Stack direction="row" sx={{ alignItems: 'center', justifyContent: 'space-between' }}>
          <Typography variant="overline" color="text.secondary" sx={{ display: 'block' }} noWrap>
            {title}
          </Typography>
          {detail && <InfoOutlinedIcon sx={{ fontSize: 14, color: 'text.disabled' }} />}
        </Stack>
        <Typography
          variant="h4"
          sx={{ fontWeight: 700, color: accent, fontFamily: '"JetBrains Mono", ui-monospace, monospace', fontSize: 28, lineHeight: 1.2, my: 0.5 }}
        >
          {value}
        </Typography>
        <Box sx={{ width: '100%', overflow: 'hidden', mb: 1 }}>
          <StatMeterChart fill={fill} />
        </Box>
        {sub && (
          <Typography variant="caption" color="text.secondary" noWrap title={sub} sx={{ display: 'block', lineHeight: 1.4 }}>
            {sub}
          </Typography>
        )}
      </CardContent>
    </Card>
  )
  if (!detail) return card
  return (
    <Tooltip title={detail} placement="bottom-start">
      {card}
    </Tooltip>
  )
}
