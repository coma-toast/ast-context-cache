import { useMemo, useState } from 'react'
import { Box, ToggleButton, ToggleButtonGroup, Typography, useTheme } from '@mui/material'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import type { TimeseriesPoint } from '../api/types'

export function ActivityTab({
  data,
  interval,
  onIntervalChange,
}: {
  data: TimeseriesPoint[] | null
  interval: 'daily' | 'hourly'
  onIntervalChange: (v: 'daily' | 'hourly') => void
}) {
  const theme = useTheme()
  const [metric, setMetric] = useState<'queries' | 'tokens_saved'>('queries')

  const chartData = useMemo(() => {
    if (!data) return []
    return data.map((d) => ({
      period: d.timestamp,
      value: metric === 'queries' ? d.queries : d.tokens_saved,
    }))
  }, [data, metric])

  const tooltipStyle = {
    background: theme.palette.background.paper,
    border: `1px solid ${theme.palette.divider}`,
  }

  return (
    <Box>
      <Box sx={{ mb: 2, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
        <ToggleButtonGroup size="small" value={interval} exclusive onChange={(_, v) => v && onIntervalChange(v)}>
          <ToggleButton value="daily">Daily</ToggleButton>
          <ToggleButton value="hourly">Hourly</ToggleButton>
        </ToggleButtonGroup>
        <ToggleButtonGroup size="small" value={metric} exclusive onChange={(_, v) => v && setMetric(v)}>
          <ToggleButton value="queries">Queries</ToggleButton>
          <ToggleButton value="tokens_saved">Tokens saved</ToggleButton>
        </ToggleButtonGroup>
      </Box>
      {!data ? (
        <Typography color="text.secondary">Loading chart…</Typography>
      ) : chartData.length === 0 ? (
        <Typography color="text.secondary">No activity yet — run MCP search tools</Typography>
      ) : (
        <Box sx={{ width: '100%', height: 360, minHeight: 360 }} role="img" aria-label={`Activity chart ${metric} ${interval}`}>
          <ResponsiveContainer>
            <LineChart data={chartData}>
              <CartesianGrid stroke={theme.palette.divider} />
              <XAxis dataKey="period" tick={{ fill: theme.palette.text.secondary, fontSize: 11 }} />
              <YAxis tick={{ fill: theme.palette.text.secondary, fontSize: 11 }} />
              <Tooltip contentStyle={tooltipStyle} />
              <Line type="monotone" dataKey="value" stroke={theme.palette.primary.main} dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </Box>
      )}
    </Box>
  )
}
