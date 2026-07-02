import { useMemo, useState } from 'react'
import { Box, ToggleButton, ToggleButtonGroup, Typography } from '@mui/material'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import type { TimeseriesPoint } from '../api/types'

export function ActivityTab({ data }: { data: TimeseriesPoint[] | null }) {
  const [interval, setInterval] = useState<'daily' | 'hourly'>('daily')
  const [metric, setMetric] = useState<'queries' | 'tokens_saved'>('queries')

  const chartData = useMemo(() => {
    if (!data) return []
    return data.map((d) => ({
      period: d.timestamp,
      value: metric === 'queries' ? d.queries : d.tokens_saved,
    }))
  }, [data, metric])

  return (
    <Box>
      <StackControls interval={interval} setInterval={setInterval} metric={metric} setMetric={setMetric} />
      {!data ? (
        <Typography color="text.secondary">Loading chart…</Typography>
      ) : chartData.length === 0 ? (
        <Typography color="text.secondary">No activity yet — run MCP search tools</Typography>
      ) : (
        <Box sx={{ width: '100%', height: 360 }} role="img" aria-label={`Activity chart ${metric} ${interval}`}>
          <ResponsiveContainer>
            <LineChart data={chartData}>
              <CartesianGrid stroke="#30363d" />
              <XAxis dataKey="period" tick={{ fill: '#8b949e', fontSize: 11 }} />
              <YAxis tick={{ fill: '#8b949e', fontSize: 11 }} />
              <Tooltip contentStyle={{ background: '#161b22', border: '1px solid #30363d' }} />
              <Line type="monotone" dataKey="value" stroke="#58a6ff" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </Box>
      )}
    </Box>
  )
}

function StackControls({
  interval,
  setInterval,
  metric,
  setMetric,
}: {
  interval: string
  setInterval: (v: 'daily' | 'hourly') => void
  metric: string
  setMetric: (v: 'queries' | 'tokens_saved') => void
}) {
  return (
    <Box sx={{ mb: 2, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
      <ToggleButtonGroup size="small" value={interval} exclusive onChange={(_, v) => v && setInterval(v)}>
        <ToggleButton value="daily">Daily</ToggleButton>
        <ToggleButton value="hourly">Hourly</ToggleButton>
      </ToggleButtonGroup>
      <ToggleButtonGroup size="small" value={metric} exclusive onChange={(_, v) => v && setMetric(v)}>
        <ToggleButton value="queries">Queries</ToggleButton>
        <ToggleButton value="tokens_saved">Tokens saved</ToggleButton>
      </ToggleButtonGroup>
    </Box>
  )
}
