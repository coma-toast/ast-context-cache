import { Box, Card, CardContent, Stack, Typography } from '@mui/material'
import { Bar, BarChart, Cell, ResponsiveContainer, XAxis, YAxis } from 'recharts'
import type { IndexHealth } from '../api/types'
import { formatNum } from '../api/client'
import { chartColors } from '../lib/chartColors'

export function CorpusChart({ data }: { data: IndexHealth }) {
  const items = [
    { name: 'Symbols', count: data.TotalSymbols, color: chartColors.accent },
    { name: 'Files', count: data.TotalFiles, color: chartColors.purple },
    { name: 'Edges', count: data.TotalEdges, color: chartColors.green },
  ]
  const max = Math.max(...items.map((i) => i.count), 1)
  const chartData = items.map((i) => ({ ...i, pct: (i.count / max) * 100 }))
  return (
    <Card variant="outlined" sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="subtitle2" gutterBottom>
          Corpus
        </Typography>
        <Box sx={{ height: 120, mb: 1 }}>
          <ResponsiveContainer>
            <BarChart data={chartData} layout="vertical" margin={{ left: 8, right: 8, top: 4, bottom: 4 }}>
              <XAxis type="number" domain={[0, 100]} hide />
              <YAxis type="category" dataKey="name" width={56} tick={{ fill: '#8b949e', fontSize: 11 }} />
              <Bar dataKey="pct" radius={[0, 4, 4, 0]} isAnimationActive={false}>
                {chartData.map((entry) => (
                  <Cell key={entry.name} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Box>
        <Stack spacing={0.5}>
          {items.map((i) => (
            <Typography key={i.name} variant="body2">
              {formatNum(i.count)} {i.name.toLowerCase()}
            </Typography>
          ))}
        </Stack>
        <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
          {data.PinnedCount} pinned projects · {formatNum(data.TotalVectors)} vectors
        </Typography>
      </CardContent>
    </Card>
  )
}
