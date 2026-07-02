import {
  Box,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import type { ToolStat } from '../api/types'
import { formatNum } from '../api/client'

export function AnalyticsTab({
  tools,
  symbols,
  imports,
}: {
  tools: ToolStat[] | null
  symbols: { kind: string; count: number }[] | null
  imports: { target: string; count: number }[] | null
}) {
  return (
    <Box>
      <Typography variant="overline" color="text.secondary">
        Tool performance
      </Typography>
      <Card variant="outlined" sx={{ mt: 1, mb: 3, overflowX: 'auto' }}>
        <Table size="small" stickyHeader>
          <TableHead>
            <TableRow>
              <TableCell>Tool</TableCell>
              <TableCell align="right">Calls</TableCell>
              <TableCell align="right" sx={{ display: { xs: 'none', md: 'table-cell' } }}>
                Avg ms
              </TableCell>
              <TableCell align="right" sx={{ display: { xs: 'none', md: 'table-cell' } }}>
                CPU ms
              </TableCell>
              <TableCell align="right">Tokens saved</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {!tools?.length && (
              <TableRow>
                <TableCell colSpan={5}>
                  <Typography color="text.secondary">No MCP calls yet</Typography>
                </TableCell>
              </TableRow>
            )}
            {tools?.map((t) => (
              <TableRow key={t.tool_name}>
                <TableCell>{t.tool_name}</TableCell>
                <TableCell align="right">{formatNum(t.calls)}</TableCell>
                <TableCell align="right" sx={{ display: { xs: 'none', md: 'table-cell' } }}>
                  {t.avg_duration_ms?.toFixed(0)}
                </TableCell>
                <TableCell align="right" sx={{ display: { xs: 'none', md: 'table-cell' } }}>
                  {t.avg_cpu_ms?.toFixed(1)}
                </TableCell>
                <TableCell align="right">{formatNum(t.tokens_saved)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Card>
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 2 }}>
        <ChartCard title="Symbols by kind" data={symbols?.map((s) => ({ name: s.kind, count: s.count })) || []} />
        <ChartCard title="Top imports" data={imports?.slice(0, 15).map((i) => ({ name: i.target, count: i.count })) || []} />
      </Box>
    </Box>
  )
}

function ChartCard({ title, data }: { title: string; data: { name: string; count: number }[] }) {
  return (
    <Card variant="outlined">
      <CardContent>
        <Typography variant="subtitle2" gutterBottom>
          {title}
        </Typography>
        {data.length === 0 ? (
          <Typography variant="body2" color="text.secondary">
            No data
          </Typography>
        ) : (
          <Box sx={{ height: 220 }}>
            <ResponsiveContainer>
              <BarChart data={data} layout="vertical" margin={{ left: 80 }}>
                <XAxis type="number" tick={{ fill: '#8b949e', fontSize: 10 }} />
                <YAxis type="category" dataKey="name" width={75} tick={{ fill: '#8b949e', fontSize: 10 }} />
                <Tooltip contentStyle={{ background: '#161b22', border: '1px solid #30363d' }} />
                <Bar dataKey="count" fill="#58a6ff" />
              </BarChart>
            </ResponsiveContainer>
          </Box>
        )}
      </CardContent>
    </Card>
  )
}
