import { useState, Fragment } from 'react'
import {
  Box,
  Button,
  Collapse,
  Tab,
  Tabs,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material'
import type { RecentQuery } from '../api/types'

export function RecentTab({
  mcp,
  indexing,
}: {
  mcp: RecentQuery[] | null
  indexing: RecentQuery[] | null
}) {
  const [sub, setSub] = useState(0)
  const [expanded, setExpanded] = useState<number | null>(null)

  const rows = sub === 0 ? mcp : sub === 1 ? indexing : []

  return (
    <Box>
      <Tabs value={sub} onChange={(_, v) => setSub(v)} sx={{ mb: 2 }}>
        <Tab label="MCP tool calls" />
        <Tab label="Indexing activity" />
      </Tabs>
      <Table size="small" stickyHeader sx={{ display: { overflowX: 'auto' } }}>
        <TableHead>
          <TableRow>
            <TableCell>Time</TableCell>
            <TableCell>Tool</TableCell>
            <TableCell sx={{ display: { xs: 'none', sm: 'table-cell' } }}>Query</TableCell>
            <TableCell align="right">ms</TableCell>
            <TableCell align="right" sx={{ display: { xs: 'none', md: 'table-cell' } }}>
              Saved
            </TableCell>
            <TableCell>Error</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {!rows?.length && (
            <TableRow>
              <TableCell colSpan={6}>
                <Typography color="text.secondary">No recent activity</Typography>
              </TableCell>
            </TableRow>
          )}
          {rows?.map((q, i) => (
            <Fragment key={`${q.Timestamp}-${i}`}>
              <TableRow>
                <TableCell sx={{ whiteSpace: 'nowrap' }}>{q.Timestamp?.slice(0, 19)}</TableCell>
                <TableCell>{q.ToolName}</TableCell>
                <TableCell sx={{ display: { xs: 'none', sm: 'table-cell' }, maxWidth: 200 }} title={q.Query}>
                  {q.Query?.slice(0, 60)}
                </TableCell>
                <TableCell align="right">{q.DurationMs?.toFixed(0)}</TableCell>
                <TableCell align="right" sx={{ display: { xs: 'none', md: 'table-cell' } }}>
                  {q.Saved ?? q.TokensSaved ?? 0}
                </TableCell>
                <TableCell>
                  {q.Error ? (
                    <Button
                      size="small"
                      aria-expanded={expanded === i}
                      onClick={() => setExpanded(expanded === i ? null : i)}
                    >
                      Error
                    </Button>
                  ) : (
                    '—'
                  )}
                </TableCell>
              </TableRow>
              {q.Error && (
                <TableRow>
                  <TableCell colSpan={6} sx={{ py: 0, border: 0 }}>
                    <Collapse in={expanded === i}>
                      <Box sx={{ p: 1, bgcolor: 'background.paper', fontFamily: 'monospace', fontSize: 12 }}>
                        {q.Error}
                      </Box>
                    </Collapse>
                  </TableCell>
                </TableRow>
              )}
            </Fragment>
          ))}
        </TableBody>
      </Table>
    </Box>
  )
}
