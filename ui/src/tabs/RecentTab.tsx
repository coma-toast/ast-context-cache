import { useState, Fragment, useEffect, useMemo, useRef, useCallback } from 'react'
import {
  Box,
  Button,
  Card,
  Checkbox,
  Collapse,
  FormControlLabel,
  Stack,
  Tab,
  Tabs,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from '@mui/material'
import type { RecentQuery, RecentLogLine } from '../api/types'
import { api } from '../api/client'

const LOG_LEVEL_COLOR: Record<string, string> = {
  error: '#f85149',
  warn: '#d29922',
  info: '#8b949e',
}

export function RecentTab({
  mcp,
  indexing,
}: {
  mcp: RecentQuery[] | null
  indexing: RecentQuery[] | null
}) {
  const [sub, setSub] = useState(0)
  const [expanded, setExpanded] = useState<number | null>(null)

  return (
    <Box>
      <Tabs value={sub} onChange={(_, v) => setSub(v)} sx={{ mb: 2 }}>
        <Tab label="MCP tool calls" />
        <Tab label="Indexing activity" />
        <Tab label="Server logs" />
      </Tabs>
      {sub === 2 ? (
        <LogsPane />
      ) : (
        <QueryTable rows={sub === 0 ? mcp : indexing} expanded={expanded} setExpanded={setExpanded} />
      )}
    </Box>
  )
}

function QueryTable({
  rows,
  expanded,
  setExpanded,
}: {
  rows: RecentQuery[] | null
  expanded: number | null
  setExpanded: (v: number | null) => void
}) {
  const [search, setSearch] = useState('')
  const [errorsOnly, setErrorsOnly] = useState(false)
  const filtered = useMemo(() => {
    if (!rows) return rows
    const q = search.trim().toLowerCase()
    return rows.filter((r) => {
      if (errorsOnly && !r.Error) return false
      if (!q) return true
      return (r.ToolName || '').toLowerCase().includes(q) || (r.Query || '').toLowerCase().includes(q)
    })
  }, [rows, search, errorsOnly])

  return (
    <Card variant="outlined" sx={{ overflowX: 'auto' }}>
      <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap" sx={{ px: 1.5, py: 1, borderBottom: '1px solid', borderColor: 'divider' }}>
        <TextField
          size="small"
          placeholder="Filter by tool or query…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          sx={{ minWidth: 220 }}
        />
        <FormControlLabel
          control={<Checkbox size="small" checked={errorsOnly} onChange={(e) => setErrorsOnly(e.target.checked)} />}
          label="Errors only"
        />
        <Typography variant="caption" color="text.secondary">
          Showing the {rows?.length ?? 0} most recent — older activity isn't kept
        </Typography>
      </Stack>
      <Table size="small" stickyHeader>
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
          {!filtered?.length && (
            <TableRow>
              <TableCell colSpan={6}>
                <Typography color="text.secondary">{rows?.length ? 'No activity matches this filter' : 'No recent activity'}</Typography>
              </TableCell>
            </TableRow>
          )}
          {filtered?.map((q, i) => (
            <Fragment key={`${q.Timestamp}-${i}`}>
              <TableRow>
                <TableCell sx={{ whiteSpace: 'nowrap' }}>{q.Timestamp?.slice(0, 19)}</TableCell>
                <TableCell>{q.ToolName}</TableCell>
                <TableCell sx={{ display: { xs: 'none', sm: 'table-cell' }, maxWidth: 200 }} title={q.Query}>
                  {q.Query?.slice(0, 60)}
                </TableCell>
                <TableCell align="right">{q.DurationMs?.toFixed(0)}</TableCell>
                <TableCell align="right" sx={{ display: { xs: 'none', md: 'table-cell' } }}>
                  {q.Saved ?? 0}
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
                      <Box
                        sx={{
                          p: 1.5,
                          my: 0.5,
                          bgcolor: 'action.hover',
                          borderLeft: '3px solid',
                          borderColor: 'error.main',
                          fontFamily: 'ui-monospace, monospace',
                          fontSize: 12,
                          whiteSpace: 'pre-wrap',
                          wordBreak: 'break-word',
                        }}
                      >
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
    </Card>
  )
}

type LogLevelFilter = 'all' | 'error' | 'warn' | 'info'

function LogsPane() {
  const [data, setData] = useState<{ lines: RecentLogLine[]; path: string; file_truncated: boolean } | null>(null)
  const [autoscroll, setAutoscroll] = useState(true)
  const [levelFilter, setLevelFilter] = useState<LogLevelFilter>('all')
  const scrollRef = useRef<HTMLDivElement>(null)
  const filteredLines = useMemo(() => {
    if (!data?.lines) return data?.lines
    if (levelFilter === 'all') return data.lines
    return data.lines.filter((l) => (l.Level || 'info') === levelFilter)
  }, [data, levelFilter])

  const load = useCallback(async () => {
    try {
      const d = await api.recentLogs()
      setData(d)
    } catch {
      setData({ lines: [], path: '', file_truncated: false })
    }
  }, [])

  useEffect(() => {
    load()
    const t = setInterval(load, 3000)
    return () => clearInterval(t)
  }, [load])

  useEffect(() => {
    if (autoscroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [filteredLines, autoscroll])

  return (
    <Card variant="outlined" sx={{ overflow: 'hidden' }}>
      <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap" sx={{ px: 2, py: 1, borderBottom: '1px solid', borderColor: 'divider' }}>
        <Typography variant="caption" color="text.secondary" sx={{ fontFamily: 'ui-monospace, monospace', overflow: 'hidden', textOverflow: 'ellipsis' }} title={data?.path}>
          {data?.path || 'Loading log path…'}
        </Typography>
        <Typography variant="caption" color="text.secondary">
          {filteredLines?.length ?? 0} of {data?.lines?.length ?? 0} lines{data?.file_truncated ? ' (truncated)' : ''}
        </Typography>
        <ToggleButtonGroup
          size="small"
          exclusive
          value={levelFilter}
          onChange={(_e, v) => v && setLevelFilter(v)}
        >
          <ToggleButton value="all">All</ToggleButton>
          <ToggleButton value="error" sx={{ color: LOG_LEVEL_COLOR.error }}>
            Error
          </ToggleButton>
          <ToggleButton value="warn" sx={{ color: LOG_LEVEL_COLOR.warn }}>
            Warn
          </ToggleButton>
          <ToggleButton value="info" sx={{ color: LOG_LEVEL_COLOR.info }}>
            Info
          </ToggleButton>
        </ToggleButtonGroup>
        <Button size="small" variant={autoscroll ? 'contained' : 'outlined'} onClick={() => setAutoscroll((v) => !v)}>
          Autoscroll {autoscroll ? 'on' : 'off'}
        </Button>
      </Stack>
      <Box
        ref={scrollRef}
        sx={{
          maxHeight: 480,
          overflowY: 'auto',
          p: 1.5,
          fontFamily: 'ui-monospace, monospace',
          fontSize: 11,
          lineHeight: 1.45,
          bgcolor: '#0d1117',
        }}
      >
        {!filteredLines?.length && (
          <Typography color="text.secondary">{data?.lines?.length ? 'No lines at this level' : 'No log lines'}</Typography>
        )}
        {filteredLines?.map((line, i) => (
          <Box key={i} sx={{ color: LOG_LEVEL_COLOR[line.Level] || LOG_LEVEL_COLOR.info, mb: 0.25 }}>
            {line.Timestamp && (
              <Box component="span" sx={{ color: 'text.secondary', mr: 1 }}>
                {line.Timestamp}
              </Box>
            )}
            {line.Message}
          </Box>
        ))}
      </Box>
    </Card>
  )
}
