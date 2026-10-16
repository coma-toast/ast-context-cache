import { useCallback, useMemo, useState } from 'react'
import {
  Box,
  Button,
  Card,
  CardContent,
  Grid,
  IconButton,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TablePagination,
  TextField,
  Typography,
} from '@mui/material'
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward'
import DeleteIcon from '@mui/icons-material/Delete'
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown'
import RefreshIcon from '@mui/icons-material/Refresh'
import type { MemoryData } from '../api/types'
import { api, formatNum } from '../api/client'
import { useToast } from '../context/ToastContext'
import { StatCard } from '../components/HealthBar'

const SORT_STORAGE_KEY = 'ast-memory-doc-sources-sort'

type SortField = 'name' | 'type' | 'age'
type SortDirection = 'asc' | 'desc'

interface SortState {
  field: SortField
  direction: SortDirection
}

function loadSort(): SortState {
  try {
    const saved = localStorage.getItem(SORT_STORAGE_KEY)
    if (saved) {
      const parsed = JSON.parse(saved)
      if (parsed && typeof parsed.field === 'string' && typeof parsed.direction === 'string') {
        if (['name', 'type', 'age'].includes(parsed.field) && ['asc', 'desc'].includes(parsed.direction)) {
          return parsed
        }
      }
    }
  } catch {
    // ignore
  }
  return { field: 'name', direction: 'asc' }
}

function saveSort(sort: SortState) {
  try {
    localStorage.setItem(SORT_STORAGE_KEY, JSON.stringify(sort))
  } catch {
    // ignore
  }
}

function getAgeSeconds(ageStr: string): number {
  if (!ageStr) return 0
  const totalSeconds = parseDuration(ageStr)
  return totalSeconds
}

function parseDuration(duration: string): number {
  const regex = /(\d+)\s*(second|minute|hour|day|week|month|year)s?/gi
  let total = 0
  let match
  while ((match = regex.exec(duration)) !== null) {
    const value = parseInt(match[1], 10)
    switch (match[2].toLowerCase()) {
      case 'second': total += value; break
      case 'minute': total += value * 60; break
      case 'hour': total += value * 60 * 60; break
      case 'day': total += value * 60 * 60 * 24; break
      case 'week': total += value * 60 * 60 * 24 * 7; break
      case 'month': total += value * 60 * 60 * 24 * 30; break
      case 'year': total += value * 60 * 60 * 24 * 365; break
    }
  }
  return total
}

export function MemoryTab({
  data,
  onRefresh,
  docSourcesPage,
  onDocSourcesPageChange,
}: {
  data: MemoryData | null
  onRefresh: () => void
  docSourcesPage: number
  onDocSourcesPageChange: (page: number) => void
}) {
  const { showToast } = useToast()
  const [docName, setDocName] = useState('')
  const [docUrl, setDocUrl] = useState('')
  const [docType, setDocType] = useState('markdown')
  const [packBusy, setPackBusy] = useState(false)
  const [sort, setSort] = useState<SortState>(loadSort)

  const handleSort = useCallback((field: SortField) => {
    const next: SortState = {
      field,
      direction: sort.field === field && sort.direction === 'asc' ? 'desc' : 'asc',
    }
    setSort(next)
    saveSort(next)
  }, [sort])

  const sortedSources = useMemo(() => {
    if (!data?.DocSources) return []
    const sorted = [...data.DocSources]
    const dir = sort.direction === 'asc' ? 1 : -1
    sorted.sort((a, b) => {
      switch (sort.field) {
        case 'name':
          return dir * a.Name.localeCompare(b.Name)
        case 'type':
          return dir * a.Type.localeCompare(b.Type)
        case 'age':
          return dir * (getAgeSeconds(a.Age) - getAgeSeconds(b.Age))
        default:
          return 0
      }
    })
    return sorted
  }, [data?.DocSources, sort])

  return (
    <Box>
      <Typography variant="overline" color="text.secondary">
        Virtual context
      </Typography>
      <Grid container spacing={2} sx={{ mt: 0.5, mb: 3 }}>
        <Grid size={{ xs: 12, sm: 4 }}>
          <StatCard title="Inventory tokens" value={formatNum(data.VirtualInventoryTokens)} sub={`${data.VirtualNotesCount} notes`} />
        </Grid>
        <Grid size={{ xs: 12, sm: 4 }}>
          <StatCard title="30d utilization" value={`${data.VirtualUtilPct30d?.toFixed(0) || 0}%`} />
        </Grid>
        <Grid size={{ xs: 12, sm: 4 }}>
          <StatCard title="Orphans" value={formatNum(data.VirtualOrphanCount)} />
        </Grid>
      </Grid>

      <Typography variant="overline" color="text.secondary">
        Knowledge base (doc sources)
      </Typography>
      <Card variant="outlined" sx={{ mt: 1, mb: 2 }}>
        <CardContent>
          <Stack direction="row" spacing={1} useFlexGap sx={{ flexWrap: 'wrap', mb: 2 }}>
            <TextField size="small" label="Name" value={docName} onChange={(e) => setDocName(e.target.value)} />
            <TextField size="small" label="URL" value={docUrl} onChange={(e) => setDocUrl(e.target.value)} sx={{ minWidth: 240 }} />
            <TextField size="small" label="Type" value={docType} onChange={(e) => setDocType(e.target.value)} />
            <Button
              variant="contained"
              size="small"
              onClick={async () => {
                try {
                  await api.addDocSource(docName, docUrl, docType)
                  showToast('Doc source added', 'success')
                  setDocName('')
                  setDocUrl('')
                  onRefresh()
                } catch (e) {
                  showToast(String(e), 'error')
                }
              }}
            >
              Add source
            </Button>
            <Button
              variant="outlined"
              size="small"
              disabled={packBusy}
              onClick={async () => {
                setPackBusy(true)
                try {
                  const r = await api.installDocPack()
                  showToast(`Starter pack: ${r.added ?? 0} sources queued`, 'success')
                  onRefresh()
                } catch (e) {
                  showToast(String(e), 'error')
                } finally {
                  setPackBusy(false)
                }
              }}
            >
              Add starter doc pack
            </Button>
          </Stack>
          {data.DocSources?.length === 0 ? (
            <Typography color="text.secondary">No doc sources — add one above or use Add starter doc pack</Typography>
          ) : (
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell
                    onClick={() => handleSort('name')}
                    sx={{ cursor: 'pointer', userSelect: 'none', '&:hover': { bgcolor: 'action.hover' } }}
                  >
                    <Stack direction="row" spacing={0.5} alignItems="center">
                      Name
                      {renderSortIcon(sort, 'name')}
                    </Stack>
                  </TableCell>
                  <TableCell
                    onClick={() => handleSort('type')}
                    sx={{ cursor: 'pointer', userSelect: 'none', '&:hover': { bgcolor: 'action.hover' } }}
                  >
                    <Stack direction="row" spacing={0.5} alignItems="center">
                      Type
                      {renderSortIcon(sort, 'type')}
                    </Stack>
                  </TableCell>
                  <TableCell
                    onClick={() => handleSort('age')}
                    sx={{ cursor: 'pointer', userSelect: 'none', '&:hover': { bgcolor: 'action.hover' } }}
                  >
                    <Stack direction="row" spacing={0.5} alignItems="center">
                      Age
                      {renderSortIcon(sort, 'age')}
                    </Stack>
                  </TableCell>
                  <TableCell align="right">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {sortedSources.map((d) => (
                  <TableRow key={d.ID}>
                    <TableCell>{d.Name}</TableCell>
                    <TableCell>{d.Type}</TableCell>
                    <TableCell>{d.Age}{d.Stale ? ' (stale)' : ''}</TableCell>
                    <TableCell align="right">
                      <IconButton size="small" aria-label="Refresh" onClick={async () => {
                        try {
                          await api.docSourceAction('refresh', d.ID)
                          showToast('Refreshing', 'info')
                          onRefresh()
                        } catch (e) {
                          showToast(String(e), 'error')
                        }
                      }}>
                        <RefreshIcon fontSize="small" />
                      </IconButton>
                      <IconButton size="small" aria-label="Delete" onClick={async () => {
                        if (!confirm(`Delete doc source "${d.Name}"?`)) return
                        try {
                          await api.docSourceAction('delete', d.ID)
                          showToast('Deleted', 'success')
                          onRefresh()
                        } catch (e) {
                          showToast(String(e), 'error')
                        }
                      }}>
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
          {(data.DocSourcesTotal ?? 0) > 0 && (
            <TablePagination
              component="div"
              count={data.DocSourcesTotal}
              page={Math.max(0, docSourcesPage - 1)}
              onPageChange={(_e, newPage) => onDocSourcesPageChange(newPage + 1)}
              rowsPerPage={data.DocSourcesPerPage || 10}
              rowsPerPageOptions={[data.DocSourcesPerPage || 10]}
            />
          )}
        </CardContent>
      </Card>
    </Box>
  )
}

function renderSortIcon(sort: SortState, field: SortField) {
  if (sort.field !== field) return <KeyboardArrowDownIcon fontSize="inherit" sx={{ opacity: 0.35 }} />
  return sort.direction === 'asc'
    ? <ArrowUpwardIcon fontSize="inherit" />
    : <KeyboardArrowDownIcon fontSize="inherit" />
}