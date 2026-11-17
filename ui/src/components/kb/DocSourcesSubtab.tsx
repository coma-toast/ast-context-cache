import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Drawer,
  IconButton,
  List,
  ListItemButton,
  ListItemText,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TablePagination,
  TableRow,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material'
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward'
import CloseIcon from '@mui/icons-material/Close'
import DeleteIcon from '@mui/icons-material/Delete'
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown'
import MenuBookIcon from '@mui/icons-material/MenuBook'
import RefreshIcon from '@mui/icons-material/Refresh'
import { api } from '../../api/client'
import type { DocContentEntry, DocSource, MemoryData } from '../../api/types'
import { useToast } from '../../context/ToastContext'
import { getScopedItem, setScopedItem } from '../../lib/scopedStorage'
import { MarkdownView } from '../MarkdownView'

const SORT_STORAGE_KEY = 'ast-memory-doc-sources-sort'
const ENTRIES_PER_PAGE = 50
const MONO = '"JetBrains Mono", ui-monospace, monospace'

type SortField = 'name' | 'type' | 'age'
type SortDirection = 'asc' | 'desc'

interface SortState {
  field: SortField
  direction: SortDirection
}

export interface DocSourcesSubtabProps {
  data: MemoryData | null
  onRefresh: () => void
  docSourcesPage: number
  onDocSourcesPageChange: (page: number) => void
}

function loadSort(): SortState {
  try {
    const saved = getScopedItem(SORT_STORAGE_KEY)
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
    setScopedItem(SORT_STORAGE_KEY, JSON.stringify(sort))
  } catch {
    // ignore
  }
}

function getAgeSeconds(ageStr: string): number {
  if (!ageStr) return 0
  return parseDuration(ageStr)
}

function parseDuration(duration: string): number {
  const regex = /(\d+)\s*(second|minute|hour|day|week|month|year)s?/gi
  let total = 0
  let match
  while ((match = regex.exec(duration)) !== null) {
    const value = parseInt(match[1], 10)
    switch (match[2].toLowerCase()) {
      case 'second':
        total += value
        break
      case 'minute':
        total += value * 60
        break
      case 'hour':
        total += value * 60 * 60
        break
      case 'day':
        total += value * 60 * 60 * 24
        break
      case 'week':
        total += value * 60 * 60 * 24 * 7
        break
      case 'month':
        total += value * 60 * 60 * 24 * 30
        break
      case 'year':
        total += value * 60 * 60 * 24 * 365
        break
    }
  }
  return total
}

function renderSortIcon(sort: SortState, field: SortField) {
  if (sort.field !== field) return <KeyboardArrowDownIcon fontSize="inherit" sx={{ opacity: 0.35 }} />
  return sort.direction === 'asc' ? <ArrowUpwardIcon fontSize="inherit" /> : <KeyboardArrowDownIcon fontSize="inherit" />
}

export function DocSourcesSubtab({
  data,
  onRefresh,
  docSourcesPage,
  onDocSourcesPageChange,
}: DocSourcesSubtabProps) {
  const { showToast } = useToast()
  const [docName, setDocName] = useState('')
  const [docUrl, setDocUrl] = useState('')
  const [docType, setDocType] = useState('markdown')
  const [packBusy, setPackBusy] = useState(false)
  const [sort, setSort] = useState<SortState>(loadSort)

  // Delete confirmation dialog state
  const [deleteTarget, setDeleteTarget] = useState<DocSource | null>(null)
  const [deletingSource, setDeletingSource] = useState(false)

  // Cached pages browser drawer state
  const [activeSource, setActiveSource] = useState<DocSource | null>(null)
  const [entriesQuery, setEntriesQuery] = useState('')
  const [entriesPage, setEntriesPage] = useState(1)
  const [entries, setEntries] = useState<DocContentEntry[]>([])
  const [entriesTotal, setEntriesTotal] = useState(0)
  const [loadingEntries, setLoadingEntries] = useState(false)
  const [entriesError, setEntriesError] = useState<string | null>(null)
  const [selectedEntry, setSelectedEntry] = useState<DocContentEntry | null>(null)

  const handleSort = useCallback(
    (field: SortField) => {
      const next: SortState = {
        field,
        direction: sort.field === field && sort.direction === 'asc' ? 'desc' : 'asc',
      }
      setSort(next)
      saveSort(next)
    },
    [sort],
  )

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

  useEffect(() => {
    if (!activeSource) return
    let cancelled = false
    setLoadingEntries(true)
    setEntriesError(null)
    api
      .docEntries(activeSource.ID, entriesQuery.trim() || undefined, entriesPage, ENTRIES_PER_PAGE)
      .then((res) => {
        if (cancelled) return
        const list = Array.isArray(res.entries) ? res.entries : []
        setEntries(list)
        setEntriesTotal(res.total ?? list.length)
        setSelectedEntry((prev) => {
          if (prev && list.some((item) => item.id === prev.id)) {
            return list.find((item) => item.id === prev.id) ?? list[0] ?? null
          }
          return list[0] ?? null
        })
      })
      .catch((err) => {
        if (cancelled) return
        setEntriesError(err instanceof Error ? err.message : String(err))
      })
      .finally(() => {
        if (!cancelled) setLoadingEntries(false)
      })
    return () => {
      cancelled = true
    }
  }, [activeSource, entriesQuery, entriesPage])

  const openSourcePages = useCallback((source: DocSource) => {
    setActiveSource(source)
    setEntriesQuery('')
    setEntriesPage(1)
    setEntries([])
    setSelectedEntry(null)
    setEntriesError(null)
  }, [])

  const handleConfirmDelete = async () => {
    if (!deleteTarget) return
    setDeletingSource(true)
    try {
      await api.docSourceAction('delete', deleteTarget.ID)
      showToast('Deleted', 'success')
      if (activeSource?.ID === deleteTarget.ID) {
        setActiveSource(null)
      }
      setDeleteTarget(null)
      onRefresh()
    } catch (e) {
      showToast(String(e), 'error')
    } finally {
      setDeletingSource(false)
    }
  }

  return (
    <Box data-testid="doc-sources-subtab">
      <Typography variant="overline" color="text.secondary">
        Knowledge base (doc sources)
      </Typography>
      <Card variant="outlined" sx={{ mt: 1, mb: 2 }}>
        <CardContent>
          <Stack direction="row" spacing={1} useFlexGap sx={{ flexWrap: 'wrap', mb: 2 }}>
            <TextField size="small" label="Name" value={docName} onChange={(e) => setDocName(e.target.value)} />
            <TextField
              size="small"
              label="URL"
              value={docUrl}
              onChange={(e) => setDocUrl(e.target.value)}
              sx={{ minWidth: 240 }}
            />
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

          {!data || data.DocSources?.length === 0 ? (
            <Typography color="text.secondary">No doc sources — add one above or use Add starter doc pack</Typography>
          ) : (
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell
                    onClick={() => handleSort('name')}
                    sx={{ cursor: 'pointer', userSelect: 'none', '&:hover': { bgcolor: 'action.hover' } }}
                  >
                    <Stack direction="row" spacing={0.5} sx={{ alignItems: 'center' }}>
                      <span>Name</span>
                      {renderSortIcon(sort, 'name')}
                    </Stack>
                  </TableCell>
                  <TableCell
                    onClick={() => handleSort('type')}
                    sx={{ cursor: 'pointer', userSelect: 'none', '&:hover': { bgcolor: 'action.hover' } }}
                  >
                    <Stack direction="row" spacing={0.5} sx={{ alignItems: 'center' }}>
                      <span>Type</span>
                      {renderSortIcon(sort, 'type')}
                    </Stack>
                  </TableCell>
                  <TableCell
                    onClick={() => handleSort('age')}
                    sx={{ cursor: 'pointer', userSelect: 'none', '&:hover': { bgcolor: 'action.hover' } }}
                  >
                    <Stack direction="row" spacing={0.5} sx={{ alignItems: 'center' }}>
                      <span>Age</span>
                      {renderSortIcon(sort, 'age')}
                    </Stack>
                  </TableCell>
                  <TableCell align="right">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {sortedSources.map((d) => (
                  <TableRow
                    key={d.ID}
                    hover
                    onClick={() => openSourcePages(d)}
                    sx={{ cursor: 'pointer' }}
                    data-testid={`doc-source-row-${d.ID}`}
                  >
                    <TableCell>
                      <Typography variant="body2" sx={{ fontWeight: 500 }}>
                        {d.Name}
                      </Typography>
                      <Typography variant="caption" color="text.secondary" sx={{ fontFamily: MONO }}>
                        {d.URL}
                      </Typography>
                    </TableCell>
                    <TableCell>{d.Type}</TableCell>
                    <TableCell>
                      {d.Age}
                      {d.Stale ? ' (stale)' : ''}
                    </TableCell>
                    <TableCell align="right" onClick={(e) => e.stopPropagation()}>
                      <Tooltip title="Browse cached pages">
                        <Button
                          size="small"
                          startIcon={<MenuBookIcon fontSize="small" />}
                          onClick={() => openSourcePages(d)}
                          sx={{ mr: 0.5 }}
                        >
                          Browse pages
                        </Button>
                      </Tooltip>
                      <IconButton
                        size="small"
                        aria-label="Refresh"
                        onClick={async () => {
                          try {
                            await api.docSourceAction('refresh', d.ID)
                            showToast('Refreshing', 'info')
                            onRefresh()
                          } catch (e) {
                            showToast(String(e), 'error')
                          }
                        }}
                      >
                        <RefreshIcon fontSize="small" />
                      </IconButton>
                      <IconButton
                        size="small"
                        aria-label="Delete"
                        onClick={() => setDeleteTarget(d)}
                      >
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}

          {(data?.DocSourcesTotal ?? 0) > 0 && (
            <TablePagination
              component="div"
              count={data!.DocSourcesTotal}
              page={Math.max(0, docSourcesPage - 1)}
              onPageChange={(_e, newPage) => onDocSourcesPageChange(newPage + 1)}
              rowsPerPage={data!.DocSourcesPerPage || 10}
              rowsPerPageOptions={[data!.DocSourcesPerPage || 10]}
            />
          )}
        </CardContent>
      </Card>

      {/* Cached doc_content browser drawer */}
      <Drawer
        anchor="right"
        open={activeSource !== null}
        onClose={() => setActiveSource(null)}
        slotProps={{
          paper: {
            sx: {
              width: { xs: '100%', md: 880 },
              maxWidth: '100vw',
              p: 2.5,
              backgroundImage: 'none',
              bgcolor: 'background.default',
              display: 'flex',
              flexDirection: 'column',
            },
          },
        }}
      >
        {activeSource && (
          <>
            <Stack direction="row" spacing={1} sx={{ mb: 1.5, alignItems: 'flex-start', justifyContent: 'space-between' }}>
              <Box sx={{ minWidth: 0, flex: 1 }}>
                <Stack direction="row" spacing={1} useFlexGap sx={{ alignItems: 'center', flexWrap: 'wrap' }}>
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    {activeSource.Name}
                  </Typography>
                  <Chip size="small" variant="outlined" label={activeSource.Type} />
                  {activeSource.Stale && <Chip size="small" color="warning" variant="outlined" label="stale" />}
                </Stack>
                <Typography variant="caption" color="text.secondary" sx={{ fontFamily: MONO, display: 'block' }} noWrap>
                  {activeSource.URL}
                </Typography>
              </Box>
              <IconButton size="small" aria-label="Close doc browser" onClick={() => setActiveSource(null)}>
                <CloseIcon fontSize="small" />
              </IconButton>
            </Stack>

            <TextField
              size="small"
              label="Search cached pages"
              placeholder="Filter by title, path, or content..."
              value={entriesQuery}
              onChange={(e) => {
                setEntriesQuery(e.target.value)
                setEntriesPage(1)
              }}
              sx={{ mb: 2 }}
              fullWidth
            />

            {entriesError && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {entriesError}
              </Alert>
            )}

            <Box
              sx={{
                display: 'grid',
                gridTemplateColumns: { xs: '1fr', md: '280px 1fr' },
                gap: 2,
                flex: 1,
                minHeight: 0,
              }}
            >
              <Box
                sx={{
                  border: '1px solid',
                  borderColor: 'divider',
                  borderRadius: 1,
                  bgcolor: 'background.paper',
                  display: 'flex',
                  flexDirection: 'column',
                  minHeight: 240,
                  overflow: 'hidden',
                }}
              >
                {loadingEntries && entries.length === 0 ? (
                  <Stack direction="row" spacing={1} sx={{ p: 2, alignItems: 'center' }}>
                    <CircularProgress size={16} />
                    <Typography variant="body2" color="text.secondary">
                      Loading pages…
                    </Typography>
                  </Stack>
                ) : entries.length === 0 ? (
                  <Typography variant="body2" color="text.secondary" sx={{ p: 2 }}>
                    No cached doc pages found.
                  </Typography>
                ) : (
                  <List dense disablePadding sx={{ flex: 1, overflowY: 'auto' }}>
                    {entries.map((entry) => (
                      <ListItemButton
                        key={entry.id}
                        selected={selectedEntry?.id === entry.id}
                        onClick={() => setSelectedEntry(entry)}
                        data-testid={`doc-entry-item-${entry.id}`}
                      >
                        <ListItemText
                          primary={
                            <Typography variant="body2" sx={{ fontWeight: 500 }} noWrap>
                              {entry.title || entry.path || `Entry #${entry.id}`}
                            </Typography>
                          }
                          secondary={
                            <Typography variant="caption" color="text.secondary" noWrap sx={{ fontFamily: MONO, display: 'block' }}>
                              {entry.path || entry.updated_at}
                            </Typography>
                          }
                        />
                      </ListItemButton>
                    ))}
                  </List>
                )}

                {entriesTotal > 0 && (
                  <TablePagination
                    component="div"
                    count={entriesTotal}
                    page={Math.max(0, entriesPage - 1)}
                    onPageChange={(_e, newPage) => setEntriesPage(newPage + 1)}
                    rowsPerPage={ENTRIES_PER_PAGE}
                    rowsPerPageOptions={[ENTRIES_PER_PAGE]}
                    sx={{ borderTop: '1px solid', borderColor: 'divider' }}
                  />
                )}
              </Box>

              <Box
                sx={{
                  border: '1px solid',
                  borderColor: 'divider',
                  borderRadius: 1,
                  bgcolor: 'background.paper',
                  p: 2,
                  overflowY: 'auto',
                  minHeight: 240,
                }}
                data-testid="doc-entry-reader"
              >
                {selectedEntry ? (
                  <>
                    <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                      {selectedEntry.title || selectedEntry.path || `Entry #${selectedEntry.id}`}
                    </Typography>
                    <Stack direction="row" spacing={1.5} sx={{ mb: 1.5, flexWrap: 'wrap' }} useFlexGap>
                      {selectedEntry.path && (
                        <Typography variant="caption" color="text.secondary" sx={{ fontFamily: MONO }}>
                          {selectedEntry.path}
                        </Typography>
                      )}
                      {selectedEntry.updated_at && (
                        <Typography variant="caption" color="text.secondary">
                          Updated {selectedEntry.updated_at}
                        </Typography>
                      )}
                    </Stack>
                    <MarkdownView content={selectedEntry.content} />
                  </>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    Select a cached page on the left to read its content.
                  </Typography>
                )}
              </Box>
            </Box>
          </>
        )}
      </Drawer>

      {/* Delete doc source confirmation dialog */}
      <Dialog
        open={deleteTarget !== null}
        onClose={() => !deletingSource && setDeleteTarget(null)}
        aria-labelledby="delete-doc-source-dialog-title"
      >
        <DialogTitle id="delete-doc-source-dialog-title">Delete doc source?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Delete doc source <strong>{deleteTarget?.Name}</strong> ({deleteTarget?.URL}) and its cached documentation
            entries?
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteTarget(null)} disabled={deletingSource}>
            Cancel
          </Button>
          <Button
            color="error"
            variant="contained"
            disabled={deletingSource}
            onClick={() => void handleConfirmDelete()}
          >
            {deletingSource ? 'Deleting…' : 'Confirm delete'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}
