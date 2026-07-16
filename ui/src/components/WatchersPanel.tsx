import {
  Box,
  Card,
  CardContent,
  IconButton,
  Pagination,
  Stack,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material'
import CheckIcon from '@mui/icons-material/Check'
import CloseIcon from '@mui/icons-material/Close'
import DeleteIcon from '@mui/icons-material/Delete'
import EditIcon from '@mui/icons-material/Edit'
import PauseIcon from '@mui/icons-material/Pause'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import RefreshIcon from '@mui/icons-material/Refresh'
import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { WatcherInfo } from '../api/types'
import { useToast } from '../context/ToastContext'
import { chartColors } from '../lib/chartColors'
import { flattenSpaceWatchers, groupWatchers } from '../lib/watcherGroups'

const PAGE_SIZE = 8

export function WatchersPanel({ watchers, onRefresh }: { watchers: WatcherInfo[]; onRefresh?: () => void }) {
  const active = watchers.filter((w) => w.Active).length
  const { local, spaces } = groupWatchers(watchers)
  const spaceFlat = flattenSpaceWatchers(spaces)
  const showSpaces = spaceFlat.length > 0

  return (
    <Card variant="outlined">
      <CardContent>
        <Typography variant="subtitle2" gutterBottom>
          File watchers ({active}/{watchers.length} active)
        </Typography>
        {watchers.length === 0 ? (
          <Typography variant="body2" color="text.secondary">
            No active watchers
          </Typography>
        ) : (
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', md: showSpaces ? '1fr 1fr' : '1fr' },
              gap: 2,
              alignItems: 'start',
            }}
          >
            <WatcherSubCard title="Projects" items={local} onRefresh={onRefresh} />
            {showSpaces && <WatcherSubCard title="Spaces" entries={spaceFlat} onRefresh={onRefresh} />}
          </Box>
        )}
      </CardContent>
    </Card>
  )
}

function WatcherSubCard({
  title,
  items,
  entries,
  onRefresh,
}: {
  title: string
  items?: WatcherInfo[]
  entries?: { watcher: WatcherInfo; groupKey: string; groupLabel: string }[]
  onRefresh?: () => void
}) {
  const rows =
    entries ??
    (items || []).map((watcher) => ({
      watcher,
      groupKey: 'local',
      groupLabel: '',
    }))
  const pageCount = Math.max(1, Math.ceil(rows.length / PAGE_SIZE))
  const [page, setPage] = useState(1)

  useEffect(() => {
    setPage((p) => Math.min(p, pageCount))
  }, [pageCount])

  const slice = rows.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)
  const range =
    rows.length === 0 ? '0' : `${(page - 1) * PAGE_SIZE + 1}–${Math.min(page * PAGE_SIZE, rows.length)} of ${rows.length}`

  return (
    <Card variant="outlined" sx={{ bgcolor: 'background.default' }}>
      <CardContent sx={{ '&:last-child': { pb: 2 } }}>
        <Stack direction="row" alignItems="baseline" justifyContent="space-between" spacing={1} sx={{ mb: 1 }}>
          <Typography variant="caption" color="text.secondary" sx={{ textTransform: 'uppercase', letterSpacing: 0.6, fontWeight: 600 }}>
            {title}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {range}
          </Typography>
        </Stack>
        {rows.length === 0 ? (
          <Typography variant="body2" color="text.secondary">
            None
          </Typography>
        ) : (
          <>
            <Stack spacing={0.75}>
              {slice.map((entry, i) => {
                const showGroup = Boolean(entry.groupLabel) && (i === 0 || slice[i - 1].groupKey !== entry.groupKey)
                return (
                  <Box key={entry.watcher.ProjectPath}>
                    {showGroup && (
                      <Typography
                        variant="caption"
                        color="text.secondary"
                        fontWeight={600}
                        display="block"
                        sx={{ mb: 0.5, mt: i === 0 ? 0 : 1 }}
                      >
                        {entry.groupLabel}
                      </Typography>
                    )}
                    <WatcherRow watcher={entry.watcher} onRefresh={onRefresh} />
                  </Box>
                )
              })}
            </Stack>
            {pageCount > 1 && (
              <Stack direction="row" justifyContent="center" sx={{ mt: 1.5 }}>
                <Pagination
                  size="small"
                  color="primary"
                  count={pageCount}
                  page={page}
                  onChange={(_, p) => setPage(p)}
                  siblingCount={0}
                  boundaryCount={1}
                />
              </Stack>
            )}
          </>
        )}
      </CardContent>
    </Card>
  )
}

function WatcherRow({ watcher, onRefresh }: { watcher: WatcherInfo; onRefresh?: () => void }) {
  const { showToast } = useToast()
  const [busy, setBusy] = useState<string | null>(null)
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(watcher.Label || watcher.Name || '')
  const label = watcher.Label || watcher.Name || watcher.ProjectPath

  useEffect(() => {
    if (!editing) setDraft(watcher.Label || watcher.Name || '')
  }, [watcher.Label, watcher.Name, editing])

  const run = async (action: string, fn: () => Promise<unknown>, okMsg: string) => {
    if (busy) return
    setBusy(action)
    try {
      await fn()
      showToast(okMsg, 'success')
      onRefresh?.()
    } catch (e) {
      showToast(e instanceof Error ? e.message : `${action} failed`, 'error')
    } finally {
      setBusy(null)
    }
  }

  const saveLabel = async () => {
    const next = draft.trim()
    if (next === label) {
      setEditing(false)
      return
    }
    await run('rename', () => api.setProjectLabel(watcher.ProjectPath, next), next ? `Renamed to ${next}` : `Reset name for ${label}`)
    setEditing(false)
  }

  if (editing) {
    return (
      <Stack direction="row" spacing={0.5} alignItems="center" sx={{ py: 0.25, px: 0.5 }}>
        <TextField
          size="small"
          fullWidth
          autoFocus
          value={draft}
          placeholder={watcher.Name || 'Display name'}
          helperText="Clear to restore auto name"
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault()
              void saveLabel()
            }
            if (e.key === 'Escape') setEditing(false)
          }}
        />
        <IconButton size="small" aria-label="Save name" disabled={busy != null} onClick={() => void saveLabel()} color="primary">
          <CheckIcon fontSize="inherit" />
        </IconButton>
        <IconButton size="small" aria-label="Cancel rename" disabled={busy != null} onClick={() => setEditing(false)}>
          <CloseIcon fontSize="inherit" />
        </IconButton>
      </Stack>
    )
  }

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 1,
        minWidth: 0,
        py: 0.5,
        px: 0.75,
        borderRadius: 1,
        opacity: watcher.Active ? 1 : 0.72,
        '&:hover .watcher-actions': { opacity: 1 },
      }}
      title={watcher.ProjectPath}
    >
      <Box
        sx={{
          width: 8,
          height: 8,
          borderRadius: '50%',
          flexShrink: 0,
          bgcolor: watcher.Active ? chartColors.green : chartColors.track,
        }}
      />
      <Typography variant="body2" sx={{ flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {label}
        {watcher.LinkedCount > 0 ? ` · ${watcher.LinkedCount} linked` : ''}
      </Typography>
      <Stack className="watcher-actions" direction="row" spacing={0} sx={{ opacity: { xs: 1, md: 0.35 }, transition: 'opacity 0.15s', flexShrink: 0 }}>
        <Tooltip title="Edit display name">
          <span>
            <IconButton
              size="small"
              aria-label={`Rename ${label}`}
              disabled={busy != null}
              onClick={() => {
                setDraft(label)
                setEditing(true)
              }}
            >
              <EditIcon fontSize="inherit" />
            </IconButton>
          </span>
        </Tooltip>
        <Tooltip title="Re-index project and queue embeddings">
          <span>
            <IconButton
              size="small"
              aria-label={`Re-index ${label}`}
              disabled={busy != null}
              onClick={() => run('index', () => api.indexProject(watcher.ProjectPath), `Indexed ${label}`)}
            >
              <RefreshIcon fontSize="inherit" />
            </IconButton>
          </span>
        </Tooltip>
        {watcher.Active ? (
          <Tooltip title="Pause watcher">
            <span>
              <IconButton
                size="small"
                aria-label={`Pause ${label}`}
                disabled={busy != null}
                onClick={() => run('stop', () => api.stopWatcher(watcher.ProjectPath), `Paused ${label}`)}
              >
                <PauseIcon fontSize="inherit" />
              </IconButton>
            </span>
          </Tooltip>
        ) : (
          <Tooltip title="Start watcher">
            <span>
              <IconButton
                size="small"
                aria-label={`Start ${label}`}
                disabled={busy != null}
                onClick={() => run('start', () => api.startWatcher(watcher.ProjectPath), `Started ${label}`)}
              >
                <PlayArrowIcon fontSize="inherit" />
              </IconButton>
            </span>
          </Tooltip>
        )}
        <Tooltip title="Delete watcher and project index data">
          <span>
            <IconButton
              size="small"
              color="error"
              aria-label={`Delete ${label}`}
              disabled={busy != null}
              onClick={() => {
                if (!confirm(`Delete watcher and index data for ${label}?`)) return
                void run('delete', () => api.deleteWatcher(watcher.ProjectPath), `Deleted ${label}`)
              }}
            >
              <DeleteIcon fontSize="inherit" />
            </IconButton>
          </span>
        </Tooltip>
      </Stack>
    </Box>
  )
}
