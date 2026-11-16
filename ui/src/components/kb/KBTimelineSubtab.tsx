import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  LinearProgress,
  MenuItem,
  Pagination,
  Stack,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from '@mui/material'
import AddIcon from '@mui/icons-material/Add'
import HistoryIcon from '@mui/icons-material/History'
import VerifiedIcon from '@mui/icons-material/Verified'

import type { KBEntry, KBKind, KBListParams, KBListResponse } from '../../api/types'
import { api } from '../../api/client'
import { KBEntryFormDialog } from './KBEntryFormDialog'
import {
  KB_KIND_LABELS,
  KB_SCOPE_LABELS,
  KB_SEVERITIES,
  formatDayLabel,
  formatShortTime,
  kbEntryDayKey,
  kbHeadline,
  kbKindChipSx,
  kbSeverityChipSx,
} from './kbHelpers'

const MONO = '"JetBrains Mono", ui-monospace, monospace'
const PER_PAGE = 50

type TimelineKindMode = 'events_and_config' | 'events' | 'config_changes' | 'all'

const KINDS_FOR_MODE: Record<TimelineKindMode, KBKind[] | undefined> = {
  events_and_config: ['event', 'config_change'],
  events: ['event'],
  config_changes: ['config_change'],
  all: undefined,
}

export function KBTimelineSubtab({
  projectId,
  kbRefreshNonce,
  onSelectRef,
}: {
  projectId?: string
  kbRefreshNonce?: number
  onSelectRef: (ref: string) => void
}) {
  const [kindMode, setKindMode] = useState<TimelineKindMode>('events_and_config')
  const [severityFilter, setSeverityFilter] = useState('')
  const [categoryFilter, setCategoryFilter] = useState('')
  const [actorFilter, setActorFilter] = useState('')
  const [sinceFilter, setSinceFilter] = useState('')
  const [untilFilter, setUntilFilter] = useState('')

  const [page, setPage] = useState(1)
  const [localRefresh, setLocalRefresh] = useState(0)

  const [data, setData] = useState<KBListResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const [logDialogOpen, setLogDialogOpen] = useState(false)

  useEffect(() => {
    setPage(1)
  }, [projectId, kindMode, actorFilter, sinceFilter, untilFilter])

  const listParams = useMemo<KBListParams>(() => {
    return {
      kinds: KINDS_FOR_MODE[kindMode],
      project_id: projectId || undefined,
      status: 'active',
      actor: actorFilter.trim() || undefined,
      since: sinceFilter.trim() || undefined,
      until: untilFilter.trim() || undefined,
      sort: 'occurred_desc',
      page,
      per_page: severityFilter || categoryFilter.trim() ? 200 : PER_PAGE,
    }
  }, [kindMode, projectId, actorFilter, sinceFilter, untilFilter, page, severityFilter, categoryFilter])

  const fetchTimeline = useCallback(async (params: KBListParams, signal?: { cancelled: boolean }) => {
    setLoading(true)
    setError('')
    try {
      const res = await api.kbList(params)
      if (!signal?.cancelled) {
        setData(res)
      }
    } catch (err) {
      if (!signal?.cancelled) {
        setError(err instanceof Error ? err.message : 'Failed to load timeline')
      }
    } finally {
      if (!signal?.cancelled) {
        setLoading(false)
      }
    }
  }, [])

  useEffect(() => {
    const signal = { cancelled: false }
    void fetchTimeline(listParams, signal)
    return () => {
      signal.cancelled = true
    }
  }, [fetchTimeline, listParams, kbRefreshNonce, localRefresh])

  const filteredEntries = useMemo(() => {
    const raw = data?.entries ?? []
    const sev = severityFilter.trim().toLowerCase()
    const cat = categoryFilter.trim().toLowerCase()
    if (!sev && !cat) return raw
    return raw.filter((e) => {
      if (sev && (e.severity || '').toLowerCase() !== sev) return false
      if (cat && !(e.category || '').toLowerCase().includes(cat)) return false
      return true
    })
  }, [data?.entries, severityFilter, categoryFilter])

  const groupedByDay = useMemo(() => {
    const groups: { dayKey: string; label: string; items: KBEntry[] }[] = []
    const indexByKey = new Map<string, number>()
    for (const entry of filteredEntries) {
      const key = kbEntryDayKey(entry)
      const existingIdx = indexByKey.get(key)
      if (existingIdx != null) {
        groups[existingIdx].items.push(entry)
      } else {
        indexByKey.set(key, groups.length)
        groups.push({ dayKey: key, label: formatDayLabel(key), items: [entry] })
      }
    }
    return groups
  }, [filteredEntries])

  const hasFilters =
    kindMode !== 'events_and_config' ||
    Boolean(severityFilter) ||
    Boolean(categoryFilter.trim()) ||
    Boolean(actorFilter.trim()) ||
    Boolean(sinceFilter.trim()) ||
    Boolean(untilFilter.trim())

  const clearFilters = () => {
    setKindMode('events_and_config')
    setSeverityFilter('')
    setCategoryFilter('')
    setActorFilter('')
    setSinceFilter('')
    setUntilFilter('')
    setPage(1)
  }

  const handleCreated = (entry: KBEntry) => {
    setLocalRefresh((n) => n + 1)
    onSelectRef(entry.ref)
  }

  const totalPages = data?.total_pages ?? 0

  return (
    <Box>
      <Card variant="outlined" sx={{ mb: 2, borderColor: '#30363d', bgcolor: '#161b22' }}>
        <CardContent sx={{ pb: '16px !important' }}>
          <Stack spacing={1.5}>
            <Stack
              direction={{ xs: 'column', md: 'row' }}
              spacing={1.25}
              sx={{ alignItems: { xs: 'stretch', md: 'center' }, justifyContent: 'space-between' }}
            >
              <ToggleButtonGroup
                size="small"
                exclusive
                value={kindMode}
                onChange={(_e, next) => {
                  if (next) setKindMode(next as TimelineKindMode)
                }}
                aria-label="Timeline activity filter"
                sx={{ flexWrap: 'wrap' }}
              >
                <ToggleButton value="events_and_config" sx={{ px: 1.5, py: 0.5, fontSize: 12.5, textTransform: 'none' }}>
                  Events &amp; Config Changes
                </ToggleButton>
                <ToggleButton value="events" sx={{ px: 1.5, py: 0.5, fontSize: 12.5, textTransform: 'none' }}>
                  Events only
                </ToggleButton>
                <ToggleButton value="config_changes" sx={{ px: 1.5, py: 0.5, fontSize: 12.5, textTransform: 'none' }}>
                  Config Changes only
                </ToggleButton>
                <ToggleButton value="all" sx={{ px: 1.5, py: 0.5, fontSize: 12.5, textTransform: 'none' }}>
                  All KB Activity
                </ToggleButton>
              </ToggleButtonGroup>

              <Button
                size="small"
                variant="contained"
                startIcon={<AddIcon fontSize="small" />}
                onClick={() => setLogDialogOpen(true)}
                sx={{ flexShrink: 0 }}
              >
                Log Event / Config Change
              </Button>
            </Stack>

            <Stack
              direction={{ xs: 'column', sm: 'row' }}
              spacing={1.25}
              useFlexGap
              sx={{ alignItems: { xs: 'stretch', sm: 'center' }, flexWrap: 'wrap' }}
            >
              <TextField
                select
                size="small"
                label="Severity"
                value={severityFilter}
                onChange={(e) => setSeverityFilter(e.target.value)}
                sx={{ minWidth: 135 }}
              >
                <MenuItem value="">All severities</MenuItem>
                {KB_SEVERITIES.map((sev) => (
                  <MenuItem key={sev} value={sev}>
                    {sev}
                  </MenuItem>
                ))}
              </TextField>

              <TextField
                size="small"
                label="Category"
                value={categoryFilter}
                onChange={(e) => setCategoryFilter(e.target.value)}
                placeholder="deploy, infra…"
                sx={{ minWidth: 145 }}
              />

              <TextField
                size="small"
                label="Actor"
                value={actorFilter}
                onChange={(e) => setActorFilter(e.target.value)}
                placeholder="human or agent"
                sx={{ minWidth: 145 }}
              />

              <TextField
                size="small"
                label="Since"
                value={sinceFilter}
                onChange={(e) => setSinceFilter(e.target.value)}
                placeholder="YYYY-MM-DD"
                sx={{ minWidth: 140 }}
              />

              <TextField
                size="small"
                label="Until"
                value={untilFilter}
                onChange={(e) => setUntilFilter(e.target.value)}
                placeholder="YYYY-MM-DD"
                sx={{ minWidth: 140 }}
              />

              {hasFilters && (
                <Button size="small" onClick={clearFilters}>
                  Reset
                </Button>
              )}
            </Stack>
          </Stack>
        </CardContent>
      </Card>

      {error && (
        <Alert
          severity="error"
          sx={{ mb: 2 }}
          action={
            <Button color="inherit" size="small" onClick={() => void fetchTimeline(listParams)}>
              Retry
            </Button>
          }
        >
          {error}
        </Alert>
      )}

      {loading && <LinearProgress sx={{ mb: 1.5, borderRadius: 1 }} />}

      {!loading && groupedByDay.length === 0 && (
        <Card variant="outlined" sx={{ borderColor: '#30363d', bgcolor: '#161b22' }}>
          <CardContent sx={{ py: 5, textAlign: 'center' }}>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
              {hasFilters
                ? 'No timeline entries match the selected filters.'
                : 'No events or configuration changes recorded yet.'}
            </Typography>
            <Stack direction="row" spacing={1} sx={{ justifyContent: 'center' }}>
              {hasFilters && (
                <Button size="small" variant="outlined" onClick={clearFilters}>
                  Reset filters
                </Button>
              )}
              <Button
                size="small"
                variant="contained"
                startIcon={<AddIcon fontSize="small" />}
                onClick={() => setLogDialogOpen(true)}
              >
                Log Event / Config Change
              </Button>
            </Stack>
          </CardContent>
        </Card>
      )}

      <Stack spacing={2.5}>
        {groupedByDay.map((group) => (
          <Box key={group.dayKey}>
            <Stack direction="row" spacing={1} sx={{ mb: 1, alignItems: 'center' }}>
              <Typography variant="overline" color="text.secondary" sx={{ fontWeight: 700 }}>
                {group.label}
              </Typography>
              <Chip size="small" variant="outlined" label={group.items.length} sx={{ height: 18, fontSize: 10.5 }} />
            </Stack>

            <Stack
              spacing={1.25}
              sx={{
                pl: 2,
                borderLeft: '2px solid #30363d',
              }}
            >
              {group.items.map((entry) => (
                <TimelineEntryCard key={entry.ref} entry={entry} onSelectRef={onSelectRef} />
              ))}
            </Stack>
          </Box>
        ))}
      </Stack>

      {totalPages > 1 && !(severityFilter || categoryFilter.trim()) && (
        <Stack direction="row" sx={{ mt: 2.5, justifyContent: 'center' }}>
          <Pagination
            size="small"
            color="primary"
            page={page}
            count={totalPages}
            onChange={(_e, nextPage) => setPage(nextPage)}
          />
        </Stack>
      )}

      <KBEntryFormDialog
        open={logDialogOpen}
        mode="create"
        defaultKind={kindMode === 'config_changes' ? 'config_change' : 'event'}
        defaultScope={projectId ? 'project' : 'global'}
        projectId={projectId}
        onClose={() => setLogDialogOpen(false)}
        onSaved={handleCreated}
      />
    </Box>
  )
}

function TimelineEntryCard({
  entry,
  onSelectRef,
}: {
  entry: KBEntry
  onSelectRef: (ref: string) => void
}) {
  const hasDiff =
    entry.kind === 'config_change' ||
    Boolean(entry.before_value || entry.after_value || entry.reason)
  const timestamp = entry.occurred_at || entry.created_at

  return (
    <Card
      variant="outlined"
      onClick={() => onSelectRef(entry.ref)}
      sx={{
        cursor: 'pointer',
        borderColor: '#30363d',
        bgcolor: '#161b22',
        transition: 'border-color 0.15s ease, background-color 0.15s ease',
        '&:hover': {
          borderColor: '#58a6ff',
          bgcolor: 'rgba(88, 166, 255, 0.04)',
        },
      }}
    >
      <CardContent sx={{ py: 1.5, px: 2, '&:last-child': { pb: 1.5 } }}>
        <Stack spacing={1}>
          <Stack
            direction="row"
            spacing={1}
            useFlexGap
            sx={{ alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap' }}
          >
            <Stack direction="row" spacing={0.75} useFlexGap sx={{ alignItems: 'center', flexWrap: 'wrap' }}>
              <Typography variant="caption" sx={{ fontFamily: MONO, color: 'text.secondary', minWidth: 56 }}>
                {formatShortTime(timestamp) || '—'}
              </Typography>
              <Chip
                size="small"
                variant="outlined"
                label={KB_KIND_LABELS[entry.kind]}
                sx={{ ...kbKindChipSx(entry.kind), height: 20, fontSize: 11 }}
              />
              {entry.severity && (
                <Chip
                  size="small"
                  variant="outlined"
                  label={entry.severity}
                  sx={{ ...kbSeverityChipSx(entry.severity), height: 20, fontSize: 11 }}
                />
              )}
              {entry.category && (
                <Chip size="small" variant="outlined" label={entry.category} sx={{ height: 20, fontSize: 11 }} />
              )}
              <Chip
                size="small"
                variant="outlined"
                label={KB_SCOPE_LABELS[entry.scope]}
                sx={{ height: 20, fontSize: 11 }}
              />
              {entry.verified && <VerifiedIcon sx={{ fontSize: 15, color: '#3fb950' }} />}
            </Stack>

            <Stack direction="row" spacing={1} sx={{ alignItems: 'center' }}>
              {entry.kind === 'config_change' && (
                <Button
                  size="small"
                  variant="outlined"
                  startIcon={<HistoryIcon fontSize="small" />}
                  onClick={(e) => {
                    e.stopPropagation()
                    onSelectRef(entry.ref)
                  }}
                  sx={{ py: 0.15, px: 1, fontSize: 11.5 }}
                >
                  Version chain
                </Button>
              )}
              <Typography variant="caption" sx={{ fontFamily: MONO, color: '#58a6ff' }}>
                {entry.ref}
              </Typography>
            </Stack>
          </Stack>

          <Typography variant="subtitle2" sx={{ fontSize: 14, fontWeight: 600, color: 'text.primary' }}>
            {kbHeadline(entry)}
          </Typography>

          <Stack direction="row" spacing={1.5} useFlexGap sx={{ flexWrap: 'wrap' }}>
            {entry.target && (
              <Typography variant="caption" color="text.secondary">
                Target:{' '}
                <Box component="span" sx={{ fontFamily: MONO, color: 'text.primary' }}>
                  {entry.target}
                </Box>
              </Typography>
            )}
            {entry.actor && (
              <Typography variant="caption" color="text.secondary">
                Actor:{' '}
                <Box component="span" sx={{ color: 'text.primary' }}>
                  {entry.actor}
                </Box>
              </Typography>
            )}
          </Stack>

          {hasDiff && (entry.before_value || entry.after_value || entry.reason) && (
            <Box
              sx={{
                p: 1.25,
                borderRadius: 1,
                border: '1px solid #30363d',
                bgcolor: '#0d1117',
              }}
            >
              {(entry.before_value || entry.after_value) && (
                <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} sx={{ alignItems: { sm: 'center' } }}>
                  {entry.before_value && (
                    <Typography
                      variant="caption"
                      sx={{
                        fontFamily: MONO,
                        color: '#f85149',
                        bgcolor: 'rgba(248, 81, 73, 0.10)',
                        px: 0.75,
                        py: 0.25,
                        borderRadius: 0.75,
                        wordBreak: 'break-word',
                      }}
                    >
                      - {entry.before_value}
                    </Typography>
                  )}
                  {entry.before_value && entry.after_value && (
                    <Typography variant="caption" color="text.secondary">
                      →
                    </Typography>
                  )}
                  {entry.after_value && (
                    <Typography
                      variant="caption"
                      sx={{
                        fontFamily: MONO,
                        color: '#3fb950',
                        bgcolor: 'rgba(63, 185, 80, 0.10)',
                        px: 0.75,
                        py: 0.25,
                        borderRadius: 0.75,
                        wordBreak: 'break-word',
                      }}
                    >
                      + {entry.after_value}
                    </Typography>
                  )}
                </Stack>
              )}
              {entry.reason && (
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: entry.before_value || entry.after_value ? 0.75 : 0 }}>
                  Reason: {entry.reason}
                </Typography>
              )}
            </Box>
          )}

          {entry.body && entry.body.trim() !== '' && (
            <Typography
              variant="body2"
              color="text.secondary"
              sx={{
                fontSize: 12.5,
                display: '-webkit-box',
                WebkitLineClamp: 2,
                WebkitBoxOrient: 'vertical',
                overflow: 'hidden',
              }}
            >
              {entry.body}
            </Typography>
          )}
        </Stack>
      </CardContent>
    </Card>
  )
}
