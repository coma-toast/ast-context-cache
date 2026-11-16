import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Checkbox,
  Chip,
  Collapse,
  FormControlLabel,
  InputAdornment,
  LinearProgress,
  Menu,
  MenuItem,
  Pagination,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  Typography,
} from '@mui/material'
import AddIcon from '@mui/icons-material/Add'
import ClearIcon from '@mui/icons-material/Clear'
import DownloadIcon from '@mui/icons-material/Download'
import FilterListIcon from '@mui/icons-material/FilterList'
import SearchIcon from '@mui/icons-material/Search'
import VerifiedIcon from '@mui/icons-material/Verified'

import type {
  KBCounts,
  KBEntry,
  KBKind,
  KBListParams,
  KBListResponse,
  KBScope,
  KBStatusFilter,
} from '../../api/types'
import { api } from '../../api/client'
import { useToast } from '../../context/ToastContext'
import { KBEntryFormDialog } from './KBEntryFormDialog'
import {
  KB_KINDS,
  KB_KIND_LABELS,
  KB_SCOPES,
  KB_SCOPE_LABELS,
  formatShortDate,
  kbEntryStatus,
  kbHeadline,
  kbKindChipSx,
  kbSeverityChipSx,
  parseTagsInput,
} from './kbHelpers'

const MONO = '"JetBrains Mono", ui-monospace, monospace'
const PER_PAGE = 50

const COUNT_KEY_BY_KIND: Record<KBKind, keyof KBCounts> = {
  fact: 'fact',
  procedure: 'procedure',
  preference: 'preference',
  event: 'event',
  config_change: 'config_change',
  discovery: 'discovery',
}

export function KBExplorerSubtab({
  projectId,
  kbRefreshNonce,
  selectedRef,
  onSelectRef,
}: {
  projectId?: string
  kbRefreshNonce?: number
  selectedRef?: string
  onSelectRef: (ref: string) => void
}) {
  const { showToast } = useToast()

  const [queryInput, setQueryInput] = useState('')
  const [debouncedQuery, setDebouncedQuery] = useState('')
  const [selectedKinds, setSelectedKinds] = useState<KBKind[]>([])
  const [scope, setScope] = useState<KBScope | ''>('')
  const [status, setStatus] = useState<KBStatusFilter>('active')

  const [showAdvanced, setShowAdvanced] = useState(false)
  const [tagsInput, setTagsInput] = useState('')
  const [actorInput, setActorInput] = useState('')
  const [sinceInput, setSinceInput] = useState('')
  const [untilInput, setUntilInput] = useState('')
  const [verifiedOnly, setVerifiedOnly] = useState(false)

  const [page, setPage] = useState(1)
  const [localRefresh, setLocalRefresh] = useState(0)

  const [data, setData] = useState<KBListResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const [createOpen, setCreateOpen] = useState(false)
  const [exportAnchor, setExportAnchor] = useState<HTMLElement | null>(null)
  const [exporting, setExporting] = useState(false)

  useEffect(() => {
    const id = window.setTimeout(() => {
      setDebouncedQuery(queryInput.trim())
    }, 250)
    return () => window.clearTimeout(id)
  }, [queryInput])

  const parsedTags = useMemo(() => parseTagsInput(tagsInput), [tagsInput])

  // Reset to page 1 when filters or project change
  useEffect(() => {
    setPage(1)
  }, [projectId, debouncedQuery, selectedKinds, scope, status, parsedTags, actorInput, sinceInput, untilInput, verifiedOnly])

  const listParams = useMemo<KBListParams>(() => {
    return {
      q: debouncedQuery || undefined,
      kinds: selectedKinds.length > 0 ? selectedKinds : undefined,
      scope: scope || undefined,
      project_id: projectId || undefined,
      status,
      tags: parsedTags.length > 0 ? parsedTags : undefined,
      actor: actorInput.trim() || undefined,
      since: sinceInput.trim() || undefined,
      until: untilInput.trim() || undefined,
      verified_only: verifiedOnly || undefined,
      sort: 'created_desc',
      page,
      per_page: PER_PAGE,
    }
  }, [
    debouncedQuery,
    selectedKinds,
    scope,
    projectId,
    status,
    parsedTags,
    actorInput,
    sinceInput,
    untilInput,
    verifiedOnly,
    page,
  ])

  const fetchList = useCallback(async (params: KBListParams, signal?: { cancelled: boolean }) => {
    setLoading(true)
    setError('')
    try {
      const res = await api.kbList(params)
      if (!signal?.cancelled) {
        setData(res)
      }
    } catch (err) {
      if (!signal?.cancelled) {
        setError(err instanceof Error ? err.message : 'Failed to load knowledge base entries')
      }
    } finally {
      if (!signal?.cancelled) {
        setLoading(false)
      }
    }
  }, [])

  useEffect(() => {
    const signal = { cancelled: false }
    void fetchList(listParams, signal)
    return () => {
      signal.cancelled = true
    }
  }, [fetchList, listParams, kbRefreshNonce, localRefresh])

  const toggleKind = (kind: KBKind) => {
    setSelectedKinds((prev) => (prev.includes(kind) ? prev.filter((k) => k !== kind) : [...prev, kind]))
  }

  const handleExport = async (format: 'markdown' | 'json') => {
    setExportAnchor(null)
    setExporting(true)
    try {
      const exportParams: KBListParams = { ...listParams, page: undefined, per_page: undefined }
      await api.kbExportDownload(exportParams, format)
      showToast(`Exported KB as ${format === 'markdown' ? 'Markdown' : 'JSON'}`, 'success')
    } catch (err) {
      showToast(err instanceof Error ? err.message : 'Export failed', 'error')
    } finally {
      setExporting(false)
    }
  }

  const handleCreated = (entry: KBEntry) => {
    setLocalRefresh((n) => n + 1)
    onSelectRef(entry.ref)
  }

  const counts = data?.counts
  const entries = data?.entries ?? []
  const total = data?.total ?? 0
  const totalPages = data?.total_pages ?? 0

  const hasActiveFilters =
    Boolean(queryInput.trim()) ||
    selectedKinds.length > 0 ||
    scope !== '' ||
    status !== 'active' ||
    parsedTags.length > 0 ||
    Boolean(actorInput.trim()) ||
    Boolean(sinceInput.trim()) ||
    Boolean(untilInput.trim()) ||
    verifiedOnly

  const clearAllFilters = () => {
    setQueryInput('')
    setDebouncedQuery('')
    setSelectedKinds([])
    setScope('')
    setStatus('active')
    setTagsInput('')
    setActorInput('')
    setSinceInput('')
    setUntilInput('')
    setVerifiedOnly(false)
    setPage(1)
  }

  return (
    <Box>
      <Card variant="outlined" sx={{ mb: 2, borderColor: '#30363d', bgcolor: '#161b22' }}>
        <CardContent sx={{ pb: '16px !important' }}>
          <Stack spacing={1.5}>
            <Stack
              direction={{ xs: 'column', md: 'row' }}
              spacing={1.25}
              sx={{ alignItems: { xs: 'stretch', md: 'center' } }}
            >
              <TextField
                size="small"
                fullWidth
                placeholder="Search knowledge base (subject, title, rule, body, target, ref)…"
                value={queryInput}
                onChange={(e) => setQueryInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    setDebouncedQuery(queryInput.trim())
                  }
                }}
                slotProps={{
                  input: {
                    startAdornment: (
                      <InputAdornment position="start">
                        <SearchIcon fontSize="small" color="action" />
                      </InputAdornment>
                    ),
                    endAdornment: queryInput ? (
                      <InputAdornment position="end">
                        <Button
                          size="small"
                          aria-label="Clear search"
                          onClick={() => {
                            setQueryInput('')
                            setDebouncedQuery('')
                          }}
                          sx={{ minWidth: 0, p: 0.25 }}
                        >
                          <ClearIcon fontSize="small" />
                        </Button>
                      </InputAdornment>
                    ) : undefined,
                  },
                }}
              />

              <Stack direction="row" spacing={1} sx={{ flexShrink: 0, alignItems: 'center' }}>
                <Button
                  size="small"
                  variant={showAdvanced ? 'contained' : 'outlined'}
                  color={showAdvanced ? 'primary' : 'inherit'}
                  startIcon={<FilterListIcon fontSize="small" />}
                  onClick={() => setShowAdvanced((v) => !v)}
                >
                  Filters
                </Button>

                <Button
                  size="small"
                  variant="outlined"
                  startIcon={<DownloadIcon fontSize="small" />}
                  disabled={exporting}
                  onClick={(e) => setExportAnchor(e.currentTarget)}
                >
                  {exporting ? 'Exporting…' : 'Export'}
                </Button>
                <Menu
                  anchorEl={exportAnchor}
                  open={Boolean(exportAnchor)}
                  onClose={() => setExportAnchor(null)}
                >
                  <MenuItem onClick={() => void handleExport('markdown')}>Export Markdown (.md)</MenuItem>
                  <MenuItem onClick={() => void handleExport('json')}>Export JSON (.json)</MenuItem>
                </Menu>

                <Button
                  size="small"
                  variant="contained"
                  startIcon={<AddIcon fontSize="small" />}
                  onClick={() => setCreateOpen(true)}
                >
                  New Entry
                </Button>
              </Stack>
            </Stack>

            <Stack
              direction="row"
              spacing={1.5}
              useFlexGap
              sx={{ alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap' }}
            >
              <Stack direction="row" spacing={0.75} useFlexGap sx={{ alignItems: 'center', flexWrap: 'wrap' }}>
                <Chip
                  size="small"
                  label={counts ? `All (${counts.total})` : 'All'}
                  clickable
                  variant={selectedKinds.length === 0 ? 'filled' : 'outlined'}
                  color={selectedKinds.length === 0 ? 'primary' : 'default'}
                  onClick={() => setSelectedKinds([])}
                />
                {KB_KINDS.map((k) => {
                  const active = selectedKinds.includes(k)
                  const countVal = counts ? counts[COUNT_KEY_BY_KIND[k]] : undefined
                  const label = countVal != null ? `${KB_KIND_LABELS[k]} (${countVal})` : KB_KIND_LABELS[k]
                  return (
                    <Chip
                      key={k}
                      size="small"
                      clickable
                      label={label}
                      variant={active ? 'filled' : 'outlined'}
                      onClick={() => toggleKind(k)}
                      sx={
                        active
                          ? { ...kbKindChipSx(k), borderWidth: 1, borderStyle: 'solid' }
                          : { borderColor: '#30363d', color: 'text.secondary' }
                      }
                    />
                  )
                })}
                {counts && counts.quarantined > 0 && (
                  <Chip
                    size="small"
                    clickable
                    color="warning"
                    variant={status === 'quarantined' ? 'filled' : 'outlined'}
                    label={`Quarantined (${counts.quarantined})`}
                    onClick={() => setStatus((prev) => (prev === 'quarantined' ? 'active' : 'quarantined'))}
                  />
                )}
              </Stack>

              <Stack direction="row" spacing={1} useFlexGap sx={{ alignItems: 'center', flexWrap: 'wrap' }}>
                <ToggleButtonGroup
                  size="small"
                  exclusive
                  value={scope}
                  onChange={(_e, val) => {
                    if (val !== null) setScope(val as KBScope | '')
                  }}
                  aria-label="Scope filter"
                >
                  <ToggleButton value="" sx={{ px: 1.25, py: 0.25, fontSize: 12, textTransform: 'none' }}>
                    All Scopes
                  </ToggleButton>
                  {KB_SCOPES.map((s) => (
                    <ToggleButton
                      key={s}
                      value={s}
                      sx={{ px: 1.25, py: 0.25, fontSize: 12, textTransform: 'none' }}
                    >
                      {KB_SCOPE_LABELS[s]}
                    </ToggleButton>
                  ))}
                </ToggleButtonGroup>

                <TextField
                  select
                  size="small"
                  value={status}
                  onChange={(e) => setStatus(e.target.value as KBStatusFilter)}
                  aria-label="Status filter"
                  sx={{ minWidth: 185 }}
                >
                  <MenuItem value="active">Active</MenuItem>
                  <MenuItem value="archived">Superseded &amp; Forgotten</MenuItem>
                  <MenuItem value="quarantined">
                    Quarantined{counts ? ` (${counts.quarantined})` : ''}
                  </MenuItem>
                </TextField>
              </Stack>
            </Stack>

            <Collapse in={showAdvanced}>
              <Stack
                direction={{ xs: 'column', sm: 'row' }}
                spacing={1.25}
                useFlexGap
                sx={{
                  pt: 1,
                  borderTop: '1px solid #30363d',
                  alignItems: { xs: 'stretch', sm: 'center' },
                  flexWrap: 'wrap',
                }}
              >
                <TextField
                  size="small"
                  label="Tags (comma-separated)"
                  value={tagsInput}
                  onChange={(e) => setTagsInput(e.target.value)}
                  placeholder="auth, perf"
                  sx={{ minWidth: 180 }}
                />
                <TextField
                  size="small"
                  label="Actor"
                  value={actorInput}
                  onChange={(e) => setActorInput(e.target.value)}
                  placeholder="human or agent"
                  sx={{ minWidth: 140 }}
                />
                <TextField
                  size="small"
                  label="Since"
                  value={sinceInput}
                  onChange={(e) => setSinceInput(e.target.value)}
                  placeholder="YYYY-MM-DD"
                  sx={{ minWidth: 145 }}
                />
                <TextField
                  size="small"
                  label="Until"
                  value={untilInput}
                  onChange={(e) => setUntilInput(e.target.value)}
                  placeholder="YYYY-MM-DD"
                  sx={{ minWidth: 145 }}
                />
                <FormControlLabel
                  control={
                    <Checkbox
                      size="small"
                      checked={verifiedOnly}
                      onChange={(e) => setVerifiedOnly(e.target.checked)}
                    />
                  }
                  label={<Typography variant="body2">Verified only</Typography>}
                />
                {hasActiveFilters && (
                  <Button size="small" onClick={clearAllFilters}>
                    Reset filters
                  </Button>
                )}
              </Stack>
            </Collapse>
          </Stack>
        </CardContent>
      </Card>

      {error && (
        <Alert
          severity="error"
          sx={{ mb: 2 }}
          action={
            <Button color="inherit" size="small" onClick={() => void fetchList(listParams)}>
              Retry
            </Button>
          }
        >
          {error}
        </Alert>
      )}

      <Card variant="outlined" sx={{ borderColor: '#30363d', bgcolor: '#161b22', overflow: 'hidden' }}>
        {loading && <LinearProgress />}
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell sx={{ width: 120 }}>Kind</TableCell>
                <TableCell>Entry</TableCell>
                <TableCell sx={{ width: 95 }}>Scope</TableCell>
                <TableCell sx={{ width: 130 }}>Status</TableCell>
                <TableCell sx={{ width: 155 }}>When</TableCell>
                <TableCell sx={{ width: 135 }}>Ref</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {!loading && entries.length === 0 && (
                <TableRow>
                  <TableCell colSpan={6} sx={{ py: 5, textAlign: 'center' }}>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      {hasActiveFilters
                        ? 'No knowledge base entries match the current filters.'
                        : 'No knowledge base entries yet.'}
                    </Typography>
                    {hasActiveFilters ? (
                      <Button size="small" variant="outlined" onClick={clearAllFilters}>
                        Clear filters
                      </Button>
                    ) : (
                      <Button
                        size="small"
                        variant="contained"
                        startIcon={<AddIcon fontSize="small" />}
                        onClick={() => setCreateOpen(true)}
                      >
                        Create first entry
                      </Button>
                    )}
                  </TableCell>
                </TableRow>
              )}

              {entries.map((entry) => {
                const isSelected = Boolean(selectedRef && entry.ref === selectedRef)
                const computedStatus = kbEntryStatus(entry)
                const subline = buildEntrySubline(entry)
                return (
                  <TableRow
                    key={entry.ref}
                    hover
                    selected={isSelected}
                    onClick={() => onSelectRef(entry.ref)}
                    sx={{
                      cursor: 'pointer',
                      bgcolor: isSelected ? 'rgba(88, 166, 255, 0.10)' : undefined,
                      '& td': { borderColor: '#30363d' },
                    }}
                  >
                    <TableCell>
                      <Chip
                        size="small"
                        variant="outlined"
                        label={KB_KIND_LABELS[entry.kind]}
                        sx={{ ...kbKindChipSx(entry.kind), height: 22, fontSize: 11 }}
                      />
                    </TableCell>
                    <TableCell sx={{ maxWidth: 460 }}>
                      <Stack spacing={0.25}>
                        <Stack direction="row" spacing={0.75} useFlexGap sx={{ alignItems: 'center', flexWrap: 'wrap' }}>
                          <Typography variant="body2" sx={{ fontWeight: 500, color: 'text.primary' }}>
                            {kbHeadline(entry)}
                          </Typography>
                          {entry.severity && (
                            <Chip
                              size="small"
                              variant="outlined"
                              label={entry.severity}
                              sx={{ ...kbSeverityChipSx(entry.severity), height: 18, fontSize: 10.5 }}
                            />
                          )}
                          {entry.category && (
                            <Chip
                              size="small"
                              variant="outlined"
                              label={entry.category}
                              sx={{ height: 18, fontSize: 10.5 }}
                            />
                          )}
                        </Stack>
                        {subline && (
                          <Typography variant="caption" color="text.secondary" noWrap>
                            {subline}
                          </Typography>
                        )}
                        {entry.tags && entry.tags.length > 0 && (
                          <Stack direction="row" spacing={0.5} useFlexGap sx={{ pt: 0.25, flexWrap: 'wrap' }}>
                            {entry.tags.slice(0, 4).map((t) => (
                              <Chip
                                key={t}
                                size="small"
                                variant="outlined"
                                label={t}
                                sx={{ height: 18, fontSize: 10.5, borderColor: '#30363d' }}
                              />
                            ))}
                            {entry.tags.length > 4 && (
                              <Typography variant="caption" color="text.secondary">
                                +{entry.tags.length - 4}
                              </Typography>
                            )}
                          </Stack>
                        )}
                      </Stack>
                    </TableCell>
                    <TableCell>
                      <Chip
                        size="small"
                        variant="outlined"
                        label={KB_SCOPE_LABELS[entry.scope]}
                        sx={{ height: 20, fontSize: 11 }}
                      />
                    </TableCell>
                    <TableCell>
                      <Stack direction="row" spacing={0.5} sx={{ alignItems: 'center' }}>
                        <Chip
                          size="small"
                          variant="outlined"
                          color={
                            computedStatus === 'active'
                              ? 'success'
                              : computedStatus === 'quarantined'
                                ? 'warning'
                                : 'default'
                          }
                          label={
                            computedStatus === 'active'
                              ? 'Active'
                              : computedStatus === 'superseded'
                                ? 'Superseded'
                                : computedStatus === 'quarantined'
                                  ? 'Quarantined'
                                  : 'Forgotten'
                          }
                          sx={{ height: 20, fontSize: 11 }}
                        />
                        {entry.verified && (
                          <Tooltip title="Verified">
                            <VerifiedIcon sx={{ fontSize: 16, color: '#3fb950' }} />
                          </Tooltip>
                        )}
                      </Stack>
                    </TableCell>
                    <TableCell>
                      <Typography variant="caption" color="text.secondary" noWrap>
                        {formatShortDate(entry.occurred_at || entry.created_at) || '—'}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="caption" sx={{ fontFamily: MONO, color: '#58a6ff' }}>
                        {entry.ref}
                      </Typography>
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </TableContainer>

        {total > 0 && (
          <Stack
            direction={{ xs: 'column', sm: 'row' }}
            spacing={1}
            sx={{ px: 2, py: 1.25, borderTop: '1px solid #30363d', justifyContent: 'space-between', alignItems: 'center' }}
          >
            <Typography variant="caption" color="text.secondary">
              Showing {(page - 1) * PER_PAGE + 1}–{Math.min(page * PER_PAGE, total)} of {total.toLocaleString()} entries
            </Typography>
            {totalPages > 1 && (
              <Pagination
                size="small"
                color="primary"
                page={page}
                count={totalPages}
                onChange={(_e, nextPage) => setPage(nextPage)}
              />
            )}
          </Stack>
        )}
      </Card>

      <KBEntryFormDialog
        open={createOpen}
        mode="create"
        projectId={projectId}
        defaultScope={scope || (projectId ? 'project' : 'global')}
        defaultKind={selectedKinds.length === 1 ? selectedKinds[0] : 'fact'}
        onClose={() => setCreateOpen(false)}
        onSaved={handleCreated}
      />
    </Box>
  )
}

function buildEntrySubline(entry: KBEntry): string {
  const parts: string[] = []
  if (entry.actor) parts.push(`actor: ${entry.actor}`)
  if (entry.target && entry.kind !== 'config_change') parts.push(`target: ${entry.target}`)
  if (entry.kind === 'config_change' && entry.reason) parts.push(entry.reason)
  if (entry.body) {
    const first = entry.body.trim().split('\n')[0]?.trim()
    if (first && first !== kbHeadline(entry)) {
      parts.push(first)
    }
  }
  return parts.join(' · ')
}
