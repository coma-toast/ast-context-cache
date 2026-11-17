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
  Divider,
  Drawer,
  Grid,
  IconButton,
  List,
  ListItemButton,
  ListItemText,
  Stack,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TablePagination,
  TableRow,
  Tabs,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from '@mui/material'
import CloseIcon from '@mui/icons-material/Close'
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutlined'
import DeleteSweepIcon from '@mui/icons-material/DeleteSweep'
import RefreshIcon from '@mui/icons-material/Refresh'
import { api, formatNum } from '../../api/client'
import type {
  ContextNoteDetailResponse,
  ContextNoteListItem,
  ContextNoteRevision,
  MemoryData,
} from '../../api/types'
import { useToast } from '../../context/ToastContext'
import { StatCard } from '../HealthBar'
import { MarkdownView } from '../MarkdownView'

const MONO = '"JetBrains Mono", ui-monospace, monospace'
const PER_PAGE = 50

export interface VirtualContextSubtabProps {
  data: MemoryData | null
  onRefresh: () => void
  projectId?: string
  kbRefreshNonce?: number
}

interface DiffLine {
  type: 'context' | 'add' | 'remove'
  text: string
  oldLine?: number
  newLine?: number
}

function computeLineDiff(beforeText: string, afterText: string): DiffLine[] {
  const a = beforeText === '' ? [] : beforeText.split('\n')
  const b = afterText === '' ? [] : afterText.split('\n')
  const m = a.length
  const n = b.length

  // Guard against very large notes blowing memory in the browser DP table.
  if (m * n > 200_000) {
    const out: DiffLine[] = []
    a.forEach((line, i) => out.push({ type: 'remove', text: line, oldLine: i + 1 }))
    b.forEach((line, i) => out.push({ type: 'add', text: line, newLine: i + 1 }))
    return out
  }

  const dp: Uint32Array[] = Array.from({ length: m + 1 }, () => new Uint32Array(n + 1))
  for (let i = m - 1; i >= 0; i--) {
    for (let j = n - 1; j >= 0; j--) {
      if (a[i] === b[j]) {
        dp[i][j] = dp[i + 1][j + 1] + 1
      } else {
        dp[i][j] = dp[i + 1][j] >= dp[i][j + 1] ? dp[i + 1][j] : dp[i][j + 1]
      }
    }
  }

  const lines: DiffLine[] = []
  let i = 0
  let j = 0
  while (i < m && j < n) {
    if (a[i] === b[j]) {
      lines.push({ type: 'context', text: a[i], oldLine: i + 1, newLine: j + 1 })
      i++
      j++
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      lines.push({ type: 'remove', text: a[i], oldLine: i + 1 })
      i++
    } else {
      lines.push({ type: 'add', text: b[j], newLine: j + 1 })
      j++
    }
  }
  while (i < m) {
    lines.push({ type: 'remove', text: a[i], oldLine: i + 1 })
    i++
  }
  while (j < n) {
    lines.push({ type: 'add', text: b[j], newLine: j + 1 })
    j++
  }
  return lines
}

function parseTagsCsv(raw: string): string[] {
  return raw
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
}

export function VirtualContextSubtab({
  data,
  onRefresh,
  projectId,
  kbRefreshNonce = 0,
}: VirtualContextSubtabProps) {
  const { showToast } = useToast()

  // Filters & pagination
  const [q, setQ] = useState('')
  const [sessionId, setSessionId] = useState('')
  const [kind, setKind] = useState('')
  const [tagsText, setTagsText] = useState('')
  const [page, setPage] = useState(1)
  const [listReloadNonce, setListReloadNonce] = useState(0)

  // List state
  const [items, setItems] = useState<ContextNoteListItem[]>([])
  const [total, setTotal] = useState(0)
  const [loadingList, setLoadingList] = useState(false)
  const [listError, setListError] = useState<string | null>(null)

  // Flush orphans dialog
  const [flushDialogOpen, setFlushDialogOpen] = useState(false)
  const [flushingOrphans, setFlushingOrphans] = useState(false)

  // Detail drawer state
  const [selectedItem, setSelectedItem] = useState<ContextNoteListItem | null>(null)
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [detail, setDetail] = useState<ContextNoteDetailResponse | null>(null)
  const [loadingDetail, setLoadingDetail] = useState(false)
  const [detailError, setDetailError] = useState<string | null>(null)
  const [drawerTab, setDrawerTab] = useState<'content' | 'revisions'>('content')

  // Revision comparison state
  const [selectedRevisionNum, setSelectedRevisionNum] = useState<number | null>(null)
  const [compareAgainst, setCompareAgainst] = useState<'current' | 'previous'>('current')
  const [diffMode, setDiffMode] = useState<'line' | 'side-by-side'>('line')

  // Delete note dialog
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const [deletingNote, setDeletingNote] = useState(false)

  const parsedTags = useMemo(() => parseTagsCsv(tagsText), [tagsText])

  useEffect(() => {
    setPage(1)
  }, [projectId])

  useEffect(() => {
    let cancelled = false
    setLoadingList(true)
    setListError(null)
    api
      .contextNotesList({
        q: q.trim() || undefined,
        session_id: sessionId.trim() || undefined,
        kind: kind.trim() || undefined,
        tags: parsedTags.length > 0 ? parsedTags : undefined,
        project_id: projectId || undefined,
        page,
        per_page: PER_PAGE,
      })
      .then((res) => {
        if (cancelled) return
        setItems(Array.isArray(res.items) ? res.items : [])
        setTotal(res.total ?? 0)
      })
      .catch((err) => {
        if (cancelled) return
        setListError(err instanceof Error ? err.message : String(err))
      })
      .finally(() => {
        if (!cancelled) setLoadingList(false)
      })
    return () => {
      cancelled = true
    }
  }, [q, sessionId, kind, parsedTags, projectId, page, kbRefreshNonce, listReloadNonce])

  const openNoteDetail = useCallback(async (item: ContextNoteListItem) => {
    setSelectedItem(item)
    setDrawerOpen(true)
    setDrawerTab('content')
    setDetail(null)
    setDetailError(null)
    setLoadingDetail(true)
    setSelectedRevisionNum(null)
    setCompareAgainst('current')
    try {
      const res = await api.contextNoteDetail(item.ref)
      setDetail(res)
      if (res.revisions && res.revisions.length > 0) {
        setSelectedRevisionNum(res.revisions[0].revision)
      }
    } catch (err) {
      setDetailError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoadingDetail(false)
    }
  }, [])

  const closeDrawer = useCallback(() => {
    setDrawerOpen(false)
    setDeleteDialogOpen(false)
  }, [])

  const handleFlushOrphans = async () => {
    setFlushingOrphans(true)
    try {
      const r = await api.flushContextOrphans(projectId)
      const freed = formatNum(r.virtual_tokens_freed ?? 0)
      const count = r.flushed_refs ?? 0
      const kept = (r.kept_recent ?? 0) > 0 ? ` · kept ${r.kept_recent} recent` : ''
      showToast(`Flushed ${count} orphan${count === 1 ? '' : 's'} (${freed} tokens)${kept}`, 'success')
      setFlushDialogOpen(false)
      setListReloadNonce((n) => n + 1)
      onRefresh()
    } catch (err) {
      showToast(err instanceof Error ? err.message : String(err), 'error')
    } finally {
      setFlushingOrphans(false)
    }
  }

  const handleDeleteNote = async () => {
    const refToDelete = detail?.note?.ref || selectedItem?.ref
    if (!refToDelete) return
    setDeletingNote(true)
    try {
      const res = await api.contextNoteDelete(refToDelete)
      showToast(
        `Deleted ${refToDelete} (${formatNum(res.virtual_tokens_freed ?? 0)} tokens freed)`,
        'success',
      )
      setDeleteDialogOpen(false)
      setDrawerOpen(false)
      setDetail(null)
      setSelectedItem(null)
      setListReloadNonce((n) => n + 1)
      onRefresh()
    } catch (err) {
      showToast(err instanceof Error ? err.message : String(err), 'error')
    } finally {
      setDeletingNote(false)
    }
  }

  const revisions = detail?.revisions ?? []
  const selectedRevIndex = revisions.findIndex((r) => r.revision === selectedRevisionNum)
  const selectedRev: ContextNoteRevision | null =
    selectedRevIndex >= 0 ? revisions[selectedRevIndex] : revisions[0] ?? null
  const previousRev: ContextNoteRevision | null =
    selectedRevIndex >= 0 && selectedRevIndex + 1 < revisions.length
      ? revisions[selectedRevIndex + 1]
      : null

  const effectiveCompareTarget =
    compareAgainst === 'previous' && previousRev ? 'previous' : 'current'

  const diffComparison = useMemo(() => {
    if (!selectedRev || !detail?.note) return null
    if (effectiveCompareTarget === 'previous' && previousRev) {
      return {
        beforeLabel: `Revision r${previousRev.revision} (${previousRev.op || 'edit'})`,
        afterLabel: `Revision r${selectedRev.revision} (${selectedRev.op || 'edit'})`,
        beforeText: previousRev.content ?? '',
        afterText: selectedRev.content ?? '',
      }
    }
    return {
      beforeLabel: `Revision r${selectedRev.revision} (${selectedRev.op || 'edit'})`,
      afterLabel: `Current body (r${detail.note.revision})`,
      beforeText: selectedRev.content ?? '',
      afterText: detail.note.content ?? '',
    }
  }, [selectedRev, previousRev, effectiveCompareTarget, detail?.note])

  const diffLines = useMemo(() => {
    if (!diffComparison) return []
    return computeLineDiff(diffComparison.beforeText, diffComparison.afterText)
  }, [diffComparison])

  const isActiveHandoffTree = Boolean(
    detail?.active_handoff_tree || selectedItem?.handoff_tree_active,
  )
  const activeNote = detail?.note
  const activeRef = activeNote?.ref || selectedItem?.ref || ''
  const activeKind = activeNote?.kind || selectedItem?.kind || ''
  const activeSession = activeNote?.session_id || selectedItem?.session_id || ''

  const hasFilters = Boolean(q.trim() || sessionId.trim() || kind.trim() || tagsText.trim())

  return (
    <Box data-testid="virtual-context-subtab">
      <Stack
        direction="row"
        spacing={1}
        sx={{ alignItems: 'center', justifyContent: 'space-between', mb: 1.5, flexWrap: 'wrap' }}
        useFlexGap
      >
        <Typography variant="overline" color="text.secondary">
          Virtual context
        </Typography>
        <Button
          size="small"
          variant="outlined"
          color="warning"
          startIcon={<DeleteSweepIcon />}
          disabled={flushingOrphans}
          onClick={() => setFlushDialogOpen(true)}
        >
          Flush orphans
        </Button>
      </Stack>

      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid size={{ xs: 12, sm: 4 }}>
          <StatCard
            title="Inventory tokens"
            value={formatNum(data?.VirtualInventoryTokens ?? 0)}
            sub={data ? `${data.VirtualNotesCount} notes` : undefined}
          />
        </Grid>
        <Grid size={{ xs: 12, sm: 4 }}>
          <StatCard
            title="30d utilization"
            value={`${data?.VirtualUtilPct30d?.toFixed(0) || 0}%`}
          />
        </Grid>
        <Grid size={{ xs: 12, sm: 4 }}>
          <StatCard
            title="Orphans"
            value={formatNum(data?.VirtualOrphanCount ?? 0)}
          />
        </Grid>
      </Grid>

      <Card variant="outlined">
        <CardContent>
          <Stack
            direction="row"
            spacing={1}
            useFlexGap
            sx={{ flexWrap: 'wrap', alignItems: 'center', mb: 2 }}
          >
            <TextField
              size="small"
              label="Search"
              placeholder="ref, label, content, session..."
              value={q}
              onChange={(e) => {
                setQ(e.target.value)
                setPage(1)
              }}
              sx={{ minWidth: 220, flex: 1 }}
            />
            <TextField
              size="small"
              label="Session ID"
              value={sessionId}
              onChange={(e) => {
                setSessionId(e.target.value)
                setPage(1)
              }}
              sx={{ minWidth: 160 }}
            />
            <TextField
              size="small"
              label="Kind"
              placeholder="handoff_result, kv_repair..."
              value={kind}
              onChange={(e) => {
                setKind(e.target.value)
                setPage(1)
              }}
              sx={{ minWidth: 150 }}
            />
            <TextField
              size="small"
              label="Tags"
              placeholder="comma-separated"
              value={tagsText}
              onChange={(e) => {
                setTagsText(e.target.value)
                setPage(1)
              }}
              sx={{ minWidth: 160 }}
            />
            {hasFilters && (
              <Button
                size="small"
                onClick={() => {
                  setQ('')
                  setSessionId('')
                  setKind('')
                  setTagsText('')
                  setPage(1)
                }}
              >
                Clear
              </Button>
            )}
            <IconButton
              size="small"
              aria-label="Reload virtual context notes"
              onClick={() => setListReloadNonce((n) => n + 1)}
            >
              <RefreshIcon fontSize="small" />
            </IconButton>
          </Stack>

          {listError && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {listError}
            </Alert>
          )}

          {loadingList && items.length === 0 ? (
            <Stack direction="row" spacing={1} sx={{ py: 3, alignItems: 'center' }}>
              <CircularProgress size={18} />
              <Typography variant="body2" color="text.secondary">
                Loading context notes…
              </Typography>
            </Stack>
          ) : items.length === 0 ? (
            <Typography color="text.secondary" sx={{ py: 2 }}>
              No virtual context notes match the current filters.
            </Typography>
          ) : (
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Ref / Summary</TableCell>
                  <TableCell>Kind</TableCell>
                  <TableCell>Session</TableCell>
                  <TableCell align="right">Tokens</TableCell>
                  <TableCell align="right">Rev</TableCell>
                  <TableCell align="right">Reads</TableCell>
                  <TableCell>Last accessed</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {items.map((item) => (
                  <TableRow
                    key={item.ref}
                    hover
                    onClick={() => void openNoteDetail(item)}
                    sx={{ cursor: 'pointer' }}
                    data-testid={`context-note-row-${item.ref}`}
                  >
                    <TableCell sx={{ maxWidth: 420 }}>
                      <Stack direction="row" spacing={1} sx={{ mb: 0.25, alignItems: 'center' }}>
                        <Typography sx={{ fontFamily: MONO, fontSize: 12, fontWeight: 600 }}>
                          {item.ref}
                        </Typography>
                        {item.label && (
                          <Typography variant="body2" sx={{ fontWeight: 500 }} noWrap>
                            {item.label}
                          </Typography>
                        )}
                      </Stack>
                      <Typography variant="caption" color="text.secondary" noWrap sx={{ display: 'block' }}>
                        {item.snippet}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Stack direction="row" spacing={0.5} sx={{ alignItems: 'center' }}>
                        <Chip
                          size="small"
                          variant="outlined"
                          label={item.kind || 'context_note'}
                          color={item.kind === 'handoff_result' ? 'secondary' : 'default'}
                        />
                        {item.handoff_tree_active && (
                          <Chip size="small" color="warning" variant="outlined" label="active tree" />
                        )}
                      </Stack>
                    </TableCell>
                    <TableCell>
                      <Typography sx={{ fontFamily: MONO, fontSize: 12 }} noWrap>
                        {item.session_id}
                      </Typography>
                    </TableCell>
                    <TableCell align="right">{formatNum(item.virtual_tokens)}</TableCell>
                    <TableCell align="right">r{item.revision}</TableCell>
                    <TableCell align="right">{item.access_count}</TableCell>
                    <TableCell>
                      <Typography variant="caption" color="text.secondary">
                        {item.last_accessed_at || item.created_at || '—'}
                      </Typography>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}

          {total > 0 && (
            <TablePagination
              component="div"
              count={total}
              page={Math.max(0, page - 1)}
              onPageChange={(_e, newPage) => setPage(newPage + 1)}
              rowsPerPage={PER_PAGE}
              rowsPerPageOptions={[PER_PAGE]}
            />
          )}
        </CardContent>
      </Card>

      {/* Slide-over detail drawer */}
      <Drawer
        anchor="right"
        open={drawerOpen}
        onClose={closeDrawer}
        slotProps={{
          paper: {
            sx: {
              width: { xs: '100%', sm: 620, md: 760 },
              maxWidth: '100vw',
              p: 2.5,
              backgroundImage: 'none',
              bgcolor: 'background.default',
            },
          },
        }}
      >
        <Stack direction="row" spacing={1} sx={{ mb: 1.5, alignItems: 'flex-start', justifyContent: 'space-between' }}>
          <Box sx={{ minWidth: 0, flex: 1 }}>
            <Stack direction="row" spacing={1} useFlexGap sx={{ alignItems: 'center', flexWrap: 'wrap' }}>
              <Typography variant="h6" sx={{ fontWeight: 600 }} noWrap>
                {activeNote?.label || selectedItem?.label || activeRef}
              </Typography>
              <Chip size="small" label={activeRef} sx={{ fontFamily: MONO }} />
              {isActiveHandoffTree && (
                <Chip size="small" color="warning" label="Active handoff tree" />
              )}
            </Stack>
          </Box>
          <Stack direction="row" spacing={1} sx={{ alignItems: 'center' }}>
            <Button
              size="small"
              color="error"
              variant="outlined"
              startIcon={<DeleteOutlineIcon />}
              disabled={!activeRef || deletingNote}
              onClick={() => setDeleteDialogOpen(true)}
            >
              Delete note
            </Button>
            <IconButton size="small" aria-label="Close note drawer" onClick={closeDrawer}>
              <CloseIcon fontSize="small" />
            </IconButton>
          </Stack>
        </Stack>

        {loadingDetail && (
          <Stack direction="row" spacing={1} sx={{ py: 3, alignItems: 'center' }}>
            <CircularProgress size={18} />
            <Typography variant="body2" color="text.secondary">
              Loading note details…
            </Typography>
          </Stack>
        )}

        {detailError && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {detailError}
          </Alert>
        )}

        {detail && (
          <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: 0, flex: 1 }}>
            <Box
              sx={{
                display: 'grid',
                gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, minmax(0, 1fr))' },
                gap: 1,
                p: 1.5,
                mb: 2,
                bgcolor: 'background.paper',
                border: '1px solid',
                borderColor: 'divider',
                borderRadius: 1,
              }}
            >
              <MetaField label="ref" value={detail.note.ref} mono />
              <MetaField label="kind" value={detail.note.kind || 'context_note'} />
              <MetaField label="session_id" value={detail.note.session_id} mono />
              <MetaField label="project_path" value={detail.note.project_path || '—'} mono />
              <MetaField
                label="virtual_tokens"
                value={formatNum(detail.note.virtual_tokens ?? 0)}
              />
              <MetaField label="access_count" value={String(detail.note.access_count ?? 0)} />
              <MetaField label="revision" value={`r${detail.note.revision}`} />
              <MetaField label="tags" value={detail.note.tags || '—'} />
              <MetaField label="created_at" value={detail.note.created_at || '—'} />
              <MetaField label="last_accessed_at" value={detail.note.last_accessed_at || '—'} />
            </Box>

            <Tabs
              value={drawerTab}
              onChange={(_e, v) => setDrawerTab(v)}
              sx={{ mb: 2, borderBottom: '1px solid', borderColor: 'divider' }}
            >
              <Tab value="content" label="Content" />
              <Tab value="revisions" label={`Revisions (${revisions.length})`} />
            </Tabs>

            {drawerTab === 'content' && (
              <Box sx={{ flex: 1, overflowY: 'auto' }}>
                {detail.note.content ? (
                  <MarkdownView content={detail.note.content} />
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    Note content is empty.
                  </Typography>
                )}
              </Box>
            )}

            {drawerTab === 'revisions' && (
              <Box sx={{ flex: 1, overflowY: 'auto' }}>
                {revisions.length === 0 ? (
                  <Typography variant="body2" color="text.secondary">
                    No prior revisions recorded for this note.
                  </Typography>
                ) : (
                  <Stack spacing={2}>
                    <Box
                      sx={{
                        border: '1px solid',
                        borderColor: 'divider',
                        borderRadius: 1,
                        bgcolor: 'background.paper',
                        maxHeight: 200,
                        overflowY: 'auto',
                      }}
                    >
                      <List dense disablePadding>
                        {revisions.map((rev) => {
                          const isSelected = selectedRev?.revision === rev.revision
                          return (
                            <ListItemButton
                              key={rev.revision}
                              selected={isSelected}
                              onClick={() => setSelectedRevisionNum(rev.revision)}
                              data-testid={`revision-item-${rev.revision}`}
                            >
                              <ListItemText
                                primary={
                                  <Stack direction="row" spacing={1} sx={{ alignItems: 'center' }}>
                                    <Typography sx={{ fontFamily: MONO, fontSize: 12, fontWeight: 600 }}>
                                      r{rev.revision}
                                    </Typography>
                                    <Chip size="small" variant="outlined" label={rev.op || 'edit'} />
                                    <Typography variant="caption" color="text.secondary">
                                      {formatNum(rev.token_est)} tokens
                                    </Typography>
                                  </Stack>
                                }
                                secondary={rev.created_at}
                              />
                            </ListItemButton>
                          )
                        })}
                      </List>
                    </Box>

                    <Stack
                      direction="row"
                      spacing={1}
                      useFlexGap
                      sx={{ alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap' }}
                    >
                      <ToggleButtonGroup
                        size="small"
                        exclusive
                        value={effectiveCompareTarget}
                        onChange={(_e, val) => {
                          if (val) setCompareAgainst(val)
                        }}
                      >
                        <ToggleButton value="current">vs Current body</ToggleButton>
                        <ToggleButton value="previous" disabled={!previousRev}>
                          vs Previous revision
                        </ToggleButton>
                      </ToggleButtonGroup>

                      <ToggleButtonGroup
                        size="small"
                        exclusive
                        value={diffMode}
                        onChange={(_e, val) => {
                          if (val) setDiffMode(val)
                        }}
                      >
                        <ToggleButton value="line">Line-by-line</ToggleButton>
                        <ToggleButton value="side-by-side">Side-by-side</ToggleButton>
                      </ToggleButtonGroup>
                    </Stack>

                    {diffComparison && (
                      <Box data-testid="revision-diff-viewer">
                        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                          Comparing <strong>{diffComparison.beforeLabel}</strong> →{' '}
                          <strong>{diffComparison.afterLabel}</strong>
                        </Typography>

                        {diffMode === 'line' ? (
                          <Box
                            sx={{
                              fontFamily: MONO,
                              fontSize: 12,
                              border: '1px solid',
                              borderColor: 'divider',
                              borderRadius: 1,
                              bgcolor: '#0d1117',
                              overflowX: 'auto',
                            }}
                          >
                            {diffLines.length === 0 ? (
                              <Typography variant="caption" color="text.secondary" sx={{ p: 1.5, display: 'block' }}>
                                No differences.
                              </Typography>
                            ) : (
                              diffLines.map((line, idx) => (
                                <Box
                                  key={idx}
                                  sx={{
                                    px: 1.25,
                                    py: 0.25,
                                    whiteSpace: 'pre-wrap',
                                    wordBreak: 'break-word',
                                    bgcolor:
                                      line.type === 'add'
                                        ? 'rgba(63, 185, 80, 0.14)'
                                        : line.type === 'remove'
                                          ? 'rgba(248, 81, 73, 0.14)'
                                          : 'transparent',
                                    color:
                                      line.type === 'add'
                                        ? '#aff5b4'
                                        : line.type === 'remove'
                                          ? '#ffdcd7'
                                          : 'text.primary',
                                  }}
                                >
                                  <Box component="span" sx={{ display: 'inline-block', width: 18, opacity: 0.7 }}>
                                    {line.type === 'add' ? '+' : line.type === 'remove' ? '-' : ' '}
                                  </Box>
                                  {line.text}
                                </Box>
                              ))
                            )}
                          </Box>
                        ) : (
                          <Grid container spacing={1.5}>
                            <Grid size={{ xs: 12, md: 6 }}>
                              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                                Before ({diffComparison.beforeLabel})
                              </Typography>
                              <Box
                                component="pre"
                                sx={{
                                  m: 0,
                                  p: 1.25,
                                  fontFamily: MONO,
                                  fontSize: 12,
                                  bgcolor: '#0d1117',
                                  border: '1px solid',
                                  borderColor: 'divider',
                                  borderRadius: 1,
                                  whiteSpace: 'pre-wrap',
                                  wordBreak: 'break-word',
                                  maxHeight: 360,
                                  overflowY: 'auto',
                                }}
                              >
                                {diffComparison.beforeText || '(empty)'}
                              </Box>
                            </Grid>
                            <Grid size={{ xs: 12, md: 6 }}>
                              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                                After ({diffComparison.afterLabel})
                              </Typography>
                              <Box
                                component="pre"
                                sx={{
                                  m: 0,
                                  p: 1.25,
                                  fontFamily: MONO,
                                  fontSize: 12,
                                  bgcolor: '#0d1117',
                                  border: '1px solid',
                                  borderColor: 'divider',
                                  borderRadius: 1,
                                  whiteSpace: 'pre-wrap',
                                  wordBreak: 'break-word',
                                  maxHeight: 360,
                                  overflowY: 'auto',
                                }}
                              >
                                {diffComparison.afterText || '(empty)'}
                              </Box>
                            </Grid>
                          </Grid>
                        )}
                      </Box>
                    )}
                  </Stack>
                )}
              </Box>
            )}
          </Box>
        )}
      </Drawer>

      {/* Flush orphans confirmation dialog */}
      <Dialog
        open={flushDialogOpen}
        onClose={() => !flushingOrphans && setFlushDialogOpen(false)}
        aria-labelledby="flush-orphans-dialog-title"
      >
        <DialogTitle id="flush-orphans-dialog-title">Flush orphaned context notes?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            This permanently deletes orphaned virtual context notes (stored notes never fetched back; notes stored in
            the last hour are kept){projectId ? ` for project ${projectId}` : ' across all projects'}.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setFlushDialogOpen(false)} disabled={flushingOrphans}>
            Cancel
          </Button>
          <Button
            color="warning"
            variant="contained"
            disabled={flushingOrphans}
            onClick={() => void handleFlushOrphans()}
          >
            {flushingOrphans ? 'Flushing…' : 'Confirm flush orphans'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete context note confirmation dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={() => !deletingNote && setDeleteDialogOpen(false)}
        aria-labelledby="delete-context-note-dialog-title"
      >
        <DialogTitle id="delete-context-note-dialog-title">Delete context note?</DialogTitle>
        <DialogContent>
          {isActiveHandoffTree && (
            <Alert severity="warning" sx={{ mb: 2 }} data-testid="active-handoff-warning">
              Warning: This {activeKind === 'handoff_result' ? 'handoff_result ' : ''}note belongs to an active
              handoff tree. Deleting it may affect active subagent coordination or parent collection.
            </Alert>
          )}
          <DialogContentText>
            Permanently delete context note{' '}
            <Box component="span" sx={{ fontFamily: MONO, fontWeight: 600 }}>
              {activeRef}
            </Box>
            {activeSession ? (
              <>
                {' '}
                (session{' '}
                <Box component="span" sx={{ fontFamily: MONO }}>
                  {activeSession}
                </Box>
                )
              </>
            ) : null}
            ? All stored revisions for this note will also be deleted. This cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)} disabled={deletingNote}>
            Cancel
          </Button>
          <Button
            color="error"
            variant="contained"
            disabled={deletingNote}
            onClick={() => void handleDeleteNote()}
          >
            {deletingNote ? 'Deleting…' : 'Confirm delete'}
          </Button>
        </DialogActions>
      </Dialog>
      <Divider sx={{ display: 'none' }} />
    </Box>
  )
}

function MetaField({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <Box sx={{ minWidth: 0 }}>
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
        {label}
      </Typography>
      <Typography
        variant="body2"
        noWrap
        title={value}
        sx={mono ? { fontFamily: MONO, fontSize: 12 } : { fontSize: 13 }}
      >
        {value}
      </Typography>
    </Box>
  )
}
