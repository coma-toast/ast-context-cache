import { useCallback, useEffect, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Divider,
  Drawer,
  IconButton,
  List,
  ListItemButton,
  Stack,
  Tooltip,
  Typography,
} from '@mui/material'
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutlined'
import CloseIcon from '@mui/icons-material/Close'
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutlined'
import EditOutlinedIcon from '@mui/icons-material/EditOutlined'
import HistoryIcon from '@mui/icons-material/History'
import RestoreFromTrashIcon from '@mui/icons-material/RestoreFromTrash'
import ThumbDownOutlinedIcon from '@mui/icons-material/ThumbDownOutlined'
import ThumbUpOutlinedIcon from '@mui/icons-material/ThumbUpOutlined'
import VerifiedIcon from '@mui/icons-material/Verified'
import WarningAmberIcon from '@mui/icons-material/WarningAmber'

import type { KBAction, KBConflict, KBDetailResponse, KBEntry } from '../../api/types'
import { api } from '../../api/client'
import { useToast } from '../../context/ToastContext'
import { MarkdownView } from '../MarkdownView'
import { KBEntryFormDialog } from './KBEntryFormDialog'
import {
  KB_KIND_LABELS,
  KB_SCOPE_LABELS,
  formatShortDate,
  isEntryActive,
  kbEntryStatus,
  kbHeadline,
  kbKindChipSx,
  kbSeverityChipSx,
} from './kbHelpers'

const MONO = '"JetBrains Mono", ui-monospace, monospace'

type ConfirmAction = 'forget' | 'reject_quarantine' | null

export function KBDetailDrawer({
  refId,
  onClose,
  onSelectRef,
  onMutated,
}: {
  refId: string
  onClose: () => void
  onSelectRef: (ref: string) => void
  onMutated?: () => void
}) {
  const { showToast } = useToast()
  const [detail, setDetail] = useState<KBDetailResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [actionBusy, setActionBusy] = useState(false)
  const [confirmAction, setConfirmAction] = useState<ConfirmAction>(null)
  const [editOpen, setEditOpen] = useState(false)

  const loadEntry = useCallback(async (targetRef: string) => {
    if (!targetRef) {
      setDetail(null)
      setError('')
      return
    }
    setLoading(true)
    setError('')
    try {
      const res = await api.kbGetEntry(targetRef)
      setDetail(res)
    } catch (err) {
      setDetail(null)
      setError(err instanceof Error ? err.message : 'Failed to load entry')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (!refId) {
      setDetail(null)
      setError('')
      setEditOpen(false)
      setConfirmAction(null)
      return
    }
    let cancelled = false
    setLoading(true)
    setError('')
    api
      .kbGetEntry(refId)
      .then((res) => {
        if (!cancelled) {
          setDetail(res)
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setDetail(null)
          setError(err instanceof Error ? err.message : 'Failed to load entry')
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false)
        }
      })
    return () => {
      cancelled = true
    }
  }, [refId])

  const entry = detail?.entry ?? null
  const history = detail?.history ?? []
  const conflicts: KBConflict[] = detail?.conflicts ?? []

  const runAction = async (action: KBAction, verified?: boolean) => {
    if (!entry) return
    setActionBusy(true)
    try {
      const res = await api.kbEntryAction({ ref: entry.ref, action, verified })
      const labelMap: Record<KBAction, string> = {
        forget: 'Entry forgotten',
        restore: 'Entry restored',
        verify: verified ? 'Entry marked verified' : 'Verification removed',
        approve_quarantine: 'Quarantined entry approved',
        reject_quarantine: 'Quarantined entry rejected',
      }
      showToast(labelMap[action] ?? 'Action completed', 'success')
      onMutated?.()

      if (res.entry) {
        await loadEntry(res.entry.ref)
      } else {
        onClose()
      }
    } catch (err) {
      showToast(err instanceof Error ? err.message : `Action "${action}" failed`, 'error')
    } finally {
      setActionBusy(false)
      setConfirmAction(null)
    }
  }

  const handleSaved = (saved: KBEntry) => {
    onMutated?.()
    if (saved.ref !== refId) {
      onSelectRef(saved.ref)
    } else {
      void loadEntry(saved.ref)
    }
  }

  const status = entry ? kbEntryStatus(entry) : 'active'
  const active = entry ? isEntryActive(entry) : false
  const isArchived = Boolean(entry?.valid_until && entry.valid_until.trim() !== '')
  const isQuarantined = (entry?.quarantined ?? 0) === 1 && !isArchived

  return (
    <>
      <Drawer
        anchor="right"
        open={Boolean(refId)}
        onClose={onClose}
        slotProps={{
          paper: {
            sx: {
              width: { xs: '100vw', sm: 560, md: 640 },
              bgcolor: '#0d1117',
              backgroundImage: 'none',
              borderRight: 'none',
              borderLeft: '1px solid #30363d',
              display: 'flex',
              flexDirection: 'column',
            },
          },
        }}
      >
        <Box sx={{ px: 2.5, py: 2, borderBottom: '1px solid #30363d', bgcolor: '#161b22' }}>
          <Stack direction="row" spacing={1.5} sx={{ justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <Box sx={{ minWidth: 0, flex: 1 }}>
              {entry ? (
                <>
                  <Typography variant="h6" sx={{ fontSize: 17, fontWeight: 600, wordBreak: 'break-word' }}>
                    {kbHeadline(entry)}
                  </Typography>
                  <Stack direction="row" spacing={0.75} useFlexGap sx={{ mt: 1, alignItems: 'center', flexWrap: 'wrap' }}>
                    <Chip
                      size="small"
                      variant="outlined"
                      label={KB_KIND_LABELS[entry.kind]}
                      sx={kbKindChipSx(entry.kind)}
                    />
                    <Chip size="small" variant="outlined" label={KB_SCOPE_LABELS[entry.scope]} />
                    <StatusChip status={status} />
                    {entry.verified && (
                      <Chip
                        size="small"
                        icon={<VerifiedIcon sx={{ fontSize: '14px !important' }} />}
                        label="Verified"
                        color="success"
                        variant="outlined"
                      />
                    )}
                    {entry.severity && (
                      <Chip
                        size="small"
                        variant="outlined"
                        label={entry.severity}
                        sx={kbSeverityChipSx(entry.severity)}
                      />
                    )}
                    {entry.category && <Chip size="small" variant="outlined" label={entry.category} />}
                    <Typography
                      component="span"
                      sx={{ fontFamily: MONO, fontSize: 11.5, color: 'text.secondary', ml: 0.5 }}
                    >
                      {entry.ref}
                    </Typography>
                  </Stack>
                </>
              ) : (
                <Typography variant="subtitle1" sx={{ fontFamily: MONO, fontSize: 14 }}>
                  {refId}
                </Typography>
              )}
            </Box>
            <IconButton size="small" onClick={onClose} aria-label="Close detail drawer">
              <CloseIcon fontSize="small" />
            </IconButton>
          </Stack>

          {entry && (
            <Stack direction="row" spacing={1} useFlexGap sx={{ mt: 1.5, flexWrap: 'wrap' }}>
              {!isArchived && (
                <Button
                  size="small"
                  variant="outlined"
                  startIcon={<EditOutlinedIcon fontSize="small" />}
                  disabled={actionBusy}
                  onClick={() => setEditOpen(true)}
                >
                  Edit
                </Button>
              )}
              {!isArchived && !isQuarantined && (
                <Button
                  size="small"
                  variant="outlined"
                  color={entry.verified ? 'inherit' : 'success'}
                  startIcon={<CheckCircleOutlineIcon fontSize="small" />}
                  disabled={actionBusy}
                  onClick={() => void runAction('verify', !entry.verified)}
                >
                  {entry.verified ? 'Unverify' : 'Verify'}
                </Button>
              )}
              {active && (
                <Button
                  size="small"
                  variant="outlined"
                  color="error"
                  startIcon={<DeleteOutlineIcon fontSize="small" />}
                  disabled={actionBusy}
                  onClick={() => setConfirmAction('forget')}
                >
                  Forget
                </Button>
              )}
              {isArchived && (
                <Button
                  size="small"
                  variant="outlined"
                  color="primary"
                  startIcon={<RestoreFromTrashIcon fontSize="small" />}
                  disabled={actionBusy}
                  onClick={() => void runAction('restore')}
                >
                  Restore
                </Button>
              )}
              {isQuarantined && (
                <>
                  <Button
                    size="small"
                    variant="contained"
                    color="success"
                    startIcon={<ThumbUpOutlinedIcon fontSize="small" />}
                    disabled={actionBusy}
                    onClick={() => void runAction('approve_quarantine')}
                  >
                    Approve
                  </Button>
                  <Button
                    size="small"
                    variant="outlined"
                    color="error"
                    startIcon={<ThumbDownOutlinedIcon fontSize="small" />}
                    disabled={actionBusy}
                    onClick={() => setConfirmAction('reject_quarantine')}
                  >
                    Reject
                  </Button>
                </>
              )}
            </Stack>
          )}
        </Box>

        <Box sx={{ flex: 1, overflowY: 'auto', p: 2.5 }}>
          {loading && (
            <Stack direction="row" spacing={1.5} sx={{ py: 4, alignItems: 'center', justifyContent: 'center' }}>
              <CircularProgress size={20} />
              <Typography variant="body2" color="text.secondary">
                Loading entry…
              </Typography>
            </Stack>
          )}

          {!loading && error && (
            <Alert
              severity="error"
              action={
                <Button color="inherit" size="small" onClick={() => void loadEntry(refId)}>
                  Retry
                </Button>
              }
            >
              {error}
            </Alert>
          )}

          {!loading && entry && (
            <Stack spacing={2.5}>
              {entry.redacted_fields && entry.redacted_fields.length > 0 && (
                <Alert severity="warning" icon={<WarningAmberIcon fontSize="small" />}>
                  Secret redaction was applied to: {entry.redacted_fields.join(', ')}.
                </Alert>
              )}

              {conflicts.length > 0 && (
                <Alert severity="warning">
                  <Typography variant="subtitle2" sx={{ fontSize: 13, fontWeight: 600, mb: 0.5 }}>
                    Potential conflicts detected ({conflicts.length})
                  </Typography>
                  <Stack spacing={0.75}>
                    {conflicts.map((c) => (
                      <Stack
                        key={c.ref}
                        direction="row"
                        spacing={1}
                        sx={{ alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap' }}
                      >
                        <Typography variant="caption" sx={{ flex: 1, minWidth: 160 }}>
                          {c.line}{' '}
                          <Typography component="span" variant="caption" color="text.secondary">
                            ({Math.round(c.similarity * 100)}% match)
                          </Typography>
                        </Typography>
                        <Button
                          size="small"
                          variant="text"
                          sx={{ fontFamily: MONO, fontSize: 11, minWidth: 0, py: 0 }}
                          onClick={() => onSelectRef(c.ref)}
                        >
                          {c.ref}
                        </Button>
                      </Stack>
                    ))}
                  </Stack>
                </Alert>
              )}

              {entry.kind === 'config_change' && (
                <Box
                  sx={{
                    p: 1.75,
                    borderRadius: 1.5,
                    border: '1px solid #30363d',
                    bgcolor: '#161b22',
                  }}
                >
                  <Typography variant="overline" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                    Configuration Diff
                  </Typography>
                  {entry.target && (
                    <Typography variant="body2" sx={{ fontFamily: MONO, fontSize: 12.5, mb: 1 }}>
                      Target: <Box component="span" sx={{ color: '#58a6ff' }}>{entry.target}</Box>
                    </Typography>
                  )}
                  <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.5}>
                    <Box
                      sx={{
                        flex: 1,
                        p: 1.25,
                        borderRadius: 1,
                        border: '1px solid rgba(248, 81, 73, 0.35)',
                        bgcolor: 'rgba(248, 81, 73, 0.08)',
                      }}
                    >
                      <Typography variant="caption" sx={{ color: '#f85149', fontWeight: 600, display: 'block', mb: 0.5 }}>
                        Before
                      </Typography>
                      <Typography
                        variant="body2"
                        sx={{ fontFamily: MONO, fontSize: 12, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}
                      >
                        {entry.before_value || '(none)'}
                      </Typography>
                    </Box>
                    <Box
                      sx={{
                        flex: 1,
                        p: 1.25,
                        borderRadius: 1,
                        border: '1px solid rgba(63, 185, 80, 0.35)',
                        bgcolor: 'rgba(63, 185, 80, 0.08)',
                      }}
                    >
                      <Typography variant="caption" sx={{ color: '#3fb950', fontWeight: 600, display: 'block', mb: 0.5 }}>
                        After
                      </Typography>
                      <Typography
                        variant="body2"
                        sx={{ fontFamily: MONO, fontSize: 12, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}
                      >
                        {entry.after_value || '(none)'}
                      </Typography>
                    </Box>
                  </Stack>
                  {entry.reason && (
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 1.25 }}>
                      <Box component="span" sx={{ fontWeight: 600, color: 'text.primary' }}>
                        Reason:{' '}
                      </Box>
                      {entry.reason}
                    </Typography>
                  )}
                </Box>
              )}

              {(entry.kind === 'fact' || entry.kind === 'preference') &&
                (entry.subject || entry.predicate || entry.object) && (
                  <Box
                    sx={{
                      p: 1.5,
                      borderRadius: 1.5,
                      border: '1px solid #30363d',
                      bgcolor: '#161b22',
                    }}
                  >
                    <Typography variant="overline" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                      Structured Assertion
                    </Typography>
                    <Typography variant="body2" sx={{ fontSize: 13.5 }}>
                      <Box component="span" sx={{ color: '#58a6ff', fontWeight: 600 }}>
                        {entry.subject || '—'}
                      </Box>{' '}
                      <Box component="span" color="text.secondary">
                        {entry.predicate || (entry.kind === 'preference' ? 'prefers' : 'is')}
                      </Box>{' '}
                      <Box component="span" sx={{ color: '#e6edf3', fontWeight: 500 }}>
                        {entry.object || '—'}
                      </Box>
                    </Typography>
                  </Box>
                )}

              {entry.kind === 'procedure' && entry.rule && (
                <Box
                  sx={{
                    p: 1.5,
                    borderRadius: 1.5,
                    border: '1px solid #30363d',
                    bgcolor: '#161b22',
                  }}
                >
                  <Typography variant="overline" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                    Procedure Rule
                  </Typography>
                  <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                    {entry.rule}
                  </Typography>
                </Box>
              )}

              {entry.body && entry.body.trim() !== '' && (
                <Box>
                  <Typography variant="overline" color="text.secondary" sx={{ display: 'block', mb: 0.75 }}>
                    Body
                  </Typography>
                  <Box
                    sx={{
                      p: 1.75,
                      borderRadius: 1.5,
                      border: '1px solid #30363d',
                      bgcolor: '#161b22',
                    }}
                  >
                    <MarkdownView content={entry.body} />
                  </Box>
                </Box>
              )}

              {entry.tags && entry.tags.length > 0 && (
                <Box>
                  <Typography variant="overline" color="text.secondary" sx={{ display: 'block', mb: 0.75 }}>
                    Tags
                  </Typography>
                  <Stack direction="row" spacing={0.75} useFlexGap sx={{ flexWrap: 'wrap' }}>
                    {entry.tags.map((tag) => (
                      <Chip key={tag} size="small" variant="outlined" label={tag} />
                    ))}
                  </Stack>
                </Box>
              )}

              <Box
                sx={{
                  p: 1.75,
                  borderRadius: 1.5,
                  border: '1px solid #30363d',
                  bgcolor: '#161b22',
                }}
              >
                <Typography variant="overline" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                  Provenance & Metadata
                </Typography>
                <Stack spacing={0.75}>
                  <MetaRow label="Actor" value={entry.actor} />
                  <MetaRow label="Target" value={entry.kind !== 'config_change' ? entry.target : undefined} mono />
                  <MetaRow label="Session ID" value={entry.session_id} mono />
                  <MetaRow label="Project path" value={entry.project_path} mono />
                  <MetaRow label="Source ref" value={entry.source_ref} mono />
                  <MetaRow label="Occurred at" value={formatShortDate(entry.occurred_at)} />
                  <MetaRow label="Created at" value={formatShortDate(entry.created_at)} />
                  <MetaRow
                    label="Access count"
                    value={entry.access_count != null ? String(entry.access_count) : undefined}
                  />
                  {entry.verified && entry.verified_at && (
                    <MetaRow label="Verified at" value={formatShortDate(entry.verified_at)} />
                  )}
                  {entry.valid_until && (
                    <MetaRow label="Invalidated at" value={formatShortDate(entry.valid_until)} />
                  )}
                  {entry.invalidated_reason && (
                    <MetaRow label="Invalidation reason" value={entry.invalidated_reason} />
                  )}
                  {entry.superseded_by && (
                    <Stack direction="row" spacing={1} sx={{ alignItems: 'baseline', justifyContent: 'space-between' }}>
                      <Typography variant="caption" color="text.secondary">
                        Superseded by
                      </Typography>
                      <Button
                        size="small"
                        variant="text"
                        sx={{ fontFamily: MONO, fontSize: 12, p: 0, minWidth: 0 }}
                        onClick={() => onSelectRef(entry.superseded_by!)}
                      >
                        {entry.superseded_by}
                      </Button>
                    </Stack>
                  )}
                </Stack>
              </Box>

              {history.length > 0 && (
                <Box>
                  <Stack direction="row" spacing={0.75} sx={{ mb: 1, alignItems: 'center' }}>
                    <HistoryIcon fontSize="small" color="action" />
                    <Typography variant="overline" color="text.secondary">
                      Version & Supersession History ({history.length})
                    </Typography>
                  </Stack>
                  <Box sx={{ border: '1px solid #30363d', borderRadius: 1.5, bgcolor: '#161b22', overflow: 'hidden' }}>
                    <List dense disablePadding>
                      {history.map((h, idx) => {
                        const isCurrent = h.ref === entry.ref
                        const hStatus = kbEntryStatus(h)
                        return (
                          <Box key={h.ref}>
                            {idx > 0 && <Divider sx={{ borderColor: '#30363d' }} />}
                            <ListItemButton
                              selected={isCurrent}
                              onClick={() => onSelectRef(h.ref)}
                              sx={{
                                borderRadius: 0,
                                mb: 0,
                                px: 1.5,
                                py: 1,
                                alignItems: 'flex-start',
                                flexDirection: 'column',
                                gap: 0.5,
                              }}
                            >
                              <Stack
                                direction="row"
                                spacing={1}
                                sx={{ width: '100%', alignItems: 'center', justifyContent: 'space-between' }}
                              >
                                <Stack direction="row" spacing={0.75} sx={{ alignItems: 'center' }}>
                                  <Typography sx={{ fontFamily: MONO, fontSize: 12, color: '#58a6ff' }}>
                                    {h.ref}
                                  </Typography>
                                  {isCurrent && <Chip size="small" label="Current" color="primary" sx={{ height: 18, fontSize: 10.5 }} />}
                                  <StatusChip status={hStatus} compact />
                                </Stack>
                                <Typography variant="caption" color="text.secondary">
                                  {formatShortDate(h.occurred_at || h.created_at)}
                                </Typography>
                              </Stack>
                              <Typography variant="body2" sx={{ fontSize: 13, color: 'text.primary' }}>
                                {kbHeadline(h)}
                              </Typography>
                              {h.invalidated_reason && (
                                <Typography variant="caption" color="text.secondary">
                                  Reason: {h.invalidated_reason}
                                </Typography>
                              )}
                            </ListItemButton>
                          </Box>
                        )
                      })}
                    </List>
                  </Box>
                </Box>
              )}
            </Stack>
          )}
        </Box>
      </Drawer>

      <KBEntryFormDialog
        open={editOpen}
        mode="edit"
        initialEntry={entry}
        projectId={entry?.project_path}
        onClose={() => setEditOpen(false)}
        onSaved={handleSaved}
      />

      <Dialog
        open={confirmAction !== null}
        onClose={() => !actionBusy && setConfirmAction(null)}
        aria-labelledby="kb-confirm-action-title"
        slotProps={{
          paper: {
            sx: {
              bgcolor: '#161b22',
              border: '1px solid #30363d',
              backgroundImage: 'none',
            },
          },
        }}
      >
        <DialogTitle id="kb-confirm-action-title">
          {confirmAction === 'forget' ? 'Forget knowledge base entry?' : 'Reject quarantined entry?'}
        </DialogTitle>
        <DialogContent>
          <DialogContentText>
            {confirmAction === 'forget' ? (
              <>
                This marks <Box component="span" sx={{ fontFamily: MONO }}>{entry?.ref}</Box> as forgotten and removes
                it from active recall and search. You can restore it later from Superseded &amp; Forgotten.
              </>
            ) : (
              <>
                This permanently rejects quarantined entry{' '}
                <Box component="span" sx={{ fontFamily: MONO }}>{entry?.ref}</Box> and archives it with reason{' '}
                <Box component="span" sx={{ fontFamily: MONO }}>quarantine</Box>.
              </>
            )}
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmAction(null)} disabled={actionBusy}>
            Cancel
          </Button>
          <Button
            color="error"
            variant="contained"
            disabled={actionBusy}
            onClick={() => confirmAction && void runAction(confirmAction)}
          >
            {actionBusy
              ? 'Working…'
              : confirmAction === 'forget'
                ? 'Forget Entry'
                : 'Reject Entry'}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  )
}

function StatusChip({ status, compact = false }: { status: ReturnType<typeof kbEntryStatus>; compact?: boolean }) {
  const labelMap = {
    active: 'Active',
    superseded: 'Superseded',
    forgotten: 'Forgotten',
    quarantined: 'Quarantined',
  } as const
  const colorMap = {
    active: 'success',
    superseded: 'default',
    forgotten: 'default',
    quarantined: 'warning',
  } as const
  return (
    <Chip
      size="small"
      variant="outlined"
      color={colorMap[status]}
      label={labelMap[status]}
      sx={compact ? { height: 18, fontSize: 10.5 } : undefined}
    />
  )
}

function MetaRow({ label, value, mono = false }: { label: string; value?: string; mono?: boolean }) {
  if (!value) return null
  return (
    <Stack direction="row" spacing={1.5} sx={{ alignItems: 'baseline', justifyContent: 'space-between' }}>
      <Typography variant="caption" color="text.secondary" sx={{ flexShrink: 0 }}>
        {label}
      </Typography>
      <Tooltip title={value}>
        <Typography
          variant="caption"
          sx={{
            fontFamily: mono ? MONO : 'inherit',
            color: 'text.primary',
            textAlign: 'right',
            wordBreak: 'break-all',
          }}
        >
          {value}
        </Typography>
      </Tooltip>
    </Stack>
  )
}
