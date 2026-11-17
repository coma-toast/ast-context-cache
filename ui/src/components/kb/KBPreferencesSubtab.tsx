import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Divider,
  IconButton,
  LinearProgress,
  MenuItem,
  Stack,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  Typography,
} from '@mui/material'
import AddIcon from '@mui/icons-material/Add'
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutlined'
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutlined'
import EditOutlinedIcon from '@mui/icons-material/EditOutlined'
import HistoryIcon from '@mui/icons-material/History'
import VerifiedIcon from '@mui/icons-material/Verified'

import type { KBCreateInput, KBEntry, KBScope } from '../../api/types'
import { api } from '../../api/client'
import { useToast } from '../../context/ToastContext'
import { KBEntryFormDialog } from './KBEntryFormDialog'
import { KB_KIND_LABELS, formatShortDate, kbHeadline, kbKindChipSx } from './kbHelpers'

const MONO = '"JetBrains Mono", ui-monospace, monospace'

type QuickKind = 'preference' | 'procedure'

export function KBPreferencesSubtab({
  projectId,
  kbRefreshNonce,
  onSelectRef,
}: {
  projectId?: string
  kbRefreshNonce?: number
  onSelectRef: (ref: string) => void
}) {
  const { showToast } = useToast()

  const [entries, setEntries] = useState<KBEntry[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [localRefresh, setLocalRefresh] = useState(0)

  // Inline quick-add bar state
  const [quickKind, setQuickKind] = useState<QuickKind>('preference')
  const [quickScope, setQuickScope] = useState<KBScope>(projectId ? 'project' : 'global')
  const [quickSubject, setQuickSubject] = useState('user')
  const [quickPredicate, setQuickPredicate] = useState('prefers')
  const [quickObject, setQuickObject] = useState('')
  const [quickRule, setQuickRule] = useState('')
  const [adding, setAdding] = useState(false)

  // Edit / Retire dialog state (isolated from background refreshes)
  const [editTarget, setEditTarget] = useState<KBEntry | null>(null)
  const [retireTarget, setRetireTarget] = useState<KBEntry | null>(null)
  const [actionBusyRef, setActionBusyRef] = useState<string | null>(null)

  useEffect(() => {
    if (!projectId && quickScope === 'project') {
      setQuickScope('global')
    }
  }, [projectId, quickScope])

  const loadPreferences = useCallback(
    async (signal?: { cancelled: boolean }) => {
      setLoading(true)
      setError('')
      try {
        const res = await api.kbList({
          kinds: ['preference', 'procedure'],
          status: 'active',
          project_id: projectId || undefined,
          sort: 'created_desc',
          per_page: 200,
        })
        if (!signal?.cancelled) {
          setEntries(res.entries ?? [])
        }
      } catch (err) {
        if (!signal?.cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load preferences & procedures')
        }
      } finally {
        if (!signal?.cancelled) {
          setLoading(false)
        }
      }
    },
    [projectId],
  )

  useEffect(() => {
    const signal = { cancelled: false }
    void loadPreferences(signal)
    return () => {
      signal.cancelled = true
    }
  }, [loadPreferences, kbRefreshNonce, localRefresh])

  const { globalEntries, projectEntries } = useMemo(() => {
    const globalList: KBEntry[] = []
    const projectList: KBEntry[] = []
    for (const e of entries) {
      if (e.scope === 'global') {
        globalList.push(e)
      } else {
        projectList.push(e)
      }
    }
    return { globalEntries: globalList, projectEntries: projectList }
  }, [entries])

  const handleQuickAdd = async (e: React.FormEvent) => {
    e.preventDefault()
    if (quickScope === 'project' && !projectId) {
      showToast('Select a project in the top bar to add a project-scoped rule', 'warning')
      return
    }
    if (quickKind === 'preference' && !quickObject.trim()) {
      showToast('Preferred value (object) is required', 'error')
      return
    }
    if (quickKind === 'procedure' && !quickRule.trim()) {
      showToast('Procedure rule is required', 'error')
      return
    }

    setAdding(true)
    try {
      const payload: KBCreateInput =
        quickKind === 'preference'
          ? {
              kind: 'preference',
              scope: quickScope,
              project_path: quickScope === 'project' ? projectId : undefined,
              subject: quickSubject.trim() || 'user',
              predicate: quickPredicate.trim() || 'prefers',
              object: quickObject.trim(),
            }
          : {
              kind: 'procedure',
              scope: quickScope,
              project_path: quickScope === 'project' ? projectId : undefined,
              rule: quickRule.trim(),
            }
      const res = await api.kbCreateEntry(payload)
      if (res.warnings?.length) {
        showToast(res.warnings.join('; '), 'warning')
      }
      showToast(`${KB_KIND_LABELS[res.entry.kind]} added (${res.entry.ref})`, 'success')
      setQuickObject('')
      setQuickRule('')
      setLocalRefresh((n) => n + 1)
    } catch (err) {
      showToast(err instanceof Error ? err.message : 'Failed to add entry', 'error')
    } finally {
      setAdding(false)
    }
  }

  const handleToggleVerify = async (entry: KBEntry) => {
    setActionBusyRef(entry.ref)
    try {
      await api.kbEntryAction({
        ref: entry.ref,
        action: 'verify',
        verified: !entry.verified,
      })
      showToast(entry.verified ? 'Verification removed' : 'Marked verified', 'success')
      setLocalRefresh((n) => n + 1)
    } catch (err) {
      showToast(err instanceof Error ? err.message : 'Failed to update verification', 'error')
    } finally {
      setActionBusyRef(null)
    }
  }

  const handleConfirmRetire = async () => {
    if (!retireTarget) return
    const targetRef = retireTarget.ref
    setActionBusyRef(targetRef)
    try {
      await api.kbEntryAction({ ref: targetRef, action: 'forget' })
      showToast(`Retired ${targetRef}`, 'success')
      setRetireTarget(null)
      setLocalRefresh((n) => n + 1)
    } catch (err) {
      showToast(err instanceof Error ? err.message : 'Failed to retire entry', 'error')
    } finally {
      setActionBusyRef(null)
    }
  }

  const showProjectSection = Boolean(projectId) || projectEntries.length > 0

  return (
    <Box>
      <Card variant="outlined" sx={{ mb: 2.5, borderColor: '#30363d', bgcolor: '#161b22' }}>
        <CardContent sx={{ pb: '16px !important' }}>
          <Typography variant="overline" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
            Quick Add Preference or Procedure
          </Typography>
          <form onSubmit={(e) => void handleQuickAdd(e)}>
            <Stack
              direction={{ xs: 'column', md: 'row' }}
              spacing={1.25}
              useFlexGap
              sx={{ alignItems: { xs: 'stretch', md: 'center' }, flexWrap: 'wrap' }}
            >
              <ToggleButtonGroup
                size="small"
                exclusive
                value={quickKind}
                onChange={(_e, val) => {
                  if (val) setQuickKind(val as QuickKind)
                }}
                aria-label="Entry kind"
              >
                <ToggleButton value="preference" sx={{ px: 1.5, py: 0.5, fontSize: 12.5, textTransform: 'none' }}>
                  Preference
                </ToggleButton>
                <ToggleButton value="procedure" sx={{ px: 1.5, py: 0.5, fontSize: 12.5, textTransform: 'none' }}>
                  Procedure
                </ToggleButton>
              </ToggleButtonGroup>

              <TextField
                select
                size="small"
                label="Scope"
                value={quickScope}
                onChange={(e) => setQuickScope(e.target.value as KBScope)}
                sx={{ minWidth: 120 }}
              >
                <MenuItem value="global">Global</MenuItem>
                <MenuItem value="project" disabled={!projectId}>
                  Project{!projectId ? ' (select project)' : ''}
                </MenuItem>
              </TextField>

              {quickKind === 'preference' ? (
                <>
                  <TextField
                    size="small"
                    label="Subject"
                    value={quickSubject}
                    onChange={(e) => setQuickSubject(e.target.value)}
                    placeholder="user"
                    sx={{ width: { xs: '100%', md: 130 } }}
                  />
                  <TextField
                    size="small"
                    label="Predicate"
                    value={quickPredicate}
                    onChange={(e) => setQuickPredicate(e.target.value)}
                    placeholder="prefers"
                    sx={{ width: { xs: '100%', md: 130 } }}
                  />
                  <TextField
                    size="small"
                    label="Preferred value (object)"
                    value={quickObject}
                    onChange={(e) => setQuickObject(e.target.value)}
                    placeholder="e.g. gofumpt formatting, concise responses"
                    sx={{ flex: 1, minWidth: 220 }}
                  />
                </>
              ) : (
                <TextField
                  size="small"
                  label="Procedure rule"
                  value={quickRule}
                  onChange={(e) => setQuickRule(e.target.value)}
                  placeholder="e.g. Run go test ./... before committing Go changes"
                  sx={{ flex: 1, minWidth: 280 }}
                />
              )}

              <Button
                type="submit"
                variant="contained"
                size="small"
                startIcon={<AddIcon fontSize="small" />}
                disabled={adding}
                sx={{ flexShrink: 0, height: 38 }}
              >
                {adding ? 'Adding…' : 'Add'}
              </Button>
            </Stack>
          </form>
        </CardContent>
      </Card>

      {error && (
        <Alert
          severity="error"
          sx={{ mb: 2 }}
          action={
            <Button color="inherit" size="small" onClick={() => void loadPreferences()}>
              Retry
            </Button>
          }
        >
          {error}
        </Alert>
      )}

      {loading && <LinearProgress sx={{ mb: 2, borderRadius: 1 }} />}

      <Stack spacing={3}>
        {showProjectSection && (
          <PreferenceSection
            title="Project Preferences & Procedures"
            subtitle={projectId ? `Scoped to ${projectId}` : 'Project-scoped rules'}
            emptyText="No project-scoped preferences or procedures yet. Use the quick-add bar above with Project scope."
            entries={projectEntries}
            busyRef={actionBusyRef}
            onEdit={(entry) => setEditTarget(entry)}
            onToggleVerify={(entry) => void handleToggleVerify(entry)}
            onSelectRef={onSelectRef}
            onRetire={(entry) => setRetireTarget(entry)}
          />
        )}

        <PreferenceSection
          title="Global Preferences & Procedures"
          subtitle="Apply across all projects and sessions"
          emptyText="No global preferences or procedures yet. Add one above to persist cross-project rules."
          entries={globalEntries}
          busyRef={actionBusyRef}
          onEdit={(entry) => setEditTarget(entry)}
          onToggleVerify={(entry) => void handleToggleVerify(entry)}
          onSelectRef={onSelectRef}
          onRetire={(entry) => setRetireTarget(entry)}
        />
      </Stack>

      <KBEntryFormDialog
        open={editTarget !== null}
        mode="edit"
        initialEntry={editTarget}
        projectId={editTarget?.project_path ?? projectId}
        onClose={() => setEditTarget(null)}
        onSaved={() => {
          setLocalRefresh((n) => n + 1)
        }}
      />

      <Dialog
        open={retireTarget !== null}
        onClose={() => !actionBusyRef && setRetireTarget(null)}
        aria-labelledby="kb-retire-title"
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
        <DialogTitle id="kb-retire-title">Retire preference / procedure?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Retire <Box component="span" sx={{ fontFamily: MONO }}>{retireTarget?.ref}</Box> (
            {retireTarget ? kbHeadline(retireTarget) : ''})? This forgets it from active recall; you can still
            inspect or restore it from the Explorer under Superseded &amp; Forgotten.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setRetireTarget(null)} disabled={Boolean(actionBusyRef)}>
            Cancel
          </Button>
          <Button
            color="error"
            variant="contained"
            disabled={Boolean(actionBusyRef)}
            onClick={() => void handleConfirmRetire()}
          >
            {actionBusyRef ? 'Retiring…' : 'Retire'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}

function PreferenceSection({
  title,
  subtitle,
  emptyText,
  entries,
  busyRef,
  onEdit,
  onToggleVerify,
  onSelectRef,
  onRetire,
}: {
  title: string
  subtitle: string
  emptyText: string
  entries: KBEntry[]
  busyRef: string | null
  onEdit: (entry: KBEntry) => void
  onToggleVerify: (entry: KBEntry) => void
  onSelectRef: (ref: string) => void
  onRetire: (entry: KBEntry) => void
}) {
  return (
    <Box>
      <Stack direction="row" spacing={1} useFlexGap sx={{ mb: 1, alignItems: 'baseline', flexWrap: 'wrap' }}>
        <Typography variant="overline" color="text.secondary" sx={{ fontWeight: 700 }}>
          {title}
        </Typography>
        <Chip size="small" variant="outlined" label={entries.length} sx={{ height: 18, fontSize: 10.5 }} />
        <Typography variant="caption" color="text.secondary" sx={{ fontFamily: MONO, fontSize: 11 }}>
          {subtitle}
        </Typography>
      </Stack>

      <Card variant="outlined" sx={{ borderColor: '#30363d', bgcolor: '#161b22' }}>
        {entries.length === 0 ? (
          <CardContent sx={{ py: 3 }}>
            <Typography variant="body2" color="text.secondary">
              {emptyText}
            </Typography>
          </CardContent>
        ) : (
          <Stack divider={<Divider sx={{ borderColor: '#30363d' }} />}>
            {entries.map((entry) => {
              const busy = busyRef === entry.ref
              return (
                <Box key={entry.ref} sx={{ px: 2, py: 1.5 }}>
                  <Stack
                    direction={{ xs: 'column', sm: 'row' }}
                    spacing={1.5}
                    sx={{ alignItems: { xs: 'flex-start', sm: 'center' }, justifyContent: 'space-between' }}
                  >
                    <Box sx={{ minWidth: 0, flex: 1 }}>
                      <Stack direction="row" spacing={0.75} useFlexGap sx={{ mb: 0.5, alignItems: 'center', flexWrap: 'wrap' }}>
                        <Chip
                          size="small"
                          variant="outlined"
                          label={KB_KIND_LABELS[entry.kind]}
                          sx={{ ...kbKindChipSx(entry.kind), height: 20, fontSize: 11 }}
                        />
                        {entry.verified && (
                          <Chip
                            size="small"
                            color="success"
                            variant="outlined"
                            icon={<VerifiedIcon sx={{ fontSize: '13px !important' }} />}
                            label="Verified"
                            sx={{ height: 20, fontSize: 11 }}
                          />
                        )}
                        <Typography variant="caption" sx={{ fontFamily: MONO, color: '#58a6ff' }}>
                          {entry.ref}
                        </Typography>
                        {entry.actor && (
                          <Typography variant="caption" color="text.secondary">
                            · by {entry.actor}
                          </Typography>
                        )}
                        {(entry.occurred_at || entry.created_at) && (
                          <Typography variant="caption" color="text.secondary">
                            · {formatShortDate(entry.occurred_at || entry.created_at)}
                          </Typography>
                        )}
                      </Stack>

                      <Typography variant="body2" sx={{ fontWeight: 500, color: 'text.primary', wordBreak: 'break-word' }}>
                        {kbHeadline(entry)}
                      </Typography>

                      {entry.body && entry.body.trim() !== '' && (
                        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.35 }}>
                          {entry.body.trim()}
                        </Typography>
                      )}

                      {entry.tags && entry.tags.length > 0 && (
                        <Stack direction="row" spacing={0.5} useFlexGap sx={{ mt: 0.75, flexWrap: 'wrap' }}>
                          {entry.tags.map((tag) => (
                            <Chip
                              key={tag}
                              size="small"
                              variant="outlined"
                              label={tag}
                              sx={{ height: 18, fontSize: 10.5, borderColor: '#30363d' }}
                            />
                          ))}
                        </Stack>
                      )}
                    </Box>

                    <Stack direction="row" spacing={0.5} sx={{ flexShrink: 0, alignItems: 'center' }}>
                      <Tooltip title="Edit">
                        <span>
                          <IconButton
                            size="small"
                            aria-label={`Edit ${entry.ref}`}
                            disabled={busy}
                            onClick={() => onEdit(entry)}
                          >
                            <EditOutlinedIcon fontSize="small" />
                          </IconButton>
                        </span>
                      </Tooltip>
                      <Tooltip title={entry.verified ? 'Remove verification' : 'Mark verified'}>
                        <span>
                          <IconButton
                            size="small"
                            color={entry.verified ? 'success' : 'default'}
                            aria-label={`${entry.verified ? 'Unverify' : 'Verify'} ${entry.ref}`}
                            disabled={busy}
                            onClick={() => onToggleVerify(entry)}
                          >
                            <CheckCircleOutlineIcon fontSize="small" />
                          </IconButton>
                        </span>
                      </Tooltip>
                      <Tooltip title="History & details">
                        <span>
                          <IconButton
                            size="small"
                            aria-label={`Details and history for ${entry.ref}`}
                            disabled={busy}
                            onClick={() => onSelectRef(entry.ref)}
                          >
                            <HistoryIcon fontSize="small" />
                          </IconButton>
                        </span>
                      </Tooltip>
                      <Tooltip title="Retire">
                        <span>
                          <IconButton
                            size="small"
                            color="error"
                            aria-label={`Retire ${entry.ref}`}
                            disabled={busy}
                            onClick={() => onRetire(entry)}
                          >
                            <DeleteOutlineIcon fontSize="small" />
                          </IconButton>
                        </span>
                      </Tooltip>
                    </Stack>
                  </Stack>
                </Box>
              )
            })}
          </Stack>
        )}
      </Card>
    </Box>
  )
}
