import { useCallback, useEffect, useState } from 'react'

import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Alert,
  AlertTitle,
  Box,
  Button,
  Card,
  CardContent,
  Checkbox,
  Chip,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  FormControlLabel,
  Stack,
  Tooltip,
  Typography,
} from '@mui/material'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'

import type {
  InstallerAction,
  InstallerBackup,
  InstallerComponentId,
  InstallerFileChange,
  InstallerOverview,
  InstallerPlan,
  InstallerPlanRequest,
  InstallerTarget,
  InstallerTargetId,
} from '../api/types'

import { useToast } from '../context/ToastContext'

import { api, formatBytes } from '../api/client'
import {
  changeKindLabel,
  componentLabel,
  defaultSelection,
  diffLines,
  hasExternallyManaged,
  isRepreviewError,
  statusChip,
  unselectableReason,
  visibleComponents,
  writableChanges,
  type DiffLineKind,
} from '../lib/diff'

/** Shown above a preview that replaced a stale one after Apply found changed files (IN-5). */
export const REPREVIEW_NOTICE = 'Files changed since preview — showing a fresh preview'

const DIFF_LINE_COLOR: Record<DiffLineKind, string> = {
  add: 'success.main',
  remove: 'error.main',
  hunk: 'text.disabled',
  header: 'text.secondary',
  context: 'text.primary',
}

const DIFF_LINE_BG: Record<DiffLineKind, string> = {
  add: 'rgba(46, 160, 67, 0.12)',
  remove: 'rgba(248, 81, 73, 0.12)',
  hunk: 'transparent',
  header: 'transparent',
  context: 'transparent',
}

interface PreviewState {
  targetName: string
  request: InstallerPlanRequest
  plan: InstallerPlan
  /** Set when this preview replaced one Apply rejected as stale. */
  notice: string
}

interface InstallerSectionProps {
  /** Changes whenever App reloads settings, so statuses refetch alongside the Settings tab. */
  refreshKey?: unknown
}

/**
 * Settings → Agent integration (W9): per-target component status from disk, preview of the exact
 * per-file diff before any write, apply with automatic re-preview when files changed, and backups.
 */
export const InstallerSection = ({ refreshKey }: InstallerSectionProps) => {
  const { showToast } = useToast()
  const [overview, setOverview] = useState<InstallerOverview | null>(null)
  const [loadError, setLoadError] = useState('')
  const [selection, setSelection] = useState<Partial<Record<InstallerTargetId, InstallerComponentId[]>>>({})
  const [replaceExternal, setReplaceExternal] = useState<Partial<Record<InstallerTargetId, boolean>>>({})
  const [busyTarget, setBusyTarget] = useState<InstallerTargetId | ''>('')
  const [preview, setPreview] = useState<PreviewState | null>(null)
  const [applying, setApplying] = useState(false)
  const [backupsKey, setBackupsKey] = useState(0)

  const load = useCallback(async () => {
    try {
      setOverview(await api.installer())
      setLoadError('')
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load, refreshKey])

  // A refresh can make a checked component unselectable (now covered, or hooks turned off).
  const selected = (t: InstallerTarget): InstallerComponentId[] => {
    const hooksEnabled = overview?.hooks_enabled ?? false
    const selectable = defaultSelection(t, hooksEnabled)
    return (selection[t.id] ?? selectable).filter((c) => selectable.includes(c))
  }

  const toggleComponent = (t: InstallerTarget, component: InstallerComponentId, checked: boolean) => {
    const current = selected(t)
    const next = checked ? [...current, component] : current.filter((c) => c !== component)
    setSelection((prev) => ({ ...prev, [t.id]: next }))
  }

  const openPreview = async (t: InstallerTarget, action: InstallerAction) => {
    const components = selected(t)
    const request: InstallerPlanRequest = {
      targets: [t.id],
      components,
      action,
      replace_external: hasExternallyManaged(t, components) && !!replaceExternal[t.id],
    }
    setBusyTarget(t.id)
    try {
      const plan = await api.installerPreview(request)
      setPreview({ targetName: t.name, request, plan, notice: '' })
    } catch (e) {
      showToast(e instanceof Error ? e.message : String(e), 'error')
    } finally {
      setBusyTarget('')
    }
  }

  const apply = async () => {
    if (!preview) return
    setApplying(true)
    try {
      const res = await api.installerApply(preview.plan.plan_id)
      setPreview(null)
      const n = res.written.length
      showToast(`${preview.request.action === 'install' ? 'Installed' : 'Uninstalled'}: ${n} file${n === 1 ? '' : 's'} written`, 'success')
      setBackupsKey((k) => k + 1)
      void load()
    } catch (e) {
      if (!isRepreviewError(e)) {
        showToast(e instanceof Error ? e.message : String(e), 'error')
        return
      }
      await repreview(preview)
    } finally {
      setApplying(false)
    }
  }

  // The plan was stale, expired, or unknown and nothing was written: preview again and say why.
  const repreview = async (stale: PreviewState) => {
    try {
      const plan = await api.installerPreview(stale.request)
      setPreview({ ...stale, plan, notice: REPREVIEW_NOTICE })
    } catch (e) {
      setPreview(null)
      showToast(e instanceof Error ? e.message : String(e), 'error')
    }
    void load()
  }

  return (
    <Card variant="outlined" id="settings-agents" sx={{ mb: 2, scrollMarginTop: { xs: 120, md: 120 } }}>
      <CardContent>
        <Typography variant="subtitle1" gutterBottom>
          Agent integration
        </Typography>
        <Typography variant="caption" color="text.secondary" component="p" sx={{ mb: 1.5 }}>
          Registers ast-context-cache with each agent host. Every change is previewed as a diff, merged into your
          files, and backed up before it is written.
        </Typography>
        {loadError && (
          <Alert severity="error" sx={{ mb: 1.5 }}>
            Installer unavailable: {loadError}
          </Alert>
        )}
        {!overview && !loadError && <Typography color="text.secondary">Loading agent hosts…</Typography>}
        {overview && (
          <>
            <LegacyWarnings warnings={overview.legacy_warnings} />
            <Box sx={{ display: 'grid', gap: 1.5, gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' } }}>
              {overview.targets.map((t) => (
                <TargetCard
                  key={t.id}
                  target={t}
                  hooksEnabled={overview.hooks_enabled}
                  selected={selected(t)}
                  replaceExternal={!!replaceExternal[t.id]}
                  busy={busyTarget === t.id}
                  disabled={busyTarget !== '' || applying}
                  onToggle={(c, checked) => toggleComponent(t, c, checked)}
                  onReplaceExternal={(v) => setReplaceExternal((prev) => ({ ...prev, [t.id]: v }))}
                  onPreview={(action) => void openPreview(t, action)}
                />
              ))}
            </Box>
            <BackupsAccordion refreshKey={backupsKey} onRestored={() => void load()} />
          </>
        )}
      </CardContent>
      <InstallerPreviewDialog
        open={!!preview}
        targetName={preview?.targetName ?? ''}
        plan={preview?.plan ?? null}
        notice={preview?.notice ?? ''}
        applying={applying}
        onApply={() => void apply()}
        onClose={() => setPreview(null)}
      />
    </Card>
  )
}

const LegacyWarnings = ({ warnings }: { warnings: string[] }) => {
  if (!warnings?.length) return null
  return (
    <Alert severity="warning" sx={{ mb: 1.5 }} data-testid="installer-legacy-warnings">
      <AlertTitle>Pre-4.0 install records</AlertTitle>
      <Box component="ul" sx={{ m: 0, pl: 2 }}>
        {warnings.map((w) => (
          <li key={w}>
            <Typography variant="body2">{w}</Typography>
          </li>
        ))}
      </Box>
    </Alert>
  )
}

interface TargetCardProps {
  target: InstallerTarget
  hooksEnabled: boolean
  selected: InstallerComponentId[]
  replaceExternal: boolean
  busy: boolean
  disabled: boolean
  onToggle: (component: InstallerComponentId, checked: boolean) => void
  onReplaceExternal: (v: boolean) => void
  onPreview: (action: InstallerAction) => void
}

const TargetCard = ({
  target,
  hooksEnabled,
  selected,
  replaceExternal,
  busy,
  disabled,
  onToggle,
  onReplaceExternal,
  onPreview,
}: TargetCardProps) => {
  const components = visibleComponents(target, hooksEnabled)
  const anySelectable = components.some((c) => unselectableReason(c) === '')
  const external = hasExternallyManaged(target, selected)
  return (
    <Box data-testid={`installer-target-${target.id}`} sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 1, p: 1.5 }}>
      <Typography sx={{ fontWeight: 600, mb: 0.5 }}>{target.name}</Typography>
      <Stack spacing={0.5}>
        {components.map((c) => {
          const reason = unselectableReason(c)
          const chip = statusChip(c.status)
          return (
            <Box key={c.component}>
              <Stack direction="row" sx={{ alignItems: 'center', gap: 1 }}>
                <Tooltip title={reason} disableHoverListener={!reason} disableFocusListener={!reason}>
                  {/* A disabled checkbox fires no pointer events, so the span carries the tooltip. */}
                  <span>
                    <FormControlLabel
                      sx={{ mr: 0 }}
                      control={
                        <Checkbox
                          size="small"
                          checked={!reason && selected.includes(c.component)}
                          disabled={!!reason || disabled}
                          onChange={(_e, checked) => onToggle(c.component, checked)}
                        />
                      }
                      label={<Typography variant="body2">{componentLabel(c.component)}</Typography>}
                    />
                  </span>
                </Tooltip>
                <Box sx={{ flex: 1 }} />
                <Tooltip title={c.status_reason ?? ''} disableHoverListener={!c.status_reason}>
                  <Chip size="small" label={chip.label} color={chip.color} variant={chip.color === 'default' ? 'outlined' : 'filled'} />
                </Tooltip>
              </Stack>
              {/* Inline as well as in the tooltip: touch devices never hover. */}
              {(reason || c.status_reason) && (
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', ml: 4, overflowWrap: 'anywhere' }}>
                  {reason || c.status_reason}
                </Typography>
              )}
              {!reason && c.path && (
                <Typography variant="caption" color="text.disabled" sx={{ display: 'block', ml: 4, fontFamily: 'monospace', overflowWrap: 'anywhere' }}>
                  {c.path}
                </Typography>
              )}
            </Box>
          )
        })}
      </Stack>
      {external && (
        <FormControlLabel
          sx={{ mt: 0.5 }}
          control={<Checkbox size="small" checked={replaceExternal} disabled={disabled} onChange={(_e, v) => onReplaceExternal(v)} />}
          label={<Typography variant="caption">Replace externally managed paths (backed up first)</Typography>}
        />
      )}
      {anySelectable && (
        <Stack direction="row" spacing={1} sx={{ mt: 1, alignItems: 'center' }}>
          <Button size="small" variant="contained" disabled={disabled || selected.length === 0} onClick={() => onPreview('install')}>
            Install…
          </Button>
          <Button size="small" variant="outlined" disabled={disabled || selected.length === 0} onClick={() => onPreview('uninstall')}>
            Uninstall…
          </Button>
          {busy && <CircularProgress size={16} />}
        </Stack>
      )}
    </Box>
  )
}

interface InstallerPreviewDialogProps {
  open: boolean
  targetName: string
  plan: InstallerPlan | null
  notice: string
  applying: boolean
  onApply: () => void
  onClose: () => void
}

/** The per-file diff a plan would write; Apply sends only its plan_id (IN-5). */
export const InstallerPreviewDialog = ({ open, targetName, plan, notice, applying, onApply, onClose }: InstallerPreviewDialogProps) => {
  const writes = plan ? writableChanges(plan.changes) : []
  const title = plan ? `${plan.action === 'install' ? 'Install' : 'Uninstall'} preview — ${targetName}` : 'Preview'
  return (
    <Dialog open={open} onClose={applying ? undefined : onClose} maxWidth="md" fullWidth aria-labelledby="installer-preview-title">
      <DialogTitle id="installer-preview-title">{title}</DialogTitle>
      <DialogContent dividers>
        {plan && (
          <Stack spacing={1.5}>
            {notice && <Alert severity="info">{notice}</Alert>}
            {plan.errors?.map((e) => (
              <Alert key={`${e.target}-${e.component}`} severity="error">
                {componentLabel(e.component)}: {e.message}. Nothing is written for this host.
              </Alert>
            ))}
            {plan.warnings?.map((w) => (
              <Alert key={w} severity="warning">
                {w}
              </Alert>
            ))}
            {writes.length === 0 && !plan.errors?.length && <Alert severity="info">Nothing to change: already up to date.</Alert>}
            {/* A pathless change only carries an aborted target's reason, already shown above. */}
            {plan.changes
              .filter((c) => c.path !== '')
              .map((c, i) => (
                <ChangeView key={`${c.path}-${c.component}-${i}`} change={c} />
              ))}
          </Stack>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} disabled={applying}>
          Cancel
        </Button>
        <Button variant="contained" onClick={onApply} disabled={applying || writes.length === 0}>
          {applyLabel(applying, writes.length)}
        </Button>
      </DialogActions>
    </Dialog>
  )
}

const applyLabel = (applying: boolean, writes: number): string => {
  if (applying) return 'Applying…'
  if (writes === 0) return 'Apply'
  return `Apply ${writes} change${writes === 1 ? '' : 's'}`
}

const ChangeView = ({ change }: { change: InstallerFileChange }) => (
  <Box data-testid="installer-change" sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 1, overflow: 'hidden' }}>
    <Stack direction="row" sx={{ alignItems: 'center', gap: 1, px: 1, py: 0.75, bgcolor: 'action.hover', flexWrap: 'wrap' }}>
      <Chip size="small" label={changeKindLabel(change.kind)} variant="outlined" color={change.skipped ? 'default' : 'primary'} />
      <Chip size="small" label={componentLabel(change.component)} variant="outlined" />
      <Typography variant="body2" sx={{ fontFamily: 'monospace', overflowWrap: 'anywhere', flex: 1, minWidth: 0 }}>
        {change.path}
      </Typography>
    </Stack>
    {change.skipped && (
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', px: 1, py: 0.5 }}>
        Skipped{change.reason ? `: ${change.reason}` : ''}
      </Typography>
    )}
    {!change.skipped && change.diff && <DiffView diff={change.diff} />}
  </Box>
)

const DiffView = ({ diff }: { diff: string }) => (
  <Box
    component="pre"
    sx={{ m: 0, py: 0.5, fontFamily: 'monospace', fontSize: 12, lineHeight: 1.5, overflowX: 'auto', maxHeight: 360 }}
  >
    {diffLines(diff).map((line, i) => (
      <Box
        key={i}
        component="span"
        data-diff-kind={line.kind}
        sx={{ display: 'block', px: 1, color: DIFF_LINE_COLOR[line.kind], bgcolor: DIFF_LINE_BG[line.kind], whiteSpace: 'pre' }}
      >
        {line.text || ' '}
      </Box>
    ))}
  </Box>
)

interface BackupsAccordionProps {
  /** Bumped after an apply so an open list picks up the new backups. */
  refreshKey: number
  onRestored: () => void
}

const BackupsAccordion = ({ refreshKey, onRestored }: BackupsAccordionProps) => {
  const { showToast } = useToast()
  const [expanded, setExpanded] = useState(false)
  const [backups, setBackups] = useState<InstallerBackup[] | null>(null)
  const [loadError, setLoadError] = useState('')
  const [confirm, setConfirm] = useState<InstallerBackup | null>(null)
  const [restoring, setRestoring] = useState(false)

  const load = useCallback(async () => {
    try {
      const res = await api.installerBackups()
      setBackups(res.backups ?? [])
      setLoadError('')
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  useEffect(() => {
    if (expanded) void load()
  }, [expanded, load, refreshKey])

  const restore = async (b: InstallerBackup) => {
    setRestoring(true)
    try {
      await api.installerRestore(b.id)
      showToast(`Restored ${b.path}`, 'success')
      setConfirm(null)
      void load()
      onRestored()
    } catch (e) {
      showToast(e instanceof Error ? e.message : String(e), 'error')
    } finally {
      setRestoring(false)
    }
  }

  return (
    <Accordion
      disableGutters
      variant="outlined"
      expanded={expanded}
      onChange={(_e, v) => setExpanded(v)}
      sx={{ mt: 2, '&::before': { display: 'none' } }}
    >
      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
        <Typography variant="body2" sx={{ fontWeight: 500 }}>
          Backups{backups ? ` (${backups.length})` : ''}
        </Typography>
      </AccordionSummary>
      <AccordionDetails>
        {loadError && <Alert severity="error">Backups unavailable: {loadError}</Alert>}
        {!backups && !loadError && <Typography color="text.secondary">Loading backups…</Typography>}
        {backups?.length === 0 && (
          <Typography variant="body2" color="text.secondary">
            No backups yet. One is taken of every file before the installer changes it.
          </Typography>
        )}
        <Stack spacing={1}>
          {backups?.map((b) => (
            <Stack key={b.id} direction="row" sx={{ alignItems: 'center', gap: 1 }} data-testid="installer-backup">
              <Box sx={{ flex: 1, minWidth: 0 }}>
                <Typography variant="body2" sx={{ fontFamily: 'monospace', overflowWrap: 'anywhere' }}>
                  {b.path}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {new Date(b.created_at).toLocaleString()} · {b.symlink ? 'symlink' : formatBytes(b.size)}
                </Typography>
              </Box>
              <Button size="small" variant="outlined" onClick={() => setConfirm(b)} disabled={restoring}>
                Restore
              </Button>
            </Stack>
          ))}
        </Stack>
      </AccordionDetails>
      <Dialog open={!!confirm} onClose={restoring ? undefined : () => setConfirm(null)} aria-labelledby="installer-restore-title">
        <DialogTitle id="installer-restore-title">Restore backup?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Write the backup from {confirm ? new Date(confirm.created_at).toLocaleString() : ''} back to{' '}
            <Box component="span" sx={{ fontFamily: 'monospace' }}>
              {confirm?.path}
            </Box>
            ? The current file is backed up first, so this can be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirm(null)} disabled={restoring}>
            Cancel
          </Button>
          <Button variant="contained" onClick={() => confirm && void restore(confirm)} disabled={restoring}>
            {restoring ? 'Restoring…' : 'Restore'}
          </Button>
        </DialogActions>
      </Dialog>
    </Accordion>
  )
}
