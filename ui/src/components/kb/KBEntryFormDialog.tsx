import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Checkbox,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  FormControlLabel,
  Grid,
  MenuItem,
  Stack,
  Switch,
  TextField,
  Typography,
} from '@mui/material'

import type { KBCreateInput, KBEntry, KBKind, KBScope, KBUpdateInput } from '../../api/types'
import { api } from '../../api/client'
import { useToast } from '../../context/ToastContext'
import {
  KB_KINDS,
  KB_KIND_LABELS,
  KB_MAX_BODY_BYTES,
  KB_SCOPES,
  KB_SCOPE_LABELS,
  KB_SEVERITIES,
  kbKindChipSx,
  parseTagsInput,
  utf8ByteLength,
} from './kbHelpers'

export interface KBEntryFormDialogProps {
  open: boolean
  mode: 'create' | 'edit'
  initialEntry?: KBEntry | null
  defaultKind?: KBKind
  defaultScope?: KBScope
  projectId?: string
  onClose: () => void
  onSaved: (entry: KBEntry) => void
}

const MONO = '"JetBrains Mono", ui-monospace, monospace'

export function KBEntryFormDialog({
  open,
  mode,
  initialEntry,
  defaultKind = 'fact',
  defaultScope,
  projectId,
  onClose,
  onSaved,
}: KBEntryFormDialogProps) {
  const { showToast } = useToast()

  const [kind, setKind] = useState<KBKind>(defaultKind)
  const [scope, setScope] = useState<KBScope>(defaultScope ?? (projectId ? 'project' : 'global'))
  const [projectPath, setProjectPath] = useState(projectId ?? '')
  const [sessionId, setSessionId] = useState('')
  const [title, setTitle] = useState('')
  const [subject, setSubject] = useState('')
  const [predicate, setPredicate] = useState('')
  const [objectVal, setObjectVal] = useState('')
  const [rule, setRule] = useState('')
  const [target, setTarget] = useState('')
  const [beforeValue, setBeforeValue] = useState('')
  const [afterValue, setAfterValue] = useState('')
  const [reason, setReason] = useState('')
  const [category, setCategory] = useState('')
  const [severity, setSeverity] = useState<string>('info')
  const [occurredAt, setOccurredAt] = useState('')
  const [tagsRaw, setTagsRaw] = useState('')
  const [sourceRef, setSourceRef] = useState('')
  const [actor, setActor] = useState('')
  const [body, setBody] = useState('')
  const [verified, setVerified] = useState(false)
  const [newVersion, setNewVersion] = useState(false)

  const [saving, setSaving] = useState(false)
  const [errorMsg, setErrorMsg] = useState('')

  const prevOpenRef = useRef(false)
  const initializedRefId = useRef<string | null>(null)

  useEffect(() => {
    const justOpened = open && !prevOpenRef.current
    const entryRef = initialEntry?.ref ?? null
    const entrySwitched = open && mode === 'edit' && entryRef !== initializedRefId.current

    if (justOpened || entrySwitched) {
      setErrorMsg('')
      setSaving(false)
      if (mode === 'edit' && initialEntry) {
        initializedRefId.current = initialEntry.ref
        setKind(initialEntry.kind)
        setScope(initialEntry.scope)
        setProjectPath(initialEntry.project_path ?? projectId ?? '')
        setSessionId(initialEntry.session_id ?? '')
        setTitle(initialEntry.title ?? '')
        setSubject(initialEntry.subject ?? '')
        setPredicate(initialEntry.predicate ?? '')
        setObjectVal(initialEntry.object ?? '')
        setRule(initialEntry.rule ?? '')
        setTarget(initialEntry.target ?? '')
        setBeforeValue(initialEntry.before_value ?? '')
        setAfterValue(initialEntry.after_value ?? '')
        setReason(initialEntry.reason ?? '')
        setCategory(initialEntry.category ?? '')
        setSeverity(initialEntry.severity || (initialEntry.kind === 'event' ? 'info' : ''))
        setOccurredAt(initialEntry.occurred_at ?? '')
        setTagsRaw((initialEntry.tags ?? []).join(', '))
        setSourceRef(initialEntry.source_ref ?? '')
        setActor(initialEntry.actor ?? '')
        setBody(initialEntry.body ?? '')
        setVerified(Boolean(initialEntry.verified))
        setNewVersion(false)
      } else {
        initializedRefId.current = null
        const resolvedKind = defaultKind
        const resolvedScope = defaultScope ?? (projectId ? 'project' : 'global')
        setKind(resolvedKind)
        setScope(resolvedScope)
        setProjectPath(projectId ?? '')
        setSessionId('')
        setTitle('')
        setSubject(resolvedKind === 'preference' ? 'user' : '')
        setPredicate(resolvedKind === 'preference' ? 'prefers' : resolvedKind === 'fact' ? 'is' : '')
        setObjectVal('')
        setRule('')
        setTarget('')
        setBeforeValue('')
        setAfterValue('')
        setReason('')
        setCategory('')
        setSeverity(resolvedKind === 'event' ? 'info' : '')
        setOccurredAt('')
        setTagsRaw('')
        setSourceRef('')
        setActor('')
        setBody('')
        setVerified(false)
        setNewVersion(false)
      }
    }
    if (!open) {
      initializedRefId.current = null
    }
    prevOpenRef.current = open
  }, [open, mode, initialEntry, defaultKind, defaultScope, projectId])

  const bodyBytes = useMemo(() => utf8ByteLength(body), [body])
  const bodyTooLarge = bodyBytes > KB_MAX_BODY_BYTES

  const handleKindChange = (nextKind: KBKind) => {
    setKind(nextKind)
    if (mode === 'create') {
      if (nextKind === 'preference') {
        if (!subject.trim()) setSubject('user')
        if (!predicate.trim() || predicate === 'is') setPredicate('prefers')
      } else if (nextKind === 'fact') {
        if (subject === 'user') setSubject('')
        if (!predicate.trim() || predicate === 'prefers') setPredicate('is')
      }
      if (nextKind === 'event' && !severity) {
        setSeverity('info')
      }
    }
  }

  const validate = (): string | null => {
    if (bodyTooLarge) {
      return `Body exceeds the 16 KB (${KB_MAX_BODY_BYTES.toLocaleString()} bytes) limit (${bodyBytes.toLocaleString()} bytes). Store large payloads with store_context and link the ctx_* ref in Source ref.`
    }
    if (mode === 'create') {
      if (scope === 'project' && !projectPath.trim()) {
        return 'Project path is required for project scope.'
      }
      if (scope === 'session' && !sessionId.trim()) {
        return 'Session ID is required for session scope.'
      }
    }
    switch (kind) {
      case 'fact':
        if (!subject.trim() || !objectVal.trim()) {
          return 'Subject and object are required for a fact.'
        }
        break
      case 'procedure':
        if (!rule.trim()) {
          return 'Rule is required for a procedure.'
        }
        break
      case 'preference':
        if (!objectVal.trim() && !rule.trim()) {
          return 'Object (preferred value) is required for a preference.'
        }
        break
      case 'event':
        if (!title.trim() && (!subject.trim() || !objectVal.trim())) {
          return 'Title (or subject and object) is required for an event.'
        }
        break
      case 'discovery':
        if (!title.trim() && (!subject.trim() || !objectVal.trim())) {
          return 'Title (or subject and object) is required for a discovery.'
        }
        break
      case 'config_change':
        if (!target.trim()) {
          return 'Target is required for a config change.'
        }
        if (!afterValue.trim() && !body.trim()) {
          return 'After value or Markdown body is required for a config change.'
        }
        break
    }
    return null
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const validationErr = validate()
    if (validationErr) {
      setErrorMsg(validationErr)
      return
    }

    setSaving(true)
    setErrorMsg('')
    try {
      const tags = parseTagsInput(tagsRaw)
      if (mode === 'create') {
        const payload: KBCreateInput = {
          kind,
          scope,
          project_path: scope === 'project' ? projectPath.trim() : projectPath.trim() || undefined,
          session_id: scope === 'session' ? sessionId.trim() : sessionId.trim() || undefined,
          title: title.trim() || undefined,
          subject: subject.trim() || undefined,
          predicate: predicate.trim() || undefined,
          object: objectVal.trim() || undefined,
          rule: rule.trim() || undefined,
          target: target.trim() || undefined,
          before_value: beforeValue.trim() || undefined,
          after_value: afterValue.trim() || undefined,
          reason: reason.trim() || undefined,
          category: category.trim() || undefined,
          severity: severity.trim() || undefined,
          occurred_at: occurredAt.trim() || undefined,
          tags,
          source_ref: sourceRef.trim() || undefined,
          actor: actor.trim() || undefined,
          body: body.trim() || undefined,
          verified,
        }
        const res = await api.kbCreateEntry(payload)
        if (res.warnings?.length) {
          showToast(res.warnings.join('; '), 'warning')
        }
        showToast(`${KB_KIND_LABELS[res.entry.kind]} created (${res.entry.ref})`, 'success')
        onSaved(res.entry)
        onClose()
      } else if (initialEntry) {
        const payload: KBUpdateInput = {
          ref: initialEntry.ref,
          title: title.trim(),
          body: body.trim(),
          subject: subject.trim(),
          predicate: predicate.trim(),
          object: objectVal.trim(),
          rule: rule.trim(),
          target: target.trim(),
          before_value: beforeValue.trim(),
          after_value: afterValue.trim(),
          reason: reason.trim(),
          category: category.trim(),
          severity: severity.trim(),
          occurred_at: occurredAt.trim(),
          tags,
          actor: actor.trim(),
          verified,
          new_version: supportsManualVersioning ? newVersion : undefined,
        }
        const res = await api.kbUpdateEntry(payload)
        if (res.warnings?.length) {
          showToast(res.warnings.join('; '), 'warning')
        }
        const versioned = res.entry.ref !== initialEntry.ref
        showToast(
          versioned
            ? `Saved new version ${res.entry.ref} (supersedes ${initialEntry.ref})`
            : `Updated ${res.entry.ref}`,
          'success',
        )
        onSaved(res.entry)
        onClose()
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to save KB entry'
      setErrorMsg(msg)
      showToast(msg, 'error')
    } finally {
      setSaving(false)
    }
  }

  const supportsManualVersioning =
    kind === 'discovery' || kind === 'event' || kind === 'fact' || kind === 'procedure'
  const isAutoVersioned = kind === 'preference' || kind === 'config_change'

  if (!open) return null

  return (
    <Dialog
      open={open}
      onClose={() => !saving && onClose()}
      maxWidth="md"
      fullWidth
      aria-labelledby="kb-entry-form-title"
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
      <form onSubmit={(e) => void handleSubmit(e)} noValidate>
        <DialogTitle id="kb-entry-form-title" sx={{ pb: 1 }}>
          <Stack direction="row" spacing={1} useFlexGap sx={{ alignItems: 'center', flexWrap: 'wrap' }}>
            <Typography variant="h6" component="span" sx={{ fontSize: 17, fontWeight: 600 }}>
              {mode === 'create' ? 'New Knowledge Base Entry' : 'Edit Knowledge Base Entry'}
            </Typography>
            {mode === 'edit' && initialEntry && (
              <>
                <Chip
                  size="small"
                  variant="outlined"
                  label={KB_KIND_LABELS[initialEntry.kind]}
                  sx={kbKindChipSx(initialEntry.kind)}
                />
                <Chip size="small" variant="outlined" label={KB_SCOPE_LABELS[initialEntry.scope]} />
                <Typography component="span" sx={{ fontFamily: MONO, fontSize: 12, color: 'text.secondary' }}>
                  {initialEntry.ref}
                </Typography>
              </>
            )}
          </Stack>
        </DialogTitle>

        <DialogContent dividers sx={{ borderColor: '#30363d' }}>
          <Stack spacing={2}>
            {errorMsg && (
              <Alert severity="error" onClose={() => setErrorMsg('')}>
                {errorMsg}
              </Alert>
            )}

            {mode === 'edit' && isAutoVersioned && (
              <Alert severity="info">
                Changing substantive values (
                {kind === 'preference' ? 'subject, predicate, or object' : 'target, before value, or after value'}
                ) automatically creates a linked new version and supersedes the prior entry. Metadata-only edits (title,
                body, tags, reason, verified) update in place.
              </Alert>
            )}

            {mode === 'create' && (
              <Grid container spacing={1.5}>
                <Grid size={{ xs: 12, sm: 6 }}>
                  <TextField
                    select
                    fullWidth
                    size="small"
                    label="Kind"
                    value={kind}
                    onChange={(e) => handleKindChange(e.target.value as KBKind)}
                  >
                    {KB_KINDS.map((k) => (
                      <MenuItem key={k} value={k}>
                        {KB_KIND_LABELS[k]}
                      </MenuItem>
                    ))}
                  </TextField>
                </Grid>
                <Grid size={{ xs: 12, sm: 6 }}>
                  <TextField
                    select
                    fullWidth
                    size="small"
                    label="Scope"
                    value={scope}
                    onChange={(e) => setScope(e.target.value as KBScope)}
                  >
                    {KB_SCOPES.map((s) => (
                      <MenuItem key={s} value={s}>
                        {KB_SCOPE_LABELS[s]}
                      </MenuItem>
                    ))}
                  </TextField>
                </Grid>
                {scope === 'project' && (
                  <Grid size={{ xs: 12 }}>
                    <TextField
                      fullWidth
                      required
                      size="small"
                      label="Project path"
                      value={projectPath}
                      onChange={(e) => setProjectPath(e.target.value)}
                      placeholder="/abs/path/to/repo"
                      helperText="Absolute repository path for project-scoped entries"
                      slotProps={{ htmlInput: { style: { fontFamily: MONO, fontSize: 12.5 } } }}
                    />
                  </Grid>
                )}
                {scope === 'session' && (
                  <Grid size={{ xs: 12 }}>
                    <TextField
                      fullWidth
                      required
                      size="small"
                      label="Session ID"
                      value={sessionId}
                      onChange={(e) => setSessionId(e.target.value)}
                      placeholder="session-id"
                      slotProps={{ htmlInput: { style: { fontFamily: MONO, fontSize: 12.5 } } }}
                    />
                  </Grid>
                )}
              </Grid>
            )}

            {(kind === 'event' || kind === 'discovery' || kind === 'config_change' || kind === 'fact' || kind === 'procedure' || kind === 'preference') && (
              <TextField
                fullWidth
                size="small"
                required={kind === 'event' || kind === 'discovery'}
                label="Title"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder={
                  kind === 'event'
                    ? 'e.g. Deploy v2.4.0 to production'
                    : kind === 'discovery'
                      ? 'e.g. SQLite WAL checkpoint stalls under heavy embed queue'
                      : 'Optional short headline'
                }
              />
            )}

            {(kind === 'fact' || kind === 'preference') && (
              <Grid container spacing={1.5}>
                <Grid size={{ xs: 12, sm: 4 }}>
                  <TextField
                    fullWidth
                    required={kind === 'fact'}
                    size="small"
                    label="Subject"
                    value={subject}
                    onChange={(e) => setSubject(e.target.value)}
                    placeholder={kind === 'preference' ? 'user' : 'e.g. auth-service'}
                  />
                </Grid>
                <Grid size={{ xs: 12, sm: 3 }}>
                  <TextField
                    fullWidth
                    size="small"
                    label="Predicate"
                    value={predicate}
                    onChange={(e) => setPredicate(e.target.value)}
                    placeholder={kind === 'preference' ? 'prefers' : 'is'}
                  />
                </Grid>
                <Grid size={{ xs: 12, sm: 5 }}>
                  <TextField
                    fullWidth
                    required
                    size="small"
                    label="Object"
                    value={objectVal}
                    onChange={(e) => setObjectVal(e.target.value)}
                    placeholder={kind === 'preference' ? 'gofumpt formatting' : 'port 8080'}
                  />
                </Grid>
              </Grid>
            )}

            {kind === 'procedure' && (
              <TextField
                fullWidth
                required
                multiline
                minRows={2}
                size="small"
                label="Rule"
                value={rule}
                onChange={(e) => setRule(e.target.value)}
                placeholder="Always run `go test ./...` before opening a PR."
              />
            )}

            {kind === 'config_change' && (
              <Stack spacing={1.5}>
                <TextField
                  fullWidth
                  required
                  size="small"
                  label="Target"
                  value={target}
                  onChange={(e) => setTarget(e.target.value)}
                  placeholder="e.g. AST_EMBED_BACKEND or nginx.conf"
                  slotProps={{ htmlInput: { style: { fontFamily: MONO, fontSize: 12.5 } } }}
                />
                <Grid container spacing={1.5}>
                  <Grid size={{ xs: 12, sm: 6 }}>
                    <TextField
                      fullWidth
                      multiline
                      minRows={2}
                      size="small"
                      label="Before value"
                      value={beforeValue}
                      onChange={(e) => setBeforeValue(e.target.value)}
                      placeholder="Previous value"
                      slotProps={{ htmlInput: { style: { fontFamily: MONO, fontSize: 12 } } }}
                    />
                  </Grid>
                  <Grid size={{ xs: 12, sm: 6 }}>
                    <TextField
                      fullWidth
                      multiline
                      minRows={2}
                      size="small"
                      label="After value"
                      value={afterValue}
                      onChange={(e) => setAfterValue(e.target.value)}
                      placeholder="New value (or provide Body below)"
                      slotProps={{ htmlInput: { style: { fontFamily: MONO, fontSize: 12 } } }}
                    />
                  </Grid>
                </Grid>
                <TextField
                  fullWidth
                  size="small"
                  label="Reason"
                  value={reason}
                  onChange={(e) => setReason(e.target.value)}
                  placeholder="Why this configuration was changed"
                />
              </Stack>
            )}

            {(kind === 'event' || kind === 'config_change' || kind === 'discovery') && (
              <Grid container spacing={1.5}>
                <Grid size={{ xs: 12, sm: 4 }}>
                  <TextField
                    fullWidth
                    size="small"
                    label="Category"
                    value={category}
                    onChange={(e) => setCategory(e.target.value)}
                    placeholder="e.g. deploy, infra, perf"
                  />
                </Grid>
                <Grid size={{ xs: 12, sm: 4 }}>
                  <TextField
                    select
                    fullWidth
                    size="small"
                    label="Severity"
                    value={severity}
                    onChange={(e) => setSeverity(e.target.value)}
                  >
                    <MenuItem value="">Default</MenuItem>
                    {KB_SEVERITIES.map((sev) => (
                      <MenuItem key={sev} value={sev}>
                        {sev}
                      </MenuItem>
                    ))}
                  </TextField>
                </Grid>
                <Grid size={{ xs: 12, sm: 4 }}>
                  <TextField
                    fullWidth
                    size="small"
                    label="Occurred at"
                    value={occurredAt}
                    onChange={(e) => setOccurredAt(e.target.value)}
                    placeholder="YYYY-MM-DD HH:MM:SS"
                    helperText="UTC or ISO-8601 (blank = now)"
                  />
                </Grid>
                {kind !== 'config_change' && (
                  <Grid size={{ xs: 12 }}>
                    <TextField
                      fullWidth
                      size="small"
                      label="Target"
                      value={target}
                      onChange={(e) => setTarget(e.target.value)}
                      placeholder="Affected service, host, file, or component"
                    />
                  </Grid>
                )}
              </Grid>
            )}

            <Box>
              <TextField
                fullWidth
                multiline
                minRows={4}
                maxRows={12}
                size="small"
                label="Body (Markdown)"
                value={body}
                onChange={(e) => setBody(e.target.value)}
                error={bodyTooLarge}
                placeholder="Detailed notes, context, snippets, or rationale (Markdown supported)..."
              />
              <Stack
                direction="row"
                spacing={1}
                sx={{ mt: 0.5, px: 0.5, justifyContent: 'space-between', alignItems: 'center' }}
              >
                <Typography variant="caption" color={bodyTooLarge ? 'error.main' : 'text.secondary'}>
                  {bodyTooLarge
                    ? 'Body exceeds 16 KB limit — store large payloads via store_context and link its ctx_* ref in Source ref.'
                    : 'Supports GitHub-flavored Markdown.'}
                </Typography>
                <Typography
                  variant="caption"
                  sx={{
                    fontFamily: MONO,
                    color: bodyTooLarge ? 'error.main' : bodyBytes > KB_MAX_BODY_BYTES * 0.85 ? 'warning.main' : 'text.secondary',
                    flexShrink: 0,
                  }}
                >
                  {bodyBytes.toLocaleString()} / {KB_MAX_BODY_BYTES.toLocaleString()} bytes
                </Typography>
              </Stack>
            </Box>

            <Grid container spacing={1.5}>
              <Grid size={{ xs: 12, sm: 6 }}>
                <TextField
                  fullWidth
                  size="small"
                  label="Tags (comma-separated)"
                  value={tagsRaw}
                  onChange={(e) => setTagsRaw(e.target.value)}
                  placeholder="auth, sqlite, release"
                />
              </Grid>
              <Grid size={{ xs: 12, sm: 3 }}>
                <TextField
                  fullWidth
                  size="small"
                  label="Actor"
                  value={actor}
                  onChange={(e) => setActor(e.target.value)}
                  placeholder="human"
                />
              </Grid>
              <Grid size={{ xs: 12, sm: 3 }}>
                <TextField
                  fullWidth
                  size="small"
                  label="Source ref"
                  value={sourceRef}
                  disabled={mode === 'edit'}
                  onChange={(e) => setSourceRef(e.target.value)}
                  placeholder="ctx_... or URL"
                  helperText={mode === 'edit' ? 'Immutable after creation' : undefined}
                  slotProps={{ htmlInput: { style: { fontFamily: MONO, fontSize: 12 } } }}
                />
              </Grid>
            </Grid>

            <Stack direction="row" spacing={2} useFlexGap sx={{ alignItems: 'center', flexWrap: 'wrap' }}>
              <FormControlLabel
                control={<Checkbox size="small" checked={verified} onChange={(e) => setVerified(e.target.checked)} />}
                label={<Typography variant="body2">Verified</Typography>}
              />
              {mode === 'edit' && supportsManualVersioning && (
                <FormControlLabel
                  control={
                    <Switch size="small" checked={newVersion} onChange={(e) => setNewVersion(e.target.checked)} />
                  }
                  label={
                    <Typography variant="body2">
                      Save as new version (supersede current entry)
                    </Typography>
                  }
                />
              )}
            </Stack>
          </Stack>
        </DialogContent>

        <DialogActions sx={{ px: 3, py: 1.5 }}>
          <Button onClick={onClose} disabled={saving}>
            Cancel
          </Button>
          <Button type="submit" variant="contained" disabled={saving || bodyTooLarge}>
            {saving ? 'Saving…' : mode === 'create' ? 'Create Entry' : newVersion ? 'Save New Version' : 'Save Changes'}
          </Button>
        </DialogActions>
      </form>
    </Dialog>
  )
}
