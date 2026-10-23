import { useCallback, useEffect, useRef, useState } from 'react'

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
  Stack,
  Switch,
  Tooltip,
  Typography,
} from '@mui/material'

import type { FlagSource, FlagState } from '../api/types'

import { useToast } from '../context/ToastContext'

import { api } from '../api/client'
import { buildFlagRows, needsToolListWarning, TOOL_LIST_WARNING, withFlagEnabled, type FlagRow } from '../lib/flags'

const SOURCE_COLOR: Record<FlagSource, 'default' | 'info' | 'warning'> = {
  default: 'default',
  setting: 'info',
  env: 'warning',
}

interface FeaturesSectionProps {
  /**
   * Changes whenever App reloads settings (on the WebSocket `settings` panel and after saves),
   * so the flags refetch alongside the rest of the Settings tab.
   */
  refreshKey?: unknown
}

/** Settings card with one switch per feature flag (FF-4). Env-locked flags are read-only. */
export const FeaturesSection = ({ refreshKey }: FeaturesSectionProps) => {
  const { showToast } = useToast()
  const [flags, setFlags] = useState<FlagState[] | null>(null)
  const [loadError, setLoadError] = useState('')
  const [pendingKey, setPendingKey] = useState('')
  // A toggle waiting on the tool-list confirmation (TS-4).
  const [confirm, setConfirm] = useState<{ key: string; enabled: boolean } | null>(null)
  // A refetch landing mid-toggle would overwrite the optimistic value with the pre-toggle state.
  const pendingRef = useRef('')

  const load = useCallback(async () => {
    try {
      const res = await api.flags()
      if (pendingRef.current) return
      setFlags(res.flags ?? [])
      setLoadError('')
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load, refreshKey])

  const toggle = async (key: string, enabled: boolean) => {
    if (!flags) return
    const previous = flags
    pendingRef.current = key
    setPendingKey(key)
    setFlags(withFlagEnabled(previous, key, enabled))
    try {
      const res = await api.setFlag(key, enabled)
      setFlags(res.flags ?? previous)
      showToast('Feature flag updated', 'success')
    } catch (e) {
      setFlags(previous)
      showToast(e instanceof Error ? e.message : String(e), 'error')
    } finally {
      pendingRef.current = ''
      setPendingKey('')
    }
  }

  const requestToggle = (key: string, enabled: boolean) => {
    if (needsToolListWarning(flags?.find((f) => f.key === key), enabled)) {
      setConfirm({ key, enabled })
      return
    }
    void toggle(key, enabled)
  }

  const confirmToggle = () => {
    if (!confirm) return
    setConfirm(null)
    void toggle(confirm.key, confirm.enabled)
  }

  return (
    <Card variant="outlined" id="settings-features" sx={{ mb: 2, scrollMarginTop: { xs: 120, md: 120 } }}>
      <CardContent>
        <Typography variant="subtitle1" gutterBottom>
          Features
        </Typography>
        <Typography variant="caption" color="text.secondary" component="p" sx={{ mb: 1.5 }}>
          Feature flags take effect immediately. A flag set by its environment variable is read-only here.
        </Typography>
        {loadError && (
          <Alert severity="error" sx={{ mb: 1.5 }}>
            Feature flags unavailable: {loadError}
          </Alert>
        )}
        {!flags && !loadError && <Typography color="text.secondary">Loading feature flags…</Typography>}
        {flags && (
          <Stack spacing={1}>
            {buildFlagRows(flags).map((row) => (
              <FlagRowView key={row.flag.key} row={row} pending={pendingKey !== ''} onToggle={requestToggle} />
            ))}
          </Stack>
        )}
      </CardContent>
      <Dialog open={confirm !== null} onClose={() => setConfirm(null)} aria-labelledby="flag-tool-list-title">
        <DialogTitle id="flag-tool-list-title">
          Turn {confirm?.enabled ? 'on' : 'off'}{' '}
          <Box component="span" sx={{ fontFamily: 'monospace' }}>
            {confirm?.key}
          </Box>
          ?
        </DialogTitle>
        <DialogContent>
          <DialogContentText>{TOOL_LIST_WARNING}</DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirm(null)}>Cancel</Button>
          <Button onClick={confirmToggle} variant="contained" data-testid="flag-tool-list-confirm">
            Change flag
          </Button>
        </DialogActions>
      </Dialog>
    </Card>
  )
}

interface FlagRowViewProps {
  row: FlagRow
  /** Any toggle in flight: one at a time, since a master toggle rewrites its children too. */
  pending: boolean
  onToggle: (key: string, enabled: boolean) => void
}

const FlagRowView = ({ row, pending, onToggle }: FlagRowViewProps) => {
  const { flag, parentKey, disabledReason, blockedByParent } = row
  const nested = parentKey !== ''
  return (
    <Box
      data-testid={`flag-row-${flag.key}`}
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 2,
        py: 0.5,
        ml: nested ? 3 : 0,
        pl: nested ? 2 : 0,
        borderLeft: nested ? '2px solid' : 'none',
        borderColor: 'divider',
        opacity: blockedByParent ? 0.6 : 1,
      }}
    >
      <Box sx={{ minWidth: 0, flex: 1 }}>
        <Typography variant="body2" sx={{ fontFamily: 'monospace', fontWeight: nested ? 400 : 600 }}>
          {flag.key}
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
          {flag.description}
        </Typography>
        {/* Shown inline as well as in the tooltip: touch devices never hover. */}
        {disabledReason && (
          <Typography variant="caption" color={flag.locked ? 'text.secondary' : 'warning.main'} sx={{ display: 'block' }}>
            {flag.locked ? `${disabledReason}; read-only here.` : `${disabledReason}; turn on ${parentKey} to change it.`}
          </Typography>
        )}
      </Box>
      <Chip label={flag.source} size="small" color={SOURCE_COLOR[flag.source] ?? 'default'} variant="outlined" />
      <Tooltip title={disabledReason} disableHoverListener={!disabledReason} disableFocusListener={!disabledReason}>
        {/* A disabled Switch fires no pointer events, so the span carries the tooltip. */}
        <span>
          <Switch
            checked={flag.enabled}
            disabled={!!disabledReason || pending}
            onChange={(_e, checked) => onToggle(flag.key, checked)}
            slotProps={{ input: { 'aria-label': flag.key } }}
          />
        </span>
      </Tooltip>
    </Box>
  )
}
