import { Box, IconButton, Stack, Tooltip, Typography } from '@mui/material'
import AddIcon from '@mui/icons-material/Add'
import RemoveIcon from '@mui/icons-material/Remove'
import { useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import { chartColors } from '../lib/chartColors'
import { workerPillCount } from '../lib/embedGauges'
import { useToast } from '../context/ToastContext'

export type WorkerPool = 'primary' | 'aux'

export function WorkerControls({
  workers,
  active,
  max,
  effective,
  live,
  pool = 'primary',
  walBadge,
  walTitle,
  onChange,
}: {
  workers: number
  active: number
  max: number
  effective?: number
  live?: number
  pool?: WorkerPool
  walBadge?: string
  walTitle?: string
  onChange?: () => void
}) {
  const { showToast } = useToast()
  const isAux = pool === 'aux'
  const { desired, adjust, syncing } = useOptimisticWorkerTarget({
    serverTarget: workers,
    max,
    send: (n) => (isAux ? api.setEmbedAuxWorkers(n) : api.setEmbedWorkers(n)),
    onSynced: () => onChange?.(),
    onError: (e) => showToast(e instanceof Error ? e.message : 'Worker adjust failed', 'error'),
  })
  const { dots, ellipsis } = workerPillCount(desired)
  const lit = Math.min(active, desired)
  const title = (() => {
    if (syncing) return `${isAux ? 'Aux workers' : 'Workers'}: applying ${desired} (currently ${workers})`
    if (workers === 0) return isAux ? 'Aux workers off — click + to enable' : 'Workers paused — click + to resume'
    if (effective != null && effective < workers) {
      return isAux
        ? `Aux workers: ${workers} target · ${effective} running (WAL throttled)`
        : `Workers: ${workers} target · ${effective} running (WAL throttled) · ${active} busy`
    }
    const draining = live != null ? Math.max(0, live - workers) : 0
    if (draining > 0) {
      return `${isAux ? 'Aux workers' : 'Workers'}: ${workers} target · ${active} busy · ${draining} draining`
    }
    return isAux ? `Aux workers: ${workers} enabled · ${active} busy` : `Workers: ${active} of ${workers} busy`
  })()
  return (
    <Stack direction="row" spacing={1} sx={{ alignItems: 'center', flexWrap: 'wrap' }} useFlexGap title={title}>
      <Stack direction="column" spacing={0}>
        <IconButton
          size="small"
          aria-label={isAux ? 'Increase aux workers' : 'Increase workers'}
          disabled={desired >= max}
          onClick={() => adjust(1)}
          sx={{ p: 0.25 }}
        >
          <AddIcon fontSize="small" />
        </IconButton>
        <IconButton
          size="small"
          aria-label={isAux ? 'Decrease aux workers' : 'Decrease workers'}
          disabled={desired <= 0}
          onClick={() => adjust(-1)}
          sx={{ p: 0.25 }}
        >
          <RemoveIcon fontSize="small" />
        </IconButton>
      </Stack>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, maxWidth: 140 }}>
        {Array.from({ length: dots }, (_, i) => (
          <Box
            key={i}
            sx={{
              width: 10,
              height: 10,
              borderRadius: '50%',
              bgcolor: i < lit ? (isAux ? chartColors.purple : chartColors.orange) : chartColors.track,
              border: '1px solid',
              borderColor: 'divider',
            }}
          />
        ))}
        {ellipsis && (
          <Typography variant="caption" color="text.secondary" sx={{ alignSelf: 'center' }}>
            …
          </Typography>
        )}
      </Box>
      <Typography
        variant="body2"
        sx={{ fontWeight: 600, fontFamily: 'ui-monospace, monospace', opacity: syncing ? 0.6 : 1, transition: 'opacity 0.15s' }}
      >
        {desired}
      </Typography>
      {walBadge && (
        <Tooltip title={walTitle || walBadge}>
          <Typography variant="caption" color="warning.main" sx={{ cursor: 'help' }}>
            {walBadge}
          </Typography>
        </Tooltip>
      )}
      {effective != null && effective < workers && !walBadge && (
        <Typography variant="caption" color="warning.main">
          WAL {effective}/{workers}
        </Typography>
      )}
    </Stack>
  )
}

const SEND_DEBOUNCE_MS = 200
// After a confirmed set, ignore server snapshots that disagree for this long — they
// come from reloads that started before the change landed and would flicker it back.
const STALE_SNAPSHOT_GRACE_MS = 1500

/**
 * Local worker target that updates on every click, with the server kept in sync in the
 * background: clicks are coalesced into one absolute "set to N" request (so rapid +/-
 * can't race each other as deltas), and clicks that land mid-request are sent as soon
 * as it finishes.
 */
function useOptimisticWorkerTarget({
  serverTarget,
  max,
  send,
  onSynced,
  onError,
}: {
  serverTarget: number
  max: number
  send: (n: number) => Promise<{ workers?: number }>
  onSynced: () => void
  onError: (e: unknown) => void
}) {
  const [desired, setDesired] = useState(serverTarget)
  const [syncing, setSyncing] = useState(false)
  const desiredRef = useRef(serverTarget)
  const serverRef = useRef(serverTarget)
  const inFlight = useRef(false)
  const timer = useRef<number | null>(null)
  const confirmedAt = useRef(0)
  const mounted = useRef(true)
  const cb = useRef({ send, onSynced, onError })
  cb.current = { send, onSynced, onError }
  serverRef.current = serverTarget

  const setLocal = (n: number) => {
    desiredRef.current = n
    if (mounted.current) setDesired(n)
  }

  useEffect(() => {
    if (inFlight.current || timer.current != null) return
    if (serverTarget !== desiredRef.current && Date.now() - confirmedAt.current < STALE_SNAPSHOT_GRACE_MS) return
    setLocal(serverTarget)
  }, [serverTarget])

  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      // Navigating away mid-debounce must not drop the click.
      if (timer.current != null) {
        window.clearTimeout(timer.current)
        timer.current = null
        cb.current.send(desiredRef.current).catch(() => {})
      }
    }
  }, [])

  const flush = async (): Promise<void> => {
    if (inFlight.current) {
      timer.current = null
      return
    }
    if (timer.current != null) window.clearTimeout(timer.current)
    timer.current = null
    const target = desiredRef.current
    inFlight.current = true
    try {
      const res = await cb.current.send(target)
      confirmedAt.current = Date.now()
      if (desiredRef.current === target && typeof res.workers === 'number') setLocal(res.workers)
    } catch (e) {
      inFlight.current = false
      cb.current.onError(e)
      setLocal(serverRef.current)
      if (mounted.current) setSyncing(false)
      return
    }
    inFlight.current = false
    if (desiredRef.current !== target) return flush()
    if (mounted.current) setSyncing(false)
    cb.current.onSynced()
  }

  const adjust = (delta: number) => {
    const next = Math.max(0, Math.min(max, desiredRef.current + delta))
    if (next === desiredRef.current) return
    setLocal(next)
    setSyncing(true)
    if (timer.current != null) window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => void flush(), SEND_DEBOUNCE_MS)
  }

  return { desired, adjust, syncing }
}
