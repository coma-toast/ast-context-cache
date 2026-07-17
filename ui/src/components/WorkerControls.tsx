import { Box, IconButton, Stack, Tooltip, Typography } from '@mui/material'
import AddIcon from '@mui/icons-material/Add'
import RemoveIcon from '@mui/icons-material/Remove'
import { useState } from 'react'
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
  const [busy, setBusy] = useState(false)
  const { dots, ellipsis } = workerPillCount(workers)
  const isAux = pool === 'aux'
  const lit = isAux ? Math.min(active, workers) : active
  const adjust = async (delta: number) => {
    if (busy) return
    setBusy(true)
    try {
      if (isAux) await api.adjustEmbedAuxWorkers(delta)
      else await api.adjustEmbedWorkers(delta)
      onChange?.()
    } catch (e) {
      showToast(e instanceof Error ? e.message : 'Worker adjust failed', 'error')
    } finally {
      setBusy(false)
    }
  }
  const title = (() => {
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
    return isAux ? `Aux workers: ${workers} enabled` : `Workers: ${active} of ${workers} busy`
  })()
  return (
    <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap title={title}>
      <Stack direction="column" spacing={0}>
        <IconButton
          size="small"
          aria-label={isAux ? 'Increase aux workers' : 'Increase workers'}
          disabled={busy || workers >= max}
          onClick={() => adjust(1)}
          sx={{ p: 0.25 }}
        >
          <AddIcon fontSize="small" />
        </IconButton>
        <IconButton
          size="small"
          aria-label={isAux ? 'Decrease aux workers' : 'Decrease workers'}
          disabled={busy || workers <= 0}
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
      <Typography variant="body2" fontWeight={600} sx={{ fontFamily: 'ui-monospace, monospace' }}>
        {workers}
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
