import { Box, Chip, Stack, Typography } from '@mui/material'
import type { Health } from '../api/types'
import { formatNum, formatUptime } from '../api/client'

export function HealthBar({ health }: { health: Health | null }) {
  if (!health) {
    return <Typography color="text.secondary">Loading health…</Typography>
  }
  const queueCap = (health.QueueHighCap || 128) + (health.QueueLowCap || 2048)
  const queuePct = queueCap > 0 ? Math.min(100, ((health.QueueQueued + health.QueuePending) / queueCap) * 100) : 0
  const queueColor = queuePct > 75 ? 'error' : queuePct > 40 ? 'warning' : 'success'
  const uptimeNs = typeof health.Uptime === 'number' ? health.Uptime : 0
  return (
    <Stack direction="row" spacing={1} flexWrap="wrap" alignItems="center" useFlexGap sx={{ flex: 1, minWidth: 0 }}>
      <Chip size="small" label={`Embed: ${health.EmbedderState || 'unknown'}`} color={health.EmbedderState === 'healthy' ? 'success' : 'default'} />
      <Chip size="small" label={`Queue ${health.QueueQueued + health.QueuePending}`} color={queueColor} />
      <Chip size="small" variant="outlined" label={`${health.QueueThroughput?.toFixed(1) || 0}/s`} />
      <Chip size="small" variant="outlined" label={`Cache ${((health.CacheHitRatio || 0) * 100).toFixed(0)}%`} />
      <Chip size="small" variant="outlined" label={`Heap ${health.HeapMB?.toFixed(0)}MB`} />
      <Chip size="small" variant="outlined" label={`CPU ${health.CPUPercent?.toFixed(0)}%`} />
      <Chip size="small" variant="outlined" label={formatUptime(uptimeNs)} />
      <Box sx={{ flex: 1 }} />
      <Typography variant="caption" color="text.secondary">
        v{health.Version}
      </Typography>
    </Stack>
  )
}

export function StatCard({ title, value, sub }: { title: string; value: string; sub?: string }) {
  return (
    <Box sx={{ p: 2, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid', borderColor: 'divider' }}>
      <Typography variant="caption" color="text.secondary" textTransform="uppercase">
        {title}
      </Typography>
      <Typography variant="h5" fontWeight={600}>
        {value}
      </Typography>
      {sub && (
        <Typography variant="caption" color="text.secondary">
          {sub}
        </Typography>
      )}
    </Box>
  )
}

export function formatStat(n: number) {
  return formatNum(n)
}
