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
  const embedState = health.EmbedderState || 'unknown'
  const embedOk = embedState === 'healthy' || embedState === 'ready' || embedState === 'ok'
  const embedColor =
    embedState === 'error' ? 'error' : embedOk ? 'success' : embedState === 'degraded' ? 'warning' : 'default'
  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        minWidth: 0,
        maxWidth: '100%',
        flex: 1,
        px: 1.75,
        py: 0.75,
        bgcolor: 'background.paper',
        border: '1px solid',
        borderColor: 'divider',
        borderRadius: 999,
        overflowX: 'auto',
        overflowY: 'hidden',
        scrollbarWidth: 'thin',
      }}
    >
      <Stack direction="row" spacing={1} alignItems="center" sx={{ flexWrap: 'nowrap', minWidth: 'min-content' }}>
        <Chip size="small" label={`Embed: ${embedState}`} color={embedColor} sx={{ flexShrink: 0 }} />
        <Chip size="small" label={`Queue ${health.QueueQueued + health.QueuePending}`} color={queueColor} sx={{ flexShrink: 0 }} />
        <Chip size="small" variant="outlined" label={`${health.QueueThroughput?.toFixed(1) || 0}/s`} sx={{ flexShrink: 0 }} />
        <Chip size="small" variant="outlined" label={`Cache ${((health.CacheHitRatio || 0) * 100).toFixed(0)}%`} sx={{ flexShrink: 0, display: { xs: 'none', sm: 'flex' } }} />
        <Chip size="small" variant="outlined" label={`Heap ${health.HeapMB?.toFixed(0)}MB`} sx={{ flexShrink: 0, display: { xs: 'none', lg: 'flex' } }} />
        <Chip size="small" variant="outlined" label={`CPU ${health.CPUPercent?.toFixed(0)}%`} sx={{ flexShrink: 0, display: { xs: 'none', lg: 'flex' } }} />
        <Chip size="small" variant="outlined" label={formatUptime(uptimeNs)} sx={{ flexShrink: 0, display: { xs: 'none', md: 'flex' } }} />
        <Typography variant="caption" color="text.secondary" sx={{ flexShrink: 0, pl: 0.5 }}>
          v{health.Version}
        </Typography>
      </Stack>
    </Box>
  )
}

export function StatCard({ title, value, sub }: { title: string; value: string; sub?: string }) {
  return (
    <Box
      sx={{
        p: 2,
        bgcolor: 'background.paper',
        borderRadius: 1,
        border: '1px solid',
        borderColor: 'divider',
        transition: 'border-color 0.15s, background 0.15s',
        '&:hover': {
          borderColor: 'primary.dark',
          bgcolor: 'rgba(88,166,255,0.04)',
        },
      }}
    >
      <Typography variant="overline" color="text.secondary" display="block">
        {title}
      </Typography>
      <Typography variant="h4" fontWeight={700} sx={{ fontFamily: '"JetBrains Mono", ui-monospace, monospace', fontSize: 28, lineHeight: 1.2 }}>
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
