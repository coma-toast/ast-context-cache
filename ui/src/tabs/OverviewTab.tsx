import { Box, Grid, Typography } from '@mui/material'
import type { Stats } from '../api/types'
import { StatCard, formatStat } from '../components/HealthBar'

export function OverviewTab({ stats }: { stats: Stats | null }) {
  if (!stats) return <Typography color="text.secondary">Loading stats…</Typography>
  return (
    <Box>
      <Typography variant="overline" color="text.secondary">
        Query activity
      </Typography>
      <Grid container spacing={2} sx={{ mt: 0.5, mb: 3 }}>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <StatCard title="Queries today" value={formatStat(stats.TodayQueries)} sub={`${formatStat(stats.TotalQueries)} total`} />
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <StatCard title="Tokens saved today" value={formatStat(stats.TodayTokens)} sub={`${formatStat(stats.TokensSaved)} total`} />
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <StatCard title="Sessions today" value={formatStat(stats.TodaySessions)} sub={`avg ${stats.TodayAvgDurationMs?.toFixed(0)}ms`} />
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <StatCard title="Dedup saved" value={formatStat(stats.DedupTokensSaved)} sub={`vs files ${formatStat(stats.SavingsVsFiles)}`} />
        </Grid>
      </Grid>
      <Typography variant="overline" color="text.secondary">
        Context memory (summary)
      </Typography>
      <Grid container spacing={2} sx={{ mt: 0.5, mb: 3 }}>
        <Grid size={{ xs: 12, sm: 4 }}>
          <StatCard title="Virtual inventory" value={formatStat(stats.VirtualInventoryTokens)} sub={`${stats.VirtualNotesCount} notes`} />
        </Grid>
        <Grid size={{ xs: 12, sm: 4 }}>
          <StatCard title="30d utilization" value={`${stats.VirtualUtilPct30d?.toFixed(0) || 0}%`} sub={`${formatStat(stats.VirtualAccessed30d)} accessed`} />
        </Grid>
        <Grid size={{ xs: 12, sm: 4 }}>
          <StatCard title="Orphans" value={formatStat(stats.VirtualOrphanCount)} sub={`stored ${formatStat(stats.VirtualStored30d)}`} />
        </Grid>
      </Grid>
    </Box>
  )
}
