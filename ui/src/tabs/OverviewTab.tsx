import { Box, Card, CardContent, Grid, Typography } from '@mui/material'
import { RingGauge } from '../components/charts/RingGauge'
import type { Stats } from '../api/types'
import { formatStat } from '../components/HealthBar'
import { MetricStatCard } from '../components/charts/MetricStatCard'
import { chartColors } from '../lib/chartColors'
import { fmtDailyAvg, meterFillSegments, todayMeterFill } from '../lib/statMeters'

export function OverviewTab({ stats }: { stats: Stats | null }) {
  if (!stats) return <Typography color="text.secondary">Loading stats…</Typography>
  return (
    <Box>
      <Typography variant="overline" color="text.secondary">
        Query activity
      </Typography>
      <Grid container spacing={2} sx={{ mt: 0.5, mb: 3 }}>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <MetricStatCard
            title="Queries"
            value={formatStat(stats.TodayQueries)}
            accent={chartColors.accent}
            fill={todayMeterFill(stats.TodayQueries, stats.TotalQueries)}
            sub={`30d: ${formatStat(stats.TotalQueries)} · avg/day: ${fmtDailyAvg(stats.TotalQueries)}`}
          />
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <MetricStatCard
            title="Tokens saved"
            value={formatStat(stats.TodayTokens)}
            accent={chartColors.green}
            fill={todayMeterFill(stats.TodayTokens, stats.TokensSaved)}
            sub={`30d: ${formatStat(stats.TokensSaved)} · avg/day: ${fmtDailyAvg(stats.TokensSaved)} · dedup: ${formatStat(stats.DedupTokensSaved)} · vs files: ${formatStat(stats.SavingsVsFiles)}`}
          />
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <MetricStatCard
            title="Avg duration"
            value={`${stats.TodayAvgDurationMs?.toFixed(1) ?? 0} ms`}
            accent={chartColors.orange}
            fill={meterFillSegments(stats.TodayAvgDurationMs ?? 0, stats.AvgDurationMs ?? 0)}
            sub={`30d avg: ${stats.AvgDurationMs?.toFixed(1) ?? 0} ms`}
          />
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <MetricStatCard
            title="Sessions"
            value={formatStat(stats.TodaySessions)}
            accent={chartColors.purple}
            fill={todayMeterFill(stats.TodaySessions, stats.Sessions)}
            sub={`30d: ${formatStat(stats.Sessions)} · avg/day: ${fmtDailyAvg(stats.Sessions)}${stats.TotalChars ? ` · chars: ${formatStat(stats.TotalChars)}` : ''}`}
          />
        </Grid>
      </Grid>

      <Typography variant="overline" color="text.secondary">
        Context memory (summary)
      </Typography>
      <Grid container spacing={2} sx={{ mt: 0.5, mb: 3 }}>
        <Grid size={{ xs: 12, sm: 4 }}>
          <VirtualGaugeCard
            title="Virtual inventory"
            value={formatStat(stats.VirtualInventoryTokens)}
            sub={`${stats.VirtualNotesCount} notes`}
            gaugePct={stats.VirtualMaxTokensGlobal > 0 ? (stats.VirtualInventoryTokens / stats.VirtualMaxTokensGlobal) * 100 : 0}
          />
        </Grid>
        <Grid size={{ xs: 12, sm: 4 }}>
          <VirtualGaugeCard
            title="30d utilization"
            value={`${stats.VirtualUtilPct30d?.toFixed(0) || 0}%`}
            sub={`${formatStat(stats.VirtualAccessed30d)} accessed · stored ${formatStat(stats.VirtualStored30d)}`}
            gaugePct={stats.VirtualUtilPct30d ?? 0}
          />
        </Grid>
        <Grid size={{ xs: 12, sm: 4 }}>
          <MetricStatCard
            title="Orphans"
            value={formatStat(stats.VirtualOrphanCount)}
            accent={stats.VirtualOrphanCount > 0 ? chartColors.orange : chartColors.green}
            fill={todayMeterFill(stats.VirtualOrphanCount, Math.max(stats.VirtualNotesCount, 1))}
            sub={`stored ${formatStat(stats.VirtualStored30d)} · flushed ${formatStat(stats.VirtualFlushed30d ?? 0)}`}
          />
        </Grid>
      </Grid>
    </Box>
  )
}

function VirtualGaugeCard({
  title,
  value,
  sub,
  gaugePct,
}: {
  title: string
  value: string
  sub?: string
  gaugePct: number
}) {
  const pct = Math.min(100, Math.max(0, Number.isFinite(gaugePct) ? gaugePct : 0))
  return (
    <Card variant="outlined" sx={{ height: '100%' }}>
      <CardContent sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
        <RingGauge value={Math.round(pct)} max={100} caption="util" size={72} />
        <Box sx={{ minWidth: 0 }}>
          <Typography variant="overline" color="text.secondary" display="block">
            {title}
          </Typography>
          <Typography variant="h5" fontWeight={700} sx={{ fontFamily: 'ui-monospace, monospace' }}>
            {value}
          </Typography>
          {sub && (
            <Typography variant="caption" color="text.secondary">
              {sub}
            </Typography>
          )}
        </Box>
      </CardContent>
    </Card>
  )
}
