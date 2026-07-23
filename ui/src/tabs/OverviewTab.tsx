import {
  Box,
  Card,
  CardContent,
  Chip,
  Grid,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material'
import { RingGauge } from '../components/charts/RingGauge'
import type { ContextSessionsResponse, Stats, WeeklyDigest } from '../api/types'
import { formatStat } from '../components/HealthBar'
import { MetricStatCard } from '../components/charts/MetricStatCard'
import { chartColors } from '../lib/chartColors'
import { fmtDailyAvg, meterFillSegments, todayMeterFill } from '../lib/statMeters'

export function OverviewTab({
  stats,
  weeklyDigest,
  contextSessions,
}: {
  stats: Stats | null
  weeklyDigest?: WeeklyDigest | null
  contextSessions?: ContextSessionsResponse | null
}) {
  if (!stats) return <Typography color="text.secondary">Loading stats…</Typography>

  const baseline = stats.ApproxBaselineTokens ?? 0
  const returned = stats.ApproxTokensReturned ?? 0
  const rounds = stats.ApproxRoundsAvoided ?? 0
  const heuristicLabel = stats.HeuristicLabel || 'approximate'

  return (
    <Box sx={{ mt: 3 }}>
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
        Value estimate ({heuristicLabel})
      </Typography>
      <Grid container spacing={2} sx={{ mt: 0.5, mb: 3 }}>
        <Grid size={{ xs: 12, sm: 4 }}>
          <MetricStatCard
            title="Approx baseline"
            value={formatStat(baseline)}
            accent={chartColors.accent}
            fill={todayMeterFill(returned, Math.max(baseline, 1))}
            sub={`≈ full-source tokens before mode/dedup · 30d · ${heuristicLabel}`}
          />
        </Grid>
        <Grid size={{ xs: 12, sm: 4 }}>
          <MetricStatCard
            title="Approx returned"
            value={formatStat(returned)}
            accent={chartColors.orange}
            fill={todayMeterFill(returned, Math.max(baseline, 1))}
            sub={`tokens actually returned · saved ${formatStat(stats.TokensSaved)}`}
          />
        </Grid>
        <Grid size={{ xs: 12, sm: 4 }}>
          <MetricStatCard
            title="Rounds avoided"
            value={rounds.toFixed(1)}
            accent={chartColors.green}
            fill={meterFillSegments(rounds, Math.max(rounds, 1))}
            sub={`≈ tokens saved ÷ 4000 · ${heuristicLabel}`}
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

      <WeeklyDigestSection digest={weeklyDigest ?? null} />
      <ContextSessionsSection data={contextSessions ?? null} />
    </Box>
  )
}

function WeeklyDigestSection({ digest }: { digest: WeeklyDigest | null }) {
  if (!digest) {
    return (
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Loading weekly digest…
      </Typography>
    )
  }
  const emb = digest.EmbedReliability
  return (
    <Box sx={{ mb: 3 }}>
      <Typography variant="overline" color="text.secondary">
        Weekly digest ({digest.WindowDays}d)
      </Typography>
      <Grid container spacing={2} sx={{ mt: 0.5 }}>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <MetricStatCard
            title="Tokens saved"
            value={formatStat(digest.TokensSaved)}
            accent={chartColors.green}
            fill={todayMeterFill(digest.TokensSaved, Math.max(digest.TokensSaved, 1))}
            sub={`${formatStat(digest.Queries)} queries`}
          />
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <MetricStatCard
            title="VC stored"
            value={formatStat(digest.VirtualStored)}
            accent={chartColors.purple}
            fill={todayMeterFill(digest.VirtualStored, Math.max(digest.VirtualStored + digest.VirtualAccessed, 1))}
            sub={`accessed ${formatStat(digest.VirtualAccessed)}`}
          />
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <MetricStatCard
            title="Rounds avoided"
            value={(digest.Heuristic?.ApproxRoundsAvoided ?? 0).toFixed(1)}
            accent={chartColors.accent}
            fill={meterFillSegments(digest.Heuristic?.ApproxRoundsAvoided ?? 0, Math.max(digest.Heuristic?.ApproxRoundsAvoided ?? 1, 1))}
            sub={`${digest.Heuristic?.HeuristicLabel || 'approximate'} · 7d`}
          />
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <Card variant="outlined" sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="overline" color="text.secondary" display="block">
                Embed reliability
              </Typography>
              {emb?.Available ? (
                <Stack spacing={0.75} sx={{ mt: 0.5 }}>
                  <Typography variant="body2" sx={{ fontFamily: 'ui-monospace, monospace' }}>
                    Pending failures: {formatStat(emb.PendingFailures ?? 0)}
                  </Typography>
                  <Typography variant="caption" color="text.secondary" display="block">
                    {emb.LastAutoRecoverUnix
                      ? `Last auto-recover: ${new Date(emb.LastAutoRecoverUnix * 1000).toLocaleString()}`
                      : 'No stuck-worker auto-recover this process'}
                  </Typography>
                  {emb.AbnormalPreviousRun && <Chip size="small" color="warning" label="Abnormal prior exit" />}
                </Stack>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  Embed reliability unavailable
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
        <Grid size={{ xs: 12 }}>
          <Card variant="outlined">
            <CardContent sx={{ pb: '16px !important' }}>
              <Typography variant="subtitle2" gutterBottom>
                Top tools (7d)
              </Typography>
              {(digest.TopTools?.length ?? 0) === 0 ? (
                <Typography variant="body2" color="text.secondary">
                  No tool activity in the last week.
                </Typography>
              ) : (
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Tool</TableCell>
                      <TableCell align="right">Calls</TableCell>
                      <TableCell align="right">Tokens saved</TableCell>
                      <TableCell align="right">Avg ms</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {digest.TopTools.map((t) => (
                      <TableRow key={t.ToolName}>
                        <TableCell sx={{ fontFamily: 'ui-monospace, monospace', fontSize: 13 }}>{t.ToolName}</TableCell>
                        <TableCell align="right">{formatStat(t.Calls)}</TableCell>
                        <TableCell align="right">{formatStat(t.TokensSaved)}</TableCell>
                        <TableCell align="right">{(t.AvgDurationMs ?? 0).toFixed(1)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  )
}

function ContextSessionsSection({ data }: { data: ContextSessionsResponse | null }) {
  if (!data) {
    return (
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Loading session stories…
      </Typography>
    )
  }
  return (
    <Box sx={{ mb: 2 }}>
      <Typography variant="overline" color="text.secondary">
        Virtual context sessions (~{data.WindowDays}d)
      </Typography>
      <Card variant="outlined" sx={{ mt: 1 }}>
        <CardContent sx={{ pb: '16px !important' }}>
          {(data.Sessions?.length ?? 0) === 0 ? (
            <Typography variant="body2" color="text.secondary">
              No virtual-context sessions with store/access activity in this window.
            </Typography>
          ) : (
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Session</TableCell>
                  <TableCell align="right">Notes</TableCell>
                  <TableCell align="right">Stored</TableCell>
                  <TableCell align="right">Accessed</TableCell>
                  <TableCell>Recovered</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {data.Sessions.map((s) => (
                  <TableRow key={s.SessionID}>
                    <TableCell sx={{ fontFamily: 'ui-monospace, monospace', fontSize: 12, maxWidth: 220 }}>
                      <Typography noWrap title={s.SessionID} sx={{ fontFamily: 'inherit', fontSize: 'inherit' }}>
                        {s.SessionID}
                      </Typography>
                      {s.ProjectPath ? (
                        <Typography variant="caption" color="text.secondary" noWrap display="block" title={s.ProjectPath}>
                          {s.ProjectPath}
                        </Typography>
                      ) : null}
                    </TableCell>
                    <TableCell align="right">
                      {formatStat(s.NotesCount)}
                      {s.ActiveNotes > 0 ? (
                        <Typography variant="caption" color="text.secondary" display="block">
                          {s.ActiveNotes} active
                        </Typography>
                      ) : null}
                    </TableCell>
                    <TableCell align="right">{formatStat(s.VirtualTokensStored)}</TableCell>
                    <TableCell align="right">{formatStat(s.VirtualTokensAccessed)}</TableCell>
                    <TableCell>
                      <Chip
                        size="small"
                        color={s.FetchedAfterStore ? 'success' : 'default'}
                        label={s.FetchedAfterStore ? 'fetch after store' : 'store only'}
                      />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
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
