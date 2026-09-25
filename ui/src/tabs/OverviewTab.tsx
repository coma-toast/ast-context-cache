import { useState } from 'react'
import {
  Alert,
  Box,
  Button,
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
import type { ContextSessionStory, ContextSessionsResponse, Stats, WeeklyDigest } from '../api/types'
import { formatStat } from '../components/HealthBar'
import { MetricStatCard } from '../components/charts/MetricStatCard'
import { chartColors } from '../lib/chartColors'
import { fmtDailyAvg, meterFillSegments, todayMeterFill } from '../lib/statMeters'

const SESSIONS_PREVIEW = 4
const TOP_TOOLS_PREVIEW = 5
const mono = '"JetBrains Mono", ui-monospace, monospace'

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
  const heuristicLabel = stats.HeuristicLabel || 'approximate'

  return (
    <Box sx={{ mb: 3 }}>
      <SectionHeader title="Activity" hint="Big number is today · bar compares today with the 30-day daily average" />
      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <MetricStatCard
            title="Queries"
            value={formatStat(stats.TodayQueries)}
            accent={chartColors.accent}
            fill={todayMeterFill(stats.TodayQueries, stats.TotalQueries)}
            sub={`${formatStat(stats.TotalQueries)} in 30d · ${formatStat(stats.TodaySessions)} sessions today`}
            detail={`30d sessions: ${formatStat(stats.Sessions)} · avg/day: ${fmtDailyAvg(stats.TotalQueries)} queries`}
          />
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <MetricStatCard
            title="Tokens saved"
            value={formatStat(stats.TodayTokens)}
            accent={chartColors.green}
            fill={todayMeterFill(stats.TodayTokens, stats.TokensSaved)}
            sub={`${formatStat(stats.TokensSaved)} in 30d · ${fmtDailyAvg(stats.TokensSaved)}/day`}
            detail={`Dedup: ${formatStat(stats.DedupTokensSaved)} · vs whole files: ${formatStat(stats.SavingsVsFiles)}${stats.TotalChars ? ` · chars returned: ${formatStat(stats.TotalChars)}` : ''}`}
          />
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <MetricStatCard
            title="Avg latency"
            value={`${stats.TodayAvgDurationMs?.toFixed(1) ?? 0} ms`}
            accent={chartColors.orange}
            fill={meterFillSegments(stats.TodayAvgDurationMs ?? 0, stats.AvgDurationMs ?? 0)}
            sub={`30d avg ${stats.AvgDurationMs?.toFixed(1) ?? 0} ms`}
          />
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <MetricStatCard
            title="Rounds avoided · 30d"
            value={(stats.ApproxRoundsAvoided ?? 0).toFixed(1)}
            accent={chartColors.purple}
            fill={meterFillSegments(returned, baseline)}
            sub={`≈${formatStat(baseline)} baseline → ${formatStat(returned)} returned`}
            detail={`${heuristicLabel}: baseline ≈ full-source tokens before mode/dedup; rounds ≈ tokens saved ÷ 4000`}
          />
        </Grid>
      </Grid>

      <Grid container spacing={2}>
        <Grid size={{ xs: 12, lg: 7 }}>
          <WeekCard digest={weeklyDigest ?? null} />
        </Grid>
        <Grid size={{ xs: 12, lg: 5 }}>
          <VirtualContextCard stats={stats} digest={weeklyDigest ?? null} sessions={contextSessions ?? null} />
        </Grid>
      </Grid>
    </Box>
  )
}

function SectionHeader({ title, hint }: { title: string; hint?: string }) {
  return (
    <Stack direction="row" spacing={1.5} sx={{ alignItems: 'baseline', mb: 1 }}>
      <Typography variant="overline" color="text.secondary">
        {title}
      </Typography>
      {hint && (
        <Typography variant="caption" color="text.disabled" sx={{ display: { xs: 'none', sm: 'block' } }}>
          {hint}
        </Typography>
      )}
    </Stack>
  )
}

function InlineStat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <Box sx={{ minWidth: 0 }}>
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }} noWrap>
        {label}
      </Typography>
      <Typography sx={{ fontFamily: mono, fontWeight: 700, fontSize: 18, lineHeight: 1.3, color }}>{value}</Typography>
    </Box>
  )
}

function WeekCard({ digest }: { digest: WeeklyDigest | null }) {
  if (!digest) {
    return (
      <Card variant="outlined" sx={{ height: '100%' }}>
        <CardContent>
          <Typography variant="body2" color="text.secondary">
            Loading weekly digest…
          </Typography>
        </CardContent>
      </Card>
    )
  }
  const pendingFailures = digest.EmbedReliability?.Available ? digest.EmbedReliability.PendingFailures ?? 0 : 0
  const tools = (digest.TopTools ?? []).slice(0, TOP_TOOLS_PREVIEW)
  return (
    <Card variant="outlined" sx={{ height: '100%' }}>
      <CardContent sx={{ pb: '16px !important' }}>
        <Typography variant="subtitle2" gutterBottom>
          Last {digest.WindowDays} days
        </Typography>
        <Stack direction="row" spacing={3} sx={{ flexWrap: 'wrap', mb: 1.5 }} useFlexGap>
          <InlineStat label="Tokens saved" value={formatStat(digest.TokensSaved)} color={chartColors.green} />
          <InlineStat label="Queries" value={formatStat(digest.Queries)} color={chartColors.accent} />
          <InlineStat
            label="Rounds avoided"
            value={(digest.Heuristic?.ApproxRoundsAvoided ?? 0).toFixed(1)}
            color={chartColors.purple}
          />
        </Stack>
        {pendingFailures > 0 && (
          <Alert severity="warning" sx={{ mb: 1.5, py: 0 }}>
            {formatStat(pendingFailures)} file{pendingFailures === 1 ? '' : 's'} waiting to retry embedding
          </Alert>
        )}
        {tools.length === 0 ? (
          <Typography variant="body2" color="text.secondary">
            No tool activity this week.
          </Typography>
        ) : (
          <Table size="small" sx={{ '& td, & th': { px: 1, py: 0.5 } }}>
            <TableHead>
              <TableRow>
                <TableCell>Top tools</TableCell>
                <TableCell align="right">Calls</TableCell>
                <TableCell align="right">Saved</TableCell>
                <TableCell align="right">Avg ms</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {tools.map((t) => (
                <TableRow key={t.ToolName} sx={{ '&:last-child td': { borderBottom: 0 } }}>
                  <TableCell sx={{ fontFamily: mono, fontSize: 12.5 }}>{t.ToolName}</TableCell>
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
  )
}

function VirtualContextCard({
  stats,
  digest,
  sessions,
}: {
  stats: Stats
  digest: WeeklyDigest | null
  sessions: ContextSessionsResponse | null
}) {
  const [showAll, setShowAll] = useState(false)
  const rows = sessions?.Sessions ?? []
  const visible = showAll ? rows : rows.slice(0, SESSIONS_PREVIEW)
  const max = stats.VirtualMaxTokensGlobal
  const capPct = max > 0 ? Math.round(Math.min(100, (stats.VirtualInventoryTokens / max) * 100)) : 0
  return (
    <Card variant="outlined" sx={{ height: '100%' }}>
      <CardContent sx={{ pb: '16px !important' }}>
        <Typography variant="subtitle2" gutterBottom>
          Virtual context
        </Typography>
        <Stack direction="row" spacing={2} sx={{ alignItems: 'center', mb: 1.5 }}>
          <RingGauge value={capPct} max={100} caption="% cap" size={64} />
          <Box sx={{ flex: 1, display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 1 }}>
            <InlineStat
              label={`Stored · ${formatStat(stats.VirtualNotesCount)} note${stats.VirtualNotesCount === 1 ? '' : 's'}`}
              value={formatStat(stats.VirtualInventoryTokens)}
            />
            <InlineStat label="30d utilization" value={`${(stats.VirtualUtilPct30d ?? 0).toFixed(0)}%`} />
            <InlineStat
              label="Orphans"
              value={formatStat(stats.VirtualOrphanCount)}
              color={stats.VirtualOrphanCount > 0 ? chartColors.orange : undefined}
            />
            {digest && (
              <InlineStat label="7d store / fetch" value={`${formatStat(digest.VirtualStored)} / ${formatStat(digest.VirtualAccessed)}`} />
            )}
          </Box>
        </Stack>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
          Recent sessions{sessions ? ` (~${sessions.WindowDays}d)` : ''} · green dot = fetched back after storing
        </Typography>
        {!sessions ? (
          <Typography variant="body2" color="text.secondary">
            Loading…
          </Typography>
        ) : rows.length === 0 ? (
          <Typography variant="body2" color="text.secondary">
            No store/access activity in this window.
          </Typography>
        ) : (
          <>
            <Stack spacing={0.5}>
              {visible.map((s) => (
                <SessionLine key={s.SessionID} s={s} />
              ))}
            </Stack>
            {rows.length > SESSIONS_PREVIEW && (
              <Button size="small" onClick={() => setShowAll((v) => !v)} sx={{ mt: 0.5, px: 0.5, minWidth: 0 }}>
                {showAll ? 'Show fewer' : `Show all ${rows.length}`}
              </Button>
            )}
          </>
        )}
      </CardContent>
    </Card>
  )
}

function SessionLine({ s }: { s: ContextSessionStory }) {
  const project = s.ProjectPath ? s.ProjectPath.split('/').filter(Boolean).pop() : ''
  return (
    <Stack direction="row" spacing={1} sx={{ alignItems: 'center', minWidth: 0 }} title={[s.SessionID, s.ProjectPath].filter(Boolean).join('\n')}>
      <Box
        sx={{
          width: 6,
          height: 6,
          borderRadius: '50%',
          flexShrink: 0,
          bgcolor: s.FetchedAfterStore ? chartColors.green : chartColors.overlap,
        }}
      />
      <Typography noWrap sx={{ fontFamily: mono, fontSize: 12, flex: 1, minWidth: 0 }}>
        {s.SessionID}
        {project && (
          <Typography component="span" color="text.secondary" sx={{ fontFamily: 'inherit', fontSize: 'inherit', ml: 1 }}>
            {project}
          </Typography>
        )}
      </Typography>
      <Chip
        size="small"
        variant="outlined"
        label={`${formatStat(s.NotesCount)} note${s.NotesCount === 1 ? '' : 's'} · ${formatStat(s.VirtualTokensStored)}`}
        sx={{ height: 20, fontSize: 11, flexShrink: 0 }}
      />
    </Stack>
  )
}
