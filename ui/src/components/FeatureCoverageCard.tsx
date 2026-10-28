import { Alert, Box, Card, CardContent, Chip, Stack, Tooltip, Typography } from '@mui/material'

import type { CoverageDigest, FeatureCoverage } from '../api/types'

import { chartColors } from '../lib/chartColors'
import { coverageRows, formatDelta, projectLabel, searchlessWarn, type CoverageRow } from '../lib/coverage'
import { formatStat } from './HealthBar'

const MONO = '"JetBrains Mono", ui-monospace, monospace'
const TOP_PROJECTS_PREVIEW = 3

interface FeatureCoverageCardProps {
  /** null shows the card loading. */
  data: FeatureCoverage | null
  /** Weekly digest coverage section; adds week-over-week deltas when it covers the same window. */
  trend?: CoverageDigest | null
}

/**
 * Feature coverage: the share of active agent sessions that used each ast-context-cache
 * feature at least once in the window, with search-less sessions called out.
 */
export const FeatureCoverageCard = ({ data, trend }: FeatureCoverageCardProps) => {
  if (!data) {
    return (
      <Card variant="outlined">
        <CardContent>
          <Typography variant="body2" color="text.secondary">
            Loading feature coverage…
          </Typography>
        </CardContent>
      </Card>
    )
  }
  const rows = coverageRows(data, trend)
  const warn = searchlessWarn(data)
  const projects = (data.TopProjects ?? []).slice(0, TOP_PROJECTS_PREVIEW)
  const other = Object.entries(data.OtherTools ?? {})
  return (
    <Card variant="outlined">
      <CardContent sx={{ pb: '16px !important' }}>
        <Stack direction="row" spacing={1} sx={{ alignItems: 'baseline', mb: 1, flexWrap: 'wrap' }} useFlexGap>
          <Typography variant="subtitle2">Feature coverage · {data.WindowDays}d</Typography>
          <Typography variant="caption" color="text.secondary">
            {formatStat(data.ActiveSessions)} active session{data.ActiveSessions === 1 ? '' : 's'} · {formatStat(data.Calls)} calls
            {data.AnonymousSessions > 0 && ` · ${formatStat(data.AnonymousSessions)} without session_id`}
          </Typography>
        </Stack>
        {data.ActiveSessions === 0 ? (
          <Typography variant="body2" color="text.secondary">
            No agent tool calls in this window.
          </Typography>
        ) : (
          <>
            <Alert severity={warn ? 'warning' : 'info'} sx={{ mb: 1.5, py: 0 }} data-testid="searchless-sessions">
              <strong>{formatStat(data.SearchlessSessions)}</strong> search-less session{data.SearchlessSessions === 1 ? '' : 's'} (
              {data.SearchlessPct.toFixed(0)}%) never called a code-search tool
            </Alert>
            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, minmax(0, 1fr))' }, columnGap: 3, rowGap: 0.75 }}>
              {rows.map((r) => (
                <CoverageBar key={r.feature} row={r} />
              ))}
            </Box>
            {projects.length > 0 && (
              <Typography variant="caption" color="text.secondary" component="div" sx={{ mt: 1.5 }}>
                Top projects:{' '}
                {projects.map((p, i) => (
                  <Tooltip key={p.ProjectPath || '(none)'} title={p.ProjectPath || 'calls without a project_path'}>
                    <span>
                      {i > 0 && ' · '}
                      <Box component="span" sx={{ fontFamily: MONO }}>
                        {projectLabel(p.ProjectPath)}
                      </Box>{' '}
                      {formatStat(p.SearchSessions)}/{formatStat(p.Sessions)} searched
                    </span>
                  </Tooltip>
                ))}
              </Typography>
            )}
            {other.length > 0 && (
              <Typography variant="caption" color="text.disabled" component="div" sx={{ mt: 0.5 }}>
                Unmapped tools: {other.map(([t, n]) => `${t} (${formatStat(n)})`).join(', ')}
              </Typography>
            )}
          </>
        )}
      </CardContent>
    </Card>
  )
}

function CoverageBar({ row }: { row: CoverageRow }) {
  return (
    <Box sx={{ minWidth: 0 }} title={`${row.label}: ${row.sessions} session${row.sessions === 1 ? '' : 's'}, ${row.calls} call${row.calls === 1 ? '' : 's'}`}>
      <Stack direction="row" spacing={1} sx={{ alignItems: 'center' }}>
        <Typography variant="caption" noWrap sx={{ flex: 1, minWidth: 0 }}>
          {row.label}
        </Typography>
        {row.delta !== null && (
          <Chip
            size="small"
            variant="outlined"
            label={formatDelta(row.delta)}
            sx={{
              height: 18,
              fontSize: 10.5,
              color: row.delta > 0 ? chartColors.green : row.delta < 0 ? chartColors.day : 'text.secondary',
            }}
          />
        )}
        <Typography sx={{ fontFamily: MONO, fontSize: 12, fontWeight: 700, minWidth: 44, textAlign: 'right' }}>
          {row.pct.toFixed(0)}%
        </Typography>
      </Stack>
      <Box sx={{ height: 6, borderRadius: 1, bgcolor: chartColors.track, overflow: 'hidden' }}>
        <Box sx={{ height: '100%', width: `${Math.min(100, row.pct)}%`, bgcolor: row.color, borderRadius: 1 }} />
      </Box>
    </Box>
  )
}
