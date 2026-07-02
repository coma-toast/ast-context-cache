import {
  Box,
  Card,
  CardContent,
  Chip,
  Grid,
  LinearProgress,
  Stack,
  Typography,
} from '@mui/material'
import type { IndexHealth } from '../api/types'
import { formatNum } from '../api/client'

export function IndexHealthSection({ data }: { data: IndexHealth | null }) {
  if (!data) return <Typography color="text.secondary">Loading index health…</Typography>
  return (
    <Box sx={{ mt: 3 }}>
      <Typography variant="overline" color="text.secondary">
        Index & runtime
      </Typography>
      <Grid container spacing={2} sx={{ mt: 0.5 }}>
        <Grid size={{ xs: 12, md: 4 }}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle2" gutterBottom>
                Server utilization
              </Typography>
              <Stack spacing={1}>
                <Typography variant="body2">CPU {data.CPUPercent?.toFixed(0)}%</Typography>
                <Typography variant="body2">Heap {data.HeapMB?.toFixed(0)} MB</Typography>
                <Typography variant="body2">Vectors {formatNum(data.TotalVectors)} ({data.VectorMemMB?.toFixed(0)} MB)</Typography>
              </Stack>
            </CardContent>
          </Card>
        </Grid>
        <Grid size={{ xs: 12, md: 4 }}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle2" gutterBottom>
                Embeddings
              </Typography>
              <Typography variant="body2" gutterBottom>
                Queue {data.EmbedQueued} · Pending {data.EmbedPending}
              </Typography>
              <LinearProgress
                variant="determinate"
                value={Math.min(100, ((data.EmbedQueued + data.EmbedPending) / 2176) * 100)}
                color={data.EmbedQueued + data.EmbedPending > 1500 ? 'error' : 'primary'}
                sx={{ mb: 1 }}
              />
              <Typography variant="caption" color="text.secondary">
                Workers {data.EmbedWorkers}/{data.EmbedWorkerMax} · {data.EmbedThroughput?.toFixed(1)}/s · {formatNum(data.EmbedComplete)} done
              </Typography>
              {data.WALMaintenanceActive && (
                <Chip size="small" color="warning" label="WAL maintenance" sx={{ mt: 1 }} />
              )}
            </CardContent>
          </Card>
        </Grid>
        <Grid size={{ xs: 12, md: 4 }}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle2" gutterBottom>
                Corpus
              </Typography>
              <Typography variant="body2">{formatNum(data.TotalSymbols)} symbols</Typography>
              <Typography variant="body2">{formatNum(data.TotalFiles)} files</Typography>
              <Typography variant="body2">{formatNum(data.TotalEdges)} edges</Typography>
              <Typography variant="caption" color="text.secondary">
                {data.PinnedCount} pinned projects
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid size={12}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle2" gutterBottom>
                File watchers ({data.Watchers?.length || 0})
              </Typography>
              {data.Watchers?.length === 0 && (
                <Typography variant="body2" color="text.secondary">
                  No active watchers
                </Typography>
              )}
              <Stack spacing={1} sx={{ maxHeight: 320, overflowY: 'auto', pr: 0.5 }}>
                {data.Watchers?.map((w) => (
                  <Box
                    key={w.ProjectPath}
                    sx={{
                      p: 1.25,
                      border: '1px solid',
                      borderColor: 'divider',
                      borderRadius: 1,
                      minWidth: 0,
                    }}
                  >
                    <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap sx={{ mb: 0.5 }}>
                      <Typography variant="body2" fontWeight={600} sx={{ minWidth: 0 }}>
                        {w.Label || w.Name}
                      </Typography>
                      <Chip size="small" label={w.Active ? 'active' : 'paused'} color={w.Active ? 'success' : 'default'} />
                      {w.LinkedCount > 0 && (
                        <Chip size="small" variant="outlined" label={`${w.LinkedCount} linked`} component="a" href="#settings-projects" />
                      )}
                    </Stack>
                    <Typography
                      variant="caption"
                      color="text.secondary"
                      sx={{
                        fontFamily: 'ui-monospace, monospace',
                        display: 'block',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                      title={w.ProjectPath}
                    >
                      {w.ProjectPath}
                    </Typography>
                  </Box>
                ))}
              </Stack>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  )
}
