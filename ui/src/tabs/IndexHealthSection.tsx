import { Box, Grid, Stack, Typography } from '@mui/material'
import type { IndexHealth } from '../api/types'
import { CorpusChart } from '../components/CorpusChart'
import { EmbeddingsPanel } from '../components/EmbeddingsPanel'
import { ResourceUtilCard } from '../components/ResourceUtilCard'
import { WatchersPanel } from '../components/WatchersPanel'

export function IndexHealthSection({ data, onRefresh }: { data: IndexHealth | null; onRefresh?: () => void }) {
  if (!data) return <Typography color="text.secondary">Loading index health…</Typography>
  return (
    <Box sx={{ mb: 3 }}>
      <Typography variant="overline" color="text.secondary">
        Index & runtime
      </Typography>
      <Grid container spacing={2} sx={{ mt: 0.5 }}>
        <Grid size={{ xs: 12, lg: 7 }}>
          <ResourceUtilCard data={data} />
        </Grid>
        <Grid size={{ xs: 12, lg: 5 }}>
          <Stack spacing={2}>
            <EmbeddingsPanel data={data} onRefresh={onRefresh} />
            <CorpusChart data={data} />
          </Stack>
        </Grid>
        <Grid size={12}>
          <WatchersPanel watchers={data.Watchers || []} onRefresh={onRefresh} />
        </Grid>
      </Grid>
    </Box>
  )
}
