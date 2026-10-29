import { Box, Card, CardContent, LinearProgress, Stack, TextField, Typography } from '@mui/material'

import type { IndexHealth, SettingsData } from '../api/types'
import { useSettingsDraft } from '../hooks/useSettingsDraft'
import { queueCapHelper, queueCapRows, validateQueueCap } from '../lib/queueCaps'
import { settingsCardSx } from '../lib/settingsSections'
import { SettingsSaveBar } from './SettingsSaveBar'

interface EmbedQueueSettingsSectionProps {
  data: SettingsData
  /** Live index health, for the current queue depth beside each limit. */
  indexHealth?: IndexHealth | null
}

/**
 * Settings card for the embed queue soft limits (retry, priority, background). Saved changes
 * apply live; a limit set by its EMBED_QUEUE_*_CAP environment variable is read-only here.
 */
export const EmbedQueueSettingsSection = ({ data, indexHealth }: EmbedQueueSettingsSectionProps) => {
  const rows = queueCapRows(data.EmbedQueueCaps, indexHealth)
  const saved = Object.fromEntries(rows.filter((r) => !r.env_locked).map((r) => [r.key, String(r.value)]))
  const draft = useSettingsDraft('embed-queue', 'Embed queue', saved, {
    validate: (changes) => {
      const out: Record<string, string> = {}
      for (const row of rows) {
        const err = row.key in changes ? validateQueueCap(changes[row.key], row.hard_max) : null
        if (err) out[row.key] = err
      }
      return out
    },
  })
  return (
    <Card variant="outlined" id="settings-embed-queue" sx={settingsCardSx(draft.dirty)}>
      <CardContent>
        <Typography variant="subtitle1" gutterBottom>
          Embed queue
        </Typography>
        <Typography variant="caption" color="text.secondary" component="p" sx={{ mb: 1.5 }}>
          How many embed jobs each channel holds in memory. Lower limits use less memory and push back on indexing
          sooner; higher limits absorb bigger bursts. Saved changes apply immediately, no restart.
        </Typography>
        {rows.length === 0 ? (
          <Typography color="text.secondary">Embed queue limits unavailable.</Typography>
        ) : (
          <>
            <Stack direction="row" spacing={2} sx={{ flexWrap: 'wrap' }} useFlexGap>
              {rows.map((row) => (
                <Box key={row.key} sx={{ width: 240 }} data-testid={`queue-cap-${row.key}`}>
                  <TextField
                    label={`${row.label} limit`}
                    type="number"
                    size="small"
                    fullWidth
                    disabled={row.env_locked}
                    value={row.env_locked ? row.value : draft.value(row.key)}
                    onChange={(e) => draft.set(row.key, e.target.value)}
                    error={!!draft.errors[row.key]}
                    slotProps={{ htmlInput: { min: 1, max: row.hard_max, 'aria-label': row.key } }}
                    helperText={draft.errors[row.key] ?? queueCapHelper(row)}
                  />
                  <LinearProgress
                    variant="determinate"
                    value={row.fillPct}
                    color={row.fillPct >= 90 ? 'warning' : 'primary'}
                    aria-label={`${row.label} depth`}
                    sx={{ mt: 0.5, height: 4, borderRadius: 2 }}
                  />
                  <Typography variant="caption" color="text.secondary">
                    {row.depth} / {row.value} queued
                  </Typography>
                </Box>
              ))}
            </Stack>
            <SettingsSaveBar draft={draft} label="Embed queue" />
          </>
        )}
      </CardContent>
    </Card>
  )
}
