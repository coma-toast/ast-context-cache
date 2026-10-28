import { useState } from 'react'
import { Box, Card, CardContent, LinearProgress, Stack, TextField, Typography } from '@mui/material'

import type { IndexHealth, SettingsData } from '../api/types'
import { queueCapHelper, queueCapRows, validateQueueCap } from '../lib/queueCaps'

interface EmbedQueueSettingsSectionProps {
  data: SettingsData
  /** Live index health, for the current queue depth beside each limit. */
  indexHealth?: IndexHealth | null
  /** SettingsTab's save: posts the setting, toasts, and reloads settings. */
  save: (key: string, value: string) => Promise<void>
}

/**
 * Settings card for the embed queue soft limits (retry, priority, background). Changes apply
 * live; a limit set by its EMBED_QUEUE_*_CAP environment variable is read-only here.
 */
export const EmbedQueueSettingsSection = ({ data, indexHealth, save }: EmbedQueueSettingsSectionProps) => {
  const [errors, setErrors] = useState<Record<string, string>>({})
  const rows = queueCapRows(data.EmbedQueueCaps, indexHealth)
  const commit = (key: string, value: number, hardMax: number, raw: string) => {
    const err = validateQueueCap(raw, hardMax)
    setErrors((prev) => {
      const { [key]: _drop, ...rest } = prev
      return err ? { ...rest, [key]: err } : rest
    })
    if (err || raw.trim() === String(value)) return
    void save(key, raw.trim())
  }
  return (
    <Card variant="outlined" id="settings-embed-queue" sx={{ mb: 2, scrollMarginTop: { xs: 120, md: 120 } }}>
      <CardContent>
        <Typography variant="subtitle1" gutterBottom>
          Embed queue
        </Typography>
        <Typography variant="caption" color="text.secondary" component="p" sx={{ mb: 1.5 }}>
          How many embed jobs each channel holds in memory. Lower limits use less memory and push back on indexing
          sooner; higher limits absorb bigger bursts. Changes apply immediately, no restart.
        </Typography>
        {rows.length === 0 ? (
          <Typography color="text.secondary">Embed queue limits unavailable.</Typography>
        ) : (
          <Stack direction="row" spacing={2} sx={{ flexWrap: 'wrap' }} useFlexGap>
            {rows.map((row) => (
              <Box key={row.key} sx={{ width: 240 }} data-testid={`queue-cap-${row.key}`}>
                <TextField
                  key={`${row.key}-${row.value}`}
                  label={`${row.label} limit`}
                  type="number"
                  size="small"
                  fullWidth
                  disabled={row.env_locked}
                  defaultValue={row.value}
                  error={!!errors[row.key]}
                  slotProps={{ htmlInput: { min: 1, max: row.hard_max, 'aria-label': row.key } }}
                  helperText={errors[row.key] ?? queueCapHelper(row)}
                  onBlur={(e) => commit(row.key, row.value, row.hard_max, e.target.value)}
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
        )}
      </CardContent>
    </Card>
  )
}
