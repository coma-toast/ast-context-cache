import { Card, CardContent, Stack, TextField, Typography } from '@mui/material'

import type { HandoffLimits, SettingsData } from '../api/types'
import { useSettingsDraft } from '../hooks/useSettingsDraft'
import { settingsCardSx } from '../lib/settingsSections'
import { SettingsSaveBar } from './SettingsSaveBar'

/** One handoff limit: its settings key, the HandoffLimits field it resolves to, and its label. */
interface HandoffKnob {
  key: string
  field: keyof HandoffLimits
  label: string
  helper: string
}

const HANDOFF_KNOBS: HandoffKnob[] = [
  { key: 'handoff_ttl_days', field: 'ttl_days', label: 'Tree TTL (days)', helper: 'Since last access' },
  { key: 'handoff_child_inactive_minutes', field: 'child_inactive_minutes', label: 'Inactivity (minutes)', helper: 'Then a child is abandoned' },
  { key: 'handoff_summary_max_tokens', field: 'summary_max_tokens', label: 'Summary cap (tokens)', helper: 'Returned to the parent' },
  { key: 'handoff_open_budget_tokens', field: 'open_budget_tokens', label: 'Open budget (tokens)', helper: 'Default open digest size' },
  { key: 'handoff_tree_max_tokens', field: 'tree_max_tokens', label: 'Tree max tokens', helper: 'Snapshots, results, scratchpad' },
  { key: 'handoff_tree_max_entries', field: 'tree_max_entries', label: 'Tree max entries', helper: 'Per tree' },
  { key: 'handoff_max_depth', field: 'max_depth', label: 'Max depth', helper: 'Nested handoffs' },
  { key: 'handoff_max_children', field: 'max_children', label: 'Max children', helper: 'Per handoff' },
]

const ENV_PREFIX = 'AST_'

interface HandoffSettingsSectionProps {
  data: SettingsData
}

/** Settings card for the handoff retention windows and caps. Each must be a positive integer. */
export const HandoffSettingsSection = ({ data }: HandoffSettingsSectionProps) => {
  const limits = data.HandoffLimits
  const locked = new Set(data.HandoffEnvLocked ?? [])
  const saved: Record<string, string> = {}
  for (const knob of HANDOFF_KNOBS) if (limits && !locked.has(knob.key)) saved[knob.key] = String(limits[knob.field])
  const draft = useSettingsDraft('handoff', 'Handoff', saved)
  return (
    <Card variant="outlined" id="settings-handoff" sx={settingsCardSx(draft.dirty)}>
      <CardContent>
        <Typography variant="subtitle1" gutterBottom>
          Handoff
        </Typography>
        <Typography variant="caption" color="text.secondary" component="p" sx={{ mb: 1.5 }}>
          Retention and caps for subagent handoff trees. Changes apply to the next handoff operation; a value set by its
          AST_HANDOFF_* environment variable is read-only here.
        </Typography>
        {!limits ? (
          <Typography color="text.secondary">Handoff limits unavailable.</Typography>
        ) : (
          <>
            <Stack direction="row" spacing={2} sx={{ flexWrap: 'wrap' }} useFlexGap>
              {HANDOFF_KNOBS.map((knob) => {
                const envLocked = locked.has(knob.key)
                return (
                  <TextField
                    key={knob.key}
                    label={knob.label}
                    type="number"
                    size="small"
                    disabled={envLocked}
                    value={envLocked ? limits[knob.field] : draft.value(knob.key)}
                    onChange={(e) => draft.set(knob.key, e.target.value)}
                    error={!!draft.errors[knob.key]}
                    slotProps={{ htmlInput: { min: 1, 'aria-label': knob.key } }}
                    helperText={draft.errors[knob.key] ?? (envLocked ? `Set by ${ENV_PREFIX}${knob.key.toUpperCase()}` : knob.helper)}
                    sx={{ width: 200 }}
                  />
                )
              })}
            </Stack>
            <SettingsSaveBar draft={draft} label="Handoff" />
          </>
        )}
      </CardContent>
    </Card>
  )
}
