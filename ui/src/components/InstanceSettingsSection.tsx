import { Card, CardContent, Checkbox, FormControlLabel, Stack, TextField, Typography } from '@mui/material'

import type { SettingsData } from '../api/types'
import { useSettingsDraft } from '../hooks/useSettingsDraft'
import { generalSaved, NAME_KEY, OVERRIDE_KEY, settingsCardSx } from '../lib/settingsSections'
import { InstanceFaviconPicker } from './InstanceFaviconPicker'
import { SettingsSaveBar } from './SettingsSaveBar'

interface InstanceSettingsSectionProps {
  data: SettingsData
}

/**
 * General settings card: the instance name and the favicon, saved together. When -name or
 * AST_INSTANCE_NAME sets the name, the field shows that value read-only next to an override box;
 * checking the box makes the dashboard name win, unchecking goes back to the external value and
 * keeps the stored name for next time. The favicon picker below it follows the same pattern.
 */
export const InstanceSettingsSection = ({ data }: InstanceSettingsSectionProps) => {
  const draft = useSettingsDraft('general', 'General', generalSaved(data))
  const external = data.InstanceNameExternal ?? ''
  const fromFlag = data.InstanceNameExternalSource === 'flag'
  const override = draft.value(OVERRIDE_KEY) === 'true'
  const editable = !external || override
  const sourceLabel = fromFlag ? 'the -name flag' : 'the AST_INSTANCE_NAME environment variable'
  const helper = !editable
    ? `Set by ${sourceLabel}. Check the box to rename it here.`
    : external
      ? `Overrides ${sourceLabel} (${external}). Uncheck to go back to it; this name is kept.`
      : '1–40 chars: letters, digits, spaces, - _ . · Shown in the tab title. Empty clears it.'
  return (
    <Card variant="outlined" id="settings-general" sx={settingsCardSx(draft.dirty)}>
      <CardContent>
        <Typography variant="subtitle1" gutterBottom>
          General
        </Typography>
        <Stack direction="row" spacing={1} sx={{ alignItems: 'flex-start', flexWrap: 'wrap' }} useFlexGap>
          <TextField
            label="Instance name"
            size="small"
            value={editable ? draft.value(NAME_KEY) : external}
            onChange={(e) => draft.set(NAME_KEY, e.target.value)}
            disabled={!editable}
            error={!!draft.errors[NAME_KEY]}
            helperText={draft.errors[NAME_KEY] ?? helper}
            slotProps={{ htmlInput: { maxLength: 40 } }}
            sx={{ minWidth: 240 }}
          />
          {external && (
            <FormControlLabel
              control={<Checkbox size="small" checked={override} onChange={(e) => draft.set(OVERRIDE_KEY, String(e.target.checked))} />}
              label={fromFlag ? 'Override the -name flag value' : 'Override the environment value'}
            />
          )}
        </Stack>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
          The name scopes the login cookie and saved UI state: renaming keeps this browser logged in, but other browsers must log in again
          and saved UI state starts fresh.
        </Typography>
        {data.InstanceNameEffective && (
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
            Active: {data.InstanceNameEffective} (from {data.InstanceNameSource || 'settings'})
          </Typography>
        )}
        <InstanceFaviconPicker data={data} draft={draft} />
        <SettingsSaveBar draft={draft} label="General" />
      </CardContent>
    </Card>
  )
}
