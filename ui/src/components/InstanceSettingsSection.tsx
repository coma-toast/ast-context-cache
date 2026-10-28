import { useEffect, useState } from 'react'
import { Button, Card, CardContent, Checkbox, FormControlLabel, Stack, TextField, Typography } from '@mui/material'

import type { SettingsData } from '../api/types'
import { InstanceFaviconPicker } from './InstanceFaviconPicker'

interface InstanceSettingsSectionProps {
  data: SettingsData
  /** SettingsTab's save: posts the setting, toasts, and reloads settings. */
  save: (key: string, value: string) => Promise<void>
}

/** Settings key that lets the dashboard name win over -name / AST_INSTANCE_NAME. */
export const OVERRIDE_KEY = 'instance_name_override'

/**
 * General settings card: the instance name. When -name or AST_INSTANCE_NAME sets it, the field
 * shows that value read-only next to an override box; checking the box makes the dashboard name
 * win (it starts as the current name), unchecking goes back to the external value and keeps the
 * stored name for next time. The favicon picker below it follows the same pattern.
 */
export const InstanceSettingsSection = ({ data, save }: InstanceSettingsSectionProps) => {
  const external = data.InstanceNameExternal ?? ''
  const fromFlag = data.InstanceNameExternalSource === 'flag'
  const override = !!data.InstanceNameOverride
  const editable = !external || override
  const initial = external ? data.InstanceName || data.InstanceNameEffective || '' : (data.InstanceName ?? '')
  const [name, setName] = useState(initial)
  useEffect(() => {
    setName(initial)
  }, [initial])
  const sourceLabel = fromFlag ? 'the -name flag' : 'the AST_INSTANCE_NAME environment variable'
  const helper = !editable
    ? `Set by ${sourceLabel}. Check the box to rename it here.`
    : external
      ? `Overrides ${sourceLabel} (${external}). Uncheck to go back to it; this name is kept.`
      : '1–40 chars: letters, digits, spaces, - _ . · Shown in the tab title. Empty clears it.'
  return (
    <Card variant="outlined" id="settings-general" sx={{ mb: 2, scrollMarginTop: { xs: 120, md: 120 } }}>
      <CardContent>
        <Typography variant="subtitle1" gutterBottom>
          General
        </Typography>
        <Stack direction="row" spacing={1} sx={{ alignItems: 'flex-start', flexWrap: 'wrap' }} useFlexGap>
          <TextField
            label="Instance name"
            size="small"
            value={editable ? name : external}
            onChange={(e) => setName(e.target.value)}
            disabled={!editable}
            helperText={helper}
            slotProps={{ htmlInput: { maxLength: 40 } }}
            sx={{ minWidth: 240 }}
          />
          <Button size="small" variant="outlined" disabled={!editable || name.trim() === initial} onClick={() => save('instance_name', name.trim())}>
            Save
          </Button>
          {external && (
            <FormControlLabel
              control={<Checkbox size="small" checked={override} onChange={(e) => save(OVERRIDE_KEY, String(e.target.checked))} />}
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
        <InstanceFaviconPicker data={data} save={save} />
      </CardContent>
    </Card>
  )
}
