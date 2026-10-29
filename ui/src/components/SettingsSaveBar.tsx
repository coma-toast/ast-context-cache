import { Button, Chip, Stack } from '@mui/material'

import type { SettingsDraft } from '../hooks/useSettingsDraft'

/** Save / Discard for a section's draft, enabled only while it differs from what is saved. */
export const SettingsSaveBar = ({ draft, label }: { draft: SettingsDraft; label: string }) => (
  <Stack direction="row" spacing={1} sx={{ mt: 2, alignItems: 'center', justifyContent: 'flex-end', flexWrap: 'wrap' }} useFlexGap>
    {draft.dirty && <Chip label="Unsaved changes" color="warning" size="small" variant="outlined" />}
    <Button size="small" disabled={!draft.dirty || draft.saving} onClick={draft.discard} aria-label={`Discard ${label} changes`}>
      Discard
    </Button>
    <Button size="small" variant="contained" disabled={!draft.dirty || draft.saving} onClick={() => void draft.save()} aria-label={`Save ${label}`}>
      {draft.saving ? 'Saving…' : 'Save'}
    </Button>
  </Stack>
)
