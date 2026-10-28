import { useEffect, useRef, useState } from 'react'
import { Box, Button, Checkbox, FormControlLabel, Radio, RadioGroup, Stack, TextField, Typography } from '@mui/material'

import type { SettingsData } from '../api/types'
import { checkFaviconFile, DEFAULT_FAVICON_COLOR, FAVICON_KEYS, readAsDataURL } from '../lib/favicon'

interface InstanceFaviconPickerProps {
  data: SettingsData
  /** SettingsTab's save: posts the setting, toasts, and reloads settings. */
  save: (key: string, value: string) => Promise<void>
}

/**
 * Favicon picker in the General card: today's icon, the icon in a chosen color with an optional
 * 1–2 character badge, or an uploaded PNG/SVG/ICO (≤ 64 KB). When AST_INSTANCE_FAVICON_COLOR or
 * AST_INSTANCE_FAVICON_BADGE sets it, the controls are read-only next to an override box, like the
 * instance name. The open tab's icon follows the new URL once settings reload.
 */
export const InstanceFaviconPicker = ({ data, save }: InstanceFaviconPickerProps) => {
  const extColor = data.InstanceFaviconExternalColor ?? ''
  const extBadge = data.InstanceFaviconExternalBadge ?? ''
  const external = !!(extColor || extBadge)
  const override = !!data.InstanceFaviconOverride
  const editable = !external || override
  const mode = editable ? data.InstanceFaviconMode || 'default' : 'color'
  const storedColor = editable ? data.InstanceFaviconColor || DEFAULT_FAVICON_COLOR : extColor || DEFAULT_FAVICON_COLOR
  const storedBadge = editable ? (data.InstanceFaviconBadge ?? '') : extBadge
  const hasUpload = !!data.InstanceFaviconUploadType
  const [color, setColor] = useState(storedColor)
  const [badge, setBadge] = useState(storedBadge)
  const [error, setError] = useState('')
  const fileRef = useRef<HTMLInputElement>(null)
  useEffect(() => setColor(storedColor), [storedColor])
  useEffect(() => setBadge(storedBadge), [storedBadge])

  const changed = color !== storedColor || badge.trim() !== storedBadge
  const apply = async () => {
    if (color !== storedColor || !data.InstanceFaviconColor) await save(FAVICON_KEYS.color, color)
    if (badge.trim() !== storedBadge) await save(FAVICON_KEYS.badge, badge.trim())
  }
  const pickMode = (next: string) => {
    setError('')
    if (next === 'upload' && !hasUpload) fileRef.current?.click()
    else void save(FAVICON_KEYS.mode, next)
  }
  const upload = async (file: File | undefined) => {
    if (!file) return
    const problem = checkFaviconFile(file)
    setError(problem)
    if (!problem) await save(FAVICON_KEYS.upload, await readAsDataURL(file))
  }
  const envNames = [extColor && 'AST_INSTANCE_FAVICON_COLOR', extBadge && 'AST_INSTANCE_FAVICON_BADGE'].filter(Boolean).join(' and ')
  const uploadLabel = hasUpload ? `Upload (${data.InstanceFaviconUploadType}, ${Math.ceil((data.InstanceFaviconUploadBytes ?? 0) / 1024)} KB)` : 'Upload'

  return (
    <Box sx={{ mt: 2 }} data-testid="favicon-picker">
      <Stack direction="row" spacing={1} sx={{ alignItems: 'center' }}>
        <Typography variant="body2">Favicon</Typography>
        {data.InstanceFaviconURL && (
          <Box component="img" src={data.InstanceFaviconURL} alt="Current favicon" sx={{ width: 24, height: 24, objectFit: 'contain' }} />
        )}
      </Stack>
      <RadioGroup row value={mode} onChange={(e) => pickMode(e.target.value)}>
        <FormControlLabel value="default" control={<Radio size="small" />} label="Default" disabled={!editable} />
        <FormControlLabel value="color" control={<Radio size="small" />} label="Color" disabled={!editable} />
        <FormControlLabel value="upload" control={<Radio size="small" />} label={uploadLabel} disabled={!editable} />
      </RadioGroup>
      {mode === 'color' && (
        <Stack direction="row" spacing={1} sx={{ alignItems: 'flex-start', flexWrap: 'wrap' }} useFlexGap>
          <TextField
            label="Color"
            type="color"
            size="small"
            value={color}
            onChange={(e) => setColor(e.target.value)}
            disabled={!editable}
            sx={{ width: 96 }}
          />
          <TextField
            label="Badge"
            size="small"
            value={badge}
            onChange={(e) => setBadge(e.target.value)}
            disabled={!editable}
            helperText="Optional, 1–2 letters or digits"
            slotProps={{ htmlInput: { maxLength: 2, 'aria-label': 'Favicon badge' } }}
            sx={{ width: 180 }}
          />
          <Button size="small" variant="outlined" disabled={!editable || (!changed && !!data.InstanceFaviconColor)} onClick={() => void apply()}>
            Apply
          </Button>
        </Stack>
      )}
      {mode === 'upload' && editable && (
        <Button size="small" variant="outlined" onClick={() => fileRef.current?.click()}>
          Replace file…
        </Button>
      )}
      <input
        ref={fileRef}
        type="file"
        hidden
        accept=".png,.svg,.ico,image/png,image/svg+xml,image/x-icon"
        data-testid="favicon-file"
        onChange={(e) => {
          void upload(e.target.files?.[0])
          e.target.value = ''
        }}
      />
      {error && (
        <Typography variant="caption" color="error" sx={{ display: 'block', mt: 0.5 }} role="alert">
          {error}
        </Typography>
      )}
      {external && (
        <FormControlLabel
          control={<Checkbox size="small" checked={override} onChange={(e) => save(FAVICON_KEYS.override, String(e.target.checked))} />}
          label="Override the environment favicon"
        />
      )}
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
        {!editable
          ? `Set by ${envNames}. Check the box to change it here.`
          : external
            ? `Overrides ${envNames}. Uncheck to go back to it; these settings are kept.`
            : 'Tells this instance’s browser tabs apart. Uploads: PNG, SVG or ICO up to 64 KB; SVG with scripts, event handlers or external links is rejected.'}
      </Typography>
    </Box>
  )
}
