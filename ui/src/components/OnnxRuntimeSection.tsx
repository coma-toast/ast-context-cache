import { useState } from 'react'
import { Alert, Button, Stack, Typography } from '@mui/material'

import { api } from '../api/client'
import type { SettingsData } from '../api/types'
import { useToast } from '../context/ToastContext'
import { auxUnavailable } from '../lib/auxEmbedder'
import { restartNow } from '../lib/restart'

interface OnnxRuntimeSectionProps {
  data: SettingsData
  onRefresh: () => void
}

/**
 * ONNX Runtime state on the Embedding card. When the runtime is missing it says why and offers
 * the verified managed install; when a managed install is older than the pinned version it
 * offers Update. Either way the new runtime loads after Restart now.
 */
export const OnnxRuntimeSection = ({ data, onRefresh }: OnnxRuntimeSectionProps) => {
  const { showToast } = useToast()
  const [installing, setInstalling] = useState(false)
  const [installed, setInstalled] = useState<string | null>(null)
  const [restarting, setRestarting] = useState(false)
  const ort = data.ONNXRuntime
  if (!ort) return null
  const install = async () => {
    setInstalling(true)
    try {
      const res = await api.installONNXRuntime()
      const ver = res.version || ort.pinned
      setInstalled(ver)
      showToast(`ONNX Runtime ${ver} installed. Restart ast-mcp to load it.`, 'success')
      onRefresh()
    } catch (e) {
      showToast(String(e), 'error')
    } finally {
      setInstalling(false)
    }
  }
  const installButton = (label: string) => (
    <Button color="inherit" size="small" disabled={installing} onClick={install}>
      {installing ? 'Installing…' : label}
    </Button>
  )
  const needsRestart = installed !== null || ort.restart_required
  const reason = data.AuxUnavailableReason || ort.reason || 'ONNX Runtime is not available'
  return (
    <Stack spacing={1} sx={{ mb: 2 }} data-testid="onnxruntime-section">
      {needsRestart ? (
        <Alert
          severity="info"
          action={
            <Button color="inherit" size="small" disabled={restarting} onClick={() => restartNow(showToast, setRestarting)}>
              {restarting ? 'Restarting…' : 'Restart now'}
            </Button>
          }
        >
          ONNX Runtime {installed || ort.installed} is installed. Restart ast-mcp to load it.
        </Alert>
      ) : !ort.available ? (
        <Alert severity="warning" action={ort.install_supported ? installButton('Install ONNX Runtime') : undefined}>
          {auxUnavailable(data) ? 'Local (aux) embedder unavailable' : 'ONNX Runtime unavailable'}: {reason}
          {!ort.install_supported && ' No managed build for this platform: set ONNXRUNTIME_LIB to ONNX Runtime 1.25 or newer.'}
        </Alert>
      ) : (
        <Alert severity="success" variant="outlined" action={ort.update_available ? installButton(`Update to ${ort.pinned}`) : undefined}>
          ONNX Runtime {ort.version || '(version unknown)'} · {ort.source}
          <Typography variant="caption" component="div" color="text.secondary" sx={{ wordBreak: 'break-all' }}>
            {ort.path}
          </Typography>
        </Alert>
      )}
    </Stack>
  )
}
