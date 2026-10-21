import { useState, useEffect } from 'react'
import { Alert, AlertTitle, Box, Button, Stack, Typography } from '@mui/material'
import { api } from '../api/client'
import { useToast } from '../context/ToastContext'

export function RecoveryBanner() {
  const [degraded, setDegraded] = useState(false)
  const [loading, setLoading] = useState(false)
  const { showToast } = useToast()

  useEffect(() => {
    api.health().then((h) => {
      if ((h as any).DataDirPhase === 'data_dir_unavailable') {
        setDegraded(true)
      }
    }).catch(() => {})
  }, [])

  if (!degraded) return null

  const handleRetry = async () => {
    setLoading(true)
    try {
      await api.restartNow()
      showToast('Restarting server…', 'success')
      setTimeout(() => window.location.reload(), 2000)
    } catch (e: any) {
      showToast(e.message || 'Failed to restart', 'error')
      setLoading(false)
    }
  }

  const handleUseDefault = async () => {
    setLoading(true)
    try {
      await api.clearDataDirOverride()
      showToast('Override cleared. Restarting…', 'success')
      await api.restartNow()
      setTimeout(() => window.location.reload(), 2000)
    } catch (e: any) {
      showToast(e.message || 'Failed to reset to default location', 'error')
      setLoading(false)
    }
  }

  return (
    <Alert severity="error" sx={{ mb: 3 }} variant="filled">
      <AlertTitle sx={{ fontWeight: 700 }}>Data Directory Unavailable</AlertTitle>
      <Typography variant="body2" sx={{ mb: 2 }}>
        The configured data directory cannot be accessed (e.g. unplugged external drive or missing path).
        The server is running in degraded recovery mode. Choose an option below to restore full functionality:
      </Typography>
      <Stack direction="row" spacing={2} alignItems="center">
        <Button variant="contained" color="inherit" size="small" onClick={handleRetry} disabled={loading}>
          Retry Restart
        </Button>
        <Button variant="outlined" color="inherit" size="small" onClick={handleUseDefault} disabled={loading}>
          Use Default Location (~/.astcache)
        </Button>
      </Stack>
    </Alert>
  )
}
