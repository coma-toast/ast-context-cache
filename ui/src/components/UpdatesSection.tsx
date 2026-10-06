import { useCallback, useEffect, useState } from 'react'
import { Alert, Box, Button, Card, CardContent, Chip, LinearProgress, Link, Stack, Step, StepLabel, Stepper, Typography } from '@mui/material'

import { api } from '../api/client'
import type { UpdateCheckResult, UpdateStatus } from '../api/types'
import { useToast } from '../context/ToastContext'
import { restartNow } from '../lib/restart'
import { formatReleaseDate, UPDATE_STEPS, updateHeadline, updateStepIndex } from '../lib/updates'

export function UpdatesSection() {
  const { showToast } = useToast()
  const [check, setCheck] = useState<UpdateCheckResult | null>(null)
  const [checking, setChecking] = useState(false)
  const [status, setStatus] = useState<UpdateStatus | null>(null)
  const [starting, setStarting] = useState(false)
  const [restarting, setRestarting] = useState(false)

  const runCheck = useCallback(async () => {
    setChecking(true)
    try {
      setCheck(await api.updateCheck())
    } catch (e) {
      showToast(String(e), 'error')
    } finally {
      setChecking(false)
    }
  }, [showToast])

  useEffect(() => {
    runCheck()
  }, [runCheck])

  useEffect(() => {
    let cancelled = false
    const poll = async () => {
      try {
        const s = await api.updateStatus()
        if (!cancelled) setStatus(s)
      } catch {
        // transient poll failure — try again next tick
      }
    }
    poll()
    const id = setInterval(poll, 1500)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  const startUpdate = async () => {
    setStarting(true)
    try {
      await api.startUpdate()
    } catch (e) {
      showToast(String(e), 'error')
    } finally {
      setStarting(false)
    }
  }

  return (
    <UpdatesCard
      check={check}
      status={status}
      checking={checking}
      starting={starting}
      restarting={restarting}
      onCheck={runCheck}
      onUpdate={startUpdate}
      onRestart={() => restartNow(showToast, setRestarting)}
    />
  )
}

export interface UpdatesCardProps {
  check: UpdateCheckResult | null
  status: UpdateStatus | null
  checking: boolean
  starting: boolean
  restarting: boolean
  onCheck: () => void
  onUpdate: () => void
  onRestart: () => void
}

export function UpdatesCard({ check, status, checking, starting, restarting, onCheck, onUpdate, onRestart }: UpdatesCardProps) {
  const active = status?.active ?? false
  const step = updateStepIndex(status?.phase)
  const released = check?.published_at ? formatReleaseDate(check.published_at) : ''
  return (
    <Card variant="outlined" id="settings-updates" sx={{ mb: 2 }}>
      <CardContent>
        <Stack direction="row" spacing={1} useFlexGap sx={{ alignItems: 'center', flexWrap: 'wrap', mb: 1.5 }}>
          <Typography variant="subtitle1" sx={{ mr: 'auto' }}>
            Updates
          </Typography>
          {check && (
            <Chip
              size="small"
              variant="outlined"
              color={check.source_build ? 'warning' : 'default'}
              label={`v${check.current_version} · ${check.source_build ? 'source build' : 'release build'}`}
            />
          )}
        </Stack>

        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={{ xs: 1, sm: 4 }} sx={{ mb: 1.5 }}>
          <Box>
            <Typography variant="caption" color="text.secondary">Installed</Typography>
            <Typography variant="body2">{check ? `v${check.current_version}` : '—'}</Typography>
          </Box>
          <Box>
            <Typography variant="caption" color="text.secondary">Latest release</Typography>
            <Typography variant="body2">
              {check?.latest_version ? (
                <>
                  <Link href={check.release_url} target="_blank" rel="noopener noreferrer" underline="hover">
                    v{check.latest_version}
                  </Link>
                  {released && <Typography component="span" variant="body2" color="text.secondary">{` · ${released}`}</Typography>}
                </>
              ) : (
                '—'
              )}
            </Typography>
          </Box>
          <Box>
            <Typography variant="caption" color="text.secondary">Status</Typography>
            <Typography variant="body2" color={check?.update_available ? 'primary' : 'text.primary'}>{updateHeadline(check)}</Typography>
          </Box>
        </Stack>

        {check?.error && (
          <Alert severity="warning" sx={{ mb: 1.5 }}>
            {check.error}
          </Alert>
        )}
        {check?.source_build && check.update_available && !active && !status?.done && (
          <Alert severity="info" sx={{ mb: 1.5 }}>
            This ast-mcp was built from source, so its version number may not match the code it runs. Updating replaces it with the
            v{check.latest_version} release build.
          </Alert>
        )}

        {(active || status?.done) && (
          <Box sx={{ mb: 1.5 }}>
            <Stepper activeStep={step} alternativeLabel sx={{ mb: 1 }}>
              {UPDATE_STEPS.map((label) => (
                <Step key={label}>
                  <StepLabel>{label}</StepLabel>
                </Step>
              ))}
            </Stepper>
            {active && <LinearProgress />}
          </Box>
        )}
        {!active && status?.done && (
          <Alert
            severity="success"
            sx={{ mb: 1.5 }}
            action={
              <Button color="inherit" size="small" disabled={restarting} onClick={onRestart}>
                {restarting ? 'Restarting…' : 'Restart now'}
              </Button>
            }
          >
            Installed v{status.to_version}
            {status.from_version ? ` (was v${status.from_version})` : ''}. Restart ast-mcp to start using it.
          </Alert>
        )}
        {!active && status?.error && (
          <Alert severity="error" sx={{ mb: 1.5 }}>
            {status.error}
          </Alert>
        )}

        <Stack direction="row" spacing={1} useFlexGap sx={{ alignItems: 'center', flexWrap: 'wrap' }}>
          <Button
            variant="contained"
            size="small"
            disabled={active || starting || !check?.update_available || !!status?.done}
            onClick={onUpdate}
          >
            {active ? 'Updating…' : check?.update_available ? `Update to v${check.latest_version}` : 'Update'}
          </Button>
          <Button variant="text" size="small" disabled={checking || active} onClick={onCheck}>
            {checking ? 'Checking…' : 'Check for updates'}
          </Button>
        </Stack>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
          Downloads {check?.asset_name || 'the release build for this platform'} from GitHub, checks it against the release's
          checksums.txt, and replaces ast-mcp in place. The previous binary is kept as ast-mcp.prev.
        </Typography>
      </CardContent>
    </Card>
  )
}
