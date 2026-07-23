import { Alert, Box, Button, Card, CardContent, Chip, LinearProgress, Stack, Typography } from '@mui/material'
import { useState } from 'react'
import type { IndexHealth } from '../api/types'
import { api, formatNum } from '../api/client'
import { useToast } from '../context/ToastContext'
import { chartColors } from '../lib/chartColors'
import { pendingRingCap, throughputPct } from '../lib/embedGauges'
import {
  formatAutoRecoverAgo,
  walCheckpointButtonDisabled,
  walMaintenanceDetail,
  walMaintenanceHeadline,
  walMaintenanceIndeterminate,
  walMaintenanceProgressPct,
  workerWalBadgeText,
  workerWalBadgeTitle,
} from '../lib/walUi'
import { ChannelBar } from './charts/ChannelBar'
import { RingGauge } from './charts/RingGauge'
import { WorkerControls } from './WorkerControls'

export function EmbeddingsPanel({ data, onRefresh }: { data: IndexHealth; onRefresh?: () => void }) {
  const queueCap = (data.EmbedHighCap ?? 128) + (data.EmbedLowCap ?? 2048)
  const pendingCap = pendingRingCap(data.EmbedPending, data.EmbedPendingPeak)
  const embedState = data.EmbedderState || 'unknown'
  const embedOk = embedState === 'healthy' || embedState === 'ready' || embedState === 'ok'
  const stateColor =
    embedState === 'error' ? 'error' : embedOk ? 'success' : embedState === 'degraded' ? 'warning' : 'default'
  const startupBusy = embedState === 'loading' || embedState === 'starting'
  const showWalBanner = Boolean(data.WALMaintenanceActive)
  const checkpointDisabled = walCheckpointButtonDisabled(showWalBanner, startupBusy)
  const effective = data.EmbedWorkersEffective ?? data.EmbedWorkers
  const walBadge = workerWalBadgeText(showWalBanner, data.EmbedWorkers, effective) || undefined
  const walTitle = walBadge ? workerWalBadgeTitle(data, data.EmbedWorkers, effective) || undefined : undefined
  const hasError = Boolean(data.EmbedderError)
  // While WAL is compacting, the maintenance banner owns the alert slot (matches templ).
  const showAlert =
    !showWalBanner &&
    (hasError || embedState === 'error' || embedState === 'degraded' || embedState === 'loading')
  const panelBusy = (data.EmbedActive ?? 0) > 0 || data.EmbedQueued > 0
  // Match HTMX ShowEmbedDismiss, plus lingering alert after recovery (ready + error text).
  const showDismiss =
    embedState === 'degraded' || (embedState === 'error' && panelBusy) || (embedOk && hasError)
  // Always offer Retry except during cold startup (probe is already in flight).
  const showRetry = embedState !== 'loading'
  const showBacklogHint =
    !panelBusy &&
    (data.EmbedThroughput ?? 0) === 0 &&
    (embedState === 'error' || embedState === 'degraded') &&
    (data.EmbedPending ?? 0) > 0
  const autoRecoverAgo = formatAutoRecoverAgo(data.LastAutoRecoverUnix ?? 0)

  return (
    <Card variant="outlined" sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="subtitle2" gutterBottom>
          Embeddings
        </Typography>
        <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap sx={{ mb: 1.5 }}>
          {data.EmbedBackend && <Chip size="small" label={data.EmbedBackend} color="primary" variant="outlined" />}
          {data.EmbedModel && (
            <Typography variant="caption" sx={{ fontFamily: 'ui-monospace, monospace', maxWidth: 180 }} noWrap title={data.EmbedModel}>
              {data.EmbedModel}
            </Typography>
          )}
          {data.EmbedInSync != null && (
            <Chip size="small" label={data.EmbedInSync ? 'in sync' : 'out of sync'} color={data.EmbedInSync ? 'success' : 'warning'} />
          )}
          <Chip size="small" label={embedState} color={stateColor} />
        </Stack>

        {showWalBanner && <WALMaintenanceBanner data={data} />}

        {showAlert && (
          <EmbedderAlertBanner
            state={embedState}
            error={data.EmbedderError || ''}
            showDismiss={showDismiss}
            showRetry={showRetry}
            showBacklogHint={showBacklogHint}
            pending={data.EmbedPending ?? 0}
            onRefresh={onRefresh}
          />
        )}

        <Box sx={{ display: 'flex', gap: 2, flexWrap: { xs: 'wrap', sm: 'nowrap' }, mb: 2 }}>
          <Stack direction="row" spacing={1}>
            <RingGauge value={data.EmbedQueued} max={queueCap} caption="queued" />
            <RingGauge value={data.EmbedPending} max={pendingCap} caption="pending" />
          </Stack>
          <Box sx={{ flex: 1, minWidth: 160 }}>
            <ChannelBar label="Priority" used={data.EmbedHighQueued ?? 0} cap={data.EmbedHighCap ?? 128} color={chartColors.priority} />
            <ChannelBar label="Background" used={data.EmbedLowQueued ?? 0} cap={data.EmbedLowCap ?? 2048} color={chartColors.background} />
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mt: 1.5, mb: 0.5 }}>
              <Typography variant="caption" color="text.secondary">
                Workers
              </Typography>
              <Typography variant="caption">
                {(data.EmbedWorkers ?? 0) === 0 ? 'paused' : `${data.EmbedActivePrimary ?? 0} active`}
              </Typography>
            </Box>
            <WorkerControls
              workers={data.EmbedWorkers}
              active={data.EmbedActivePrimary ?? 0}
              max={data.EmbedWorkerMax}
              effective={data.EmbedWorkersEffective}
              live={data.EmbedWorkersLive}
              pool="primary"
              walBadge={walBadge}
              walTitle={walTitle}
              onChange={onRefresh}
            />
            {data.EmbedAuxEnabled && (
              <>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mt: 1.5, mb: 0.5 }}>
                  <Typography variant="caption" color="text.secondary">
                    {data.EmbedAuxBackend ? `Aux (${data.EmbedAuxBackend})` : 'Aux workers'}
                  </Typography>
                  <Typography variant="caption" title={data.EmbedAuxModel || undefined}>
                    {(data.EmbedAuxWorkers ?? 0) === 0
                      ? 'off'
                      : (data.EmbedAuxActive ?? 0) > 0
                        ? `${data.EmbedAuxActive} active`
                        : `${data.EmbedAuxWorkers} enabled`}
                  </Typography>
                </Box>
                <WorkerControls
                  workers={data.EmbedAuxWorkers ?? 0}
                  active={data.EmbedAuxActive ?? 0}
                  max={data.EmbedAuxWorkerMax ?? 10}
                  effective={data.EmbedAuxWorkersEffective}
                  live={data.EmbedAuxWorkersLive}
                  pool="aux"
                  onChange={onRefresh}
                />
              </>
            )}
            <Box sx={{ mt: 1.5 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                <Typography variant="caption" color="text.secondary">
                  Throughput
                </Typography>
                <Typography variant="caption" sx={{ fontFamily: 'ui-monospace, monospace' }}>
                  {data.EmbedThroughput?.toFixed(1) ?? 0}/s
                </Typography>
              </Box>
              <Box sx={{ height: 6, bgcolor: chartColors.track, borderRadius: 1, overflow: 'hidden' }}>
                <Box
                  sx={{
                    height: '100%',
                    width: `${throughputPct(data.EmbedThroughput ?? 0)}%`,
                    bgcolor: chartColors.green,
                    borderRadius: 1,
                    transition: 'width 0.35s',
                  }}
                />
              </Box>
            </Box>
          </Box>
        </Box>

        {(data.EmbedInProgress?.length ?? 0) > 0 && (
          <Box sx={{ mb: 1.5 }}>
            <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
              In progress
            </Typography>
            <Stack spacing={0.5}>
              {data.EmbedInProgress?.slice(0, 4).map((item) => (
                <Typography key={`${item.ProjectPath}-${item.File}`} variant="caption" sx={{ fontFamily: 'ui-monospace, monospace' }} noWrap>
                  {shortProject(item.ProjectPath)} · {item.File}
                </Typography>
              ))}
            </Stack>
          </Box>
        )}

        {(data.EmbedRecent?.length ?? 0) > 0 && (
          <Box sx={{ mb: 1.5 }}>
            <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
              Recent
            </Typography>
            <Stack spacing={0.5}>
              {data.EmbedRecent?.slice(0, 3).map((item) => (
                <Typography
                  key={`${item.ProjectPath}-${item.File}`}
                  variant="caption"
                  color="text.secondary"
                  sx={{ fontFamily: 'ui-monospace, monospace' }}
                  noWrap
                >
                  {shortProject(item.ProjectPath)} · {item.File}
                </Typography>
              ))}
            </Stack>
          </Box>
        )}

        <Stack direction="row" spacing={2} flexWrap="wrap" useFlexGap sx={{ pt: 1, borderTop: '1px solid', borderColor: 'divider' }}>
          <EmbedStat value={formatNum(data.TotalVectors)} label="Vectors cached" />
          <EmbedStat value={formatNum(data.EmbedComplete)} label="Total embedded" />
          <EmbedStat value={String(data.PinnedCount)} label="Pinned" />
        </Stack>

        {autoRecoverAgo && (
          <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
            Last auto-recover · {autoRecoverAgo}
          </Typography>
        )}

        {!showWalBanner && (
          <Box sx={{ mt: 1.5, pt: 1, borderTop: '1px solid', borderColor: 'divider' }}>
            <WALCheckpointButton
              disabled={checkpointDisabled}
              title={startupBusy ? 'Unavailable during startup' : undefined}
              onRefresh={onRefresh}
            />
          </Box>
        )}
      </CardContent>
    </Card>
  )
}

function WALMaintenanceBanner({ data }: { data: IndexHealth }) {
  const pct = walMaintenanceProgressPct(data)
  const indeterminate = walMaintenanceIndeterminate(data)
  return (
    <Alert severity="warning" sx={{ mb: 1.5, py: 0.75, alignItems: 'flex-start' }} role="status">
      <Typography variant="subtitle2" component="div">
        {walMaintenanceHeadline()}
      </Typography>
      <Typography
        variant="caption"
        component="div"
        sx={{ mt: 0.25, fontFamily: 'ui-monospace, monospace', wordBreak: 'break-word' }}
      >
        {walMaintenanceDetail(data)}
      </Typography>
      <LinearProgress
        variant={indeterminate ? 'indeterminate' : 'determinate'}
        value={indeterminate ? undefined : pct}
        color="warning"
        sx={{ mt: 1, height: 6, borderRadius: 1 }}
      />
      <Box sx={{ mt: 1 }}>
        <Button color="inherit" size="small" variant="outlined" disabled aria-disabled>
          Checkpoint running…
        </Button>
      </Box>
    </Alert>
  )
}

function WALCheckpointButton({
  disabled,
  title,
  onRefresh,
}: {
  disabled?: boolean
  title?: string
  onRefresh?: () => void
}) {
  const { showToast } = useToast()
  const [busy, setBusy] = useState(false)
  const run = async () => {
    if (busy || disabled) return
    setBusy(true)
    try {
      await api.walCheckpoint()
      showToast('WAL checkpoint started — embed workers will pause briefly', 'info')
      onRefresh?.()
    } catch (e) {
      showToast(e instanceof Error ? e.message : 'Could not start checkpoint', 'error')
      onRefresh?.()
    } finally {
      setBusy(false)
    }
  }
  return (
    <Button
      size="small"
      variant="outlined"
      color="warning"
      disabled={busy || disabled}
      title={title || 'Force WAL checkpoint (pauses embed workers briefly)'}
      onClick={() => void run()}
    >
      Checkpoint WAL now
    </Button>
  )
}

function EmbedderRetryButton({ onRefresh, disabled }: { onRefresh?: () => void; disabled?: boolean }) {
  const { showToast } = useToast()
  const [busy, setBusy] = useState(false)
  const retry = async () => {
    if (busy || disabled) return
    setBusy(true)
    try {
      const res = await api.embedderRetry()
      if (res.skipped) showToast(res.error || 'Probe deferred while embed work is in flight', 'info')
      else if (res.ok === false) showToast(res.error || 'Retry failed', 'error')
      else showToast('Embedder probe succeeded', 'success')
      onRefresh?.()
    } catch (e) {
      showToast(e instanceof Error ? e.message : 'Retry failed', 'error')
      onRefresh?.()
    } finally {
      setBusy(false)
    }
  }
  return (
    <Button color="inherit" size="small" variant="outlined" disabled={busy || disabled} onClick={() => void retry()}>
      Retry now
    </Button>
  )
}

function EmbedderAlertBanner({
  state,
  error,
  showDismiss,
  showRetry,
  showBacklogHint,
  pending,
  onRefresh,
}: {
  state: string
  error: string
  showDismiss: boolean
  showRetry: boolean
  showBacklogHint: boolean
  pending: number
  onRefresh?: () => void
}) {
  const { showToast } = useToast()
  const [busy, setBusy] = useState(false)
  const headline =
    state === 'loading'
      ? 'Starting up'
      : state === 'error'
        ? 'Embedder unreachable'
        : state === 'degraded'
          ? 'Recent embed errors'
          : 'Embedder alert'
  const detail = error.length > 160 ? `${error.slice(0, 157)}…` : error
  const severity = state === 'error' ? 'error' : state === 'loading' ? 'info' : 'warning'

  const dismiss = async () => {
    if (busy) return
    setBusy(true)
    try {
      const res = await api.embedderDismissAlert()
      if (res.ok === false) showToast(res.error || 'Dismiss failed', 'error')
      else showToast('Alert dismissed', 'success')
      onRefresh?.()
    } catch (e) {
      showToast(e instanceof Error ? e.message : 'Dismiss failed', 'error')
    } finally {
      setBusy(false)
    }
  }

  // Buttons live in the body (not Alert action) so they stay visible on the narrow Overview column.
  return (
    <Alert severity={severity} sx={{ mb: 1.5, py: 0.75, alignItems: 'flex-start' }}>
      <Typography variant="subtitle2" component="div">
        {headline}
      </Typography>
      {detail && (
        <Typography
          variant="caption"
          component="div"
          sx={{ mt: 0.25, fontFamily: 'ui-monospace, monospace', wordBreak: 'break-word' }}
          title={error}
        >
          {detail}
        </Typography>
      )}
      {showBacklogHint && (
        <Typography variant="caption" component="div" sx={{ mt: 0.5 }}>
          {pending.toLocaleString()} files awaiting retry after embed failure
        </Typography>
      )}
      {(showDismiss || showRetry) && (
        <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
          {showDismiss && (
            <Button color="inherit" size="small" disabled={busy} onClick={() => void dismiss()}>
              Dismiss
            </Button>
          )}
          {showRetry && <EmbedderRetryButton onRefresh={onRefresh} disabled={busy} />}
        </Stack>
      )}
    </Alert>
  )
}

function EmbedStat({ value, label }: { value: string; label: string }) {
  return (
    <Box>
      <Typography variant="body2" fontWeight={700} color="primary.main" sx={{ fontFamily: 'ui-monospace, monospace' }}>
        {value}
      </Typography>
      <Typography variant="caption" color="text.secondary">
        {label}
      </Typography>
    </Box>
  )
}

function shortProject(path: string) {
  const parts = path.split(/[/\\]/)
  return parts[parts.length - 1] || path
}
