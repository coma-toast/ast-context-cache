/** Port of internal/dashboard/components/wal_ui.go string/progress helpers. */

export const WAL_PHASE_PAUSING = 'pausing'
export const WAL_PHASE_DRAINING = 'draining'
export const WAL_PHASE_CHECKPOINT = 'checkpoint'
export const WAL_PHASE_RESTORING = 'restoring'

/** Matches db.DefaultLogPath display hint when HOME is unavailable in the browser. */
export const DEFAULT_LOG_PATH = '~/.astcache/ast-mcp.log'

export type WalHealthFields = {
  WALMaintenanceActive?: boolean
  WALMaintenancePhase?: string
  WALMaintenanceMode?: string
  WALMaintenanceStarted?: string
  WALWalStartBytes?: number
  WALWalCurrentBytes?: number
  WALBusyStreak?: number
  WALInFlight?: number
  WALLastBusy?: number
}

/** Matches db.FormatFileSize. */
export function formatFileSize(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '0 B'
  if (bytes >= 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  if (bytes >= 1024) return `${Math.floor(bytes / 1024)} KB`
  return `${Math.floor(bytes)} B`
}

export function isZeroTime(iso?: string | null): boolean {
  if (!iso) return true
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return true
  return d.getUTCFullYear() <= 1
}

export function walMaintenanceElapsed(started?: string | null, nowMs = Date.now()): string {
  if (isZeroTime(started)) return '0s'
  const startMs = new Date(started!).getTime()
  const sec = Math.max(0, Math.round((nowMs - startMs) / 1000))
  if (sec < 60) return `${sec}s`
  const m = Math.floor(sec / 60)
  const s = sec % 60
  return `${m}m${String(s).padStart(2, '0')}s`
}

export function walMaintenanceHeadline(): string {
  return 'Compacting database WAL'
}

export function walMaintenanceDetail(h: WalHealthFields, nowMs = Date.now(), logPath = DEFAULT_LOG_PATH): string {
  const elapsed = walMaintenanceElapsed(h.WALMaintenanceStarted, nowMs)
  switch (h.WALMaintenancePhase) {
    case WAL_PHASE_PAUSING: {
      const inFlight = h.WALInFlight ?? 0
      if (inFlight > 0) return `Pausing embed workers… (${inFlight} in flight) · ${elapsed}`
      return `Pausing embed workers… · ${elapsed}`
    }
    case WAL_PHASE_DRAINING:
      return `Flushing write buffers… · ${elapsed}`
    case WAL_PHASE_CHECKPOINT:
      return walCheckpointDetail(h, elapsed, nowMs, logPath)
    case WAL_PHASE_RESTORING:
      return `Restoring embed workers… · ${elapsed}`
    default:
      return `Checkpoint in progress… · ${elapsed}`
  }
}

function walCheckpointDetail(h: WalHealthFields, elapsed: string, nowMs: number, logPath: string): string {
  const start = formatFileSize(h.WALWalStartBytes ?? 0)
  const cur = formatFileSize(h.WALWalCurrentBytes ?? 0)
  const mode = h.WALMaintenanceMode || 'TRUNCATE'
  if (h.WALLastBusy === 1) {
    if (!isZeroTime(h.WALMaintenanceStarted) && nowMs - new Date(h.WALMaintenanceStarted!).getTime() > 2 * 60 * 1000) {
      return `TRUNCATE blocked — deferring until readers idle · ${start} → ${cur} · streak ${h.WALBusyStreak ?? 0} · ${elapsed}`
    }
    return `Waiting for DB readers (busy) · ${start} → ${cur} · streak ${h.WALBusyStreak ?? 0} · ${elapsed} · logs: tail ${logPath}`
  }
  const startBytes = h.WALWalStartBytes ?? 0
  const curBytes = h.WALWalCurrentBytes ?? 0
  if (startBytes > 0 && curBytes < startBytes) {
    return `Checkpoint ${mode} · ${start} → ${cur} · ${elapsed}`
  }
  return `Checkpoint ${mode} · ${cur} · ${elapsed}`
}

export function walMaintenanceProgressPct(h: WalHealthFields): number {
  const startBytes = h.WALWalStartBytes ?? 0
  if (!h.WALMaintenanceActive || startBytes <= 0) return 0
  const curBytes = h.WALWalCurrentBytes ?? 0
  if (h.WALLastBusy === 1 && curBytes >= startBytes) return 0
  let shrink = 1 - curBytes / startBytes
  if (shrink < 0) shrink = 0
  if (shrink > 1) shrink = 1
  return shrink * 100
}

export function walMaintenanceIndeterminate(h: WalHealthFields): boolean {
  return Boolean(h.WALMaintenanceActive) && h.WALLastBusy === 1 && walMaintenanceProgressPct(h) < 1
}

export function walCheckpointButtonDisabled(walActive: boolean, startupBusy: boolean): boolean {
  return walActive || startupBusy
}

export function workerWalBadgeText(walActive: boolean, target: number, effective: number): string {
  if (walActive) return 'checkpointing'
  if (effective < target && target > 0) return `WAL ${effective}/${target}`
  return ''
}

export function workerWalBadgeTitle(h: WalHealthFields, target: number, effective: number, nowMs = Date.now()): string {
  if (h.WALMaintenanceActive) {
    return `${walMaintenanceHeadline()} — ${walMaintenanceDetail(h, nowMs)}`
  }
  if (effective >= target) return ''
  return `SQLite WAL throttled: ${effective} of ${target} worker goroutines running`
}

export function formatAutoRecoverAgo(unixSec: number, nowMs = Date.now()): string {
  if (!unixSec || unixSec <= 0) return ''
  const agoSec = Math.max(0, Math.floor(nowMs / 1000 - unixSec))
  if (agoSec < 60) return `${agoSec}s ago`
  if (agoSec < 3600) return `${Math.floor(agoSec / 60)}m ago`
  if (agoSec < 86400) return `${Math.floor(agoSec / 3600)}h ago`
  return `${Math.floor(agoSec / 86400)}d ago`
}
