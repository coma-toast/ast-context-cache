/** Gauge width helpers — mirror internal/dashboard/components/metrics.go soft caps. */

import { formatFileSize } from './walUi'

export function cpuGaugePct(p: number) {
  return Math.min(100, Math.max(0, p))
}

export function memoryPct(mb: number) {
  const softMax = 512
  return Math.min(100, (mb / softMax) * 100)
}

export function diskFootprintPct(diskMb: number, walMb: number) {
  const softMax = 2048
  return Math.min(100, ((diskMb + walMb) / softMax) * 100)
}

export function diskIoPct(mbps: number) {
  const softMax = 200
  return Math.min(100, (mbps / softMax) * 100)
}

export function loadAvgBarWidth(load1: number, cpus: number) {
  const util = cpus > 0 ? (load1 / cpus) * 100 : load1 * 100
  return Math.min(100, Math.max(0, util))
}

export function loadAvgLabel(load1: number, load5: number, load15: number, cpus: number) {
  const per = (l: number) => (cpus > 0 ? l / cpus : l)
  return `${per(load1).toFixed(2)}× · ${per(load5).toFixed(2)}× · ${per(load15).toFixed(2)}× · ${cpus}c`
}

export function formatDiskIo(mbps: number) {
  if (mbps <= 0) return '0 MB/s'
  if (mbps >= 1024) return `${(mbps / 1024).toFixed(1)} GB/s`
  if (mbps >= 10) return `${Math.round(mbps)} MB/s`
  return `${mbps.toFixed(1)} MB/s`
}

export function diskSizeLabel(
  diskSize: string,
  walSize: string,
  walMaintenance: boolean,
  walStart?: number,
  walCurrent?: number,
): string {
  if (diskSize === '-' || !diskSize) return '-'
  if (walMaintenance && walStart && walStart > 0 && walCurrent != null) {
    const start = formatFileSize(walStart)
    const cur = formatFileSize(walCurrent)
    if (cur !== start) return `${diskSize} · WAL ${start} → ${cur}`
    return `${diskSize} · WAL ${cur} · compacting`
  }
  if (!walSize || walSize === '0 B') return diskSize
  return `${diskSize} · WAL ${walSize}`
}

/** Open descriptors as a share of the process limit (0 when the limit is unknown). */
export function fdPct(open: number, limit: number) {
  if (limit <= 0) return 0
  return Math.min(100, Math.max(0, (open / limit) * 100))
}

export function fdLabel(open: number, limit: number) {
  const n = (v: number) => v.toLocaleString('en-US')
  return limit > 0 ? `${n(open)} / ${n(limit)}` : n(open)
}

/** Levels come from the server (internal/sys/fds.go: warning ≥70%, critical ≥90%). */
export function fdColor(level: string | undefined) {
  if (level === 'critical') return '#f85149'
  if (level === 'warning') return '#d29922'
  return '#3fb950'
}

export function gaugeLevel(pct: number): 'ok' | 'warn' | 'critical' {
  if (pct >= 85) return 'critical'
  if (pct >= 60) return 'warn'
  return 'ok'
}

export function ssdSmartTone(status: string): 'success' | 'warning' | 'error' | 'default' {
  const s = status.toLowerCase()
  if (s.includes('verified') || s.includes('ok') || s.includes('normal')) return 'success'
  if (s.includes('fail') || s.includes('error') || s.includes('critical')) return 'error'
  return 'default'
}

export function diskTone(status?: string): 'success' | 'warning' | 'error' | 'default' {
  if (!status) return 'default'
  const s = status.toLowerCase()
  if (s.includes('verified') || s.includes('ok') || s.includes('normal')) return 'success'
  if (s.includes('fail') || s.includes('error') || s.includes('critical')) return 'error'
  if (s.includes('warn') || s.includes('degraded')) return 'warning'
  return 'default'
}

export function diskHealthLevel(data: DiskInfo): 'ok' | 'warn' | 'critical' {
  if (data.WearUsedPct != null) {
    if (data.WearUsedPct >= 80) return 'critical'
    if (data.WearUsedPct >= 50) return 'warn'
  }
  if (data.TemperatureC != null) {
    if (data.TemperatureC >= 65) return 'critical'
    if (data.TemperatureC >= 55) return 'warn'
  }
  if (data.WarningLevel === 'critical' || data.WarningMsg?.includes('critical')) return 'critical'
  if (data.WarningLevel === 'warning' || data.WarningMsg?.includes('warning')) return 'warn'
  return 'ok'
}

export function diskLabel(data: DiskInfo): string {
  const parts = []
  if (data.Device) parts.push(data.Device)
  if (data.Model) parts.push(data.Model)
  return parts.join(' · ') || '-'
}

export function diskSubtitle(data: DiskInfo): string {
  const parts = []
  if (data.Protocol) parts.push(data.Protocol)
  if (data.Capacity) parts.push(data.Capacity)
  if (data.FreeSpace && data.FreeSpace !== data.Capacity) parts.push(`Free ${data.FreeSpace}`)
  return parts.join(' · ') || '-'
}

export function diskTempColor(tempC?: number): string {
  if (tempC == null) return '#3fb950'
  if (tempC >= 65) return '#f85149'
  if (tempC >= 55) return '#d29922'
  return '#3fb950'
}

export function diskTempLabel(tempC?: number): string {
  if (tempC == null) return 'N/A'
  return `${tempC}°C`
}

export function formatPowerOnHours(hh?: number): string {
  if (hh == null) return '-'
  const days = Math.floor(hh / 24)
  if (days > 365) return `${Math.floor(days / 365)} yr`
  if (days > 30) return `${Math.floor(days / 30)} mo`
  return `${days} days`
}

export function diskHasWarning(data: DiskInfo): boolean {
  if (data.TemperatureC != null && (data.TemperatureC >= 65 || data.TemperatureC >= 55)) return true
  if (data.WearUsedPct != null && (data.WearUsedPct >= 80 || data.WearUsedPct >= 50)) return true
  if (data.WarningLevel || data.WarningMsg) return true
  return false
}

export function diskWarningMessage(data: DiskInfo): string {
  if (!data.WarningMsg) return ''
  return data.WarningMsg
}

export function ssdWearColor(pct: number) {
  if (pct >= 80) return '#f85149'
  if (pct >= 50) return '#d29922'
  return '#3fb950'
}
