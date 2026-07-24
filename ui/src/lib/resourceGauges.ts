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

export function ssdWearColor(pct: number) {
  if (pct >= 80) return '#f85149'
  if (pct >= 50) return '#d29922'
  return '#3fb950'
}
