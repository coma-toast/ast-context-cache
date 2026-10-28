import type { CoverageDigest, FeatureCoverage } from '../api/types'
import { chartColors } from './chartColors'

/** Search-less share at or above this is flagged: most sessions never searched code. */
export const SEARCHLESS_WARN_PCT = 50

export interface CoverageRow {
  feature: string
  label: string
  sessions: number
  pct: number
  calls: number
  /** Week-over-week change in points; null when there is no comparable prior window. */
  delta: number | null
  color: string
}

/** Bar colour by session share: green at half or more, blue at a fifth, orange below. */
export function coverageColor(pct: number): string {
  if (pct >= 50) return chartColors.green
  if (pct >= 20) return chartColors.accent
  return chartColors.orange
}

/**
 * Card rows in feature order. The digest trend only applies when it covers the same
 * window as the report (the digest is always 7 days).
 */
export function coverageRows(report: FeatureCoverage, trend?: CoverageDigest | null): CoverageRow[] {
  const deltas = new Map<string, number>()
  if (trend && trend.WindowDays === report.WindowDays && trend.PrevActiveSessions > 0) {
    for (const f of trend.Features ?? []) deltas.set(f.Feature, f.DeltaPct)
  }
  return (report.Features ?? []).map((f) => ({
    feature: f.Feature,
    label: f.Label,
    sessions: f.Sessions,
    pct: f.Pct,
    calls: f.Calls,
    delta: deltas.has(f.Feature) ? (deltas.get(f.Feature) as number) : null,
    color: coverageColor(f.Pct),
  }))
}

/** Signed point change, e.g. "+5.2 pts", "−3 pts", "±0". */
export function formatDelta(delta: number): string {
  if (Math.abs(delta) < 0.05) return '±0'
  const abs = Math.abs(delta)
  const n = Number.isInteger(abs) ? String(abs) : abs.toFixed(1)
  return `${delta > 0 ? '+' : '−'}${n} pts`
}

/** True when too many sessions never used code search. */
export function searchlessWarn(report: FeatureCoverage): boolean {
  return report.ActiveSessions > 0 && report.SearchlessPct >= SEARCHLESS_WARN_PCT
}

/** Last path segment of a project, or a placeholder for calls without one. */
export function projectLabel(path: string): string {
  return path.split('/').filter(Boolean).pop() || '(no project)'
}
