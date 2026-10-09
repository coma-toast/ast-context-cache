import { formatNum } from '../api/client'
import type { HostUsageDay, HostUsageResponse, Stats } from '../api/types'

/** Order the definitions appear in the Tokens saved tooltip. */
export const DEFINITION_ORDER = ['compression', 'dedup', 'conservative', 'estimated'] as const

/** Compression and dedup savings; older servers without ledgers fall back to TokensSaved. */
export const ledgerSplit = (s: Stats): { compression: number; dedup: number; total: number } => {
  if (s.CompressionSaved == null) {
    const dedup = s.DedupTokensSaved ?? 0
    return { compression: Math.max(0, (s.TokensSaved ?? 0) - dedup), dedup, total: s.TokensSaved ?? 0 }
  }
  const compression = s.CompressionSaved
  const dedup = s.DedupSaved ?? 0
  return { compression, dedup, total: compression + dedup }
}

/** "Conservative: X" line, or '' when the server has no conservative baseline yet. */
export const conservativeLine = (s: Stats): string =>
  s.ConservativeSaved == null ? '' : `conservative: ${formatNum(s.ConservativeSaved)}`

/** Footnote shown when some rows were counted as bytes/4 estimates. */
export const estimatedRowsNote = (s: Stats): string => {
  const n = s.EstimatedRows ?? 0
  if (n <= 0) return ''
  return `${formatNum(n)} earlier row${n === 1 ? '' : 's'} estimated`
}

/** Tokens saved card tooltip: the split, then one line per baseline definition. */
export const tokensSavedDetail = (s: Stats): string => {
  const { compression, dedup } = ledgerSplit(s)
  const lines = [`Compression: ${formatNum(compression)} · Dedup: ${formatNum(dedup)} · vs whole files: ${formatNum(s.SavingsVsFiles ?? 0)}`]
  const conservative = conservativeLine(s)
  if (conservative) lines.push(conservative.charAt(0).toUpperCase() + conservative.slice(1))
  const defs = s.BaselineDefinitions ?? {}
  for (const key of DEFINITION_ORDER) {
    if (defs[key]) lines.push(`${key}: ${defs[key]}`)
  }
  const note = estimatedRowsNote(s)
  if (note) lines.push(note)
  return lines.join('\n')
}

/** Tokens saved card sub-line: 30d total plus the conservative figure and footnote. */
export const tokensSavedSub = (s: Stats, dailyAvg: string): string =>
  [`${formatNum(ledgerSplit(s).total)} in 30d · ${dailyAvg}/day`, conservativeLine(s), estimatedRowsNote(s)].filter(Boolean).join(' · ')

/** Virtual ledger stats for the Virtual context card, or null on servers without ledgers. */
export const virtualLedger = (s: Stats): { stored: number; fetched: number; recalled: number } | null =>
  s.VirtualStoredTokens == null
    ? null
    : { stored: s.VirtualStoredTokens, fetched: s.VirtualFetchedTokens ?? 0, recalled: s.VirtualRecalledTokens ?? 0 }

/** Host usage card is shown only when transcript ingest is on. */
export const showHostUsage = (h: HostUsageResponse | null | undefined): h is HostUsageResponse => !!h?.Enabled

/** Totals over the host usage window. */
export const hostUsageTotals = (days: HostUsageDay[]): Omit<HostUsageDay, 'Day'> =>
  days.reduce(
    (acc, d) => ({
      Input: acc.Input + d.Input,
      Output: acc.Output + d.Output,
      CacheRead: acc.CacheRead + d.CacheRead,
      CacheWrite: acc.CacheWrite + d.CacheWrite,
    }),
    { Input: 0, Output: 0, CacheRead: 0, CacheWrite: 0 },
  )

/** Bar heights (0–100) of each day's total tokens relative to the busiest day. */
export const hostUsageBars = (days: HostUsageDay[]): { day: string; pct: number; total: number }[] => {
  const totals = days.map((d) => d.Input + d.Output + d.CacheRead + d.CacheWrite)
  const max = Math.max(0, ...totals)
  return days.map((d, i) => ({ day: d.Day, total: totals[i], pct: max > 0 ? (totals[i] / max) * 100 : 0 }))
}
