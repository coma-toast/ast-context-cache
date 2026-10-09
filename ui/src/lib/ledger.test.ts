import { describe, expect, it } from 'vitest'

import type { Stats } from '../api/types'

import {
  conservativeLine,
  estimatedRowsNote,
  hostUsageBars,
  hostUsageTotals,
  ledgerSplit,
  showHostUsage,
  tokensSavedDetail,
  tokensSavedSub,
  virtualLedger,
} from './ledger'

const stats = (overrides: Partial<Stats> = {}): Stats =>
  ({ TokensSaved: 1500, DedupTokensSaved: 300, SavingsVsFiles: 4000, ...overrides }) as Stats

describe('ledgerSplit', () => {
  it('uses the ledger fields when present', () => {
    expect(ledgerSplit(stats({ CompressionSaved: 1000, DedupSaved: 500 }))).toEqual({ compression: 1000, dedup: 500, total: 1500 })
  })
  it('falls back to TokensSaved minus dedup on older servers', () => {
    expect(ledgerSplit(stats())).toEqual({ compression: 1200, dedup: 300, total: 1500 })
  })
})

describe('conservative and estimated lines', () => {
  it('omits both when absent', () => {
    expect(conservativeLine(stats())).toBe('')
    expect(estimatedRowsNote(stats({ EstimatedRows: 0 }))).toBe('')
  })
  it('formats both', () => {
    expect(conservativeLine(stats({ ConservativeSaved: 2500 }))).toBe('conservative: 2.5k')
    expect(estimatedRowsNote(stats({ EstimatedRows: 1 }))).toBe('1 earlier row estimated')
    expect(estimatedRowsNote(stats({ EstimatedRows: 42 }))).toBe('42 earlier rows estimated')
  })
})

describe('tokensSaved text', () => {
  it('lists the split, conservative line, definitions in order, and the footnote', () => {
    const s = stats({
      CompressionSaved: 1000,
      DedupSaved: 500,
      ConservativeSaved: 700,
      EstimatedRows: 3,
      BaselineDefinitions: { dedup: 'D', compression: 'C', virtual: 'V' },
    })
    expect(tokensSavedDetail(s).split('\n')).toEqual([
      'Compression: 1.0k · Dedup: 500 · vs whole files: 4.0k',
      'Conservative: 700',
      'compression: C',
      'dedup: D',
      '3 earlier rows estimated',
    ])
    expect(tokensSavedSub(s, '50')).toBe('1.5k in 30d · 50/day · conservative: 700 · 3 earlier rows estimated')
  })
})

describe('virtualLedger', () => {
  it('is null without ledger fields', () => {
    expect(virtualLedger(stats())).toBeNull()
  })
  it('defaults missing parts to zero', () => {
    expect(virtualLedger(stats({ VirtualStoredTokens: 9000, VirtualFetchedTokens: 800 }))).toEqual({ stored: 9000, fetched: 800, recalled: 0 })
  })
})

describe('host usage', () => {
  const days = [
    { Day: '2026-10-01', Input: 10, Output: 5, CacheRead: 80, CacheWrite: 5 },
    { Day: '2026-10-02', Input: 5, Output: 5, CacheRead: 40, CacheWrite: 0 },
  ]
  it('is shown only when enabled', () => {
    expect(showHostUsage(null)).toBe(false)
    expect(showHostUsage({ Enabled: false, WindowDays: 30, Days: [] })).toBe(false)
    expect(showHostUsage({ Enabled: true, WindowDays: 30, Days: [] })).toBe(true)
  })
  it('totals and scales days', () => {
    expect(hostUsageTotals(days)).toEqual({ Input: 15, Output: 10, CacheRead: 120, CacheWrite: 5 })
    expect(hostUsageBars(days)).toEqual([
      { day: '2026-10-01', total: 100, pct: 100 },
      { day: '2026-10-02', total: 50, pct: 50 },
    ])
    expect(hostUsageBars([])).toEqual([])
  })
})
