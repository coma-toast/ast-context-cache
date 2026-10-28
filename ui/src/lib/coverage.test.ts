import { describe, expect, it } from 'vitest'

import type { CoverageDigest, FeatureCoverage } from '../api/types'
import { chartColors } from './chartColors'
import { coverageColor, coverageRows, formatDelta, projectLabel, searchlessWarn } from './coverage'

const report: FeatureCoverage = {
  WindowDays: 7,
  Since: '2026-10-21T12:00:00Z',
  Until: '2026-10-28T12:00:00Z',
  ActiveSessions: 4,
  AnonymousSessions: 1,
  Calls: 9,
  ErrorCalls: 0,
  SearchlessSessions: 3,
  SearchlessPct: 75,
  Features: [
    { Feature: 'search', Label: 'Code search', Sessions: 1, Pct: 25, Calls: 1 },
    { Feature: 'memory', Label: 'Structured memory', Sessions: 2, Pct: 50, Calls: 3 },
    { Feature: 'impact', Label: 'Impact analysis', Sessions: 0, Pct: 0, Calls: 0 },
  ],
  TopProjects: [],
}

const trend: CoverageDigest = {
  WindowDays: 7,
  ActiveSessions: 4,
  PrevActiveSessions: 2,
  SearchlessSessions: 3,
  PrevSearchlessSessions: 1,
  SearchlessPct: 75,
  PrevSearchlessPct: 50,
  Features: [
    { Feature: 'search', Label: 'Code search', Sessions: 1, Pct: 25, PrevPct: 50, DeltaPct: -25 },
    { Feature: 'memory', Label: 'Structured memory', Sessions: 2, Pct: 50, PrevPct: 50, DeltaPct: 0 },
  ],
}

describe('coverageRows', () => {
  it('joins week-over-week deltas by feature', () => {
    const rows = coverageRows(report, trend)
    expect(rows.map((r) => [r.feature, r.delta])).toEqual([
      ['search', -25],
      ['memory', 0],
      ['impact', null],
    ])
    expect(rows[1].color).toBe(chartColors.green)
  })

  it.each([
    ['no trend', null],
    ['different window', { ...trend, WindowDays: 30 }],
    ['empty prior window', { ...trend, PrevActiveSessions: 0 }],
  ])('drops deltas with %s', (_, t) => {
    expect(coverageRows(report, t).every((r) => r.delta === null)).toBe(true)
  })
})

describe('coverageColor', () => {
  it.each([
    [80, chartColors.green],
    [50, chartColors.green],
    [20, chartColors.accent],
    [5, chartColors.orange],
  ])('%d%% is %s', (pct, want) => {
    expect(coverageColor(pct)).toBe(want)
  })
})

describe('formatDelta', () => {
  it.each([
    [5.2, '+5.2 pts'],
    [-3, '−3 pts'],
    [0, '±0'],
    [12, '+12 pts'],
  ])('%d → %s', (d, want) => {
    expect(formatDelta(d)).toBe(want)
  })
})

describe('searchlessWarn', () => {
  it('flags a mostly search-less window', () => {
    expect(searchlessWarn(report)).toBe(true)
    expect(searchlessWarn({ ...report, SearchlessPct: 20 })).toBe(false)
    expect(searchlessWarn({ ...report, ActiveSessions: 0 })).toBe(false)
  })
})

describe('projectLabel', () => {
  it.each([
    ['/Users/jason/git/ast-context-cache', 'ast-context-cache'],
    ['', '(no project)'],
  ])('%s → %s', (p, want) => {
    expect(projectLabel(p)).toBe(want)
  })
})
