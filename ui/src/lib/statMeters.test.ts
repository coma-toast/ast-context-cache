import { describe, expect, it } from 'vitest'

import { fmtDailyAvg } from './statMeters'

describe('fmtDailyAvg', () => {
  it.each([
    [0, '0'],
    [45, '1.5'],
    [300, '10'],
    [3_000, '100'],
    [1_820_000, '60.7k'],
    [360_000, '12k'],
    [45_000_000, '1.5M'],
  ])('averages %d over 30 days as %s', (total, want) => {
    expect(fmtDailyAvg(total)).toBe(want)
  })
})
