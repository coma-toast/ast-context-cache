import { describe, expect, it } from 'vitest'

import { diskTone, sampleAge } from './resourceGauges'

describe('diskTone', () => {
  it.each([
    ['PASSED', 'success'],
    ['Verified', 'success'],
    ['FAILED', 'error'],
    ['', 'default'],
  ])('%s → %s', (status, want) => {
    expect(diskTone(status)).toBe(want)
  })
})

describe('sampleAge', () => {
  const now = 1_800_000_000_000
  it.each([
    [0, ''],
    [now / 1000 - 30, '30s ago'],
    [now / 1000 - 600, '10 min ago'],
    [now / 1000 - 7200, '2 h ago'],
    [now / 1000 - 3 * 86400, '3 d ago'],
  ])('%d → %s', (unix, want) => {
    expect(sampleAge(unix, now)).toBe(want)
  })
})
