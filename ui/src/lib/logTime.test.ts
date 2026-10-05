import { describe, expect, it } from 'vitest'

import { shortLogTime } from './logTime'

describe('shortLogTime', () => {
  it.each([
    ['2026-10-05T12:34:56.789123-05:00', '12:34:56'],
    ['2026-10-05T07:08:09Z', '07:08:09'],
    ['2026-10-05 23:59:01', '23:59:01'],
    [' 2026-10-05T01:02:03+02:00 ', '01:02:03'],
    ['12:34:56', '12:34:56'],
    ['not a time', 'not a time'],
    ['', ''],
  ])('%j -> %j', (ts, want) => {
    expect(shortLogTime(ts)).toBe(want)
  })
})
