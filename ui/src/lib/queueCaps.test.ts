import { describe, expect, it } from 'vitest'

import type { IndexHealth, QueueCapState } from '../api/types'

import { queueCapHelper, queueCapRows, validateQueueCap } from './queueCaps'

const cap = (overrides: Partial<QueueCapState> = {}): QueueCapState => ({
  key: 'embed_queue_low_cap',
  env: 'EMBED_QUEUE_LOW_CAP',
  label: 'Background',
  default: 2048,
  hard_max: 16384,
  value: 2048,
  used: 10,
  env_locked: false,
  ...overrides,
})

describe('queueCapRows', () => {
  it('uses live depth from index health per channel', () => {
    const health = { EmbedPendingQueued: 3, EmbedHighQueued: 64, EmbedLowQueued: 1024 } as IndexHealth
    const rows = queueCapRows(
      [
        cap({ key: 'embed_queue_pending_cap', value: 128 }),
        cap({ key: 'embed_queue_high_cap', value: 128 }),
        cap({ key: 'embed_queue_low_cap', value: 2048 }),
      ],
      health,
    )
    expect(rows.map((r) => [r.depth, r.fillPct])).toEqual([
      [3, 2],
      [64, 50],
      [1024, 50],
    ])
  })

  it('falls back to the settings depth and caps fill at 100%', () => {
    const [row] = queueCapRows([cap({ value: 4, used: 9 })], null)
    expect(row.depth).toBe(9)
    expect(row.fillPct).toBe(100)
  })

  it('handles missing caps', () => {
    expect(queueCapRows(undefined)).toEqual([])
    expect(queueCapRows(null, null)).toEqual([])
  })
})

describe('validateQueueCap', () => {
  it.each([
    ['1', null],
    ['16384', null],
    [' 300 ', null],
    ['0', 'Must be 1–16384'],
    ['16385', 'Must be 1–16384'],
    ['-2', 'Enter a whole number 1–16384'],
    ['1.5', 'Enter a whole number 1–16384'],
    ['', 'Enter a whole number 1–16384'],
    ['lots', 'Enter a whole number 1–16384'],
  ])('%j -> %j', (raw, want) => {
    expect(validateQueueCap(raw, 16384)).toBe(want)
  })
})

describe('queueCapHelper', () => {
  it('names the env var when locked', () => {
    const [row] = queueCapRows([cap({ env_locked: true })])
    expect(queueCapHelper(row)).toBe('Set by EMBED_QUEUE_LOW_CAP')
  })

  it('shows depth and hard max otherwise', () => {
    const [row] = queueCapRows([cap({ used: 7 })])
    expect(queueCapHelper(row)).toBe('7 queued now · max 16384')
  })
})
