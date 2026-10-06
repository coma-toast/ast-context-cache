import { describe, expect, it } from 'vitest'

import type { UpdateCheckResult } from '../api/types'

import { formatReleaseDate, UPDATE_STEPS, updateHeadline, updateStepIndex } from './updates'

const check = (overrides: Partial<UpdateCheckResult> = {}): UpdateCheckResult => ({
  current_version: '4.0.6',
  build: 'release',
  source_build: false,
  latest_version: '4.0.7',
  release_url: 'https://github.com/coma-toast/ast-context-cache/releases/tag/v4.0.7',
  published_at: '2026-10-06T12:00:00Z',
  asset_name: 'ast-context-cache_4.0.7_darwin_arm64.tar.gz',
  update_available: true,
  ...overrides,
})

describe('updateStepIndex', () => {
  it('walks the steps in order and ends past the last one', () => {
    expect(['downloading', 'verifying', 'installing', 'installed'].map(updateStepIndex)).toEqual([0, 1, 2, UPDATE_STEPS.length])
  })

  it('is idle for no phase or an error', () => {
    expect(updateStepIndex(undefined)).toBe(-1)
    expect(updateStepIndex('error')).toBe(-1)
  })
})

describe('updateHeadline', () => {
  it('names the newer release', () => {
    expect(updateHeadline(check())).toBe('v4.0.7 available')
  })

  it('says up to date when nothing newer exists', () => {
    expect(updateHeadline(check({ current_version: '4.0.7', update_available: false }))).toBe('Up to date with v4.0.7')
  })

  it('offers the release build to a source build of the same version', () => {
    expect(updateHeadline(check({ current_version: '4.0.7', build: 'source', source_build: true }))).toBe('v4.0.7 release build available')
  })

  it('reports a failed lookup and a pending one', () => {
    expect(updateHeadline(check({ error: 'offline' }))).toBe('Could not check for updates')
    expect(updateHeadline(null)).toBe('Checking for updates…')
  })
})

describe('formatReleaseDate', () => {
  it('formats a valid timestamp and drops a bad one', () => {
    expect(formatReleaseDate('2026-10-06T12:00:00Z')).toMatch(/2026/)
    expect(formatReleaseDate('')).toBe('')
  })
})
