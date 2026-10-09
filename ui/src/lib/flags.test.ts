import { describe, expect, it } from 'vitest'

import type { FlagState } from '../api/types'

import { buildFlagRows, envLockReason, HANDOFF_MASTER_KEY, needsToolListWarning, parentFlagKey, withFlagEnabled } from './flags'

const flag = (key: string, overrides: Partial<FlagState> = {}): FlagState => ({
  key,
  description: `${key} description`,
  source: 'default',
  env: `AST_${key.toUpperCase()}`,
  enabled: true,
  default: true,
  locked: false,
  affects_tools: false,
  ...overrides,
})

const keys = (flags: FlagState[]) => buildFlagRows(flags).map((r) => r.flag.key)

describe('parentFlagKey', () => {
  it.each([
    ['feature_handoff', ''],
    ['feature_handoff_hooks', HANDOFF_MASTER_KEY],
    ['feature_handoff_new_child', HANDOFF_MASTER_KEY],
    ['feature_handoffish', ''],
    ['feature_shared_query_cache', ''],
  ])('%s -> %j', (key, want) => {
    expect(parentFlagKey(key)).toBe(want)
  })
})

describe('buildFlagRows', () => {
  it('nests children directly under their master and keeps registry order otherwise', () => {
    const flags = [
      flag('feature_shared_query_cache'),
      flag('feature_handoff_claims'),
      flag('feature_handoff'),
      flag('feature_handoff_hooks'),
    ]
    expect(keys(flags)).toEqual(['feature_shared_query_cache', 'feature_handoff', 'feature_handoff_claims', 'feature_handoff_hooks'])
    const rows = buildFlagRows(flags)
    expect(rows.map((r) => r.parentKey)).toEqual(['', '', HANDOFF_MASTER_KEY, HANDOFF_MASTER_KEY])
  })

  it('shows an orphan child top-level instead of dropping it', () => {
    const rows = buildFlagRows([flag('feature_handoff_hooks')])
    expect(rows).toHaveLength(1)
    expect(rows[0].parentKey).toBe('')
    expect(rows[0].disabledReason).toBe('')
  })

  it.each([
    { name: 'master on', master: true, child: {}, reason: '', blocked: false },
    { name: 'master off', master: false, child: { enabled: false }, reason: 'Off while feature_handoff is off', blocked: true },
    {
      name: 'env lock wins over master off',
      master: false,
      child: { enabled: false, locked: true, source: 'env' as const },
      reason: 'Set by AST_FEATURE_HANDOFF_HOOKS in the environment',
      blocked: true,
    },
    {
      name: 'env lock with master on',
      master: true,
      child: { locked: true, source: 'env' as const },
      reason: 'Set by AST_FEATURE_HANDOFF_HOOKS in the environment',
      blocked: false,
    },
  ])('child read-only reason: $name', ({ master, child, reason, blocked }) => {
    const rows = buildFlagRows([flag('feature_handoff', { enabled: master }), flag('feature_handoff_hooks', child)])
    expect(rows[0].disabledReason).toBe('')
    expect(rows[1].disabledReason).toBe(reason)
    expect(rows[1].blockedByParent).toBe(blocked)
  })

  it('locks an env-set master', () => {
    const master = flag('feature_handoff', { locked: true, source: 'env' })
    expect(buildFlagRows([master])[0].disabledReason).toBe(envLockReason(master))
  })
})

describe('withFlagEnabled', () => {
  it('flips only the toggled flag and marks it as a setting', () => {
    const before = [flag('feature_handoff'), flag('feature_shared_query_cache')]
    const after = withFlagEnabled(before, 'feature_handoff', false)
    expect(after[0]).toMatchObject({ enabled: false, source: 'setting' })
    expect(after[1]).toBe(before[1])
    expect(before[0].enabled).toBe(true)
  })
})

describe('needsToolListWarning', () => {
  it.each([
    ['tool flag turned off', { affects_tools: true, enabled: true }, false, true],
    ['tool flag turned on', { affects_tools: true, enabled: false }, true, true],
    ['tool flag set to its current value', { affects_tools: true, enabled: true }, true, false],
    ['flag that does not affect tools', { affects_tools: false, enabled: true }, false, false],
  ])('%s', (_name, overrides, enabled, want) => {
    expect(needsToolListWarning(flag('feature_x', overrides), enabled)).toBe(want)
  })

  it('is false for an unknown flag', () => {
    expect(needsToolListWarning(undefined, false)).toBe(false)
  })
})
