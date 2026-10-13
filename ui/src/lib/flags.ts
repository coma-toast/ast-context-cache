import type { FlagState } from '../api/types'

/** Master switch: while it is off every `feature_handoff_*` child is effectively off. */
export const HANDOFF_MASTER_KEY = 'feature_handoff'

const HANDOFF_CHILD_PREFIX = `${HANDOFF_MASTER_KEY}_`

/** One rendered row of the Features section, in display order. */
export interface FlagRow {
  flag: FlagState
  /** Key of the master switch this flag is nested under, or '' for a top-level flag. */
  parentKey: string
  /** Why the switch is read-only, or '' when it can be toggled. */
  disabledReason: string
  /** True when the row is read-only because its master switch is off (shown as an inline hint). */
  blockedByParent: boolean
}

/** The master switch that implies `key`, mirroring `parentOf` in internal/flags. */
export const parentFlagKey = (key: string): string =>
  key.startsWith(HANDOFF_CHILD_PREFIX) ? HANDOFF_MASTER_KEY : ''

export const envLockReason = (flag: FlagState): string => `Set by ${flag.env} in the environment`

/**
 * Orders flags for display (each master followed by its children, otherwise registry order) and
 * works out which switches are read-only. A child whose master is missing from the list is shown
 * top-level rather than dropped.
 */
export const buildFlagRows = (flags: FlagState[]): FlagRow[] => {
  const byKey = new Map(flags.map((f) => [f.key, f]))
  const parentOf = (f: FlagState): FlagState | undefined => byKey.get(parentFlagKey(f.key))
  const row = (flag: FlagState): FlagRow => {
    const parent = parentOf(flag)
    const blockedByParent = !!parent && !parent.enabled
    let disabledReason = ''
    if (flag.locked) disabledReason = envLockReason(flag)
    else if (blockedByParent) disabledReason = `Off while ${parent.key} is off`
    return { flag, parentKey: parent?.key ?? '', disabledReason, blockedByParent }
  }
  const rows: FlagRow[] = []
  for (const f of flags) {
    if (parentOf(f)) continue
    rows.push(row(f))
    for (const child of flags) {
      if (parentOf(child)?.key === f.key) rows.push(row(child))
    }
  }
  return rows
}

/** Confirmation shown before toggling a flag that changes the tool list (TS-4). */
export const TOOL_LIST_WARNING =
  'Changing this updates the tool list. Connected agents lose their prompt cache; sessions that started before the change keep the old list until they reconnect.'

/** Whether setting `flag` to `enabled` changes the tool list, so the toggle needs confirming first. */
export const needsToolListWarning = (flag: FlagState | undefined, enabled: boolean): boolean =>
  !!flag?.affects_tools && flag.enabled !== enabled

/** Optimistically applies a toggle; the server's response replaces it (including children). */
export const withFlagEnabled = (flags: FlagState[], key: string, enabled: boolean): FlagState[] =>
  flags.map((f) => (f.key === key ? { ...f, enabled, source: 'setting' } : f))
