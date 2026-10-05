import { describe, expect, it } from 'vitest'

import type { InstallerComponent, InstallerFileChange, InstallerStatus, InstallerTarget } from '../api/types'

import {
  changeKindLabel,
  classifyDiffLine,
  componentLabel,
  defaultSelection,
  diffLines,
  hasExternallyManaged,
  isRepreviewError,
  statusChip,
  unselectableReason,
  visibleComponents,
  writableChanges,
} from './diff'

const comp = (overrides: Partial<InstallerComponent> = {}): InstallerComponent => ({
  component: 'mcp',
  supported: true,
  path: '/Users/demo/.cursor/mcp.json',
  status: 'not_installed',
  ...overrides,
})

const target = (components: InstallerComponent[]): InstallerTarget => ({ id: 'claude_code', name: 'Claude Code', components })

const change = (overrides: Partial<InstallerFileChange> = {}): InstallerFileChange => ({
  target: 'cursor',
  component: 'mcp',
  path: '/Users/demo/.cursor/mcp.json',
  kind: 'modify',
  diff: '',
  skipped: false,
  ...overrides,
})

describe('classifyDiffLine', () => {
  it.each([
    ['+  "url": "x"', false, 'add'],
    ['-  "url": "y"', false, 'remove'],
    ['@@ -1,3 +1,4 @@', false, 'hunk'],
    ['--- a/Users/demo/.cursor/mcp.json', false, 'header'],
    ['+++ b/Users/demo/.cursor/mcp.json', false, 'header'],
    ['--- removed markdown rule', true, 'remove'],
    ['+++ added text', true, 'add'],
    ['   unchanged', true, 'context'],
    ['', true, 'context'],
  ] as const)('%j (in hunk %s) is %s', (line, inHunk, want) => {
    expect(classifyDiffLine(line, inHunk)).toBe(want)
  })
})

describe('diffLines', () => {
  it('classifies a whole diff, tracking hunks, and drops the trailing empty line', () => {
    const diff = [
      '--- a/Users/demo/AGENTS.md',
      '+++ b/Users/demo/AGENTS.md',
      '@@ -1,2 +1,3 @@',
      ' # Notes',
      '--- old rule',
      '+new rule',
      '',
    ].join('\n')
    expect(diffLines(diff).map((l) => l.kind)).toEqual(['header', 'header', 'hunk', 'context', 'remove', 'add'])
  })

  it('returns nothing for an empty diff', () => {
    expect(diffLines('')).toEqual([])
  })
})

describe('statusChip', () => {
  it.each([
    ['installed', 'Installed', 'success'],
    ['outdated', 'Outdated', 'warning'],
    ['modified_by_user', 'Modified by user', 'info'],
    ['missing', 'Missing', 'error'],
    ['not_installed', 'Not installed', 'default'],
    ['externally_managed', 'Externally managed', 'secondary'],
    ['unsupported', 'Unsupported', 'default'],
    ['covered', 'Covered', 'default'],
  ] as const)('%s → %s / %s', (status, label, color) => {
    expect(statusChip(status)).toEqual({ label, color })
  })

  it('renders an unknown status neutrally', () => {
    expect(statusChip('brand_new' as InstallerStatus)).toEqual({ label: 'brand_new', color: 'default' })
  })
})

describe('labels', () => {
  it('names components and change kinds', () => {
    expect(componentLabel('mcp')).toBe('MCP server')
    expect(componentLabel('hooks')).toBe('Hooks')
    expect(changeKindLabel('remove-block')).toBe('Remove block')
    expect(changeKindLabel('none')).toBe('No change')
  })
})

describe('component selection', () => {
  const hooks = comp({ component: 'hooks', path: '/Users/demo/.claude/settings.json' })
  const skills = comp({ component: 'skills', status: 'covered', status_reason: 'Claude Code skills cover it' })
  const rules = comp({ component: 'rules', supported: false, status: 'unsupported', reason: 'no rules location' })
  const mcp = comp({ status: 'installed' })
  const t = target([mcp, skills, rules, hooks])

  it('hides the hooks row unless the hooks flag is on', () => {
    expect(visibleComponents(t, false).map((c) => c.component)).toEqual(['mcp', 'skills', 'rules'])
    expect(visibleComponents(t, true).map((c) => c.component)).toEqual(['mcp', 'skills', 'rules', 'hooks'])
  })

  it.each([
    ['supported', mcp, ''],
    ['covered uses its status reason', skills, 'Claude Code skills cover it'],
    ['covered without a reason', comp({ status: 'covered' }), "Already loaded from another target's skills directory"],
    ['unsupported uses its reason', rules, 'no rules location'],
    ['unsupported without a reason', comp({ supported: false, status: 'unsupported' }), 'Not supported by this host'],
    ['externally managed stays selectable', comp({ status: 'externally_managed' }), ''],
  ])('%s', (_name, c, want) => {
    expect(unselectableReason(c)).toBe(want)
  })

  it('selects every selectable visible component by default', () => {
    expect(defaultSelection(t, false)).toEqual(['mcp'])
    expect(defaultSelection(t, true)).toEqual(['mcp', 'hooks'])
  })

  it('offers replace-external only when an externally managed component is selected', () => {
    const ext = target([mcp, comp({ component: 'skills', status: 'externally_managed' })])
    expect(hasExternallyManaged(ext, ['mcp', 'skills'])).toBe(true)
    expect(hasExternallyManaged(ext, ['mcp'])).toBe(false)
  })
})

describe('writableChanges', () => {
  it('drops skipped changes', () => {
    const kept = change()
    expect(writableChanges([kept, change({ skipped: true, kind: 'none', reason: 'same file as cursor' })])).toEqual([kept])
  })
})

describe('isRepreviewError', () => {
  it.each([
    ['repreview error', Object.assign(new Error('file changed since preview'), { repreview: true }), true],
    ['other api error', Object.assign(new Error('boom'), { repreview: false }), false],
    ['plain error', new Error('boom'), false],
    ['string', 'boom', false],
    ['null', null, false],
  ])('%s', (_name, e, want) => {
    expect(isRepreviewError(e)).toBe(want)
  })
})
