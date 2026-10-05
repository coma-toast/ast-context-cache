import { describe, expect, it } from 'vitest'

import type { HandoffChildView, HandoffTreeView, HandoffView } from '../api/types'

import {
  buildHandoffNodes,
  claimsLabel,
  formatAge,
  formatRepeatRate,
  parseHandoffTime,
  sortTrees,
  statusCounts,
} from './handoffTree'

const child = (session_id: string, overrides: Partial<HandoffChildView> = {}): HandoffChildView => ({
  session_id,
  status: 'open',
  depth: 1,
  opened_at: '2026-10-05 10:00:00',
  last_activity_at: '2026-10-05 10:00:00',
  search_calls: 0,
  repeat_calls: 0,
  repeat_rate: 0,
  tokens_available: 0,
  tokens_delivered: 0,
  tokens_saved: 0,
  active_claims: 0,
  queued_claims: 0,
  ...overrides,
})

const handoff = (ref: string, children: HandoffChildView[], overrides: Partial<HandoffView> = {}): HandoffView => ({
  handoff: ref,
  mode: 'fresh',
  depth: 1,
  parent_session_id: 'root',
  created_at: '2026-10-05 09:00:00',
  children,
  ...overrides,
})

const tree = (tree_id: string, handoffs: HandoffView[], overrides: Partial<HandoffTreeView> = {}): HandoffTreeView => ({
  tree_id,
  root_session_id: 'root',
  created_at: '2026-10-05 09:00:00',
  last_access_at: '2026-10-05 10:00:00',
  expires_at: '2026-10-12 10:00:00',
  expired: false,
  tokens_used: 0,
  tokens_max: 64000,
  entries_used: 0,
  entries_max: 300,
  active_claims: 0,
  queued_claims: 0,
  search_calls: 0,
  repeat_calls: 0,
  repeat_rate: 0,
  tokens_delivered: 0,
  tokens_saved: 0,
  handoffs,
  ...overrides,
})

describe('buildHandoffNodes', () => {
  it('nests a handoff under the child that created it', () => {
    const t = tree('hft_a', [
      handoff('hof_nested', [child('hof_nested.c1', { depth: 2 })], {
        depth: 2,
        parent_session_id: 'hof_top.c1',
        parent_child_session_id: 'hof_top.c1',
        created_at: '2026-10-05 09:30:00',
      }),
      handoff('hof_top', [child('hof_top.c1'), child('hof_top.c2')]),
    ])
    const nodes = buildHandoffNodes(t)
    expect(nodes.map((n) => n.handoff.handoff)).toEqual(['hof_top'])
    const c1 = nodes[0].children.find((c) => c.child.session_id === 'hof_top.c1')
    expect(c1?.handoffs.map((n) => n.handoff.handoff)).toEqual(['hof_nested'])
    expect(c1?.handoffs[0].children.map((c) => c.child.session_id)).toEqual(['hof_nested.c1'])
    const c2 = nodes[0].children.find((c) => c.child.session_id === 'hof_top.c2')
    expect(c2?.handoffs).toEqual([])
  })

  it('keeps a nested handoff whose creator is missing at the top level', () => {
    const t = tree('hft_a', [handoff('hof_lost', [], { parent_child_session_id: 'gone.c1', depth: 2 }), handoff('hof_top', [])])
    expect(buildHandoffNodes(t).map((n) => n.handoff.handoff)).toEqual(['hof_lost', 'hof_top'])
  })

  it('orders handoffs by creation and children by status, then latest activity', () => {
    const t = tree('hft_a', [
      handoff(
        'hof_b',
        [
          child('done-old', { status: 'done', last_activity_at: '2026-10-05 09:10:00' }),
          child('abandoned', { status: 'abandoned' }),
          child('done-new', { status: 'done', last_activity_at: '2026-10-05 09:50:00' }),
          child('open', { status: 'open' }),
          child('failed', { status: 'failed' }),
          child('partial', { status: 'partial' }),
        ],
        { created_at: '2026-10-05 09:20:00' },
      ),
      handoff('hof_a', [], { created_at: '2026-10-05 09:05:00' }),
    ])
    const nodes = buildHandoffNodes(t)
    expect(nodes.map((n) => n.handoff.handoff)).toEqual(['hof_a', 'hof_b'])
    expect(nodes[1].children.map((c) => c.child.session_id)).toEqual(['open', 'failed', 'partial', 'done-new', 'done-old', 'abandoned'])
  })

  it('keeps handoffs in a malformed creation loop, each once', () => {
    const t = tree('hft_a', [
      handoff('hof_x', [child('hof_x.c1')], { parent_child_session_id: 'hof_y.c1' }),
      handoff('hof_y', [child('hof_y.c1')], { parent_child_session_id: 'hof_x.c1', created_at: '2026-10-05 09:10:00' }),
    ])
    const nodes = buildHandoffNodes(t)
    expect(nodes.map((n) => n.handoff.handoff)).toEqual(['hof_x'])
    expect(nodes[0].children[0].handoffs.map((n) => n.handoff.handoff)).toEqual(['hof_y'])
    expect(nodes[0].children[0].handoffs[0].children[0].handoffs).toEqual([])
  })

  it('does not mutate its input', () => {
    const t = tree('hft_a', [handoff('hof_b', [child('z'), child('a', { status: 'done' })], { created_at: '2026-10-05 09:30:00' }), handoff('hof_a', [])])
    const before = JSON.stringify(t)
    buildHandoffNodes(t)
    expect(JSON.stringify(t)).toBe(before)
  })
})

describe('sortTrees', () => {
  it('puts live trees first, newest first', () => {
    const trees = [
      tree('old', [], { created_at: '2026-10-01 00:00:00' }),
      tree('expired-new', [], { created_at: '2026-10-05 00:00:00', expired: true }),
      tree('new', [], { created_at: '2026-10-04 00:00:00' }),
    ]
    expect(sortTrees(trees).map((t) => t.tree_id)).toEqual(['new', 'old', 'expired-new'])
    expect(trees.map((t) => t.tree_id)).toEqual(['old', 'expired-new', 'new'])
  })
})

describe('statusCounts', () => {
  it('counts children across handoffs', () => {
    const t = tree('hft_a', [
      handoff('hof_a', [child('a'), child('b', { status: 'done' })]),
      handoff('hof_b', [child('c', { status: 'done' }), child('d', { status: 'abandoned' })]),
    ])
    expect(statusCounts(t)).toEqual({ open: 1, failed: 0, partial: 0, done: 2, abandoned: 1 })
  })
})

describe('formatters', () => {
  it.each([
    [0.25, 4, '25%'],
    [2 / 3, 3, '67%'],
    [0, 5, '0%'],
    [0, 0, '—'],
  ])('formatRepeatRate(%d, %d) -> %s', (rate, searches, want) => {
    expect(formatRepeatRate(rate, searches)).toBe(want)
  })

  it.each([
    [0, 0, ''],
    [2, 0, '2 held'],
    [0, 3, '3 queued'],
    [1, 2, '1 held · 2 queued'],
  ])('claimsLabel(%d, %d) -> %j', (active, queued, want) => {
    expect(claimsLabel(active, queued)).toBe(want)
  })

  it('parses handoff times as UTC', () => {
    expect(parseHandoffTime('2026-10-05 12:00:00')).toBe(Date.UTC(2026, 9, 5, 12, 0, 0))
    expect(parseHandoffTime('soon')).toBeNaN()
  })

  it.each([
    ['2026-10-05 11:59:15', '45s'],
    ['2026-10-05 11:48:00', '12m'],
    ['2026-10-05 09:00:00', '3h'],
    ['2026-10-03 12:00:00', '2d'],
    ['2026-10-05 12:00:30', '0s'],
    ['bad', ''],
  ])('formatAge(%s) -> %j', (at, want) => {
    expect(formatAge(at, Date.UTC(2026, 9, 5, 12, 0, 0))).toBe(want)
  })
})
