import type { HandoffChildView, HandoffStatus, HandoffTreeView, HandoffView } from '../api/types'

/** A handoff with its children, each carrying the handoffs it created in turn. */
export interface HandoffNode {
  handoff: HandoffView
  children: ChildNode[]
}

/** A child session and the nested handoffs it created (HO-8). */
export interface ChildNode {
  child: HandoffChildView
  handoffs: HandoffNode[]
}

/** Status order for children: live work first, then outcomes that need a look, then the rest. */
const STATUS_ORDER: Record<HandoffStatus, number> = {
  open: 0,
  failed: 1,
  partial: 2,
  done: 3,
  abandoned: 4,
}

export const HANDOFF_STATUSES: HandoffStatus[] = ['open', 'failed', 'partial', 'done', 'abandoned']

/** Chip color per child status. */
export const STATUS_COLOR: Record<HandoffStatus, 'info' | 'error' | 'warning' | 'success' | 'default'> = {
  open: 'info',
  failed: 'error',
  partial: 'warning',
  done: 'success',
  abandoned: 'default',
}

const compareChildren = (a: HandoffChildView, b: HandoffChildView): number =>
  STATUS_ORDER[a.status] - STATUS_ORDER[b.status] ||
  b.last_activity_at.localeCompare(a.last_activity_at) ||
  a.session_id.localeCompare(b.session_id)

const compareHandoffs = (a: HandoffView, b: HandoffView): number =>
  a.created_at.localeCompare(b.created_at) || a.handoff.localeCompare(b.handoff)

/**
 * Nests a tree's flat handoff list: handoffs the root created are the top level, and a nested
 * handoff hangs under the child session that created it. A handoff that can't be reached that
 * way (its creating child isn't in the tree, or malformed data loops) is kept at the top level
 * rather than dropped. Handoffs are in creation order; children are sorted by status (open
 * first), then most recent activity.
 */
export const buildHandoffNodes = (tree: HandoffTreeView): HandoffNode[] => {
  const handoffs = [...tree.handoffs].sort(compareHandoffs)
  const childIds = new Set(handoffs.flatMap((h) => h.children.map((c) => c.session_id)))
  const byCreator = new Map<string, HandoffView[]>()
  const top: HandoffView[] = []
  for (const h of handoffs) {
    const creator = h.parent_child_session_id ?? ''
    if (creator === '' || !childIds.has(creator)) {
      top.push(h)
      continue
    }
    byCreator.set(creator, [...(byCreator.get(creator) ?? []), h])
  }
  const placed = new Set<string>()
  const build = (h: HandoffView): HandoffNode => {
    placed.add(h.handoff)
    return {
      handoff: h,
      children: [...h.children].sort(compareChildren).map((child) => ({
        child,
        handoffs: (byCreator.get(child.session_id) ?? []).filter((n) => !placed.has(n.handoff)).map(build),
      })),
    }
  }
  const nodes = top.map(build)
  for (const h of handoffs) {
    if (!placed.has(h.handoff)) nodes.push(build(h))
  }
  return nodes
}

/** Sorts trees for display: live trees before expired ones, then newest first. */
export const sortTrees = (trees: HandoffTreeView[]): HandoffTreeView[] =>
  [...trees].sort(
    (a, b) =>
      Number(a.expired) - Number(b.expired) || b.created_at.localeCompare(a.created_at) || a.tree_id.localeCompare(b.tree_id),
  )

/** Counts a tree's children by status, across every handoff. */
export const statusCounts = (tree: HandoffTreeView): Record<HandoffStatus, number> => {
  const counts: Record<HandoffStatus, number> = { open: 0, failed: 0, partial: 0, done: 0, abandoned: 0 }
  for (const h of tree.handoffs) {
    for (const c of h.children) counts[c.status] = (counts[c.status] ?? 0) + 1
  }
  return counts
}

/** Formats a repeat rate (0–1) as a whole percentage, or "—" when there were no searches. */
export const formatRepeatRate = (rate: number, searches: number): string =>
  searches > 0 ? `${Math.round(rate * 100)}%` : '—'

/** Claims badge text: "2 held · 1 queued", or "" when there are none. */
export const claimsLabel = (active: number, queued: number): string =>
  [active > 0 ? `${active} held` : '', queued > 0 ? `${queued} queued` : ''].filter(Boolean).join(' · ')

/** Parses a handoff time ("YYYY-MM-DD HH:MM:SS", UTC) to epoch ms; NaN when malformed. */
export const parseHandoffTime = (s: string): number => Date.parse(`${s.replace(' ', 'T')}Z`)

/** Short relative age ("45s", "12m", "3h", "2d") of a handoff time; "" when malformed. */
export const formatAge = (s: string, now: number = Date.now()): string => {
  const at = parseHandoffTime(s)
  if (Number.isNaN(at)) return ''
  const sec = Math.max(0, Math.round((now - at) / 1000))
  if (sec < 60) return `${sec}s`
  if (sec < 3600) return `${Math.floor(sec / 60)}m`
  if (sec < 86400) return `${Math.floor(sec / 3600)}h`
  return `${Math.floor(sec / 86400)}d`
}
