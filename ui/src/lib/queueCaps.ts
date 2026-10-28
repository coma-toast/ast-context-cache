import type { IndexHealth, QueueCapState } from '../api/types'

/** A queue cap with the depth to show beside it: live from index health when available. */
export interface QueueCapRow extends QueueCapState {
  depth: number
  /** Depth as a percentage of the limit, 0–100. */
  fillPct: number
}

const liveDepth = (key: string, health: IndexHealth | null | undefined): number | undefined => {
  if (!health) return undefined
  switch (key) {
    case 'embed_queue_pending_cap':
      return health.EmbedPendingQueued
    case 'embed_queue_high_cap':
      return health.EmbedHighQueued
    case 'embed_queue_low_cap':
      return health.EmbedLowQueued
    default:
      return undefined
  }
}

/** Joins the settings caps with live queue depth from index health (falling back to settings). */
export const queueCapRows = (caps: QueueCapState[] | null | undefined, health?: IndexHealth | null): QueueCapRow[] =>
  (caps ?? []).map((c) => {
    const depth = liveDepth(c.key, health) ?? c.used
    const fillPct = c.value > 0 ? Math.min(100, Math.round((depth / c.value) * 100)) : 0
    return { ...c, depth, fillPct }
  })

/** Returns an error message for a cap entry, or null when it is an integer 1..hardMax. */
export const validateQueueCap = (raw: string, hardMax: number): string | null => {
  const v = raw.trim()
  if (!/^\d+$/.test(v)) return `Enter a whole number 1–${hardMax}`
  const n = Number(v)
  if (n < 1 || n > hardMax) return `Must be 1–${hardMax}`
  return null
}

/** Helper text under a cap field: the env lock, else the live depth against the limit. */
export const queueCapHelper = (row: QueueCapRow): string =>
  row.env_locked ? `Set by ${row.env}` : `${row.depth} queued now · max ${row.hard_max}`
