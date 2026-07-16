export function pendingRingCap(pending: number, peak?: number): number {
  if (pending <= 0) return 1
  let cap = peak ?? pending
  if (cap < pending) cap = pending
  return cap < 1 ? 1 : cap
}

export function fillPct(used: number, cap: number): number {
  if (cap <= 0) return 0
  const p = (used / cap) * 100
  return p > 100 ? 100 : p
}

export function throughputPct(rate: number): number {
  const max = 80
  const p = (rate / max) * 100
  return p > 100 ? 100 : p
}

export function workerPillCount(total: number, visibleMax = 20): { dots: number; ellipsis: boolean } {
  if (total > visibleMax) return { dots: visibleMax - 1, ellipsis: true }
  return { dots: total, ellipsis: false }
}
