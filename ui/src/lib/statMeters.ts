export interface MeterFill {
  overlapPct: number
  dayOnlyPct: number
  avgOnlyPct: number
  gaugePct: number
}

export function meterFillSegments(day: number, avg: number): MeterFill {
  if (avg <= 0) return { overlapPct: 0, dayOnlyPct: 0, avgOnlyPct: 0, gaugePct: 0 }
  const max = Math.max(day, avg)
  const overlap = Math.min(day, avg)
  const dayOnly = Math.max(0, day - overlap)
  const avgOnly = Math.max(0, avg - overlap)
  return {
    overlapPct: (overlap / max) * 100,
    dayOnlyPct: (dayOnly / max) * 100,
    avgOnlyPct: (avgOnly / max) * 100,
    gaugePct: (day / avg) * 100,
  }
}

export function todayMeterFill(today: number, total30d: number): MeterFill {
  if (total30d <= 0) return { overlapPct: 0, dayOnlyPct: 0, avgOnlyPct: 0, gaugePct: 0 }
  return meterFillSegments(today, total30d / 30)
}

export function fmtDailyAvg(total30d: number): string {
  if (total30d <= 0) return '0'
  const avg = total30d / 30
  if (avg >= 100) return String(Math.round(avg))
  if (avg === Math.floor(avg)) return String(Math.floor(avg))
  return avg.toFixed(1)
}

export function gaugeLevelFromPct(pct: number): 'ok' | 'warn' | 'critical' {
  if (pct >= 85) return 'critical'
  if (pct >= 60) return 'warn'
  return 'ok'
}
