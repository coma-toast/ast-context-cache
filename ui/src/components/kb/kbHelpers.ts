import type { KBEntry, KBKind, KBScope, KBSeverity } from '../../api/types'

export const KB_MAX_BODY_BYTES = 16384

export const KB_KINDS: readonly KBKind[] = [
  'fact',
  'procedure',
  'preference',
  'event',
  'config_change',
  'discovery',
]

export const KB_SCOPES: readonly KBScope[] = ['global', 'project', 'session']

export const KB_SEVERITIES: readonly KBSeverity[] = ['info', 'warning', 'error', 'critical']

export const KB_KIND_LABELS: Record<KBKind, string> = {
  fact: 'Fact',
  procedure: 'Procedure',
  preference: 'Preference',
  event: 'Event',
  config_change: 'Config Change',
  discovery: 'Discovery',
}

export const KB_KIND_COLORS: Record<KBKind, string> = {
  fact: '#58a6ff',
  procedure: '#bc8cff',
  preference: '#3fb950',
  event: '#f0883e',
  config_change: '#d29922',
  discovery: '#79c0ff',
}

export const KB_SCOPE_LABELS: Record<KBScope, string> = {
  global: 'Global',
  project: 'Project',
  session: 'Session',
}

export const KB_SEVERITY_COLORS: Record<KBSeverity, string> = {
  info: '#58a6ff',
  warning: '#d29922',
  error: '#f85149',
  critical: '#ff7b72',
}

export function kbKindChipSx(kind: KBKind) {
  const color = KB_KIND_COLORS[kind] ?? '#8b949e'
  return {
    color,
    borderColor: `${color}55`,
    bgcolor: `${color}18`,
    fontWeight: 500,
  }
}

export function kbSeverityChipSx(severity?: string) {
  const normalized = (severity || '').toLowerCase() as KBSeverity
  const color = KB_SEVERITY_COLORS[normalized] ?? '#8b949e'
  const isCritical = normalized === 'critical'
  return {
    color: isCritical ? '#0d1117' : color,
    borderColor: isCritical ? color : `${color}66`,
    bgcolor: isCritical ? color : `${color}1a`,
    fontWeight: 600,
  }
}

export function isEntryActive(entry: KBEntry): boolean {
  const hasValidUntil = Boolean(entry.valid_until && entry.valid_until.trim() !== '')
  return !hasValidUntil && (entry.quarantined ?? 0) === 0
}

export type KBComputedStatus = 'active' | 'superseded' | 'forgotten' | 'quarantined'

export function kbEntryStatus(entry: KBEntry): KBComputedStatus {
  if ((entry.quarantined ?? 0) === 1 && (!entry.valid_until || entry.valid_until.trim() === '')) {
    return 'quarantined'
  }
  if (entry.valid_until && entry.valid_until.trim() !== '') {
    return entry.superseded_by && entry.superseded_by.trim() !== '' ? 'superseded' : 'forgotten'
  }
  return 'active'
}

function firstLine(s?: string): string {
  if (!s) return ''
  const trimmed = s.trim()
  if (!trimmed) return ''
  const idx = trimmed.indexOf('\n')
  return idx >= 0 ? trimmed.slice(0, idx).trim() : trimmed
}

function formatTriple(entry: KBEntry, defaultPredicate = 'is'): string {
  const subj = entry.subject?.trim() || ''
  const pred = entry.predicate?.trim() || ''
  const obj = entry.object?.trim() || ''
  if (!subj && !obj) return ''
  const effectivePred = pred || defaultPredicate
  if (subj && obj) return `${subj} ${effectivePred} ${obj}`
  return subj || obj
}

export function kbHeadline(entry: KBEntry): string {
  const title = entry.title?.trim() || ''
  const rule = entry.rule?.trim() || ''
  const bodyLine = firstLine(entry.body)

  switch (entry.kind) {
    case 'fact':
      return title || formatTriple(entry, 'is') || rule || bodyLine || entry.ref
    case 'procedure':
      return title || rule || formatTriple(entry, 'is') || bodyLine || entry.ref
    case 'preference':
      return title || formatTriple(entry, 'prefers') || rule || bodyLine || entry.ref
    case 'event':
      return title || formatTriple(entry, '') || entry.target?.trim() || bodyLine || entry.ref
    case 'discovery':
      return title || formatTriple(entry, '') || entry.target?.trim() || bodyLine || entry.ref
    case 'config_change': {
      if (title) return title
      const target = entry.target?.trim() || ''
      const before = entry.before_value?.trim() || ''
      const after = entry.after_value?.trim() || ''
      if (target) {
        if (before && after) return `${target}: ${before} → ${after}`
        if (after) return `${target}: ${after}`
        if (before) return `${target}: ${before} → (unset)`
        return target
      }
      return bodyLine || entry.ref
    }
    default:
      return title || rule || formatTriple(entry) || bodyLine || entry.ref
  }
}

const utf8Encoder = new TextEncoder()

export function utf8ByteLength(s: string): number {
  return utf8Encoder.encode(s).length
}

export function parseDateString(s?: string): Date | null {
  if (!s) return null
  const trimmed = s.trim()
  if (!trimmed) return null
  // Handle SQLite UTC timestamps like "2026-11-15 13:13:49"
  const normalized = /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}/.test(trimmed)
    ? `${trimmed.replace(' ', 'T')}Z`
    : trimmed
  const d = new Date(normalized)
  return Number.isNaN(d.getTime()) ? null : d
}

export function formatShortDate(s?: string): string {
  if (!s || !s.trim()) return ''
  const d = parseDateString(s)
  if (!d) return s
  return d.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function formatShortTime(s?: string): string {
  if (!s || !s.trim()) return ''
  const d = parseDateString(s)
  if (!d) return s
  return d.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function kbEntryDayKey(entry: KBEntry): string {
  const raw = (entry.occurred_at || entry.created_at || '').trim()
  if (!raw) return 'unknown'
  const d = parseDateString(raw)
  if (d) {
    const year = d.getFullYear()
    const month = String(d.getMonth() + 1).padStart(2, '0')
    const day = String(d.getDate()).padStart(2, '0')
    return `${year}-${month}-${day}`
  }
  const m = /^(\d{4}-\d{2}-\d{2})/.exec(raw)
  return m ? m[1] : 'unknown'
}

export function formatDayLabel(dayKey: string): string {
  if (!dayKey || dayKey === 'unknown') return 'Undated'
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(dayKey)
  if (!m) return dayKey
  const d = new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]))
  if (Number.isNaN(d.getTime())) return dayKey

  const now = new Date()
  const todayKey = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`
  const yesterday = new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1)
  const yesterdayKey = `${yesterday.getFullYear()}-${String(yesterday.getMonth() + 1).padStart(2, '0')}-${String(yesterday.getDate()).padStart(2, '0')}`

  const formatted = d.toLocaleDateString(undefined, {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
  if (dayKey === todayKey) return `Today · ${formatted}`
  if (dayKey === yesterdayKey) return `Yesterday · ${formatted}`
  return formatted
}

export function parseTagsInput(raw: string): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  for (const part of raw.split(',')) {
    const trimmed = part.trim()
    if (!trimmed) continue
    const key = trimmed.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    out.push(trimmed)
  }
  return out
}
