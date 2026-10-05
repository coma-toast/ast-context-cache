import type {
  InstallerChangeKind,
  InstallerComponent,
  InstallerComponentId,
  InstallerFileChange,
  InstallerStatus,
  InstallerTarget,
} from '../api/types'

/** How one unified-diff line is colored in the installer preview. */
export type DiffLineKind = 'add' | 'remove' | 'hunk' | 'header' | 'context'

export interface DiffLine {
  kind: DiffLineKind
  text: string
}

/** MUI Chip colors used for installer statuses. */
export type StatusChipColor = 'default' | 'success' | 'warning' | 'error' | 'info' | 'secondary'

export interface StatusChip {
  label: string
  color: StatusChipColor
}

const STATUS_CHIPS: Record<InstallerStatus, StatusChip> = {
  installed: { label: 'Installed', color: 'success' },
  outdated: { label: 'Outdated', color: 'warning' },
  modified_by_user: { label: 'Modified by user', color: 'info' },
  missing: { label: 'Missing', color: 'error' },
  not_installed: { label: 'Not installed', color: 'default' },
  externally_managed: { label: 'Externally managed', color: 'secondary' },
  unsupported: { label: 'Unsupported', color: 'default' },
  covered: { label: 'Covered', color: 'default' },
}

const COMPONENT_LABELS: Record<InstallerComponentId, string> = {
  mcp: 'MCP server',
  skills: 'Skills',
  rules: 'Rules',
  hooks: 'Hooks',
}

const CHANGE_KIND_LABELS: Record<InstallerChangeKind, string> = {
  create: 'Create',
  modify: 'Modify',
  'remove-block': 'Remove block',
  delete: 'Delete',
  none: 'No change',
}

/**
 * Classifies one unified-diff line. Outside a hunk, `---`/`+++` lines are file headers; inside
 * one they are a removed or added line whose own text starts with `--` or `++`.
 */
export const classifyDiffLine = (line: string, inHunk = false): DiffLineKind => {
  if (line.startsWith('@@')) return 'hunk'
  if (!inHunk && (line.startsWith('--- ') || line.startsWith('+++ '))) return 'header'
  if (line.startsWith('+')) return 'add'
  if (line.startsWith('-')) return 'remove'
  return 'context'
}

/** Splits a unified diff into classified lines, dropping the trailing newline's empty line. */
export const diffLines = (diff: string): DiffLine[] => {
  if (!diff) return []
  const lines = diff.split('\n')
  if (lines[lines.length - 1] === '') lines.pop()
  let inHunk = false
  return lines.map((text) => {
    const kind = classifyDiffLine(text, inHunk)
    if (kind === 'hunk') inHunk = true
    return { kind, text }
  })
}

/** Chip label and color for a status; an unknown status from a newer server renders neutrally. */
export const statusChip = (status: InstallerStatus): StatusChip =>
  STATUS_CHIPS[status] ?? { label: status, color: 'default' }

export const componentLabel = (component: InstallerComponentId): string => COMPONENT_LABELS[component] ?? component

export const changeKindLabel = (kind: InstallerChangeKind): string => CHANGE_KIND_LABELS[kind] ?? kind

/** The components a target card shows: the Hooks row only while feature_handoff_hooks is on. */
export const visibleComponents = (target: InstallerTarget, hooksEnabled: boolean): InstallerComponent[] =>
  target.components.filter((c) => hooksEnabled || c.component !== 'hooks')

/**
 * Why a component's checkbox is disabled, or '' when it can be selected. Unsupported components
 * have no location on the host; covered skills already load from another target's directory.
 */
export const unselectableReason = (c: InstallerComponent): string => {
  if (!c.supported || c.status === 'unsupported') return c.reason || c.status_reason || 'Not supported by this host'
  if (c.status === 'covered') return c.status_reason || "Already loaded from another target's skills directory"
  return ''
}

/** Components checked by default on a card: every selectable visible one. */
export const defaultSelection = (target: InstallerTarget, hooksEnabled: boolean): InstallerComponentId[] =>
  visibleComponents(target, hooksEnabled)
    .filter((c) => unselectableReason(c) === '')
    .map((c) => c.component)

/** True when any selected component is externally managed, so the card offers to replace it (IN-9). */
export const hasExternallyManaged = (target: InstallerTarget, selected: InstallerComponentId[]): boolean =>
  target.components.some((c) => c.status === 'externally_managed' && selected.includes(c.component))

/** Changes Apply would actually write; skipped ones are shown with their reason only. */
export const writableChanges = (changes: InstallerFileChange[]): InstallerFileChange[] =>
  changes.filter((c) => !c.skipped)

/** True when an API error means the plan is stale, expired, or unknown and must be previewed again. */
export const isRepreviewError = (e: unknown): boolean =>
  typeof e === 'object' && e !== null && (e as { repreview?: unknown }).repreview === true
