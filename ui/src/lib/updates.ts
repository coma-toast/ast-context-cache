import type { UpdateCheckResult } from '../api/types'

export const UPDATE_STEPS = ['Download', 'Verify checksum', 'Install'] as const

// updateStepIndex maps the server's update phase to the active step in
// UPDATE_STEPS (UPDATE_STEPS.length once installed, -1 when idle or unknown).
export function updateStepIndex(phase: string | undefined): number {
  switch (phase) {
    case 'downloading':
      return 0
    case 'verifying':
      return 1
    case 'installing':
      return 2
    case 'installed':
      return UPDATE_STEPS.length
    default:
      return -1
  }
}

export function updateHeadline(check: UpdateCheckResult | null): string {
  if (!check) return 'Checking for updates…'
  if (check.error) return 'Could not check for updates'
  if (!check.update_available) return `Up to date with v${check.latest_version}`
  if (check.source_build && check.latest_version === check.current_version) return `v${check.latest_version} release build available`
  return `v${check.latest_version} available`
}

export function formatReleaseDate(iso: string): string {
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
}
