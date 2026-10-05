/**
 * Storybook-only stand-in for `../api/client`, aliased in `.storybook/main.ts`.
 * Panels rendered with fixture data don't need real network calls; button clicks
 * in stories resolve harmlessly instead of hitting a live ast-mcp server.
 *
 * NOTE: formatNum/formatUptime/formatBytes are duplicated (not re-exported) from
 * `../api/client` because that import specifier is itself aliased to this file —
 * re-exporting it would create a self-import loop.
 */
import type { InstallerPlanRequest } from '../api/types'
import {
  fixtureFlags,
  fixtureInstaller,
  fixtureInstallerBackups,
  fixtureInstallerErrorPlan,
  fixtureInstallerPlan,
} from './fixtures'

export function formatUptime(ns: number): string {
  const sec = Math.floor(ns / 1e9)
  const h = Math.floor(sec / 3600)
  const m = Math.floor((sec % 3600) / 60)
  if (h > 0) return `${h}h ${m}m`
  return `${m}m`
}

export function formatNum(n: number): string {
  if (n == null || !Number.isFinite(n)) return '0'
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`
  return String(n)
}

export function formatBytes(n: number): string {
  if (n == null || !Number.isFinite(n)) return '0 B'
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MB`
  if (n >= 1024) return `${Math.round(n / 1024)} KB`
  return `${n} B`
}

const noop = async (): Promise<never> => {
  throw new Error('This is a Storybook preview — actions are disabled.')
}

export const api = {
  health: noop,
  stats: noop,
  weeklyDigest: noop,
  contextSessions: noop,
  indexHealth: noop,
  memory: noop,
  settings: noop,
  projects: noop,
  recentSplit: noop,
  recentLogs: noop,
  tools: noop,
  symbolKinds: noop,
  languageStats: noop,
  topImports: noop,
  timeseries: noop,
  mcpTier: noop,
  // Read-only fixture so the Features section renders; toggling still hits `noop`.
  flags: async () => ({ flags: fixtureFlags }),
  setFlag: noop,
  saveSetting: noop,
  saveEmbedSettings: noop,
  pinProject: noop,
  resetProject: noop,
  deleteWatcher: noop,
  startWatcher: noop,
  startWatcherSpace: noop,
  reconcileSpaces: noop,
  stopWatcher: noop,
  indexProject: noop,
  setProjectLabel: noop,
  setProjectExcludes: noop,
  linkProject: noop,
  unlinkProject: noop,
  flushContextAll: noop,
  flushContextOrphans: noop,
  flushContextSession: noop,
  docSourceAction: noop,
  addDocSource: noop,
  installDocPack: noop,
  // Read-only fixtures so the installer cards, preview, and backups render; Apply and Restore hit `noop`.
  installer: async () => fixtureInstaller,
  installerPreview: async (req: InstallerPlanRequest) =>
    req.targets[0] === 'vscode' ? fixtureInstallerErrorPlan : { ...fixtureInstallerPlan, action: req.action },
  installerApply: noop,
  installerBackups: async () => ({ backups: fixtureInstallerBackups }),
  installerRestore: noop,
  embedderTest: noop,
  embedderRetry: noop,
  embedderDismissAlert: noop,
  retryPendingEmbeds: noop,
  walCheckpoint: noop,
  adjustEmbedWorkers: noop,
  adjustEmbedAuxWorkers: noop,
  setEmbedWorkers: noop,
  setEmbedAuxWorkers: noop,
  updateCheck: noop,
  startUpdate: noop,
  updateStatus: noop,
  restartNow: noop,
}
