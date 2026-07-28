/**
 * Storybook-only stand-in for `../api/client`, aliased in `.storybook/main.ts`.
 * Panels rendered with fixture data don't need real network calls; button clicks
 * in stories resolve harmlessly instead of hitting a live ast-mcp server.
 *
 * NOTE: formatNum/formatUptime are duplicated (not re-exported) from `../api/client`
 * because that import specifier is itself aliased to this file — re-exporting it
 * would create a self-import loop.
 */
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
  saveSetting: noop,
  saveEmbedSettings: noop,
  pinProject: noop,
  resetProject: noop,
  deleteWatcher: noop,
  startWatcher: noop,
  stopWatcher: noop,
  indexProject: noop,
  setProjectLabel: noop,
  linkProject: noop,
  unlinkProject: noop,
  flushContextAll: noop,
  flushContextSession: noop,
  docSourceAction: noop,
  addDocSource: noop,
  installDocPack: noop,
  agentInstall: noop,
  agentUninstall: noop,
  embedderTest: noop,
  embedderRetry: noop,
  embedderDismissAlert: noop,
  walCheckpoint: noop,
  adjustEmbedWorkers: noop,
  adjustEmbedAuxWorkers: noop,
}
