import type {
  Health,
  IndexHealth,
  MCPTier,
  MemoryData,
  Project,
  SettingsData,
  Stats,
  TimeseriesPoint,
  ToolStat,
} from './types'

function qs(projectId?: string, extra?: Record<string, string>): string {
  const p = new URLSearchParams()
  if (projectId) p.set('project_id', projectId)
  if (extra) {
    for (const [k, v] of Object.entries(extra)) p.set(k, v)
  }
  const s = p.toString()
  return s ? `?${s}` : ''
}

/** Absolute path from site root — works when SPA is served at /dashboard/ */
export function apiUrl(path: string): string {
  if (path.startsWith('http://') || path.startsWith('https://')) return path
  const normalized = path.startsWith('/') ? path : `/${path}`
  if (typeof window !== 'undefined') return `${window.location.origin}${normalized}`
  return normalized
}

async function get<T>(path: string): Promise<T> {
  const url = apiUrl(path)
  let r: Response
  try {
    r = await fetch(url, { headers: { Accept: 'application/json' } })
  } catch (e) {
    throw new Error(`${path}: network error (${e instanceof Error ? e.message : 'fetch failed'})`)
  }
  const text = await r.text()
  if (!r.ok) {
    throw new Error(`${path}: HTTP ${r.status}${text ? ` — ${text.slice(0, 120)}` : ''}`)
  }
  if (!text.trim()) {
    throw new Error(`${path}: empty response`)
  }
  try {
    return JSON.parse(text) as T
  } catch {
    throw new Error(`${path}: invalid JSON`)
  }
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const url = apiUrl(path)
  let r: Response
  try {
    r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(body),
    })
  } catch (e) {
    throw new Error(`${path}: network error (${e instanceof Error ? e.message : 'fetch failed'})`)
  }
  const text = await r.text()
  let data: T & { error?: string }
  try {
    data = JSON.parse(text) as T & { error?: string }
  } catch {
    throw new Error(`${path}: invalid JSON (HTTP ${r.status})`)
  }
  if (!r.ok || data.error) throw new Error(data.error || `${path}: HTTP ${r.status}`)
  return data
}

/** POST that returns JSON even when the server signals failure via ok:false / HTTP 4xx/5xx. */
async function postEmbedderAction<T extends { ok?: boolean; error?: string }>(path: string): Promise<T> {
  const url = apiUrl(path)
  let r: Response
  try {
    r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: '{}',
    })
  } catch (e) {
    throw new Error(`${path}: network error (${e instanceof Error ? e.message : 'fetch failed'})`)
  }
  const text = await r.text()
  try {
    return JSON.parse(text) as T
  } catch {
    throw new Error(`${path}: invalid JSON (HTTP ${r.status})`)
  }
}

export const api = {
  health: () => get<Health>('/api/dashboard/health'),
  stats: (projectId?: string) => get<Stats>(`/api/dashboard/stats${qs(projectId)}`),
  indexHealth: (projectId?: string) => get<IndexHealth>(`/api/dashboard/index-health${qs(projectId)}`),
  memory: (projectId?: string, page = 1) =>
    get<MemoryData>(`/api/dashboard/memory${qs(projectId, { doc_sources_page: String(page) })}`),
  settings: () => get<SettingsData>('/api/dashboard/settings'),
  projects: () => get<Project[]>('/api/projects'),
  recentSplit: (projectId?: string) =>
    get<{ mcp: import('./types').RecentQuery[]; indexing: import('./types').RecentQuery[] }>(
      `/api/dashboard/recent-split${qs(projectId)}`,
    ),
  recentLogs: () =>
    get<{ lines: import('./types').RecentLogLine[]; path: string; file_truncated: boolean; tail_lines: number }>(
      '/api/dashboard/recent-logs',
    ),
  tools: (projectId?: string) => get<ToolStat[]>(`/api/tools${qs(projectId)}`),
  symbolKinds: (projectId?: string) => get<{ kind: string; count: number }[]>(`/api/symbol-kinds${qs(projectId)}`),
  languageStats: (projectId?: string) =>
    get<{ language: string; count: number; symbols?: number; files?: number }[]>(
      `/api/language-stats${qs(projectId)}`,
    ),
  topImports: (projectId?: string) => get<{ target: string; count: number }[]>(`/api/top-imports${qs(projectId)}`),
  timeseries: (projectId?: string, interval = 'daily', days = 30) =>
    get<TimeseriesPoint[]>(`/api/timeseries${qs(projectId, { interval, days: String(days) })}`),
  mcpTier: () => get<MCPTier>('/api/dashboard/mcp-tier'),
  saveSetting: (key: string, value: string) => post<{ status?: string; error?: string }>('/api/settings', { key, value }),
  saveEmbedSettings: (settings: Record<string, string>) =>
    post<{ status?: string; error?: string }>('/api/settings/embed', { settings }),
  pinProject: (project_path: string, pinned: boolean) => post('/api/pin-project', { project_path, pinned }),
  resetProject: (project_path: string) => post('/api/reset-project', { project_path }),
  deleteWatcher: (project_path: string) => post('/api/delete-watcher', { project_path }),
  startWatcher: (project_path: string) => post<{ status?: string }>('/api/start-watcher', { project_path }),
  stopWatcher: (project_path: string) => post<{ status?: string }>('/api/stop-watcher', { project_path }),
  indexProject: (project_path: string) =>
    post<{ status?: string; symbols?: number }>('/api/index-project', { project_path }),
  setProjectLabel: (project_path: string, label: string) =>
    post<{ status?: string; label?: string; custom?: boolean }>('/api/project-label', { project_path, label }),
  linkProject: (parent_path: string, child_path: string) =>
    post('/api/project-links', { parent_path, child_path }),
  unlinkProject: (parent_path: string, child_path: string) =>
    fetch(apiUrl('/api/project-links'), {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ parent_path, child_path }),
    }).then(async (r) => {
      const d = await r.json()
      if (!r.ok || d.error) throw new Error(d.error || 'unlink failed')
      return d
    }),
  flushContextAll: () => post('/api/flush-context', { all: true }),
  flushContextSession: (session_id: string) => post('/api/flush-context', { session_id }),
  docSourceAction: (action: string, id: number) => post('/api/doc-sources', { action, id }),
  addDocSource: (name: string, url: string, type: string, version = '') =>
    post('/api/doc-sources', { action: 'add', name, url, type, version }),
  agentInstall: (agent_type: string, is_global: boolean) => post('/api/agent-install', { agent_type, is_global }),
  agentUninstall: (agent_type: string, is_global: boolean) => post('/api/agent-uninstall', { agent_type, is_global }),
  embedderTest: () => post<{ status?: string; error?: string }>('/api/embedder/test', {}),
  embedderRetry: () =>
    postEmbedderAction<{ ok?: boolean; state?: string; error?: string; skipped?: boolean }>('/api/embedder/retry'),
  embedderDismissAlert: () =>
    postEmbedderAction<{ ok?: boolean; state?: string; error?: string }>('/api/embedder/dismiss-alert'),
  walCheckpoint: () => post('/api/wal-checkpoint', {}),
  adjustEmbedWorkers: (delta: number) => post<{ status?: string; workers?: number; error?: string }>('/api/embed-workers', { delta }),
  adjustEmbedAuxWorkers: (delta: number) =>
    post<{ status?: string; workers?: number; error?: string }>('/api/embed-aux-workers', { delta }),
}

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
