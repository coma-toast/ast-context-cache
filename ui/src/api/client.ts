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

async function get<T>(path: string): Promise<T> {
  const r = await fetch(path)
  if (!r.ok) throw new Error(`${path}: ${r.status}`)
  return r.json() as Promise<T>
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  const data = (await r.json()) as T & { error?: string }
  if (!r.ok || data.error) throw new Error(data.error || `${path}: ${r.status}`)
  return data
}

export const api = {
  health: () => get<Health>('/api/dashboard/health'),
  stats: (projectId?: string) => get<Stats>(`/api/dashboard/stats${qs(projectId)}`),
  indexHealth: (projectId?: string) => get<IndexHealth>(`/api/dashboard/index-health${qs(projectId)}`),
  memory: (projectId?: string, page = 1) =>
    get<MemoryData>(`/api/dashboard/memory${qs(projectId, { doc_sources_page: String(page) })}`),
  settings: () => get<SettingsData>('/api/dashboard/settings'),
  projects: () => get<{ path: string; name: string; query_count?: number; symbol_count?: number; file_count?: number }[]>('/api/projects'),
  recentSplit: (projectId?: string) =>
    get<{ mcp: import('./types').RecentQuery[]; indexing: import('./types').RecentQuery[] }>(
      `/api/dashboard/recent-split${qs(projectId)}`,
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
  linkProject: (parent_path: string, child_path: string) =>
    post('/api/project-links', { parent_path, child_path }),
  unlinkProject: (parent_path: string, child_path: string) =>
    fetch('/api/project-links', {
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
  walCheckpoint: () => post('/api/wal-checkpoint', {}),
}

export function formatUptime(ns: number): string {
  const sec = Math.floor(ns / 1e9)
  const h = Math.floor(sec / 3600)
  const m = Math.floor((sec % 3600) / 60)
  if (h > 0) return `${h}h ${m}m`
  return `${m}m`
}

export function formatNum(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`
  return String(n)
}
