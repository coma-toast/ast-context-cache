import type { WatcherInfo } from '../api/types'

export type WatcherSpaceGroup = { space: string; watchers: WatcherInfo[] }

export type WatcherDisplayEntry = {
  watcher: WatcherInfo
  groupKey: string
  groupLabel: string
}

export function groupWatchers(watchers: WatcherInfo[]): { local: WatcherInfo[]; spaces: WatcherSpaceGroup[] } {
  const local: WatcherInfo[] = []
  const bySpace = new Map<string, WatcherInfo[]>()
  for (const w of watchers) {
    const space = (w.Workspace || '').trim()
    if (!space) {
      local.push(w)
      continue
    }
    const list = bySpace.get(space) || []
    list.push(w)
    bySpace.set(space, list)
  }
  sortWatchers(local)
  const spaces = [...bySpace.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([space, items]) => ({ space, watchers: sortWatchers([...items]) }))
  return { local, spaces }
}

/** Flat space watchers with per-workspace group labels for paginated Spaces column. */
export function flattenSpaceWatchers(spaces: WatcherSpaceGroup[]): WatcherDisplayEntry[] {
  const out: WatcherDisplayEntry[] = []
  for (const g of spaces) {
    for (const w of g.watchers) {
      out.push({ watcher: w, groupKey: `space:${g.space}`, groupLabel: g.space })
    }
  }
  return out
}

function sortWatchers(ws: WatcherInfo[]): WatcherInfo[] {
  ws.sort((a, b) => {
    const la = (a.Label || a.Name || '').toLowerCase()
    const lb = (b.Label || b.Name || '').toLowerCase()
    if (la !== lb) return la < lb ? -1 : 1
    return a.ProjectPath < b.ProjectPath ? -1 : a.ProjectPath > b.ProjectPath ? 1 : 0
  })
  return ws
}
