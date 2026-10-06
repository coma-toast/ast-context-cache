import { useCallback, useEffect, useRef } from 'react'

const panelToQuery: Record<string, string[]> = {
  'health-bar': ['health'],
  stats: ['stats', 'weeklyDigest', 'contextSessions'],
  'index-health': ['indexHealth'],
  memory: ['memory', 'contextSessions'],
  recent: ['recent'],
  'symbol-chart': ['symbolKinds'],
  'language-chart': ['languageStats'],
  'tool-chart': ['tools'],
  'import-chart': ['topImports'],
  settings: ['settings'],
  handoffs: ['handoffTrees'],
}

export function useWebSocket(
  onRefresh: (panels: string[]) => void,
  onToast?: (data: Record<string, string>) => void,
  onIndexDrained?: (data: Record<string, string>) => void,
) {
  const onRefreshRef = useRef(onRefresh)
  const onToastRef = useRef(onToast)
  const onIndexDrainedRef = useRef(onIndexDrained)
  onRefreshRef.current = onRefresh
  onToastRef.current = onToast
  onIndexDrainedRef.current = onIndexDrained

  const connect = useCallback(() => {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${proto}//${location.host}/ws`)
    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)
        if (msg.type === 'refresh' && msg.data?.panels) {
          onRefreshRef.current(msg.data.panels as string[])
        } else if (msg.type === 'toast' && msg.data) {
          onToastRef.current?.(msg.data as Record<string, string>)
        } else if (msg.type === 'index-drained' && msg.data) {
          onIndexDrainedRef.current?.(msg.data as Record<string, string>)
        }
      } catch {
        /* ignore */
      }
    }
    ws.onclose = () => {
      setTimeout(connect, 3000)
    }
    return ws
  }, [])

  useEffect(() => {
    const ws = connect()
    return () => ws.close()
  }, [connect])
}

export function panelsToKeys(panels: string[]): string[] {
  const keys = new Set<string>()
  for (const p of panels) {
    for (const k of panelToQuery[p] || []) keys.add(k)
  }
  return [...keys]
}

export { panelToQuery }
