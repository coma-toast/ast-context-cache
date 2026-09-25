import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Autocomplete,
  Box,
  Button,
  Drawer,
  IconButton,
  List,
  ListItemButton,
  ListItemText,
  TextField,
  Toolbar,
  Typography,
} from '@mui/material'
import MenuIcon from '@mui/icons-material/Menu'
import { api } from './api/client'
import type { ContextSessionsResponse, Health, IndexHealth, MCPTier, MemoryData, SettingsData, Stats, TimeseriesPoint, ToolStat, WeeklyDigest } from './api/types'
import { HealthBar } from './components/HealthBar'
import { ErrorBoundary } from './components/ErrorBoundary'
import { ToastProvider, useToast } from './context/ToastContext'
import { panelsToKeys, useWebSocket } from './hooks/useWebSocket'
import { useResizableSidebar } from './hooks/useResizableSidebar'
import { OverviewTab } from './tabs/OverviewTab'
import { IndexHealthSection } from './tabs/IndexHealthSection'
import { WatchersPanel } from './components/WatchersPanel'
import { MemoryTab } from './tabs/MemoryTab'
import { ActivityTab } from './tabs/ActivityTab'
import { AnalyticsTab } from './tabs/AnalyticsTab'
import { RecentTab } from './tabs/RecentTab'
import { SettingsTab } from './tabs/SettingsTab'

const NAV = [
  { id: 'overview', label: 'Overview' },
  { id: 'memory', label: 'Memory' },
  { id: 'activity', label: 'Activity' },
  { id: 'analytics', label: 'Analytics' },
  { id: 'recent', label: 'Recent' },
  { id: 'settings', label: 'Settings' },
] as const

type TabId = (typeof NAV)[number]['id']

const TAB_HINTS: Record<TabId, string> = {
  overview: 'Server and embedder health, activity, the last 7 days, virtual context, and file watchers',
  memory: 'Virtual context inventory and cached documentation sources',
  activity: 'Query volume and token savings over time',
  analytics: 'Tool performance, symbol mix, and import graph',
  recent: 'MCP calls, indexing events, and server logs',
  settings: 'Embedding, watchers, retention, projects, and agent integration',
}

function DashboardInner() {
  const { showToast } = useToast()
  const { width: sidebarWidth, onPointerDown: onSidebarResize } = useResizableSidebar()
  const [tab, setTab] = useState<TabId>('overview')
  const [mobileOpen, setMobileOpen] = useState(false)
  const [projectId, setProjectId] = useState('')
  const [projects, setProjects] = useState<{ path: string; name: string }[]>([])
  const [health, setHealth] = useState<Health | null>(null)
  const [stats, setStats] = useState<Stats | null>(null)
  const [weeklyDigest, setWeeklyDigest] = useState<WeeklyDigest | null>(null)
  const [contextSessions, setContextSessions] = useState<ContextSessionsResponse | null>(null)
  const [indexHealth, setIndexHealth] = useState<IndexHealth | null>(null)
  const [memory, setMemory] = useState<MemoryData | null>(null)
  const [settings, setSettings] = useState<SettingsData | null>(null)
  const [mcpTier, setMcpTier] = useState<MCPTier | null>(null)
  const [timeseries, setTimeseries] = useState<TimeseriesPoint[] | null>(null)
  const [timeseriesInterval, setTimeseriesInterval] = useState<'daily' | 'hourly'>('daily')
  const [tools, setTools] = useState<ToolStat[] | null>(null)
  const [symbols, setSymbols] = useState<{ kind: string; count: number }[] | null>(null)
  const [languages, setLanguages] = useState<{ language: string; count: number }[] | null>(null)
  const [imports, setImports] = useState<{ target: string; count: number }[] | null>(null)
  const [recentMcp, setRecentMcp] = useState<import('./api/types').RecentQuery[] | null>(null)
  const [recentIdx, setRecentIdx] = useState<import('./api/types').RecentQuery[] | null>(null)
  const [docSourcesPage, setDocSourcesPage] = useState(1)
  const [loadErrors, setLoadErrors] = useState<string[]>([])
  const [abnormalDismissed, setAbnormalDismissed] = useState(false)

  const pid = projectId || undefined

  const load = useCallback(
    async (keys: string[]) => {
      const failures: string[] = []
      const run = <T,>(label: string, task: Promise<T>, apply: (v: T) => void) =>
        task
          .then(apply)
          .catch((e) => {
            failures.push(`${label}: ${e instanceof Error ? e.message : String(e)}`)
          })

      const tasks: Promise<void>[] = []
      if (keys.includes('health')) tasks.push(run('health', api.health(), setHealth))
      if (keys.includes('stats')) tasks.push(run('stats', api.stats(pid), setStats))
      if (keys.includes('weeklyDigest')) tasks.push(run('weeklyDigest', api.weeklyDigest(pid), setWeeklyDigest))
      if (keys.includes('contextSessions')) {
        tasks.push(run('contextSessions', api.contextSessions(pid), setContextSessions))
      }
      if (keys.includes('indexHealth')) tasks.push(run('indexHealth', api.indexHealth(pid), setIndexHealth))
      if (keys.includes('memory')) tasks.push(run('memory', api.memory(pid, docSourcesPage), setMemory))
      if (keys.includes('settings')) tasks.push(run('settings', api.settings(), setSettings))
      if (keys.includes('mcpTier')) tasks.push(run('mcpTier', api.mcpTier(), setMcpTier))
      if (keys.includes('timeseries')) tasks.push(run('timeseries', api.timeseries(pid, timeseriesInterval), setTimeseries))
      if (keys.includes('tools')) tasks.push(run('tools', api.tools(pid), setTools))
      if (keys.includes('symbolKinds')) tasks.push(run('symbolKinds', api.symbolKinds(pid), setSymbols))
      if (keys.includes('languageStats')) tasks.push(run('languageStats', api.languageStats(pid), setLanguages))
      if (keys.includes('topImports')) tasks.push(run('topImports', api.topImports(pid), setImports))
      if (keys.includes('recent')) {
        tasks.push(
          run('recent', api.recentSplit(pid), (r) => {
            setRecentMcp(r.mcp)
            setRecentIdx(r.indexing)
          }),
        )
      }
      if (keys.includes('projects')) {
        tasks.push(
          run('projects', api.projects(), (list) => {
            const rows = Array.isArray(list) ? list : []
            setProjects(
              rows.map((p) => ({
                path: p.path ?? p.Path ?? '',
                name: p.name ?? p.Name ?? p.path ?? p.Path ?? '',
              })),
            )
          }),
        )
      }
      await Promise.all(tasks)
      setLoadErrors(failures)
      if (failures.length) {
        failures.forEach((f) => showToast(f, 'error'))
      }
    },
    [pid, timeseriesInterval, docSourcesPage, showToast],
  )

  const loadAll = useCallback(() => {
    load(['health', 'stats', 'weeklyDigest', 'contextSessions', 'indexHealth', 'memory', 'settings', 'mcpTier', 'timeseries', 'tools', 'symbolKinds', 'languageStats', 'topImports', 'recent', 'projects'])
  }, [load])

  useEffect(() => {
    loadAll()
  }, [loadAll])

  useEffect(() => {
    const starting = health?.EmbedderState === 'loading' || health?.EmbedderState === 'starting'
    if (!starting) return
    const id = window.setInterval(() => load(['health', 'stats', 'indexHealth', 'memory', 'settings']), 2500)
    return () => window.clearInterval(id)
  }, [health, load])

  useEffect(() => {
    load(['timeseries'])
  }, [timeseriesInterval, load])

  useEffect(() => {
    load(['memory'])
  }, [docSourcesPage, load])

  useEffect(() => {
    setDocSourcesPage(1)
  }, [pid])

  useWebSocket(
    (panels) => {
      const keys = panelsToKeys(panels)
      if (keys.length) load(keys)
    },
    (data) => {
      showToast(`${data.toolName}: ${data.query?.slice(0, 40)}`, 'info')
    },
  )

  const drawer = (
    <Box sx={{ width: '100%', p: 2, boxSizing: 'border-box', height: '100%' }}>
      <Typography variant="h6" sx={{ fontWeight: 700 }} gutterBottom>
        AST Context Cache
      </Typography>
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 2 }}>
        Operator dashboard
      </Typography>
      <List dense>
        {NAV.map((item) => (
          <ListItemButton key={item.id} selected={tab === item.id} onClick={() => { setTab(item.id); setMobileOpen(false) }} aria-current={tab === item.id ? 'page' : undefined}>
            <ListItemText primary={item.label} />
          </ListItemButton>
        ))}
      </List>
    </Box>
  )

  const title = useMemo(() => NAV.find((n) => n.id === tab)?.label ?? 'Dashboard', [tab])

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', width: '100%' }}>
      <Drawer variant="temporary" open={mobileOpen} onClose={() => setMobileOpen(false)} ModalProps={{ keepMounted: true }} sx={{ display: { xs: 'block', md: 'none' } }}>
        {drawer}
      </Drawer>
      <Drawer
        variant="permanent"
        open
        sx={{
          display: { xs: 'none', md: 'block' },
          width: sidebarWidth,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: sidebarWidth,
            boxSizing: 'border-box',
            position: 'relative',
            height: '100vh',
            borderRight: '1px solid',
            borderColor: 'divider',
            overflowY: 'auto',
            overflowX: 'hidden',
          },
        }}
      >
        {drawer}
        <Box
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize sidebar"
          onPointerDown={onSidebarResize}
          sx={{
            position: 'absolute',
            top: 0,
            right: 0,
            width: 6,
            height: '100%',
            cursor: 'col-resize',
            zIndex: 1200,
            touchAction: 'none',
            '&:hover, &:active': {
              bgcolor: 'primary.main',
              opacity: 0.35,
            },
          }}
        />
      </Drawer>
      <Box
        component="main"
        sx={{
          flex: '1 1 auto',
          minWidth: 0,
          overflowX: 'hidden',
        }}
      >
        <Toolbar
          sx={{
            gap: 2,
            flexWrap: 'nowrap',
            borderBottom: '1px solid',
            borderColor: 'divider',
            minHeight: 56,
            position: 'sticky',
            top: 0,
            zIndex: 200,
            bgcolor: 'rgba(13,17,23,0.92)',
            backdropFilter: 'blur(10px)',
          }}
        >
          <IconButton edge="start" sx={{ display: { md: 'none' }, flexShrink: 0 }} onClick={() => setMobileOpen(true)} aria-label="Open navigation">
            <MenuIcon />
          </IconButton>
          <Box sx={{ flex: 1, minWidth: 0, maxWidth: '100%' }}>
            <HealthBar health={health} />
          </Box>
          <Autocomplete
            size="small"
            sx={{ minWidth: { xs: 160, md: 220 }, maxWidth: { md: 280 }, flexShrink: 0 }}
            options={projects}
            getOptionLabel={(p) => p.name}
            isOptionEqualToValue={(a, b) => a.path === b.path}
            value={projects.find((p) => p.path === projectId) || null}
            onChange={(_e, v) => setProjectId(v?.path || '')}
            renderInput={(params) => <TextField {...params} placeholder="All projects" />}
          />
        </Toolbar>
        <Box sx={{ p: 3, maxWidth: 1200, mx: 'auto' }}>
          {loadErrors.length > 0 && (
            <Alert
              severity="warning"
              sx={{ mb: 2 }}
              action={
                <Button color="inherit" size="small" onClick={() => { setLoadErrors([]); loadAll() }}>
                  Retry
                </Button>
              }
            >
              Some dashboard APIs failed. Ensure ast-mcp is running on port 7830 ({window.location.origin}).
              {loadErrors.slice(0, 2).map((e) => (
                <Box component="div" key={e} sx={{ fontFamily: 'ui-monospace, monospace', fontSize: 11, mt: 0.5 }}>
                  {e}
                </Box>
              ))}
            </Alert>
          )}
          <Box sx={{ mb: 2 }}>
            <Typography variant="h5" sx={{ fontWeight: 600 }}>
              {title}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {TAB_HINTS[tab]}
            </Typography>
          </Box>
          {tab === 'overview' && (
            <ErrorBoundary
              label="Overview"
              onRetry={() => load(['indexHealth', 'health', 'projects', 'settings', 'stats', 'weeklyDigest', 'contextSessions'])}
            >
              {!!health?.AbnormalPreviousRun && !abnormalDismissed && (
                <Alert severity="warning" sx={{ mb: 2 }} onClose={() => setAbnormalDismissed(true)}>
                  Restarted after abnormal exit
                </Alert>
              )}
              <IndexHealthSection data={indexHealth} onRefresh={() => load(['indexHealth', 'health', 'projects', 'settings'])} />
              <OverviewTab
                stats={stats}
                weeklyDigest={weeklyDigest}
                contextSessions={contextSessions}
              />
              {indexHealth && (
                <WatchersPanel
                  watchers={indexHealth.Watchers || []}
                  onRefresh={() => load(['indexHealth', 'health', 'projects', 'settings'])}
                />
              )}
            </ErrorBoundary>
          )}
          {tab === 'memory' && (
            <ErrorBoundary label="Memory" onRetry={() => load(['memory'])}>
              <MemoryTab
                data={memory}
                onRefresh={() => load(['memory'])}
                docSourcesPage={docSourcesPage}
                onDocSourcesPageChange={setDocSourcesPage}
              />
            </ErrorBoundary>
          )}
          {tab === 'activity' && (
            <ErrorBoundary label="Activity" onRetry={() => load(['timeseries'])}>
              <ActivityTab data={timeseries} interval={timeseriesInterval} onIntervalChange={setTimeseriesInterval} />
            </ErrorBoundary>
          )}
          {tab === 'analytics' && (
            <ErrorBoundary label="Analytics" onRetry={() => load(['tools', 'symbolKinds', 'languageStats', 'topImports'])}>
              <AnalyticsTab tools={tools} symbols={symbols} languages={languages} imports={imports} />
            </ErrorBoundary>
          )}
          {tab === 'recent' && (
            <ErrorBoundary label="Recent" onRetry={() => load(['recent'])}>
              <RecentTab mcp={recentMcp} indexing={recentIdx} />
            </ErrorBoundary>
          )}
          {tab === 'settings' && (
            <ErrorBoundary label="Settings" onRetry={() => load(['settings', 'indexHealth', 'projects', 'mcpTier'])}>
              <SettingsTab data={settings} mcpTier={mcpTier} onRefresh={() => load(['settings', 'indexHealth', 'projects'])} />
            </ErrorBoundary>
          )}
        </Box>
      </Box>
    </Box>
  )
}

export default function App() {
  return (
    <ToastProvider>
      <DashboardInner />
    </ToastProvider>
  )
}
