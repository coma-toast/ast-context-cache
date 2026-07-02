import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Box,
  Drawer,
  FormControl,
  IconButton,
  List,
  ListItemButton,
  ListItemText,
  MenuItem,
  Select,
  Toolbar,
  Typography,
} from '@mui/material'
import MenuIcon from '@mui/icons-material/Menu'
import { api } from './api/client'
import type { Health, IndexHealth, MCPTier, MemoryData, SettingsData, Stats, TimeseriesPoint, ToolStat } from './api/types'
import { HealthBar } from './components/HealthBar'
import { ToastProvider, useToast } from './context/ToastContext'
import { panelsToKeys, useWebSocket } from './hooks/useWebSocket'
import { OverviewTab } from './tabs/OverviewTab'
import { IndexHealthSection } from './tabs/IndexHealthSection'
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

function DashboardInner() {
  const { showToast } = useToast()
  const [tab, setTab] = useState<TabId>('overview')
  const [mobileOpen, setMobileOpen] = useState(false)
  const [projectId, setProjectId] = useState('')
  const [projects, setProjects] = useState<{ path: string; name: string }[]>([])
  const [health, setHealth] = useState<Health | null>(null)
  const [stats, setStats] = useState<Stats | null>(null)
  const [indexHealth, setIndexHealth] = useState<IndexHealth | null>(null)
  const [memory, setMemory] = useState<MemoryData | null>(null)
  const [settings, setSettings] = useState<SettingsData | null>(null)
  const [mcpTier, setMcpTier] = useState<MCPTier | null>(null)
  const [timeseries, setTimeseries] = useState<TimeseriesPoint[] | null>(null)
  const [timeseriesInterval] = useState<'daily' | 'hourly'>('daily')
  const [tools, setTools] = useState<ToolStat[] | null>(null)
  const [symbols, setSymbols] = useState<{ kind: string; count: number }[] | null>(null)
  const [imports, setImports] = useState<{ target: string; count: number }[] | null>(null)
  const [recentMcp, setRecentMcp] = useState<import('./api/types').RecentQuery[] | null>(null)
  const [recentIdx, setRecentIdx] = useState<import('./api/types').RecentQuery[] | null>(null)

  const pid = projectId || undefined

  const load = useCallback(
    async (keys: string[]) => {
      const tasks: Promise<void>[] = []
      if (keys.includes('health')) tasks.push(api.health().then(setHealth))
      if (keys.includes('stats')) tasks.push(api.stats(pid).then(setStats))
      if (keys.includes('indexHealth')) tasks.push(api.indexHealth(pid).then(setIndexHealth))
      if (keys.includes('memory')) tasks.push(api.memory(pid).then(setMemory))
      if (keys.includes('settings')) tasks.push(api.settings().then(setSettings))
      if (keys.includes('mcpTier')) tasks.push(api.mcpTier().then(setMcpTier))
      if (keys.includes('timeseries')) tasks.push(api.timeseries(pid, timeseriesInterval).then(setTimeseries))
      if (keys.includes('tools')) tasks.push(api.tools(pid).then(setTools))
      if (keys.includes('symbolKinds')) tasks.push(api.symbolKinds(pid).then(setSymbols))
      if (keys.includes('topImports')) tasks.push(api.topImports(pid).then(setImports))
      if (keys.includes('recent')) {
        tasks.push(
          api.recentSplit(pid).then((r) => {
            setRecentMcp(r.mcp)
            setRecentIdx(r.indexing)
          }),
        )
      }
      if (keys.includes('projects')) {
        tasks.push(
          api.projects().then((list) => {
            setProjects(list.map((p) => ({ path: p.path, name: p.name || p.path })))
          }),
        )
      }
      await Promise.allSettled(tasks)
    },
    [pid, timeseriesInterval],
  )

  const loadAll = useCallback(() => {
    load(['health', 'stats', 'indexHealth', 'memory', 'settings', 'mcpTier', 'timeseries', 'tools', 'symbolKinds', 'topImports', 'recent', 'projects'])
  }, [load])

  useEffect(() => {
    loadAll()
  }, [loadAll])

  useEffect(() => {
    load(['timeseries'])
  }, [timeseriesInterval, load])

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
    <Box sx={{ width: 240, p: 2 }}>
      <Typography variant="h6" fontWeight={700} gutterBottom>
        AST Context Cache
      </Typography>
      <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 2 }}>
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
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <Drawer variant="temporary" open={mobileOpen} onClose={() => setMobileOpen(false)} ModalProps={{ keepMounted: true }} sx={{ display: { xs: 'block', md: 'none' } }}>
        {drawer}
      </Drawer>
      <Drawer variant="permanent" sx={{ display: { xs: 'none', md: 'block' }, '& .MuiDrawer-paper': { width: 240, boxSizing: 'border-box', borderRight: '1px solid', borderColor: 'divider' } }} open>
        {drawer}
      </Drawer>
      <Box component="main" sx={{ flex: 1, minWidth: 0 }}>
        <Toolbar sx={{ gap: 2, flexWrap: 'wrap', borderBottom: '1px solid', borderColor: 'divider', minHeight: 56 }}>
          <IconButton edge="start" sx={{ display: { md: 'none' } }} onClick={() => setMobileOpen(true)} aria-label="Open navigation">
            <MenuIcon />
          </IconButton>
          <Box sx={{ flex: 1, minWidth: 200 }}>
            <HealthBar health={health} />
          </Box>
          <FormControl size="small" sx={{ minWidth: { xs: 140, md: 220 } }}>
            <Select
              displayEmpty
              value={projectId}
              onChange={(e) => setProjectId(e.target.value)}
              renderValue={(v) => (v ? projects.find((p) => p.path === v)?.name || v : 'All projects')}
            >
              <MenuItem value="">All projects</MenuItem>
              {projects.map((p) => (
                <MenuItem key={p.path} value={p.path}>
                  {p.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          {projectId && (
            <Typography component="button" variant="caption" sx={{ cursor: 'pointer', border: 0, bgcolor: 'transparent', color: 'primary.main' }} onClick={() => setProjectId('')}>
              Clear filter
            </Typography>
          )}
        </Toolbar>
        <Box sx={{ p: 3, maxWidth: 1200, mx: 'auto' }}>
          <Typography variant="h5" fontWeight={600} gutterBottom>
            {title}
          </Typography>
          {tab === 'overview' && (
            <>
              <OverviewTab stats={stats} />
              <IndexHealthSection data={indexHealth} />
            </>
          )}
          {tab === 'memory' && <MemoryTab data={memory} onRefresh={() => load(['memory'])} />}
          {tab === 'activity' && <ActivityTab data={timeseries} />}
          {tab === 'analytics' && <AnalyticsTab tools={tools} symbols={symbols} imports={imports} />}
          {tab === 'recent' && <RecentTab mcp={recentMcp} indexing={recentIdx} />}
          {tab === 'settings' && <SettingsTab data={settings} mcpTier={mcpTier} onRefresh={() => load(['settings', 'indexHealth'])} />}
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
