export interface Health {
  EmbedderState: string
  EmbedderError: string
  EmbedderLast: string
  EmbedBackend: string
  QueueWorkers: number
  QueueWorkersEffective: number
  QueueWorkersLive: number
  QueueThroughput: number
  QueueQueued: number
  QueuePending: number
  QueuePendingPeak: number
  QueueInFlight: number
  QueueHighCap: number
  QueueLowCap: number
  CacheHitRatio: number
  HeapMB: number
  CPUPercent: number
  Uptime: number
  Version: string
}

export interface Stats {
  TotalQueries: number
  TodayQueries: number
  TokensSaved: number
  TodayTokens: number
  DedupTokensSaved: number
  SavingsVsFiles: number
  AvgDurationMs: number
  TodayAvgDurationMs: number
  Sessions: number
  TodaySessions: number
  VirtualInventoryTokens: number
  VirtualNotesCount: number
  VirtualUtilPct30d: number
  VirtualOrphanCount: number
  VirtualStored30d: number
  VirtualAccessed30d: number
  VirtualTodayStored: number
  VirtualTodayAccessed: number
  VirtualMaxNotesGlobal: number
  VirtualMaxTokensGlobal: number
  KvRepairArchivesActive: number
  KvRepairRepairsTotal30d: number
  KvRepairUtilPct30d: number
}

export interface WatcherInfo {
  ProjectPath: string
  Name: string
  Label: string
  Workspace: string
  Active: boolean
  LinkedCount: number
}

export interface IndexHealth {
  TotalSymbols: number
  TotalFiles: number
  TotalEdges: number
  TotalVectors: number
  VectorMemMB: number
  EmbedQueued: number
  EmbedPending: number
  EmbedWorkers: number
  EmbedWorkerMax: number
  EmbedThroughput: number
  EmbedComplete: number
  PinnedCount: number
  Watchers: WatcherInfo[]
  CPUPercent: number
  HeapMB: number
  WALMaintenanceActive: boolean
  WALPressure: number
  FilteredProject: string
}

export interface Project {
  Path: string
  Label: string
  Name: string
  QueryCount: number
  SymbolCount: number
  FileCount: number
  Pinned: boolean
  LinkedChildren: string[]
  LinkedParent: string
}

export interface SettingsData {
  IdleUnloadMinutes: number
  WatcherIgnoreGlobs: string
  ProjectExcludePaths: string
  IndexLogFiles: boolean
  LogRetentionEnabled: boolean
  LogRetentionRoots: string
  LogRetentionMaxAgeDays: number
  LogRetentionMaxTotalMB: number
  LogRetentionDryRun: boolean
  QueryRetentionEnabled: boolean
  QueryRetentionMaxAgeDays: number
  ContextMaxNotesSession: number
  ContextMaxTokensSession: number
  ContextMaxNotesGlobal: number
  ContextMaxTokensGlobal: number
  ContextLimitPolicy: string
  EmbedBackend: string
  EmbedModelDir: string
  EmbedHTTPURL: string
  EmbedHTTPBearer: string
  EmbedOllamaHost: string
  EmbedOllamaModel: string
  EmbedOpenAIBaseURL: string
  EmbedOpenAIAPIKey: string
  EmbedOpenAIModel: string
  EmbedOpenAIDimensions: string
  EmbedDockerURL: string
  EmbedDockerModel: string
  EmbedActiveBackend: string
  EmbedActiveModel: string
  EmbedInSync: boolean
  EmbedderState: string
  EmbedderError: string
  Projects: Project[]
  ProjectsLoading: boolean
  Agents: AgentInfo[]
}

export interface AgentInfo {
  Type: string
  Name: string
  Description: string
  GlobalPath: string
  ProjectPath: string
  GlobalInstalled: boolean
  ProjectInstalled: boolean
}

export interface MemoryData {
  TotalSymbols: number
  TotalVectors: number
  VectorMemMB: number
  VirtualInventoryTokens: number
  VirtualNotesCount: number
  VirtualUtilPct30d: number
  VirtualOrphanCount: number
  VirtualStored30d: number
  VirtualAccessed30d: number
  DocSources: DocSource[]
  DocSourcesTotal: number
  DocSourcesPage: number
  DocSourcesPerPage: number
}

export interface DocSource {
  ID: number
  Name: string
  Type: string
  URL: string
  Age: string
  Stale: boolean
  Refreshing: boolean
}

export interface RecentQuery {
  Timestamp: string
  ToolName: string
  Query: string
  ProjectPath: string
  DurationMs: number
  CPUMs: number
  TokensSaved: number
  Error: string
  Mode: string
}

export interface ToolStat {
  tool_name: string
  calls: number
  avg_duration_ms: number
  avg_cpu_ms: number
  tokens_saved: number
}

export interface BarItem {
  kind?: string
  language?: string
  target?: string
  count: number
}

export interface TimeseriesPoint {
  timestamp: string
  queries: number
  tokens_saved: number
  avg_duration_ms: number
}

export interface MCPTier {
  tier: string
  tools_json_path: string
  tools_json_exists: boolean
}

export interface RecentLogLine {
  Timestamp?: string
  Level: string
  Message: string
  Raw?: string
  MsgTruncated?: boolean
}
