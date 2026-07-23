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
  AbnormalPreviousRun?: boolean
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
  TotalChars?: number
  AvgDurationMs?: number
  VirtualFlushed30d?: number
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
  /** Approximate full-source baseline (tokens-first heuristic). */
  ApproxBaselineTokens?: number
  ApproxTokensReturned?: number
  ApproxRoundsAvoided?: number
  HeuristicApproximate?: boolean
  HeuristicLabel?: string
}

export interface WeeklyDigestTool {
  ToolName: string
  Calls: number
  TokensSaved: number
  AvgDurationMs: number
}

export interface WeeklyDigestEmbedReliability {
  PendingFailures: number
  LastAutoRecoverUnix: number
  AbnormalPreviousRun: boolean
  Available: boolean
  Note?: string
}

export interface WeeklyDigest {
  WindowDays: number
  TokensSaved: number
  Queries: number
  VirtualStored: number
  VirtualAccessed: number
  TopTools: WeeklyDigestTool[]
  EmbedReliability: WeeklyDigestEmbedReliability
  Heuristic: {
    ApproxBaselineTokens: number
    ApproxTokensReturned: number
    ApproxTokensSaved: number
    ApproxRoundsAvoided: number
    HeuristicApproximate: boolean
    HeuristicLabel: string
    WindowDays: number
  }
}

export interface ContextSessionStory {
  SessionID: string
  ProjectPath?: string
  NotesCount: number
  VirtualTokensStored: number
  VirtualTokensAccessed: number
  ActiveNotes: number
  ActiveTokens: number
  FetchedAfterStore: boolean
  LastStoreAt?: string
  LastAccessAt?: string
}

export interface ContextSessionsResponse {
  WindowDays: number
  Sessions: ContextSessionStory[]
}

export interface WatcherInfo {
  ProjectPath: string
  Name: string
  Label: string
  Workspace: string
  Active: boolean
  LinkedCount: number
}

export interface EmbedActivityItem {
  File: string
  ProjectPath: string
}

export interface IndexHealth {
  TotalSymbols: number
  TotalFiles: number
  TotalEdges: number
  TotalVectors: number
  VectorMemMB: number
  MemoryMB?: number
  DiskMB?: number
  WalMB?: number
  WalSize?: string
  CPUPercent: number
  HeapMB?: number
  LoadAvgAvailable?: boolean
  LoadAvg1?: number
  LoadAvg5?: number
  LoadAvg15?: number
  DiskSize?: string
  DiskReadMBps?: number
  DiskWriteMBps?: number
  SSDModel?: string
  SSDSmartStatus?: string
  SSDProtocol?: string
  SSDCapacity?: string
  SSDSolidState?: boolean
  SSDTrim?: boolean
  SSDAvailable?: boolean
  SSDWearUsedPct?: number
  SSDSparePct?: number
  SSDDataWrittenTB?: number
  SSDTemperatureC?: number
  EmbedQueued: number
  EmbedPending: number
  EmbedPendingPeak?: number
  EmbedFailed?: number
  EmbedHighQueued?: number
  EmbedLowQueued?: number
  EmbedHighCap?: number
  EmbedLowCap?: number
  EmbedActive?: number
  EmbedWorkers: number
  EmbedWorkersEffective?: number
  EmbedWorkersLive?: number
  EmbedWorkerMax: number
  EmbedAuxWorkers?: number
  EmbedAuxWorkersEffective?: number
  EmbedAuxWorkersLive?: number
  EmbedAuxWorkerMax?: number
  EmbedAuxBackend?: string
  EmbedAuxModel?: string
  EmbedAuxEnabled?: boolean
  EmbedComplete: number
  EmbedThroughput: number
  PinnedCount: number
  FilteredProject: string
  EmbedBackend?: string
  EmbedModel?: string
  EmbedRuntime?: string
  EmbedEndpoint?: string
  EmbedDim?: number
  EmbedderState?: string
  EmbedderError?: string
  EmbedLoaded?: boolean
  EmbedRecent?: EmbedActivityItem[]
  EmbedInProgress?: EmbedActivityItem[]
  EmbedConfiguredBackend?: string
  EmbedConfiguredModel?: string
  EmbedInSync?: boolean
  WALMaintenanceActive: boolean
  WALMaintenancePhase?: string
  WALMaintenanceMode?: string
  WALMaintenanceReason?: string
  /** RFC3339 from Go time.Time; zero time may be "0001-01-01T00:00:00Z". */
  WALMaintenanceStarted?: string
  WALWalStartBytes?: number
  WALWalCurrentBytes?: number
  WALBusyStreak?: number
  WALInFlight?: number
  WALLastBusy?: number
  WALPressure?: string | number
  /** Unix seconds of last stuck-worker auto-recover, or 0. */
  LastAutoRecoverUnix?: number
  Watchers: WatcherInfo[]
}

export interface Project {
  Path: string
  path?: string
  Label?: string
  label?: string
  Name: string
  name?: string
  QueryCount: number
  query_count?: number
  SymbolCount: number
  symbol_count?: number
  FileCount: number
  file_count?: number
  Pinned: boolean
  LinkedChildren: string[]
  LinkedParent: string
}

export interface SettingsData {
  IdleUnloadMinutes: number
  EmbedWorkerMax?: number
  EmbedAuxWorkerMax?: number
  EmbedAuxWorkers?: number
  EmbedAuxBackend?: string
  EmbedProbeIntervalSec?: number
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
