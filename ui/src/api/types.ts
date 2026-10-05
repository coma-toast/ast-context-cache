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
  /** Files currently waiting for an embed retry. */
  PendingFailures: number
  /** Cumulative failed embed attempts this process, including ones since fixed. */
  FailedSinceStart?: number
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

/** Result of starting watchers on every repo checkout in a WTG space. */
export interface StartWatcherSpaceResult {
  status?: string
  space?: string
  started?: string[]
  already_running?: string[]
  skipped?: string[]
  errors?: string[]
}

/** Result of reconciling indexed space projects against what's still on disk. */
export interface ReconcileSpacesResult {
  status?: string
  purged: string[]
}

export interface EmbedActivityItem {
  File: string
  ProjectPath: string
  /** "primary" | "aux" for in-progress rows; omitted/empty for recent */
  Pool?: string
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
  FDAvailable?: boolean
  OpenFDs?: number
  FDSoftLimit?: number
  FDLevel?: 'ok' | 'warning' | 'critical' | 'unknown'
  WatchBackend?: string
  LoadAvgAvailable?: boolean
  LoadAvg1?: number
  LoadAvg5?: number
  LoadAvg15?: number
  DiskSize?: string
  DiskReadMBps?: number
  DiskWriteMBps?: number
  SSDModel?: string
  SSDSmartStatus?: string
  SSDSmartSource?: string
  SSDProtocol?: string
  SSDCapacity?: string
  SSDFreeSpace?: string
  SSDExternal?: boolean
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
  EmbedActivePrimary?: number
  EmbedAuxActive?: number
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
  DriveDisconnected?: boolean
  DriveDisconnectedPath?: string
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
  DataDir?: string
  DataDirSize?: string
  WatcherIgnoreGlobs: string
  ProjectExcludePaths: string
  /** Project path → per-project exclude patterns (gitignore syntax, relative to the project root). */
  ProjectIndexExcludes?: Record<string, string[]> | null
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
  /** Effective handoff limits (env > setting > default). */
  HandoffLimits?: HandoffLimits
  /** Handoff limit settings keys locked by their AST_HANDOFF_* env var. */
  HandoffEnvLocked?: string[] | null
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
  TimestampTitle: string
  ToolName: string
  Query: string
  Mode: string
  Budget: number
  Saved: number
  DedupTokensSaved: number
  Project: string
  DurationMs: number
  CpuMs: number
  Error: string
  Event: string
  File: string
  FileTitle: string
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
  code_mode: boolean
  tool_overrides: Record<string, { enabled: boolean; tier: string }>
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

export interface DataDirMoveStatus {
  active: boolean
  done: boolean
  phase: string
  target_dir: string
  started_at: string
  finished_at: string
  error: string
  /** db filenames whose source was missing and got started fresh instead of copied. */
  recreated?: string[] | null
  /** db filenames that already existed at the target and were kept as-is instead of copied. */
  kept?: string[] | null
}

export interface UpdateCheckResult {
  branch: string
  clean: boolean
  current_commit: string
  latest_commit: string
  commits_behind: number
  update_available: boolean
  error?: string
}

export interface UpdateStatus {
  active: boolean
  done: boolean
  phase: string
  error: string
  started_at: string
  finished_at: string
  from_commit: string
  to_commit: string
}

export interface PruneStatus {
  active: boolean
  done: boolean
  phase: string
  started_at: string
  finished_at: string
  error: string
  size_before_bytes: number
  size_after_bytes: number
  projects_purged: number
  orphan_vectors: number
  queries_pruned: number
  memory_pruned: number
}

export interface BrowseDirEntry {
  name: string
  path: string
}

export interface BrowseDirResult {
  path: string
  parent?: string
  entries: BrowseDirEntry[]
  shortcuts?: BrowseDirEntry[]
  error?: string
}

/** Where a feature flag's own value came from; env locks the flag. */
export type FlagSource = 'env' | 'setting' | 'default'

/** One feature flag's resolved state (`internal/flags.FlagState`). */
export interface FlagState {
  key: string
  description: string
  source: FlagSource
  /** Env var that overrides (and locks) the flag. */
  env: string
  /** Effective value: a feature_handoff_* child reads false while feature_handoff is off. */
  enabled: boolean
  default: boolean
  locked: boolean
}

export interface FlagsResponse {
  flags: FlagState[]
}

export interface SetFlagResponse extends FlagsResponse {
  status: string
}

/** Handoff retention windows and caps (`internal/handoff.Limits`). */
export interface HandoffLimits {
  ttl_days: number
  summary_max_tokens: number
  child_inactive_minutes: number
  tree_max_tokens: number
  tree_max_entries: number
  max_depth: number
  max_children: number
  open_budget_tokens: number
}

export type HandoffStatus = 'open' | 'done' | 'partial' | 'failed' | 'abandoned'

export type HandoffMode = 'fresh' | 'fork'

/** One child session of a handoff (`internal/handoff.ChildView`). */
export interface HandoffChildView {
  session_id: string
  label?: string
  status: HandoffStatus
  depth: number
  opened_at: string
  last_activity_at: string
  result_ref?: string
  summary?: string
  search_calls: number
  repeat_calls: number
  /** repeat_calls / search_calls (OB-1); 0 with no searches. */
  repeat_rate: number
  tokens_available: number
  tokens_delivered: number
  /** tokens_available − tokens_delivered, floored at 0 (OB-2). */
  tokens_saved: number
  active_claims: number
  queued_claims: number
}

/** One handoff in a tree (`internal/handoff.HandoffView`). */
export interface HandoffView {
  handoff: string
  label?: string
  mode: HandoffMode
  depth: number
  parent_session_id: string
  /** Set on a nested handoff: the child session that created it. */
  parent_child_session_id?: string
  created_at: string
  children: HandoffChildView[]
}

/** One handoff tree (`internal/handoff.TreeView`). Times are UTC "YYYY-MM-DD HH:MM:SS". */
export interface HandoffTreeView {
  tree_id: string
  root_session_id: string
  project_path?: string
  created_at: string
  last_access_at: string
  expires_at: string
  expired: boolean
  tokens_used: number
  tokens_max: number
  entries_used: number
  entries_max: number
  active_claims: number
  queued_claims: number
  search_calls: number
  repeat_calls: number
  repeat_rate: number
  tokens_delivered: number
  tokens_saved: number
  handoffs: HandoffView[]
}

export interface HandoffTreesResponse {
  trees: HandoffTreeView[]
  limits: HandoffLimits
  repeat_search_ratio_24h: number
}

export interface FlushHandoffTreeResponse {
  status: string
  flushed: {
    tree_id: string
    handoffs: number
    children: number
    notes_deleted: number
    memory_deleted: number
  }
}

/** Agent host the installer configures (`internal/installer.Target`). */
export type InstallerTargetId =
  | 'claude_code'
  | 'cursor'
  | 'opencode'
  | 'codex'
  | 'claude_desktop'
  | 'vscode'
  | 'jetbrains'

/** Installable piece of a target, in display and apply order. */
export type InstallerComponentId = 'mcp' | 'skills' | 'rules' | 'hooks'

export type InstallerAction = 'install' | 'uninstall'

/** A component's installed state, computed from the files on disk (IN-8). */
export type InstallerStatus =
  | 'installed'
  | 'outdated'
  | 'modified_by_user'
  | 'missing'
  | 'not_installed'
  | 'externally_managed'
  | 'unsupported'
  | 'covered'

/** What applying a change does to its file. */
export type InstallerChangeKind = 'create' | 'modify' | 'remove-block' | 'delete' | 'none'

/** One target × component cell of `GET /api/dashboard/installer`. */
export interface InstallerComponent {
  component: InstallerComponentId
  supported: boolean
  path?: string
  /** Why the component is unsupported. */
  reason?: string
  status: InstallerStatus
  /** Detail on the status, e.g. a parse error or which target already covers it. */
  status_reason?: string
}

export interface InstallerTarget {
  id: InstallerTargetId
  name: string
  components: InstallerComponent[]
}

export interface InstallerOverview {
  targets: InstallerTarget[]
  /** Findings of the one-time pre-4.0 install-record check (IN-13). */
  legacy_warnings: string[]
  hooks_enabled: boolean
}

export interface InstallerPlanRequest {
  targets: InstallerTargetId[]
  components: InstallerComponentId[]
  action: InstallerAction
  replace_external: boolean
}

/** One file a plan would touch (`internal/installer.FileChange`). */
export interface InstallerFileChange {
  target: InstallerTargetId
  component: InstallerComponentId
  path: string
  kind: InstallerChangeKind
  /** Unified diff; empty for a skipped change. */
  diff: string
  skipped: boolean
  reason?: string
}

export interface InstallerComponentStatus {
  target: InstallerTargetId
  component: InstallerComponentId
  status: InstallerStatus
  path: string
  reason?: string
}

/** A target aborted because one of its files could not be edited safely (IN-3). */
export interface InstallerPlanError {
  target: InstallerTargetId
  component: InstallerComponentId
  code: string
  message: string
}

export interface InstallerPlan {
  plan_id: string
  action: InstallerAction
  changes: InstallerFileChange[]
  status: InstallerComponentStatus[]
  warnings: string[]
  errors?: InstallerPlanError[]
  expires_at: string
}

export interface InstallerBackup {
  /** `<timestamp dir>/<encoded name>`, passed back to restore. */
  id: string
  path: string
  created_at: string
  size: number
  symlink?: boolean
}

export interface InstallerApplyResult {
  plan_id: string
  written: string[]
  backups: InstallerBackup[]
  status: InstallerComponentStatus[]
  warnings: string[]
}

export interface InstallerBackupsResponse {
  backups: InstallerBackup[]
}

/** Error thrown by API POSTs; installer apply sets `repreview` when the plan is stale (IN-5). */
export interface ApiError extends Error {
  code: string
  repreview: boolean
}
