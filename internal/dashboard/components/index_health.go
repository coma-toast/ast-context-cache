package components

import "time"

type EmbedActivityItem struct {
	File        string
	ProjectPath string
}

type IndexHealth struct {
	TotalSymbols             int
	TotalFiles               int
	TotalEdges               int
	TotalVectors             int
	VectorMemMB              float64
	MemoryMB                 float64
	DiskMB                   float64
	WalMB                    float64
	WalSize                  string
	CPUPercent               float64
	LoadAvgAvailable         bool
	LoadAvg1                 float64
	LoadAvg5                 float64
	LoadAvg15                float64
	DiskSize                 string
	DiskReadMBps             float64
	DiskWriteMBps            float64
	SSDModel                 string
	SSDSmartStatus           string
	SSDProtocol              string
	SSDCapacity              string
	SSDSolidState            bool
	SSDTrim                  bool
	SSDAvailable             bool
	SSDWearUsedPct           int
	SSDSparePct              int
	SSDDataWrittenTB         float64
	SSDTemperatureC          float64
	Watchers                 []WatcherInfo
	EmbedQueued              int
	EmbedPending             int
	EmbedPendingPeak         int
	EmbedFailed              int64
	EmbedHighQueued          int
	EmbedLowQueued           int
	EmbedHighCap             int
	EmbedLowCap              int
	EmbedActive              int
	EmbedWorkers             int
	EmbedWorkersEffective    int
	EmbedWorkersLive         int
	EmbedWorkerMax           int
	EmbedAuxWorkers          int
	EmbedAuxWorkersEffective int
	EmbedAuxWorkersLive      int
	EmbedAuxWorkerMax        int
	EmbedAuxBackend          string
	EmbedAuxModel            string
	EmbedAuxEnabled          bool
	EmbedComplete            int64
	EmbedThroughput          int64
	PinnedCount              int
	FilteredProject          string
	EmbedBackend             string
	EmbedModel               string
	EmbedRuntime             string
	EmbedEndpoint            string
	EmbedDim                 int
	EmbedderState            string
	EmbedderError            string
	EmbedLoaded              bool
	EmbedRecent              []EmbedActivityItem
	EmbedInProgress          []EmbedActivityItem
	EmbedConfiguredBackend   string
	EmbedConfiguredModel     string
	EmbedInSync              bool
	WALMaintenanceActive     bool
	WALMaintenancePhase      string
	WALMaintenanceMode       string
	WALMaintenanceReason     string
	WALMaintenanceStarted    time.Time
	WALWalStartBytes         int64
	WALWalCurrentBytes       int64
	WALBusyStreak            int32
	WALInFlight              int64
	WALLastBusy              int
	WALPressure              string
	LastAutoRecoverUnix      int64
}

type WatcherInfo struct {
	ProjectPath string
	Name        string
	Label       string
	Workspace   string
	Active      bool
	LinkedCount int
}

type IndexDocSource struct {
	ID         int
	Name       string
	Type       string
	URL        string
	Age        string
	Stale      bool
	Refreshing bool
}
