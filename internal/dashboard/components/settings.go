package components

type AgentInfo struct {
	Type             string
	Name             string
	GlobalPath       string
	ProjectPath      string
	Description      string
	GlobalInstalled  bool
	ProjectInstalled bool
}

type SettingsData struct {
	IdleUnloadMinutes        int
	WatcherIgnoreGlobs       string
	ProjectExcludePaths      string
	IndexLogFiles            bool
	LogRetentionEnabled      bool
	LogRetentionRoots        string
	LogRetentionMaxAgeDays   int
	LogRetentionMaxTotalMB   int
	LogRetentionDryRun       bool
	LogRetentionLastRun      string
	QueryRetentionEnabled    bool
	QueryRetentionMaxAgeDays int
	QueryRetentionLastRun    string
	EmbedBackend             string
	EmbedModelDir            string
	EmbedHTTPURL             string
	EmbedHTTPBearer          string
	EmbedOllamaHost          string
	EmbedOllamaModel         string
	EmbedOpenAIBaseURL       string
	EmbedOpenAIAPIKey        string
	EmbedOpenAIModel         string
	EmbedOpenAIDimensions    string
	EmbedDockerURL           string
	EmbedDockerModel         string
	EmbedDockerDimensions    string
	EmbedModels              []string
	EmbedModelsErr           string
	EmbedActiveBackend       string
	EmbedActiveModel         string
	EmbedActiveRuntime       string
	EmbedActiveEndpoint      string
	EmbedActiveDim           int
	EmbedActiveLoaded        bool
	EmbedActive              int
	EmbedderState            string
	EmbedderError            string
	EmbedConfiguredBackend   string
	EmbedConfiguredModel     string
	EmbedInSync              bool
	EmbedEnvOverrides        []string
	Projects                 []Project
	ProjectsLoading          bool
	Agents                   []AgentInfo
	ContextMaxNotesSession   int
	ContextMaxTokensSession  int
	ContextMaxNotesGlobal    int
	ContextMaxTokensGlobal   int
	ContextLimitPolicy       string
	EmbedWorkerMax           int
	EmbedAuxWorkerMax        int
	EmbedAuxWorkers          int
	EmbedAuxBackend          string
	EmbedProbeIntervalSec    int
}
