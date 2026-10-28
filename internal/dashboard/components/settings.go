package components

import "github.com/coma-toast/ast-context-cache/internal/handoff"

type SettingsData struct {
	IdleUnloadMinutes        int
	WatcherIgnoreGlobs       string
	ProjectExcludePaths      string
	ProjectIndexExcludes     map[string][]string // project path → gitignore-syntax excludes
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
	DataDir                  string
	DataDirSize              string
	// HandoffLimits are the effective handoff limits (env > setting > default).
	HandoffLimits handoff.Limits
	// HandoffEnvLocked lists the handoff limit settings keys whose AST_HANDOFF_* env var is set:
	// saving them from the dashboard has no effect.
	HandoffEnvLocked []string
	// TranscriptUsageIngest is the transcript_usage_ingest setting (TL-5, default off).
	TranscriptUsageIngest bool
	// InstanceName is the raw instance_name DB setting (what the field edits).
	InstanceName string
	// InstanceNameEffective is the live name (flag > env > setting > empty; with
	// InstanceNameOverride a non-empty setting comes first).
	InstanceNameEffective string
	// InstanceScope is the cookie/localStorage scope for the effective name.
	InstanceScope string
	// InstanceNameSource is flag|env|settings|override|default.
	InstanceNameSource string
	// InstanceNameLocked is true when the flag or env var sets the name and the
	// override is off: the dashboard may not change instance_name.
	InstanceNameLocked bool
	// InstanceNameOverride is the instance_name_override setting: the setting wins
	// over the flag and env var.
	InstanceNameOverride bool
	// InstanceNameExternal is the -name flag or AST_INSTANCE_NAME value, if any.
	InstanceNameExternal string
	// InstanceNameExternalSource is flag|env for InstanceNameExternal.
	InstanceNameExternalSource string
}
