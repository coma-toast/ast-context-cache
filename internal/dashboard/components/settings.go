package components

import (
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/embedqueue"
	"github.com/coma-toast/ast-context-cache/internal/handoff"
)

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
	EmbedHTTPBearerSet       bool
	EmbedOllamaHost          string
	EmbedOllamaModel         string
	EmbedOpenAIBaseURL       string
	EmbedOpenAIAPIKey        string
	EmbedOpenAIAPIKeySet     bool
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
	// AuxAvailable is false when the aux backend is onnx and ONNX Runtime can't load; the
	// dashboard then hides the aux controls and offers the managed install.
	AuxAvailable         bool
	AuxUnavailableReason string
	// ONNXRuntime is the ONNX Runtime load state, managed install and pinned version.
	ONNXRuntime           embedder.ORTStatus
	EmbedProbeIntervalSec int
	DataDir               string
	DataDirSize           string
	// EmbedQueueCaps are the embed queue soft limits with their current depth and env lock.
	EmbedQueueCaps []embedqueue.QueueCapState
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
	// InstanceFaviconMode, InstanceFaviconColor and InstanceFaviconBadge are the
	// stored favicon settings (mode default|color|upload).
	InstanceFaviconMode, InstanceFaviconColor, InstanceFaviconBadge string
	// InstanceFaviconUploadType and InstanceFaviconUploadBytes describe the stored upload, if any.
	InstanceFaviconUploadType  string
	InstanceFaviconUploadBytes int
	// InstanceFaviconEffectiveMode is the mode in effect after the env vars and fallbacks.
	InstanceFaviconEffectiveMode string
	// InstanceFaviconSource is env|override|settings|default.
	InstanceFaviconSource string
	// InstanceFaviconURL is the favicon path with its ?v= content hash.
	InstanceFaviconURL string
	// InstanceFaviconOverride is the instance_favicon_override setting: the settings win over the env vars.
	InstanceFaviconOverride bool
	// InstanceFaviconLocked is true when an AST_INSTANCE_FAVICON_* env var sets the
	// favicon and the override is off.
	InstanceFaviconLocked bool
	// InstanceFaviconExternalColor and InstanceFaviconExternalBadge are the valid env values, if any.
	InstanceFaviconExternalColor, InstanceFaviconExternalBadge string
}
