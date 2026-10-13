package dashboard

import (
	"os"
	"strconv"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/embedqueue"
	"github.com/coma-toast/ast-context-cache/internal/handoff"
	"github.com/coma-toast/ast-context-cache/internal/ignorepatterns"
	"github.com/coma-toast/ast-context-cache/internal/projectmeta"
	"github.com/coma-toast/ast-context-cache/internal/transcripts"
)

// handoffEnvLocked lists the handoff limit keys overridden by a non-empty environment variable.
func handoffEnvLocked() []string {
	locked := []string{}
	for _, ls := range handoff.LimitSettings() {
		if strings.TrimSpace(os.Getenv(handoff.EnvKey(ls.Key))) != "" {
			locked = append(locked, ls.Key)
		}
	}
	return locked
}

// dataDirSize formats the on-disk size of the three databases for display in Settings.
func dataDirSize() string {
	return db.FormatFileSize(db.DataDirSizeBytes())
}

type settingsBuildOpts struct {
	loadEmbedModels bool
}

func buildSettingsData(opts settingsBuildOpts) components.SettingsData {
	settings := db.GetAllSettings()
	idleMinutes := 1
	if v, ok := settings["idle_unload_minutes"]; ok {
		if parsed, err := strconv.Atoi(v); err == nil {
			idleMinutes = parsed
		}
	}
	watcherIgn := ignorepatterns.JSONForSettings(settings["watcher_ignore_globs"])
	projectExclude := projectmeta.ExcludeJSONForSettings(settings["project_exclude_paths"])
	indexLog := settings["index_log_files"] == "true"
	logRoots := settings["log_retention_roots"]
	if logRoots == "" {
		logRoots = "[]"
	}
	logRetentionMaxAge := 0
	if v, ok := settings["log_retention_max_age_days"]; ok && v != "" {
		logRetentionMaxAge, _ = strconv.Atoi(v)
	}
	logRetentionMaxMB := 0
	if v, ok := settings["log_retention_max_total_mib"]; ok && v != "" {
		logRetentionMaxMB, _ = strconv.Atoi(v)
	}
	logRetentionEn := settings["log_retention_enabled"] == "true"
	logDry := settings["log_retention_dry_run"] == "true"
	logLast := settings["log_retention_last_run"]
	queryRetentionEn := settings["query_retention_enabled"] != "false"
	queryRetentionMaxAge := 90
	if v, ok := settings["query_retention_max_age_days"]; ok && v != "" {
		queryRetentionMaxAge, _ = strconv.Atoi(v)
	}
	queryRetentionLast := settings["query_retention_last_run"]

	projects, projectsLoading := loadProjectsForPage()
	data := components.SettingsData{
		IdleUnloadMinutes:        idleMinutes,
		WatcherIgnoreGlobs:       watcherIgn,
		ProjectExcludePaths:      projectExclude,
		ProjectIndexExcludes:     db.GetProjectIndexExcludes(),
		IndexLogFiles:            indexLog,
		TranscriptUsageIngest:    settings[transcripts.SettingKey] == "true",
		LogRetentionEnabled:      logRetentionEn,
		LogRetentionRoots:        logRoots,
		LogRetentionMaxAgeDays:   logRetentionMaxAge,
		LogRetentionMaxTotalMB:   logRetentionMaxMB,
		LogRetentionDryRun:       logDry,
		LogRetentionLastRun:      logLast,
		QueryRetentionEnabled:    queryRetentionEn,
		QueryRetentionMaxAgeDays: queryRetentionMaxAge,
		QueryRetentionLastRun:    queryRetentionLast,
		Projects:                 projects,
		ProjectsLoading:          projectsLoading,
		EmbedWorkerMax:           embedqueue.MaxWorkers(),
		EmbedAuxWorkerMax:        embedqueue.AuxMaxWorkers(),
		EmbedAuxWorkers:          embedqueue.AuxWorkerTarget(),
		EmbedAuxBackend:          strings.TrimSpace(db.GetSetting("EMBED_AUX_BACKEND", "onnx")),
		DataDir:                  db.GetDataDir(),
		DataDirSize:              dataDirSize(),
	}
	if data.EmbedAuxBackend == "" {
		data.EmbedAuxBackend = "onnx"
	}
	data.EmbedProbeIntervalSec = int(embedder.ProbeInterval().Seconds())
	data.HandoffLimits, data.HandoffEnvLocked = handoff.LoadLimits(), handoffEnvLocked()
	PopulateEmbedSettings(settings, &data)
	populateContextSettings(settings, &data)
	applyActiveEmbedderSettings(&data)
	if opts.loadEmbedModels {
		loadEmbedModels(&data)
	}
	return data
}
