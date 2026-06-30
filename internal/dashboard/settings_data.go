package dashboard

import (
	"strconv"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedqueue"
	"github.com/coma-toast/ast-context-cache/internal/ignorepatterns"
	"github.com/coma-toast/ast-context-cache/internal/projectmeta"
)

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
	configs, _ := db.GetAgentConfigs()
	var agents []components.AgentInfo
	for _, sa := range supportedAgents {
		a := components.AgentInfo{
			Type:        sa.Type,
			Name:        sa.Name,
			GlobalPath:  sa.GlobalPath,
			ProjectPath: sa.ProjectPath,
			Description: sa.Description,
		}
		for _, c := range configs {
			if c.AgentType == sa.Type {
				if c.IsGlobal {
					a.GlobalInstalled = true
				} else {
					a.ProjectInstalled = true
				}
			}
		}
		agents = append(agents, a)
	}
	data := components.SettingsData{
		IdleUnloadMinutes:        idleMinutes,
		WatcherIgnoreGlobs:       watcherIgn,
		ProjectExcludePaths:      projectExclude,
		IndexLogFiles:            indexLog,
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
		Agents:                   agents,
		EmbedWorkerMax:           embedqueue.MaxWorkers(),
		EmbedAuxWorkerMax:        embedqueue.AuxMaxWorkers(),
		EmbedAuxWorkers:          embedqueue.AuxWorkerTarget(),
		EmbedAuxBackend:          strings.TrimSpace(db.GetSetting("EMBED_AUX_BACKEND", "onnx")),
	}
	if data.EmbedAuxBackend == "" {
		data.EmbedAuxBackend = "onnx"
	}
	PopulateEmbedSettings(settings, &data)
	populateContextSettings(settings, &data)
	applyActiveEmbedderSettings(&data)
	if opts.loadEmbedModels {
		loadEmbedModels(&data)
	}
	return data
}
