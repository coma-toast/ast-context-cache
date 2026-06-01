package dashboard

import (
	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/embedqueue"
	"github.com/coma-toast/ast-context-cache/internal/mcp"
)

func applyActiveEmbedder(h *components.IndexHealth) {
	h.EmbedBackend, h.EmbedModel, h.EmbedRuntime, h.EmbedEndpoint, h.EmbedDim = embedder.ActiveSnapshot()
	h.EmbedRecent = embedActivityItems(embedqueue.RecentActivity())
	h.EmbedInProgress = embedActivityItems(embedqueue.CurrentJobs())
	state, _ := mcp.EmbedderState()
	h.EmbedLoaded = state == "ready" || embedder.IsNetworkBackend(h.EmbedBackend) || h.EmbedActive > 0 || h.EmbedQueued > 0
	applyEmbedAlignmentToIndexHealth(h)
}

func embedActivityItems(in []embedqueue.ActivityEntry) []components.EmbedActivityItem {
	if len(in) == 0 {
		return nil
	}
	out := make([]components.EmbedActivityItem, len(in))
	for i, e := range in {
		out[i] = components.EmbedActivityItem{File: e.File, ProjectPath: e.ProjectPath}
	}
	return out
}

func applyActiveEmbedderSettings(data *components.SettingsData) {
	data.EmbedActiveBackend, data.EmbedActiveModel, data.EmbedActiveRuntime, data.EmbedActiveEndpoint, data.EmbedActiveDim = embedder.ActiveSnapshot()
	state, _ := mcp.EmbedderState()
	data.EmbedActiveLoaded = state == "ready"
	applyEmbedAlignmentToSettings(data)
}

func applyEmbedAlignmentToSettings(data *components.SettingsData) {
	a := embedder.AlignmentStatus()
	data.EmbedConfiguredBackend = a.ConfiguredBackend
	data.EmbedConfiguredModel = a.ConfiguredModel
	data.EmbedInSync = a.InSync
	data.EmbedEnvOverrides = a.EnvOverrides
}

func applyEmbedAlignmentToIndexHealth(h *components.IndexHealth) {
	a := embedder.AlignmentStatus()
	h.EmbedConfiguredBackend = a.ConfiguredBackend
	h.EmbedConfiguredModel = a.ConfiguredModel
	h.EmbedInSync = a.InSync
}

func applyActiveEmbedderHealth(h *components.Health) {
	h.EmbedBackend, h.EmbedModel, h.EmbedRuntime, _, h.EmbedDim = embedder.ActiveSnapshot()
}
