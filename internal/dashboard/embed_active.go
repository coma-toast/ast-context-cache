package dashboard

import (
	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/embedqueue"
	"github.com/coma-toast/ast-context-cache/internal/mcp"
	"github.com/coma-toast/ast-context-cache/internal/startup"
)

func applyActiveEmbedder(h *components.IndexHealth) {
	h.EmbedBackend, h.EmbedModel, h.EmbedRuntime, h.EmbedEndpoint, h.EmbedDim = embedder.WiredSnapshot()
	h.EmbedRecent = embedActivityItems(embedqueue.RecentActivity())
	h.EmbedInProgress = embedActivityItems(embedqueue.CurrentJobs())
	state, _ := mcp.EmbedderState()
	h.EmbedderState = state
	h.EmbedderError = mcp.EmbedderError()
	overlayStartupEmbedder(&h.EmbedderState, &h.EmbedderError, &h.EmbedBackend)
	h.EmbedLoaded = h.EmbedderState == "ready"
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
	data.EmbedActiveBackend, data.EmbedActiveModel, data.EmbedActiveRuntime, data.EmbedActiveEndpoint, data.EmbedActiveDim = embedder.WiredSnapshot()
	eq := embedqueue.Snapshot()
	data.EmbedActive = int(eq.InFlight)
	state, _ := mcp.EmbedderState()
	data.EmbedderState = state
	data.EmbedderError = mcp.EmbedderError()
	overlayStartupEmbedder(&data.EmbedderState, &data.EmbedderError, &data.EmbedActiveBackend)
	data.EmbedActiveLoaded = data.EmbedderState == "ready"
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
	h.EmbedBackend, h.EmbedModel, h.EmbedRuntime, _, h.EmbedDim = embedder.WiredSnapshot()
}

func overlayStartupEmbedder(state, err *string, backend *string) {
	if startup.Starting() {
		*state = "loading"
		*backend = "…"
		return
	}
	if startup.Failed() {
		*state = "error"
		if *err == "" {
			*err = startup.Error()
		}
	}
}
