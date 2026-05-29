package dashboard

import (
	"github.com/coma-toast/ast-context-cache/internal/dashboard/components"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/mcp"
)

func applyActiveEmbedder(h *components.IndexHealth) {
	h.EmbedBackend, h.EmbedModel, h.EmbedRuntime, h.EmbedEndpoint, h.EmbedDim = embedder.ActiveSnapshot()
	state, _ := mcp.EmbedderState()
	h.EmbedLoaded = state == "ready"
	applyEmbedAlignmentToIndexHealth(h)
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
