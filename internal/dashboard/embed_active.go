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
}

func applyActiveEmbedderSettings(data *components.SettingsData) {
	data.EmbedActiveBackend, data.EmbedActiveModel, data.EmbedActiveRuntime, data.EmbedActiveEndpoint, data.EmbedActiveDim = embedder.ActiveSnapshot()
	state, _ := mcp.EmbedderState()
	data.EmbedActiveLoaded = state == "ready"
}

func applyActiveEmbedderHealth(h *components.Health) {
	h.EmbedBackend, h.EmbedModel, h.EmbedRuntime, _, h.EmbedDim = embedder.ActiveSnapshot()
}
