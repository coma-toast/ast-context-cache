package dashboard

import "github.com/coma-toast/ast-context-cache/internal/dashboard/components"

// PopulateEmbedSettings copies embedding-related settings keys (same names as env vars) into UI data.
func PopulateEmbedSettings(settings map[string]string, data *components.SettingsData) {
	data.EmbedBackend = settings["EMBED_BACKEND"]
	data.EmbedModelDir = settings["MODEL_DIR"]
	data.EmbedHTTPURL = settings["EMBED_HTTP_URL"]
	data.EmbedHTTPBearer = settings["EMBED_HTTP_BEARER"]
	data.EmbedOllamaHost = settings["OLLAMA_HOST"]
	data.EmbedOllamaModel = settings["OLLAMA_EMBED_MODEL"]
	data.EmbedOpenAIBaseURL = settings["EMBED_OPENAI_BASE_URL"]
	data.EmbedOpenAIAPIKey = settings["EMBED_OPENAI_API_KEY"]
	data.EmbedOpenAIModel = settings["EMBED_OPENAI_MODEL"]
	data.EmbedOpenAIDimensions = settings["EMBED_OPENAI_DIMENSIONS"]
	data.EmbedDockerURL = settings["EMBED_DOCKER_URL"]
	data.EmbedDockerModel = settings["EMBED_DOCKER_MODEL"]
	data.EmbedDockerDimensions = settings["EMBED_DOCKER_DIMENSIONS"]
}
