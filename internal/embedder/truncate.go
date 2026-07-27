package embedder

// maxRemoteEmbedInputRunes matches indexer.maxEmbedInputRunes: remote GGUF /
// LiteLLM paths often use a 512-token context; dense code ≈ 1 token/rune.
const maxRemoteEmbedInputRunes = 480

// TruncateRemoteEmbedInput shortens a single embedding input for remote APIs.
func TruncateRemoteEmbedInput(s string) string {
	r := []rune(s)
	if len(r) <= maxRemoteEmbedInputRunes {
		return s
	}
	return string(r[:maxRemoteEmbedInputRunes])
}
