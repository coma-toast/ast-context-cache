package indexer

// maxEmbedInputRunes caps the assembled embed string (kind + name + source).
// Remote stacks (LiteLLM → llama.cpp GGUF) often default to 512 context; dense
// code can approach one BPE token per rune, so a 500-char source alone can
// exceed the limit once the prefix is included.
const maxEmbedInputRunes = 480

// BuildEmbedText formats and truncates symbol text for embedding and content hashing.
func BuildEmbedText(kind, name, src string) string {
	return truncateEmbedText(kind + " " + name + ": " + src)
}

func truncateEmbedText(s string) string {
	r := []rune(s)
	if len(r) <= maxEmbedInputRunes {
		return s
	}
	return string(r[:maxEmbedInputRunes])
}
