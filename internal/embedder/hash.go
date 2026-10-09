package embedder

import (
	"math"
	"strings"
	"unicode"

	"github.com/cespare/xxhash/v2"
)

// hashEmbedder is a deterministic offline bag-of-words embedder for tests and benchmarks.
type hashEmbedder struct {
	dims int
}

// NewHashEmbedder returns an embedder that hashes word and identifier-part tokens into
// dims buckets (Dimensions when dims <= 0) and L2-normalizes the result.
func NewHashEmbedder(dims int) Interface {
	if dims <= 0 {
		dims = Dimensions
	}
	return &hashEmbedder{dims: dims}
}

func (h *hashEmbedder) Embed(texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = h.vector(t)
	}
	return out, nil
}

func (h *hashEmbedder) EmbedSingle(text string) ([]float32, error) {
	return h.vector(text), nil
}

func (h *hashEmbedder) vector(text string) []float32 {
	v := make([]float32, h.dims)
	for _, tok := range hashTokens(text) {
		sum := xxhash.Sum64String(tok)
		sign := float32(1)
		if sum>>63 == 1 {
			sign = -1
		}
		v[sum%uint64(h.dims)] += sign
	}
	var norm float64
	for _, x := range v {
		norm += float64(x) * float64(x)
	}
	if norm == 0 {
		return v
	}
	inv := float32(1 / math.Sqrt(norm))
	for i := range v {
		v[i] *= inv
	}
	return v
}

// hashTokens splits text into lowercase words; identifiers also contribute their
// camelCase and snake_case parts.
func hashTokens(text string) []string {
	words := strings.FieldsFunc(text, func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '_' })
	var toks []string
	for _, w := range words {
		parts := splitIdentifier(w)
		if whole := strings.ToLower(strings.Trim(w, "_")); whole != "" && (len(parts) != 1 || parts[0] != whole) {
			toks = append(toks, whole)
		}
		toks = append(toks, parts...)
	}
	return toks
}

// splitIdentifier splits on underscores and camelCase boundaries (HTTPServer -> http, server).
func splitIdentifier(w string) []string {
	var parts []string
	for _, seg := range strings.Split(w, "_") {
		rs := []rune(seg)
		start := 0
		for i := 1; i < len(rs); i++ {
			lowerToUpper := unicode.IsUpper(rs[i]) && !unicode.IsUpper(rs[i-1])
			acronymEnd := unicode.IsUpper(rs[i-1]) && unicode.IsUpper(rs[i]) && i+1 < len(rs) && unicode.IsLower(rs[i+1])
			if lowerToUpper || acronymEnd {
				parts = append(parts, strings.ToLower(string(rs[start:i])))
				start = i
			}
		}
		if start < len(rs) {
			parts = append(parts, strings.ToLower(string(rs[start:])))
		}
	}
	return parts
}
