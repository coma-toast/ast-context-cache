package embedder

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var _ Interface = (*hashEmbedder)(nil)

func cosine(a, b []float32) float64 {
	var dot float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
	}
	return dot
}

func TestHashEmbedderDeterministic(t *testing.T) {
	e1, e2 := NewHashEmbedder(0), NewHashEmbedder(0)
	v1, err := e1.EmbedSingle("func LoadConfig(path string) (*Config, error)")
	require.NoError(t, err)
	v2, err := e2.EmbedSingle("func LoadConfig(path string) (*Config, error)")
	require.NoError(t, err)
	assert.Len(t, v1, Dimensions)
	assert.Equal(t, v1, v2)
	batch, err := e1.Embed([]string{"func LoadConfig(path string) (*Config, error)", "other"})
	require.NoError(t, err)
	require.Len(t, batch, 2)
	assert.Equal(t, v1, batch[0])
}

func TestHashEmbedderNormalized(t *testing.T) {
	e := NewHashEmbedder(64)
	for _, text := range []string{"hello", "parse_http_request handles HTTPServer", "a b c d e f g"} {
		v, err := e.EmbedSingle(text)
		require.NoError(t, err)
		require.Len(t, v, 64)
		assert.InDelta(t, 1.0, math.Sqrt(cosine(v, v)), 1e-5, text)
	}
	v, err := e.EmbedSingle("  ...  ")
	require.NoError(t, err)
	assert.Equal(t, make([]float32, 64), v)
}

func TestHashEmbedderSimilarity(t *testing.T) {
	e := NewHashEmbedder(0)
	vecs, err := e.Embed([]string{"loadConfigFile reads the config file", "load_config_file reads a config", "render the dashboard chart widget"})
	require.NoError(t, err)
	assert.Greater(t, cosine(vecs[0], vecs[1]), cosine(vecs[0], vecs[2]))
}

func TestHashTokens(t *testing.T) {
	tests := []struct {
		in   string
		want []string
	}{
		{"LoadConfig", []string{"loadconfig", "load", "config"}},
		{"parse_http_request", []string{"parse_http_request", "parse", "http", "request"}},
		{"HTTPServer run", []string{"httpserver", "http", "server", "run"}},
	}
	for _, tc := range tests {
		assert.Equal(t, tc.want, hashTokens(tc.in), tc.in)
	}
}
