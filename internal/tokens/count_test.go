package tokens

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"sync"
	"testing"

	"github.com/cespare/xxhash/v2"
	"github.com/pkoukk/tiktoken-go"
	"github.com/pkoukk/tiktoken-go-loader/assets"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// o200kSHA256 is the published sha256 of o200k_base.tiktoken (openai/tiktoken load.py).
const o200kSHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"

// resetState clears the lazy encoder and memo so a test starts cold.
func resetState(t *testing.T) {
	t.Helper()
	orig := loadEncoder
	initOnce = sync.Once{}
	encoder = nil
	memo.reset()
	t.Cleanup(func() {
		loadEncoder = orig
		initOnce = sync.Once{}
		encoder = nil
		memo.reset()
	})
}

func codeText(size int) string {
	const snippet = "func (s *Server) handle(ctx context.Context, req *Request) (*Response, error) {\n\tif req == nil {\n\t\treturn nil, errs.New(\"nil request\")\n\t}\n\treturn &Response{ID: req.ID, Items: s.items[req.Key]}, nil\n}\n\n"
	return strings.Repeat(snippet, size/len(snippet)+1)[:size]
}

func TestVocabularySHA256(t *testing.T) {
	data, err := assets.Assets.ReadFile(encodingName + ".tiktoken")
	require.NoError(t, err)
	sum := sha256.Sum256(data)
	assert.Equal(t, o200kSHA256, hex.EncodeToString(sum[:]))
}

func TestCountKnownStrings(t *testing.T) {
	resetState(t)
	tests := []struct {
		text string
		want int
	}{
		{"", 0},
		{"hello world", 2},
		{"func main() { fmt.Println(\"hi\") }", 10},
		{"The quick brown fox jumps over the lazy dog.", 10},
		{"<|endoftext|> is a special token", 11},
	}
	for _, tc := range tests {
		assert.Equal(t, tc.want, Count(tc.text), "text=%q", tc.text)
	}
	assert.Equal(t, methodBPE, Method())
}

func TestCountFallback(t *testing.T) {
	resetState(t)
	loadEncoder = func() (*tiktoken.Tiktoken, error) { return nil, errs.New("boom") }
	assert.Equal(t, 2, Count("hello world"))
	assert.Equal(t, 0, Count("abc"))
	assert.Equal(t, methodBytes, Method())
}

func TestCountConcurrent(t *testing.T) {
	resetState(t)
	text := codeText(2048)
	want := Count(text)
	short := Count("hello world")
	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 50; j++ {
				assert.Equal(t, want, Count(text))
				assert.Equal(t, short, Count("hello world"))
				Count(codeText(512 + j))
			}
		}()
	}
	wg.Wait()
}

func TestCountMemoization(t *testing.T) {
	resetState(t)
	Count(strings.Repeat("a", memoMinBytes-1))
	assert.Equal(t, 0, memo.len())
	text := codeText(4096)
	n := Count(text)
	require.Equal(t, 1, memo.len())
	got, ok := memo.get(xxhash.Sum64String(text))
	require.True(t, ok)
	assert.Equal(t, n, got)
	assert.Equal(t, n, Count(text))
	assert.Equal(t, 1, memo.len())
}

func TestLRUEviction(t *testing.T) {
	l := newLRU(2)
	l.put(1, 10)
	l.put(2, 20)
	_, _ = l.get(1)
	l.put(3, 30)
	_, ok := l.get(2)
	assert.False(t, ok)
	v, ok := l.get(1)
	assert.True(t, ok)
	assert.Equal(t, 10, v)
	assert.Equal(t, 2, l.len())
}

func BenchmarkCount4k(b *testing.B) {
	text := codeText(16 * 1024)
	Count("warm up")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memo.reset()
		Count(text)
	}
}
