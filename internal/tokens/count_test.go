package tokens

import (
	"crypto/sha256"
	"encoding/hex"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"unicode/utf8"

	"github.com/cespare/xxhash/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// resetState clears the lazy vocabulary and memo so a test starts cold.
func resetState(t *testing.T) {
	t.Helper()
	orig := loadRanks
	initOnce = sync.Once{}
	ranks = nil
	memo.reset()
	t.Cleanup(func() {
		loadRanks = orig
		initOnce = sync.Once{}
		ranks = nil
		memo.reset()
	})
}

func codeText(size int) string {
	const snippet = "func (s *Server) handle(ctx context.Context, req *Request) (*Response, error) {\n\tif req == nil {\n\t\treturn nil, errs.New(\"nil request\")\n\t}\n\treturn &Response{ID: req.ID, Items: s.items[req.Key]}, nil\n}\n\n"
	return strings.Repeat(snippet, size/len(snippet)+1)[:size]
}

func TestVocabularySHA256(t *testing.T) {
	data, err := vocabulary()
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
	loadRanks = func() (map[string]int, error) { return nil, errs.New("boom") }
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

func TestApproxCalibrated(t *testing.T) {
	assert.Equal(t, 0, Approx(""))
	assert.Equal(t, 100, Approx(strings.Repeat("x", 364)))
	approx, exact := 0, 0
	err := filepath.WalkDir("../..", func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || filepath.Ext(p) != ".go" {
			return err
		}
		data, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		approx += Approx(string(data))
		exact += Count(string(data))
		return nil
	})
	require.NoError(t, err)
	require.Positive(t, exact)
	assert.InEpsilon(t, exact, approx, 0.05, "approx=%d exact=%d", approx, exact)
}

func TestTruncate(t *testing.T) {
	resetState(t)
	words := strings.Repeat("alpha beta gamma delta ", 50)
	tests := []struct {
		name string
		in   string
		max  int
	}{
		{"words", words, 20},
		{"code", codeText(4096), 100},
		{"no spaces", strings.Repeat("x", 400), 10},
		{"multibyte", strings.Repeat("é", 300), 10},
		{"cjk", strings.Repeat("中文字符", 100), 25},
		{"emoji", strings.Repeat("😀🎉", 100), 15},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := Truncate(tc.in, tc.max)
			assert.True(t, strings.HasPrefix(tc.in, got))
			assert.True(t, utf8.ValidString(got))
			assert.LessOrEqual(t, Count(got), tc.max)
			assert.Greater(t, Count(got), tc.max*3/4, "the cut keeps most of the budget")
		})
	}
	got := Truncate(words, 20)
	assert.Equal(t, byte(' '), words[len(got)], "cut lands on a word boundary")
	assert.Equal(t, "short text", Truncate("short text", 10))
	assert.Equal(t, "", Truncate("short text", 0))
	assert.Equal(t, "", Truncate("", 5))
}

func TestTruncateFallback(t *testing.T) {
	resetState(t)
	loadRanks = func() (map[string]int, error) { return nil, errs.New("boom") }
	got := Truncate(strings.Repeat("abcd", 10), 2)
	assert.LessOrEqual(t, len(got), 11)
	assert.LessOrEqual(t, Count(got), 2)
}
