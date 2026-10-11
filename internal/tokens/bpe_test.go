package tokens

import (
	"io/fs"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/dlclark/regexp2"
	"github.com/pkoukk/tiktoken-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var _ tiktoken.BpeLoader = embeddedLoader{}

// o200kPattern is tiktoken-go's o200k_base pre-tokenizer regex.
var o200kPattern = regexp2.MustCompile(strings.Join([]string{
	`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?`,
	`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?`,
	`\p{N}{1,3}`,
	` ?[^\s\p{L}\p{N}]+[\r\n/]*`,
	`\s*[\r\n]+`,
	`\s+(?!\S)`,
	`\s+`,
}, "|"), regexp2.None)

// regexPieces splits valid UTF-8 text into pre-tokens with o200kPattern.
func regexPieces(t *testing.T, text string) []string {
	t.Helper()
	runes := []rune(text)
	var out []string
	m, err := o200kPattern.FindStringMatch(text)
	for ; m != nil && err == nil; m, err = o200kPattern.FindNextMatch(m) {
		out = append(out, string(runes[m.Index:m.Index+m.Length]))
	}
	require.NoError(t, err)
	return out
}

func scanPieces(text string) []string {
	var out []string
	for i := 0; i < len(text); {
		e := pieceEnd(text, i)
		out = append(out, text[i:e])
		i = e
	}
	return out
}

var (
	oracleOnce sync.Once
	oracleEnc  *tiktoken.Tiktoken
	oracleErr  error
)

// oracle is tiktoken-go's regex-based o200k encoder served by embeddedLoader.
func oracle(t *testing.T) *tiktoken.Tiktoken {
	t.Helper()
	oracleOnce.Do(func() {
		tiktoken.SetBpeLoader(embeddedLoader{})
		oracleEnc, oracleErr = tiktoken.GetEncoding(encodingName)
	})
	require.NoError(t, oracleErr)
	return oracleEnc
}

func TestLoaderRejectsOtherVocabularies(t *testing.T) {
	_, err := embeddedLoader{}.LoadTiktokenBpe("https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken")
	assert.Error(t, err)
}

func TestCountMatchesTiktokenOnRepo(t *testing.T) {
	enc := oracle(t)
	r := ranksInstance()
	require.NotNil(t, r)
	exts := map[string]bool{".go": true, ".md": true, ".ts": true, ".tsx": true, ".js": true, ".json": true, ".yaml": true, ".yml": true, ".py": true, ".sh": true, ".html": true, ".css": true}
	files := 0
	err := filepath.WalkDir("../..", func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() && (d.Name() == ".git" || d.Name() == "node_modules") {
			return filepath.SkipDir
		}
		if d.IsDir() || !exts[filepath.Ext(p)] {
			return nil
		}
		data, err := os.ReadFile(p)
		if err != nil || len(data) > 256*1024 {
			return err
		}
		text := string(data)
		assert.Equal(t, len(enc.EncodeOrdinary(text)), count(r, text), "file=%s", p)
		files++
		return nil
	})
	require.NoError(t, err)
	assert.Greater(t, files, 100)
}

func TestCountMatchesTiktokenOnRandomText(t *testing.T) {
	enc := oracle(t)
	r := ranksInstance()
	require.NotNil(t, r)
	alphabet := []string{
		"a", "Z", "ǅ", "ʰ", "中", "́", "ः", "5", "٣", "Ⅻ", "½", " ", " ", "\t", "\n", "\r", " ",
		" ", "\u0085", "　", "'", "s", "S", "t", "R", "e", "V", "m", "l", "L", "d", "D", "/", "!", "{",
		"😀", "\xff", "ß", "İ", "K", "_", "-", ".", "�", "é", "Ω", "ω", "1", "9",
	}
	rng := rand.New(rand.NewSource(1))
	var sb strings.Builder
	for i := 0; i < 20000; i++ {
		sb.Reset()
		for n := 1 + rng.Intn(40); n > 0; n-- {
			sb.WriteString(alphabet[rng.Intn(len(alphabet))])
		}
		text := sb.String()
		require.Equal(t, len(enc.EncodeOrdinary(text)), count(r, text), "text=%q", text)
		valid := string([]rune(text))
		require.Equal(t, regexPieces(t, valid), scanPieces(valid), "text=%q", text)
	}
}
