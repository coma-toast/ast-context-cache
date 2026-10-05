package installer

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMarkdownBlockUpsertAndRemove(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name, in string
		exact    bool // removal restores the input byte-for-byte
	}{
		{"empty", "", true},
		{"user content", "# Mine\n\nhello\n", true},
		{"CRLF", "a\r\nb\r\n", true},
		{"no trailing newline", "# Mine", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			nl := newlineOf(tt.in)
			block := renderBlock("4.0.0", "body\n", nl)
			out, err := upsertBlock(tt.in, block, nl)
			require.NoError(t, err)
			assert.True(t, strings.HasPrefix(out, strings.TrimRight(tt.in, "\r\n")))
			b, found, err := findBlock(out)
			require.NoError(t, err)
			require.True(t, found)
			assert.Equal(t, StatusInstalled, blockStatus(b, "body", "4.0.0"))
			again, err := upsertBlock(out, block, nl)
			require.NoError(t, err)
			assert.Equal(t, out, again)
			back, removed, err := removeBlock(out)
			require.NoError(t, err)
			assert.True(t, removed)
			if tt.exact {
				assert.Equal(t, tt.in, back)
			} else {
				assert.Equal(t, tt.in+"\n", back)
			}
		})
	}
}

func TestMarkdownBlockStatus(t *testing.T) {
	t.Parallel()
	block := renderBlock("4.0.0", "body", "\n")
	tests := []struct {
		name, file, version string
		want                Status
	}{
		{"current", "x\n\n" + block + "\n", "4.0.0", StatusInstalled},
		{"user edited the body", "x\n\n" + strings.Replace(block, "body", "body edited", 1) + "\n", "4.0.0", StatusModifiedByUser},
		{"newer version ships", "x\n\n" + block + "\n", "4.1.0", StatusOutdated},
		{"body text changed upstream", "x\n\n" + renderBlock("4.0.0", "older body", "\n") + "\n", "4.0.0", StatusOutdated},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			b, found, err := findBlock(tt.file)
			require.NoError(t, err)
			require.True(t, found)
			assert.Equal(t, tt.want, blockStatus(b, "body", tt.version))
		})
	}
}

func TestMarkdownBlockMalformed(t *testing.T) {
	t.Parallel()
	block := renderBlock("4.0.0", "body", "\n")
	tests := []struct {
		name, file string
	}{
		{"two blocks", block + "\n" + block + "\n"},
		{"no end marker", strings.Replace(block, blockEndMarker, "", 1)},
		{"bad begin marker", strings.Replace(block, "sha=", "hash=", 1)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			_, _, err := findBlock(tt.file)
			assert.Error(t, err)
		})
	}
}

func TestSplitFrontmatter(t *testing.T) {
	t.Parallel()
	front, body := splitFrontmatter("---\na: 1\n---\n\nbody\n")
	assert.Equal(t, "---\na: 1\n---\n", front)
	assert.Equal(t, "\nbody\n", body)
	front, body = splitFrontmatter("no frontmatter")
	assert.Empty(t, front)
	assert.Equal(t, "no frontmatter", body)
}
