package installer

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

const tomlKey = "mcp_servers.ast-context-cache"

func testTOMLBlock(nl string) string {
	return tomlTableUnit{parent: "mcp_servers", url: "http://127.0.0.1:7821/mcp"}.block(nl)
}

func TestTOMLUpsert(t *testing.T) {
	t.Parallel()
	block := testTOMLBlock("\n")
	tests := []struct {
		name, in, want string
	}{
		{"empty file", "", block},
		{"no trailing newline", "a = 1", "a = 1\n\n" + block},
		{"appends after comments and tables", "# top\n[desktop]\nx = 1 # c\n", "# top\n[desktop]\nx = 1 # c\n\n" + block},
		{
			name: "replaces only our table and keeps the next table's comment",
			in:   "[mcp_servers.ast-context-cache]\nurl = \"old\"\n\n# next\n[other]\ny = 2\n",
			want: block + "\n# next\n[other]\ny = 2\n",
		},
		{
			name: "replaces our sub-tables too",
			in:   "a = 1\n\n# managed by ast-context-cache v3.9.0\n[mcp_servers.ast-context-cache]\nurl = \"old\"\n[mcp_servers.ast-context-cache.env]\nK = \"v\"\n[b]\nz = 1\n",
			want: "a = 1\n\n" + block + "[b]\nz = 1\n",
		},
		{
			name: "array element line is not a header",
			in:   "[mcp_servers.ast-context-cache]\nurl = \"old\"\nargs = [\n  [\"x\"]\n]\n[b]\nz = 1\n",
			want: block + "[b]\nz = 1\n",
		},
		{"CRLF", "a = 1\r\n", "a = 1\r\n\r\n" + testTOMLBlock("\r\n")},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			nl := newlineOf(tt.in)
			got := upsertTOMLTable(tt.in, tomlKey, testTOMLBlock(nl))
			assert.Equal(t, tt.want, got)
			_, err := decodeTOML(got)
			require.NoError(t, err)
		})
	}
}

func TestTOMLRemove(t *testing.T) {
	t.Parallel()
	block := testTOMLBlock("\n")
	tests := []struct {
		name, in, want string
	}{
		{"last table", "a = 1\n\n" + block, "a = 1\n"},
		{"only table", block, ""},
		{"middle table", "a = 1\n\n" + block + "\n[b]\nz = 1\n", "a = 1\n\n[b]\nz = 1\n"},
		{"first table", block + "\n# b\n[b]\nz = 1\n", "# b\n[b]\nz = 1\n"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, ok := removeTOMLTable(tt.in, tomlKey)
			assert.True(t, ok)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestTOMLMalformed(t *testing.T) {
	t.Parallel()
	_, err := decodeTOML("[mcp_servers\nurl = ")
	assert.True(t, errs.HasCode(err, errs.CodeInvalidInput))
}

func TestTOMLHeaderKey(t *testing.T) {
	t.Parallel()
	tests := []struct {
		line, key string
		ok        bool
	}{
		{"[mcp_servers.ast-context-cache]", "mcp_servers.ast-context-cache", true},
		{`[ mcp_servers . "ast-context-cache" ] # c`, "mcp_servers.ast-context-cache", true},
		{"[[hooks.Stop]]", "hooks.Stop", true},
		{`url = "x"`, "", false},
		{`  ["a"],`, "", false},
	}
	for _, tt := range tests {
		t.Run(tt.line, func(t *testing.T) {
			t.Parallel()
			key, ok := tomlHeaderKey(tt.line)
			assert.Equal(t, tt.ok, ok)
			assert.Equal(t, tt.key, key)
		})
	}
}
