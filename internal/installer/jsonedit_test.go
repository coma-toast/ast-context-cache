package installer

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

var testEntry = jsonObj{{"type", "http"}, {"url", "http://127.0.0.1:7821/mcp"}}

func TestJSONSetEntry(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name, in, want string
	}{
		{
			name: "empty file",
			in:   "",
			want: "{\n  \"mcpServers\": {\n    \"ast-context-cache\": {\n      \"type\": \"http\",\n      \"url\": \"http://127.0.0.1:7821/mcp\"\n    }\n  }\n}\n",
		},
		{
			name: "appends after foreign servers keeping comments and trailing commas",
			in:   "{\n  // c\n  \"mcpServers\": {\n    \"a\": {\"url\": \"x\"}, // a\n  },\n}\n",
			want: "{\n  // c\n  \"mcpServers\": {\n    \"a\": {\"url\": \"x\"}, // a\n    \"ast-context-cache\": {\n      \"type\": \"http\",\n      \"url\": \"http://127.0.0.1:7821/mcp\"\n    },\n  },\n}\n",
		},
		{
			name: "comment after last member stays on its line",
			in:   "{\n\t\"mcpServers\": {\n\t\t\"a\": 1 // last\n\t}\n}",
			want: "{\n\t\"mcpServers\": {\n\t\t\"a\": 1, // last\n\t\t\"ast-context-cache\": {\n\t\t\t\"type\": \"http\",\n\t\t\t\"url\": \"http://127.0.0.1:7821/mcp\"\n\t\t}\n\t}\n}",
		},
		{
			name: "compact file stays compact",
			in:   `{"mcpServers":{"a":1}}`,
			want: `{"mcpServers":{"a":1,"ast-context-cache":{"type":"http","url":"http://127.0.0.1:7821/mcp"}}}`,
		},
		{
			name: "replaces our entry in place",
			in:   "{\n  \"mcpServers\": {\n    \"ast-context-cache\": {\"url\": \"old\"}, // ours\n    \"b\": 2\n  }\n}\n",
			want: "{\n  \"mcpServers\": {\n    \"ast-context-cache\": {\n      \"type\": \"http\",\n      \"url\": \"http://127.0.0.1:7821/mcp\"\n    }, // ours\n    \"b\": 2\n  }\n}\n",
		},
		{
			name: "CRLF and BOM preserved",
			in:   "\xEF\xBB\xBF{\r\n  \"mcpServers\": {}\r\n}\r\n",
			want: "\xEF\xBB\xBF{\r\n  \"mcpServers\": {\r\n    \"ast-context-cache\": {\r\n      \"type\": \"http\",\r\n      \"url\": \"http://127.0.0.1:7821/mcp\"\r\n    }\r\n  }\r\n}\r\n",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			d, err := parseJSONDoc([]byte(tt.in))
			require.NoError(t, err)
			_, err = d.set([]string{"mcpServers", serverName}, testEntry)
			require.NoError(t, err)
			assert.Equal(t, tt.want, string(d.bytes()))
		})
	}
}

// TestJSONRoundTrip checks that removing the entry we added restores the original bytes.
func TestJSONRoundTrip(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name, in string
	}{
		{"trailing commas and comments", "{\n  // c\n  \"mcpServers\": {\n    \"a\": {\"url\": \"x\"}, // a\n    \"b\": 1,\n  },\n}\n"},
		{"comment before closing brace", "{\n\t\"mcpServers\": {\n\t\t\"a\": 1 // last\n\t}\n}"},
		{"block comment before closing brace", "{\n  \"mcpServers\": {\n    \"a\": 1\n    /* end */\n  }\n}\n"},
		{"compact", `{"mcpServers":{"a":1}}`},
		{"empty block", "{\n  \"mcpServers\": {}\n}\n"},
		{"CRLF", "{\r\n  \"mcpServers\": {\r\n    \"a\": 1\r\n  }\r\n}\r\n"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			d, err := parseJSONDoc([]byte(tt.in))
			require.NoError(t, err)
			_, err = d.set([]string{"mcpServers", serverName}, testEntry)
			require.NoError(t, err)
			assertValidJSONC(t, d.bytes())
			d2, err := parseJSONDoc(d.bytes())
			require.NoError(t, err)
			removed, err := d2.remove("mcpServers", serverName)
			require.NoError(t, err)
			assert.True(t, removed)
			assert.Equal(t, tt.in, string(d2.bytes()))
		})
	}
}

func TestJSONParseErrors(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name, in string
	}{
		{"stray double comma", "{\n  \"servers\": {\n    \"a\": 1,,\n  }\n}\n"},
		{"truncated", `{"mcpServers": {`},
		{"top-level array", `[]`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			_, err := parseJSONDoc([]byte(tt.in))
			require.Error(t, err)
			assert.True(t, errs.HasCode(err, errs.CodeInvalidInput))
		})
	}
}

func TestJSONDuplicateKeyRejected(t *testing.T) {
	t.Parallel()
	d, err := parseJSONDoc([]byte(`{"mcpServers": {"ast-context-cache": 1, "ast-context-cache": 2}}`))
	require.NoError(t, err)
	_, err = d.set([]string{"mcpServers", serverName}, testEntry)
	assert.True(t, errs.HasCode(err, errs.CodeConflict))
}

func TestJSONArrayAppendAndRemove(t *testing.T) {
	t.Parallel()
	in := "{\n  \"hooks\": {\n    \"Stop\": [\n      {\"command\": \"mine\"}\n    ]\n  }\n}\n"
	d, err := parseJSONDoc([]byte(in))
	require.NoError(t, err)
	n, ok := d.find("hooks", "Stop")
	require.True(t, ok)
	require.NoError(t, d.appendElem(n, jsonObj{{"command", "ours"}}))
	want := "{\n  \"hooks\": {\n    \"Stop\": [\n      {\"command\": \"mine\"},\n      {\n        \"command\": \"ours\"\n      }\n    ]\n  }\n}\n"
	assert.Equal(t, want, string(d.bytes()))
}

func assertValidJSONC(t *testing.T, b []byte) {
	t.Helper()
	d, err := parseJSONDoc(b)
	require.NoError(t, err)
	std, err := standardJSON(d.root)
	require.NoError(t, err)
	assert.True(t, json.Valid(std))
}
