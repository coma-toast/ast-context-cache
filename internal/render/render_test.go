package render

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// update rewrites the testdata golden files: AST_RENDER_UPDATE=1 make test TEST_PKGS=./internal/render/...
var update = os.Getenv("AST_RENDER_UPDATE") == "1"

func goldenResponses() map[string]Response {
	alpha := map[string]any{
		"file": "/proj/pkg/widgets.go", "start_line": 3, "end_line": 5, "kind": "function", "name": "WidgetAlpha",
		"source": "func WidgetAlpha() int {\n\treturn 1\n}\n", "score": 0.032786,
	}
	beta := map[string]any{
		"file": "/proj/pkg/widgets.go", "start_line": float64(7), "end_line": float64(7), "kind": "method", "name": "Beta",
		"qualified_name": "Widget.Beta", "skeleton": "func (w *Widget) Beta() int", "similarity": 0.61,
	}
	summary := map[string]any{
		"file": "/proj/web/app.ts", "start_line": 10, "end_line": 40, "kind": "class", "name": "App",
		"summary": "Root application component.", "score": 0.5,
	}
	explicit := map[string]any{
		"file": "/proj/main.py", "start_line": 1, "end_line": 2, "kind": "function", "name": "main",
		"source": "def main():\n    pass", "mode": "edit",
	}
	fenced := map[string]any{"file": "/proj/README.md", "start_line": 1, "end_line": 3, "kind": "section", "name": "Usage", "source": "```sh\nmake\n```"}
	bare := map[string]any{"file": "/elsewhere/lib.rs", "kind": "function", "name": "outside"}
	return map[string]Response{
		"basic": {Tool: "get_context_capsule", Query: "widget", Root: "/proj", Results: []map[string]any{alpha, beta, summary}},
		"withheld_collapsed": {
			Tool: "search_semantic", Query: "widget", Root: "/proj", Results: []map[string]any{alpha},
			Withheld: 4, WithheldTokens: 120,
			Collapsed: []Collapse{{Into: "WidgetAlpha", Paths: []string{"pkg/widgets_test.go", "mocks/widgets.go"}, Count: 2}},
			Notes:     []string{"note: deduped 1 symbol already returned this session"},
		},
		"no_match": {
			Tool: "get_context_capsule", Query: "quantum flux", Root: "/proj", Results: []map[string]any{beta},
			NoMatch: &NoMatch{BestScore: 0.1234, Hint: "hint: try search_semantic or a symbol name"},
		},
		"modes_and_fences": {Tool: "retrieve", Query: "main", Root: "/proj", Results: []map[string]any{explicit, fenced, bare}},
		"empty":            {Tool: "get_context_capsule", Query: "nothing"},
	}
}

func TestGolden(t *testing.T) {
	t.Parallel()
	renderers := map[string]func(Response) string{"text": Text, "locations": Locations}
	for name, resp := range goldenResponses() {
		for kind, fn := range renderers {
			t.Run(name+"_"+kind, func(t *testing.T) {
				t.Parallel()
				path := filepath.Join("testdata", name+"."+kind+".golden")
				got := fn(resp)
				if update {
					require.NoError(t, os.WriteFile(path, []byte(got), 0o644))
				}
				want, err := os.ReadFile(path)
				require.NoError(t, err)
				assert.Equal(t, string(want), got)
			})
		}
	}
}

func TestLangFromExt(t *testing.T) {
	t.Parallel()
	tests := []struct{ file, want string }{
		{"a.go", "go"},
		{"a.py", "python"},
		{"a.TS", "typescript"},
		{"a.tsx", "typescript"},
		{"a.js", "javascript"},
		{"a.jsx", "javascript"},
		{"a.rs", "rust"},
		{"a.rb", "ruby"},
		{"a.java", "java"},
		{"a.sh", "bash"},
		{"a.fish", "fish"},
		{"a.yml", "yaml"},
		{"a.yaml", "yaml"},
		{"a.md", ""},
		{"Makefile", ""},
		{"dir.go/file", ""},
	}
	for _, tt := range tests {
		t.Run(tt.file, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.want, LangFromExt(tt.file))
		})
	}
}
