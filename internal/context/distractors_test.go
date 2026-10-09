package context

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/render"
)

func TestIsDistractorPath(t *testing.T) {
	t.Parallel()
	tests := []struct {
		rel   string
		extra []string
		want  bool
	}{
		{rel: "pkg/widgets.go"},
		{rel: "pkg/widgets_test.go", want: true},
		{rel: "tests/test_widgets.py", want: true},
		{rel: "widgets_test.py", want: true},
		{rel: "web/app.spec.ts", want: true},
		{rel: "web/app.test.ts", want: true},
		{rel: "web/app.spec.js", want: true},
		{rel: "web/app.test.js", want: true},
		{rel: "web/__mocks__/api.ts", want: true},
		{rel: "internal/mocks/store.go", want: true},
		{rel: "mocks/store.go", want: true},
		{rel: "pkg/mock_store.go", want: true},
		{rel: "vendor/x/y.go", want: true},
		{rel: "web/node_modules/lib/index.js", want: true},
		{rel: "internal/render/testdata/a.go", want: true},
		{rel: "api/v1/api.pb.go", want: true},
		{rel: "pkg/enum_gen.go", want: true},
		{rel: "pkg/mocksy/store.go"},
		{rel: "pkg/latest.go"},
		{rel: "gen/out.go", extra: []string{"gen/"}, want: true},
		{rel: "pkg/schema.sql.go", extra: []string{" *.sql.go "}, want: true},
		{rel: "pkg/x.go", extra: []string{"pkg/*.go"}, want: true},
		{rel: "pkg/sub/x.go", extra: []string{"pkg/*.go", ""}},
	}
	for _, tt := range tests {
		t.Run(tt.rel, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.want, IsDistractorPath(tt.rel, tt.extra))
		})
	}
}

func res(name, file string, extra ...string) map[string]any {
	m := map[string]any{"name": name, "kind": "function", "file": file}
	for i := 0; i+1 < len(extra); i += 2 {
		m[extra[i]] = extra[i+1]
	}
	return m
}

func resultFiles(rs []map[string]any) []string {
	var out []string
	for _, r := range rs {
		out = append(out, r["file"].(string))
	}
	return out
}

func TestCollapseDistractors(t *testing.T) {
	dbtest.Init(t)
	require.NoError(t, db.SetSetting(settingCollapseGlobs, "generated/, *.sql.go"))
	results := []map[string]any{
		res("Load", "/p/load_test.go"),
		res("Load", "/p/load.go", "skeleton", "func Load() error"),
		res("Load", "/p/copy/load.go", "skeleton", "func Load() error"),
		res("Load", "/p/mocks/load.go"),
		res("Other", "/p/other_test.go"),
		res("Query", "/p/generated/query.go"),
		res("Store", "/p/store.go", "source", "func Store() {}"),
		res("Store", "/p/store.sql.go"),
	}
	tests := []struct {
		name      string
		results   []map[string]any
		query     string
		enabled   bool
		kept      []string
		collapsed []render.Collapse
	}{
		{
			name: "groups distractors and duplicates", results: results, query: "load", enabled: true,
			kept: []string{"/p/load.go", "/p/store.go"},
			collapsed: []render.Collapse{
				{Into: "Load", Paths: []string{"/p/load_test.go", "/p/copy/load.go", "/p/mocks/load.go"}, Count: 3},
				{Into: testsGroup, Paths: []string{"/p/other_test.go", "/p/generated/query.go"}, Count: 2},
				{Into: "Store", Paths: []string{"/p/store.sql.go"}, Count: 1},
			},
		},
		{name: "disabled", results: results, query: "load", kept: resultFiles(results)},
		{name: "test query", results: results, query: "Load tests", enabled: true, kept: resultFiles(results)},
		{name: "mock query", results: results, query: "MockStore", enabled: true, kept: resultFiles(results)},
		{
			name: "only distractors kept as is", results: []map[string]any{res("A", "/p/a_test.go"), res("B", "/p/vendor/b.go")},
			query: "a", enabled: true, kept: []string{"/p/a_test.go", "/p/vendor/b.go"},
		},
		{
			name: "same name different signature kept", results: []map[string]any{res("New", "/p/a.go", "skeleton", "func New() *A"), res("New", "/p/b.go", "skeleton", "func New() *B")},
			query: "new", enabled: true, kept: []string{"/p/a.go", "/p/b.go"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			kept, collapsed := CollapseDistractors(tt.results, tt.query, tt.enabled)
			assert.Equal(t, tt.kept, resultFiles(kept))
			assert.Equal(t, tt.collapsed, collapsed)
		})
	}
}
