package context

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/db/dbtest"
	"github.com/coma-toast/ast-context-cache/internal/errs"
)

const selectSymbolSpanForTestQuery = "SELECT file, kind, start_line, end_line FROM symbols WHERE project_path = ? AND name = ? LIMIT 1"

const editWidgetsGo = `package widgets

type Widget struct{}

func helper(n int) int { return n + 1 }

func (w *Widget) Beta() int { return 2 }

func Target(w *Widget) int {
	x := helper(1)
	if len("a") > 0 {
		x += w.Beta()
	}
	return x + Target2() + Target(nil)
}
`

const editOtherGo = `package widgets

func helper(n int) int { return n }

func Target2() int { return 3 }

func unused() {}
`

// manyCallsGo has a target calling twelve project functions, to exercise the callee cap.
func manyCallsGo() string {
	var b strings.Builder
	b.WriteString("package many\n\n")
	for i := 0; i < 12; i++ {
		b.WriteString("func f" + string(rune('a'+i)) + "() {}\n\n")
	}
	b.WriteString("func Many() {\n")
	for i := 0; i < 12; i++ {
		b.WriteString("\tf" + string(rune('a'+i)) + "()\n")
	}
	b.WriteString("}\n")
	return b.String()
}

func editTarget(t *testing.T, project, name string) map[string]any {
	t.Helper()
	conn, err := db.IndexReader()
	require.NoError(t, err)
	var file, kind string
	var start, end int
	require.NoError(t, conn.QueryRow(selectSymbolSpanForTestQuery, project, name).Scan(&file, &kind, &start, &end))
	return map[string]any{"file": file, "name": name, "kind": kind, "start_line": start, "end_line": end, "skeleton": "stale"}
}

func calleeKeys(cs []map[string]any) []string {
	var out []string
	for _, c := range cs {
		out = append(out, filepath.Base(c["file"].(string))+":"+c["name"].(string))
	}
	return out
}

func TestEditView(t *testing.T) {
	dbtest.Init(t)
	project := t.TempDir()
	writeAndIndex(t, filepath.Join(project, "widgets.go"), project, editWidgetsGo)
	writeAndIndex(t, filepath.Join(project, "other.go"), project, editOtherGo)
	writeAndIndex(t, filepath.Join(project, "many.go"), project, manyCallsGo())
	tests := []struct {
		name, target string
		wantCallees  []string
	}{
		{name: "same file first, own name and keywords dropped", target: "Target", wantCallees: []string{"widgets.go:helper", "widgets.go:Beta", "other.go:Target2"}},
		{name: "no calls", target: "Target2"},
		{name: "capped at ten", target: "Many", wantCallees: []string{
			"many.go:fa", "many.go:fb", "many.go:fc", "many.go:fd", "many.go:fe",
			"many.go:ff", "many.go:fg", "many.go:fh", "many.go:fi", "many.go:fj",
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			target := editTarget(t, project, tt.target)
			out, callees, err := EditView(project, target, map[string][]string{})
			require.NoError(t, err)
			assert.Equal(t, "edit", out["mode"])
			assert.Contains(t, out["source"], "func "+tt.target)
			assert.NotContains(t, out, "skeleton")
			assert.Equal(t, "stale", target["skeleton"], "the caller's map is not modified")
			assert.Equal(t, tt.wantCallees, calleeKeys(callees))
			for _, c := range callees {
				assert.Equal(t, "skeleton", c["mode"])
				assert.NotEmpty(t, c["skeleton"], c["name"])
				assert.NotContains(t, c, "source")
			}
		})
	}
}

func TestEditViewErrors(t *testing.T) {
	dbtest.Init(t)
	missing := filepath.Join(t.TempDir(), "gone.go")
	require.NoFileExists(t, missing)
	tests := []struct {
		name   string
		target map[string]any
		code   errs.Code
	}{
		{name: "no file", target: map[string]any{"name": "x", "start_line": 1, "end_line": 2}, code: errs.CodeInvalidInput},
		{name: "no span", target: map[string]any{"name": "x", "file": missing}, code: errs.CodeInvalidInput},
		{name: "unreadable", target: map[string]any{"name": "x", "file": missing, "start_line": 1, "end_line": 2}, code: errs.CodeNotFound},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, err := EditView(t.TempDir(), tt.target, map[string][]string{})
			require.Error(t, err)
			assert.True(t, errs.HasCode(err, tt.code), err.Error())
		})
	}
}
