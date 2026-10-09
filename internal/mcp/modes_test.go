package mcp

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/context"
)

const editTargetGo = `package widgets

func helper(n int) int { return n + 1 }

func Target() int {
	return helper(1)
}

func Unrelated() {}
`

type fileContextReply struct {
	Mode     string                   `json:"mode"`
	Symbols  []map[string]interface{} `json:"symbols"`
	Withheld *context.Withheld        `json:"withheld"`
	Error    string                   `json:"error"`
}

func fileContext(t *testing.T, file, project, mode, symbol string, budget int) fileContextReply {
	t.Helper()
	fc := handleFileContextWithMeta(file, project, mode, symbol, "", budget)
	var out fileContextReply
	require.NoError(t, json.Unmarshal([]byte(fc.JSON), &out), fc.JSON)
	return out
}

func symbolNames(syms []map[string]interface{}) []string {
	var out []string
	for _, s := range syms {
		out = append(out, s["name"].(string))
	}
	return out
}

// AC8: get_file_context mode=edit returns the named symbol's source and its callees' skeletons.
func TestFileContextEditMode(t *testing.T) {
	project, file := indexedPython(t, "widgets.go", editTargetGo)
	got := fileContext(t, file, project, "edit", "Target", 0)
	require.Empty(t, got.Error)
	assert.Equal(t, []string{"Target", "helper"}, symbolNames(got.Symbols))
	assert.Equal(t, "edit", got.Symbols[0]["mode"])
	assert.Contains(t, got.Symbols[0]["source"], "return helper(1)")
	assert.Equal(t, "skeleton", got.Symbols[1]["mode"])
	assert.NotContains(t, got.Symbols[1], "source")

	assert.Contains(t, fileContext(t, file, project, "edit", "", 0).Error, "symbol is required")
	assert.Contains(t, fileContext(t, file, project, "edit", "Missing", 0).Error, "not found")
}

// MO-1: get_file_context auto is skeleton under mode v2.
func TestFileContextAutoIsSkeleton(t *testing.T) {
	project, file := indexedPython(t, "widgets.go", editTargetGo)
	got := fileContext(t, file, project, "auto", "", 0)
	assert.Equal(t, "skeleton", got.Mode)
	require.Len(t, got.Symbols, 3)
	for _, s := range got.Symbols {
		assert.NotContains(t, s, "source", s["name"])
	}
}

// PR-6: get_file_context skips a symbol over the budget and keeps packing smaller ones.
func TestFileContextBudgetSkipsAndWithholds(t *testing.T) {
	src := "package sizes\n\nfunc Big() string {\n\treturn \"" + strings.Repeat("x", 4000) + "\"\n}\n\nfunc Small() int { return 1 }\n"
	project, file := indexedPython(t, "sizes.go", src)
	got := fileContext(t, file, project, "full", "", 200)
	assert.Equal(t, []string{"Small"}, symbolNames(got.Symbols))
	require.NotNil(t, got.Withheld)
	assert.Equal(t, 1, got.Withheld.Count)
	assert.Positive(t, got.Withheld.Tokens)
}

// MO-2: retrieve does not support mode=edit.
func TestRetrieveRejectsEditMode(t *testing.T) {
	out := HandleRetrieve(map[string]interface{}{"query": "Target", "mode": "edit"}, t.TempDir())
	assert.Contains(t, out["error"], "mode=edit")
}
