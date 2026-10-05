package logging

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

type testRef string

func (r testRef) LogValue() slog.Value {
	return slog.GroupValue(slog.String("handoff", string(r)))
}

func TestKeyedErrorExpandsFields(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	logger := slog.New(NewHandler(&buf, FormatText, slog.LevelInfo))
	logger.Warn("Failed to flush", "error", errs.NewCode(errs.CodeConflict, "busy", "tree", "hft_1"))
	out := buf.String()
	assert.Contains(t, out, "error=busy")
	assert.Contains(t, out, "error_codes=[conflict]")
	assert.Contains(t, out, "tree=hft_1")
}

// Bare arguments fail go vet's slog check at call sites, so the repo convention is keyed
// ("error", err); the handler still rewrites bare values defensively. Spreading a slice keeps
// vet from rejecting this test.
func TestBareErrorRendersAsErrorAttr(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	logger := slog.New(NewHandler(&buf, FormatText, slog.LevelInfo))
	args := []any{errs.NewCode(errs.CodeConflict, "busy", "tree", "hft_1")}
	logger.Warn("Failed to flush", args...)
	out := buf.String()
	assert.Contains(t, out, `msg="Failed to flush"`)
	assert.Contains(t, out, "error=busy")
	assert.Contains(t, out, "error_codes=[conflict]")
	assert.Contains(t, out, "tree=hft_1")
	assert.NotContains(t, out, badKey)
}

func TestBareLogValuerGroupIsInlined(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	logger := slog.New(NewHandler(&buf, FormatText, slog.LevelInfo))
	args := []any{testRef("hof_abc"), "tokens", 12}
	logger.Info("Created handoff", args...)
	assert.Contains(t, buf.String(), "handoff=hof_abc tokens=12")
}

func TestJSONFormatAndLevel(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	logger := slog.New(NewHandler(&buf, "JSON", ParseLevel("warn")))
	logger.Info("Dropped")
	logger.Error("Kept", "n", 1)
	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	require.Len(t, lines, 1)
	var rec map[string]any
	require.NoError(t, json.Unmarshal([]byte(lines[0]), &rec))
	assert.Equal(t, "Kept", rec["msg"])
}

func TestParseLevel(t *testing.T) {
	t.Parallel()
	tests := map[string]slog.Level{"": slog.LevelInfo, "debug": slog.LevelDebug, "WARNING": slog.LevelWarn, "error": slog.LevelError, "bogus": slog.LevelInfo}
	for in, want := range tests {
		assert.Equal(t, want, ParseLevel(in), in)
	}
}

// Not parallel: swaps the process-wide default logger.
func TestTaggedFollowsCurrentDefault(t *testing.T) {
	prev := slog.Default()
	t.Cleanup(func() { slog.SetDefault(prev) })
	logger := Tagged("db").With("pool", "usage")
	var buf bytes.Buffer
	slog.SetDefault(slog.New(NewHandler(&buf, FormatText, slog.LevelInfo)))
	logger.Info("Opened pool", "conns", 4)
	assert.Contains(t, buf.String(), `msg="Opened pool" tag=db pool=usage conns=4`)
}
