package handoff

import (
	"bytes"
	"log/slog"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

func TestIDFormats(t *testing.T) {
	t.Parallel()
	seen := map[string]bool{}
	for range 100 {
		ref, err := NewHandoffRef()
		require.NoError(t, err)
		assert.Regexp(t, `^hof_[0-9a-f]{16}$`, string(ref))
		tree, err := NewTreeID()
		require.NoError(t, err)
		assert.Regexp(t, `^hft_[0-9a-f]{16}$`, string(tree))
		assert.False(t, seen[string(ref)], "unique")
		seen[string(ref)] = true
	}
	ref, _ := NewHandoffRef()
	assert.Equal(t, SessionID(string(ref)+".c3"), ChildSessionID(ref, 3))
	parsed, err := ParseHandoffRef(string(ref))
	require.NoError(t, err)
	assert.Equal(t, ref, parsed)
	for _, bad := range []string{"", "hof_123", "hft_0123456789abcdef", "hof_0123456789ABCDEF", "ctx_0123456789ab"} {
		_, err := ParseHandoffRef(bad)
		assert.True(t, errs.HasCode(err, errs.CodeInvalidInput), bad)
	}
	_, err = ParseTreeID("hft_0123456789abcdef")
	assert.NoError(t, err)
	_, err = ParseTreeID("hof_0123456789abcdef")
	assert.True(t, errs.HasCode(err, errs.CodeInvalidInput))
}

func TestIDLogValues(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name  string
		value slog.LogValuer
		attr  slog.Attr
		key   string
		want  string
	}{
		{"handoff", HandoffRef("hof_0123456789abcdef"), HandoffRef("hof_0123456789abcdef").Attr(), "handoff", "hof_0123456789abcdef"},
		{"tree", TreeID("hft_0123456789abcdef"), TreeID("hft_0123456789abcdef").Attr(), "tree", "hft_0123456789abcdef"},
		{"session", SessionID("hof_0123456789abcdef.c1"), SessionID("hof_0123456789abcdef.c1").Attr(), "session", "hof_0123456789abcdef.c1"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			v := tt.value.LogValue()
			require.Equal(t, slog.KindGroup, v.Kind())
			group := v.Group()
			require.Len(t, group, 1)
			assert.Equal(t, tt.key, group[0].Key)
			assert.Equal(t, tt.want, group[0].Value.String())
			var buf bytes.Buffer
			slog.New(slog.NewTextHandler(&buf, nil)).Info("x", tt.attr)
			assert.Contains(t, buf.String(), " "+tt.key+"="+tt.want, "Attr renders inline")
		})
	}
}

func TestEnumValidity(t *testing.T) {
	t.Parallel()
	assert.True(t, ModeFork.Valid())
	assert.False(t, Mode("clone").Valid())
	assert.True(t, StatusAbandoned.Valid())
	assert.False(t, StatusAbandoned.Completed())
	assert.True(t, StatusPartial.Completed())
	assert.True(t, SectionPointer.Valid())
	assert.False(t, Section("scratchpad").Valid())
	assert.True(t, EntryTypeDeadEnd.Valid())
	assert.False(t, EntryType("note").Valid())
}

func TestSQLTimeMatchesSQLiteFormat(t *testing.T) {
	t.Parallel()
	assert.Regexp(t, regexp.MustCompile(`^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$`), sqlTime(time.Now()))
	assert.False(t, strings.Contains(sqlTime(time.Now()), "T"))
}
