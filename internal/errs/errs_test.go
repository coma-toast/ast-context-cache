package errs

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestErrorString(t *testing.T) {
	t.Parallel()
	cause := errors.New("disk full")
	tests := []struct {
		name string
		err  error
		want string
	}{
		{name: "new", err: New("unable to store note"), want: "unable to store note"},
		{name: "wrap message", err: WrapMessage("unable to store note", cause), want: "unable to store note: disk full"},
		{name: "wrap keeps cause text", err: Wrap(cause, "ref", "ctx_1"), want: "disk full"},
		{name: "nested", err: WrapMessage("outer", WrapMessage("inner", cause)), want: "outer: inner: disk full"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.want, tt.err.Error())
		})
	}
}

func TestWrapNilReturnsNil(t *testing.T) {
	t.Parallel()
	assert.NoError(t, Wrap(nil))
	assert.NoError(t, WrapCode(CodeInternal, nil))
	assert.NoError(t, WrapMessage("x", nil))
	assert.NoError(t, WrapCodeMessage(CodeInternal, "x", nil))
}

func TestStdlibCompatibility(t *testing.T) {
	t.Parallel()
	err := WrapMessage("open index", fmt.Errorf("stat: %w", fs.ErrNotExist), "path", "/tmp/x")
	assert.ErrorIs(t, err, fs.ErrNotExist)
	var e *Error
	require.ErrorAs(t, err, &e)
	assert.Equal(t, "open index", e.Message())
}

func TestCodesAccumulateOutermostFirst(t *testing.T) {
	t.Parallel()
	inner := NewCode(CodeLimitExceeded, "too many notes")
	err := WrapCodeMessage(CodeConflict, "store failed", fmt.Errorf("batch: %w", inner))
	assert.Equal(t, Codes{CodeConflict, CodeLimitExceeded}, CodesOf(err))
	assert.Equal(t, CodeConflict, CodeOf(err))
	assert.True(t, HasCode(err, CodeLimitExceeded))
	assert.False(t, HasCode(err, CodeNotFound))
	assert.Equal(t, Code(""), CodeOf(errors.New("plain")))
}

func TestFieldsMergeOuterWins(t *testing.T) {
	t.Parallel()
	inner := New("inner", "ref", "ctx_inner", "session", "s1")
	err := WrapMessage("outer", inner, "ref", "ctx_outer", 42, "answer", "dangling")
	assert.Equal(t, map[string]any{"ref": "ctx_outer", "session": "s1", "42": "answer", badKey: "dangling"}, FieldsOf(err))
	assert.Nil(t, FieldsOf(errors.New("plain")))
}

func TestMarshalJSON(t *testing.T) {
	t.Parallel()
	b, err := json.Marshal(NewCode(CodeNotFound, "handoff not found", "handoff", "hof_1"))
	require.NoError(t, err)
	assert.JSONEq(t, `{"message":"handoff not found","codes":["not_found"],"fields":{"handoff":"hof_1"}}`, string(b))
}
