package handoff

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/coma-toast/ast-context-cache/internal/contextnotes"
	"github.com/coma-toast/ast-context-cache/internal/errs"
)

func TestErrorMap(t *testing.T) {
	t.Parallel()
	for _, code := range []errs.Code{
		CodeHandoffNotFound, CodeHandoffExpired, CodeHandoffDepthExceeded, CodeHandoffChildrenExceeded,
		CodeHandoffTreeLimitExceeded, CodeClaimDeadlockRisk, CodeFeatureDisabled,
	} {
		t.Run(string(code), func(t *testing.T) {
			t.Parallel()
			m := ErrorMap(errs.NewCode(code, "it failed", "handoff", "hof_0123456789abcdef"))
			assert.Equal(t, string(code), m["error"])
			assert.Equal(t, "it failed", m["message"])
			assert.Equal(t, map[string]any{"handoff": "hof_0123456789abcdef"}, m["details"])
			assert.NotEmpty(t, m["suggestions"], "every handoff code suggests a next action")
		})
	}
}

func TestErrorMapCodeSelection(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name string
		err  error
		want string
	}{
		{"handoff code under a generic one", errs.WrapCode(errs.CodeConflict, errs.NewCode(CodeClaimDeadlockRisk, "cycle")), "claim_deadlock_risk"},
		{"generic code", errs.NewCode(errs.CodeUnsupported, "not implemented yet"), "unsupported"},
		{"no code", errors.New("disk on fire"), "internal"},
		{"wrapped plain error", errs.WrapMessage("failed to open", errors.New("eof"), "tree", "hft_x"), "internal"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			m := ErrorMap(tt.err)
			assert.Equal(t, tt.want, m["error"])
			assert.Equal(t, tt.err.Error(), m["message"])
			assert.NotNil(t, m["details"])
			assert.NotNil(t, m["suggestions"])
		})
	}
	assert.Equal(t, map[string]any{"tree": "hft_x"}, ErrorMap(tests[3].err)["details"])
	assert.Nil(t, ErrorMap(nil))
}

func TestErrorMapDelegatesContextLimitErrors(t *testing.T) {
	t.Parallel()
	err := errs.WrapMessage("failed to store result", errs.WrapCode(errs.CodeLimitExceeded,
		&contextnotes.LimitError{Limit: "session_notes", Current: 50, Max: 50, WouldAdd: 1}))
	m := ErrorMap(err)
	assert.Equal(t, contextnotes.LimitErrorMap(err), m)
	assert.Equal(t, "context_limit_exceeded", m["error"])
	assert.Equal(t, "session_notes", m["limit"])
}
