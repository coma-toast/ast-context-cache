package memory

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

func TestPeek(t *testing.T) {
	testMemoryDB(t)
	ref := storeEntry(t, StoreInput{Kind: KindProcedure, Scope: ScopeGlobal, Rule: "peek rule"})
	e, err := Peek(" " + ref + " ")
	require.NoError(t, err)
	assert.Equal(t, "peek rule", e.Rule)
	assert.Zero(t, e.AccessCount, "peeking records no access")
	_, err = Peek("mem_missing")
	assert.True(t, errs.HasCode(err, errs.CodeNotFound), "%v", err)
	_, err = Peek("")
	assert.True(t, errs.HasCode(err, errs.CodeInvalidInput), "%v", err)
}
