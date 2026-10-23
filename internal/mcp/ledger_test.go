package mcp

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestToolLedger(t *testing.T) {
	for tool, want := range map[string]string{
		"get_context_capsule": ledgerCompression, "execute_code": ledgerCompression,
		"store_context": ledgerVirtual, "recall_memory": ledgerVirtual, "handoff": ledgerVirtual,
		"list_context": ledgerNone, "index_files": ledgerNone,
	} {
		assert.Equal(t, want, toolLedger(tool), tool)
	}
}
