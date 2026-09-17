package mcp

import "testing"

// A misconfigured/misspelled AST_MCP_TIER (or per-tool tier override) must fail
// closed to the least-privileged tier, not silently grant more access than
// whoever configured it intended.
func TestParseTierFailsClosedOnUnknownValue(t *testing.T) {
	for _, s := range []string{"", "cor", "Complete-ish", "unknown", "  "} {
		if got := ParseTier(s); got != TierCore {
			t.Fatalf("ParseTier(%q)=%q want %q (fail closed)", s, got, TierCore)
		}
	}
}

func TestParseTierRecognizesValidValues(t *testing.T) {
	cases := map[string]Tier{
		"core":     TierCore,
		"CORE":     TierCore,
		"extended": TierExtended,
		"complete": TierComplete,
		"Complete": TierComplete,
	}
	for s, want := range cases {
		if got := ParseTier(s); got != want {
			t.Fatalf("ParseTier(%q)=%q want %q", s, got, want)
		}
	}
}
