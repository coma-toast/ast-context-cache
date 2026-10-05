package flags

// Flag keys double as their settings-table keys.
const (
	KeyHandoff            = "feature_handoff"
	KeyHandoffScratchpad  = "feature_handoff_scratchpad"
	KeyHandoffClaims      = "feature_handoff_claims"
	KeyHandoffLiveTrail   = "feature_handoff_live_trail"
	KeyHandoffHooks       = "feature_handoff_hooks"
	KeySharedQueryCache   = "feature_shared_query_cache"
	handoffChildKeyPrefix = KeyHandoff + "_"
)

// registry is the FF-6 flag table. Order is the order State and All report.
var registry = []Flag{
	{
		Key:         KeyHandoff,
		Env:         "AST_FEATURE_HANDOFF",
		Description: "Master switch for subagent handoff: the handoff, open_handoff, and scratchpad tools.",
		Default:     true,
		Tools:       []string{"handoff", "open_handoff", "scratchpad"},
	},
	{
		Key:         KeyHandoffScratchpad,
		Env:         "AST_FEATURE_HANDOFF_SCRATCHPAD",
		Description: "Shared scratchpad tool and its digest sections for handoff trees.",
		Default:     true,
		Tools:       []string{"scratchpad"},
	},
	{
		Key:         KeyHandoffClaims,
		Env:         "AST_FEATURE_HANDOFF_CLAIMS",
		Description: "Scratchpad claim and release actions for coordinating work across agents.",
		Default:     true,
		Actions:     map[string][]string{"scratchpad": {"claim", "release"}},
	},
	{
		Key:         KeyHandoffLiveTrail,
		Env:         "AST_FEATURE_HANDOFF_LIVE_TRAIL",
		Description: "Automatic sharing of each agent's search trail into the handoff tree.",
		Default:     true,
	},
	{
		Key:         KeyHandoffHooks,
		Env:         "AST_FEATURE_HANDOFF_HOOKS",
		Description: "Installer offers Claude Code hooks that create and open handoffs.",
		Default:     false,
	},
	{
		Key:         KeySharedQueryCache,
		Env:         "AST_FEATURE_SHARED_QUERY_CACHE",
		Description: "Cross-session cache of search results shared between agents.",
		Default:     true,
	},
}
