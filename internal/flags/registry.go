package flags

// Flag keys double as their settings-table keys.
const (
	KeyHandoff            = "feature_handoff"
	KeyHandoffScratchpad  = "feature_handoff_scratchpad"
	KeyHandoffClaims      = "feature_handoff_claims"
	KeyHandoffLiveTrail   = "feature_handoff_live_trail"
	KeyHandoffHooks       = "feature_handoff_hooks"
	KeyContextEdit        = "feature_context_edit"
	KeyContextFn          = "feature_context_fn"
	KeySharedQueryCache   = "feature_shared_query_cache"
	KeyStableResponses    = "feature_stable_responses"
	KeyTextFormat         = "feature_text_format"
	KeyRelevanceFloor     = "feature_relevance_floor"
	KeyModeV2             = "feature_mode_v2"
	KeyResultOffload      = "feature_result_offload"
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
	{
		Key:         KeyContextEdit,
		Env:         "AST_FEATURE_CONTEXT_EDIT",
		Description: "In-place editing of stored virtual context: the edit_context tool, so an agent can rewrite a ctx_* note without losing its ref.",
		Default:     true,
		Tools:       []string{"edit_context"},
	},
	{
		Key:         KeyContextFn,
		Env:         "AST_FEATURE_CONTEXT_FN",
		Description: "Model-defined reusable context functions: define_context_fn, apply_context_fn, list_context_fns. Off by default because an agent-authored context transform is a prompt-injection surface (arXiv 2609.37725 Discussion).",
		Default:     false,
		Tools:       []string{"define_context_fn", "apply_context_fn", "list_context_fns"},
	},
	{
		Key:         KeyStableResponses,
		Env:         "AST_FEATURE_STABLE_RESPONSES",
		Description: "Byte-stable tool responses: timings and cache stats move to _meta and stable results come before variable stats, so identical calls return identical text.",
		Default:     true,
	},
	{
		Key:         KeyTextFormat,
		Env:         "AST_FEATURE_TEXT_FORMAT",
		Description: "Code tools default to compact text (one header line per symbol plus a fenced source block); output=json keeps the JSON structure.",
		Default:     true,
	},
	{
		Key:         KeyRelevanceFloor,
		Env:         "AST_FEATURE_RELEVANCE_FLOOR",
		Description: "Code search drops hits far below the top score, returns an explicit no-match for weak queries, and collapses test, mock, vendored and duplicate hits.",
		Default:     true,
	},
	{
		Key:         KeyModeV2,
		Env:         "AST_FEATURE_MODE_V2",
		Description: "Revised modes: auto returns full source for the top 3 hits and skeletons for the rest, and mode=edit returns a symbol plus its callee signatures.",
		Default:     true,
	},
	{
		Key:         KeyResultOffload,
		Env:         "AST_FEATURE_RESULT_OFFLOAD",
		Description: "Tool results over the offload threshold are stored as a ctx_* note and returned as a head plus the ref.",
		Default:     true,
	},
}
