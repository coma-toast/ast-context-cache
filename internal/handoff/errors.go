package handoff

import (
	"errors"

	"github.com/coma-toast/ast-context-cache/internal/contextnotes"
	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// Stable error codes for handoff, open_handoff, and scratchpad responses (TS-5).
const (
	CodeHandoffNotFound          errs.Code = "handoff_not_found"
	CodeHandoffExpired           errs.Code = "handoff_expired"
	CodeHandoffDepthExceeded     errs.Code = "handoff_depth_exceeded"
	CodeHandoffChildrenExceeded  errs.Code = "handoff_children_exceeded"
	CodeHandoffTreeLimitExceeded errs.Code = "handoff_tree_limit_exceeded"
	CodeClaimDeadlockRisk        errs.Code = "claim_deadlock_risk"
	CodeFeatureDisabled          errs.Code = "feature_disabled"
)

// codeInternal is reported for errors that carry no code.
const codeInternal = "internal"

var errNoContextDB = errs.NewCode(errs.CodeInternal, "context database unavailable")

// suggestions are the next actions offered with each code.
var suggestions = map[errs.Code][]string{
	CodeHandoffNotFound: {
		"check the hof_ ref, or call handoff(action=list, session_id=<parent>) to find it",
	},
	CodeHandoffExpired: {
		"ask the parent to create a new handoff; trees expire after handoff_ttl_days without access",
	},
	CodeHandoffDepthExceeded: {
		"do the work in this session instead of nesting another handoff",
		"raise handoff_max_depth in dashboard settings",
	},
	CodeHandoffChildrenExceeded: {
		"resume an existing child with open_handoff(action=resume, session_id=<child>)",
		"create a new handoff for more children, or raise handoff_max_children",
	},
	CodeHandoffTreeLimitExceeded: {
		"prune the snapshot: exclude_trail, exclude_trail_query, include_manifest=false, or fewer ctx_refs",
		"flush finished trees with handoff(action=flush), or raise handoff_tree_max_tokens / handoff_tree_max_entries",
	},
	CodeClaimDeadlockRisk: {
		"release one of your claims before claiming this key, or work on another key",
	},
	CodeFeatureDisabled: {
		"enable the feature flag in dashboard settings, or unset its AST_ environment override",
	},
	errs.CodeInvalidInput: {
		"check the required arguments in the tool's input schema",
	},
	errs.CodeNotFound: {
		"check the ref or session_id",
	},
	errs.CodeUnsupported: {
		"this action is not available in this build",
	},
}

// ErrorMap renders err as the structured tool error {"error", "message", "details",
// "suggestions"} (TS-5). "error" is the most specific handoff code err carries, else its
// outermost code, else "internal". A contextnotes limit error keeps contextnotes' own shape.
func ErrorMap(err error) map[string]any {
	if err == nil {
		return nil
	}
	var le *contextnotes.LimitError
	if errors.As(err, &le) {
		return contextnotes.LimitErrorMap(err)
	}
	code := errorCode(err)
	details := errs.FieldsOf(err)
	if details == nil {
		details = map[string]any{}
	}
	sugg := suggestions[errs.Code(code)]
	if sugg == nil {
		sugg = []string{}
	}
	return map[string]any{
		"error":       code,
		"message":     err.Error(),
		"details":     details,
		"suggestions": sugg,
	}
}

func errorCode(err error) string {
	codes := errs.CodesOf(err)
	for _, c := range codes {
		if isHandoffCode(c) {
			return string(c)
		}
	}
	if len(codes) > 0 {
		return string(codes[0])
	}
	return codeInternal
}

func isHandoffCode(c errs.Code) bool {
	switch c {
	case CodeHandoffNotFound, CodeHandoffExpired, CodeHandoffDepthExceeded, CodeHandoffChildrenExceeded,
		CodeHandoffTreeLimitExceeded, CodeClaimDeadlockRisk, CodeFeatureDisabled:
		return true
	}
	return false
}
