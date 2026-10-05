package mcp

import (
	"github.com/coma-toast/ast-context-cache/internal/flags"
	"github.com/coma-toast/ast-context-cache/internal/search"
	"github.com/coma-toast/ast-context-cache/internal/trail"
)

// recordSearch adds e to sessionID's search trail. Nothing is recorded without a session, while
// feature_handoff is off, or for a call that failed before it searched (e has no tool).
func recordSearch(sessionID string, e trail.Entry) {
	if sessionID == "" || e.Tool == "" || !flags.Enabled(flags.KeyHandoff) {
		return
	}
	e.SessionID = sessionID
	trail.Record(e)
}

// sessionArg returns a tool call's session_id argument, or "".
func sessionArg(args map[string]interface{}) string {
	sid, _ := args["session_id"].(string)
	return sid
}

// retrieveTrail returns the trail entry HandleRetrieve attached to its result.
func retrieveTrail(r map[string]interface{}) trail.Entry {
	e, _ := r["trail"].(trail.Entry)
	return e
}

// semanticTrail completes the entry PackScoredResults returned for a search_semantic call.
func semanticTrail(e trail.Entry, query, docType, projectPath string, filters *search.SearchFilters) trail.Entry {
	e.Tool, e.Query, e.DocType = "search_semantic", query, docType
	e.FiltersKey = filters.NormalizedKey(projectPath)
	return e
}
