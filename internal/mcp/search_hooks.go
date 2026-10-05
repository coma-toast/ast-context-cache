package mcp

import (
	"encoding/json"

	"github.com/coma-toast/ast-context-cache/internal/flags"
	"github.com/coma-toast/ast-context-cache/internal/handoff"
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

// annotateSearch matches a recorded search against sessionID's handoff tree (OP-9, OP-10, SP-6):
// it marks results the parent already explored with "parent_explored" in place and returns the
// fields to merge under the response's "handoff" key, or nil. Call it after recordSearch and
// before the response is marshaled; annotations are per session, so they are never cached.
func annotateSearch(sessionID string, e trail.Entry, results []map[string]any) map[string]any {
	svc := treeService(sessionID, e)
	if svc == nil {
		return nil
	}
	return svc.Annotate(handoff.SessionID(sessionID), handoff.SearchEventFor(e), results)
}

// annotateFileContext annotates a get_file_context response, which lists its symbols without a
// file, so each is matched with the response's file. Outside a tree it returns data unparsed.
func annotateFileContext(sessionID string, e trail.Entry, data string) string {
	svc := treeService(sessionID, e)
	if svc == nil {
		return data
	}
	var resp map[string]any
	if json.Unmarshal([]byte(data), &resp) != nil {
		return data
	}
	file, _ := resp["file"].(string)
	symbols := resultMaps(resp["symbols"])
	keyed := make([]map[string]any, len(symbols))
	for i, s := range symbols {
		keyed[i] = map[string]any{"file": file, "name": s["name"], "start_line": s["start_line"]}
	}
	ann := svc.Annotate(handoff.SessionID(sessionID), handoff.SearchEventFor(e), keyed)
	for i, k := range keyed {
		if explored, _ := k["parent_explored"].(bool); explored {
			symbols[i]["parent_explored"] = true
		}
	}
	if ann != nil {
		resp["handoff"] = ann
	}
	out, err := json.Marshal(resp)
	if err != nil {
		return data
	}
	return string(out)
}

// treeService returns the handoff service when sessionID's search e should be annotated: a
// search ran, feature_handoff is on, and the session is in a tree. A session outside every
// tree costs one in-memory lookup (NFR-2).
func treeService(sessionID string, e trail.Entry) handoff.Service {
	if sessionID == "" || e.Tool == "" || !flags.Enabled(flags.KeyHandoff) {
		return nil
	}
	svc := handoff.Default()
	if svc == nil || !svc.IsTreeSession(handoff.SessionID(sessionID)) {
		return nil
	}
	return svc
}

// resultMaps returns a decoded results array's object elements; they share the response's maps,
// so marks set on them land in the response.
func resultMaps(v any) []map[string]any {
	switch rs := v.(type) {
	case []map[string]any:
		return rs
	case []any:
		out := make([]map[string]any, 0, len(rs))
		for _, r := range rs {
			if m, ok := r.(map[string]any); ok {
				out = append(out, m)
			}
		}
		return out
	}
	return nil
}

// sessionArg returns a tool call's session_id argument, or "".
func sessionArg(args map[string]interface{}) string {
	sid, _ := args["session_id"].(string)
	return sid
}

// semanticTrail completes the entry PackScoredResults returned for a search_semantic call.
func semanticTrail(e trail.Entry, query, docType, projectPath string, filters *search.SearchFilters) trail.Entry {
	e.Tool, e.Query, e.DocType = "search_semantic", query, docType
	e.FiltersKey = filters.NormalizedKey(projectPath)
	return e
}
