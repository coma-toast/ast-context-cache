package dashboard

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/purge"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
)

// handlePruneSpaceRoots lists indexed projects that are wtg space roots (GET, or
// POST without apply) and, on POST with apply=true, prunes the ones fully covered
// by their own checkouts. Backs `ast-mcp prune-projects --space-roots`.
func handlePruneSpaceRoots(w http.ResponseWriter, r *http.Request) {
	apply := false
	switch r.Method {
	case http.MethodGet:
	case http.MethodPost:
		apply = parseApplyFromRequest(r)
	default:
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	roots := purge.SpaceRoots(apply)
	if roots == nil {
		roots = []purge.SpaceRoot{}
	}
	if apply {
		realtime.Notify(realtime.IndexCommitted | realtime.IndexHealth)
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"apply": apply, "space_roots": roots})
}

func parseApplyFromRequest(r *http.Request) bool {
	if strings.Contains(r.Header.Get("Content-Type"), "application/json") {
		var req struct {
			Apply bool `json:"apply"`
		}
		if json.NewDecoder(r.Body).Decode(&req) != nil {
			return false
		}
		return req.Apply
	}
	r.ParseForm()
	return r.FormValue("apply") == "true" || r.FormValue("apply") == "1"
}
