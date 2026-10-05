package dashboard

import (
	"encoding/json"
	"net/http"
	"strconv"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/handoff"
)

// handoffTreesResponse is GET /api/dashboard/handoff-trees: the newest trees, the caps they're
// measured against, and the aggregate 24h repeat-search ratio (OB-1, OB-4).
type handoffTreesResponse struct {
	Trees             []handoff.TreeView `json:"trees"`
	Limits            handoff.Limits     `json:"limits"`
	RepeatSearchRatio float64            `json:"repeat_search_ratio_24h"`
}

type flushHandoffTreeRequest struct {
	TreeID string `json:"tree_id"`
}

func registerHandoffAPI(mux *http.ServeMux) {
	mux.HandleFunc("/api/dashboard/handoff-trees", handleHandoffTreesJSON)
	mux.HandleFunc("/api/dashboard/handoff-trees/flush", handleFlushHandoffTree)
}

// handleHandoffTreesJSON lists the newest handoff trees, ?limit= of them (default 20, max 100).
func handleHandoffTreesJSON(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method != http.MethodGet {
		writeFlagsError(w, http.StatusMethodNotAllowed, "GET required")
		return
	}
	limit := handoff.DefaultTreeViewLimit
	if v := r.URL.Query().Get("limit"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 1 {
			writeFlagsError(w, http.StatusBadRequest, "limit must be a positive integer")
			return
		}
		limit = n
	}
	res := handoffTreesResponse{Trees: []handoff.TreeView{}, Limits: handoff.LoadLimits()}
	if db.ContextDB == nil {
		json.NewEncoder(w).Encode(res)
		return
	}
	trees, err := handoff.TreeViews(limit)
	if err != nil {
		logger.Warn("Failed to list handoff trees", "error", err)
		writeFlagsError(w, http.StatusInternalServerError, "failed to list handoff trees")
		return
	}
	res.Trees, res.RepeatSearchRatio = trees, handoff.RepeatSearchRatio()
	json.NewEncoder(w).Encode(res)
}

// handleFlushHandoffTree deletes one tree and everything it owns (POST {"tree_id"}), as the
// handoff tool's flush action does (RQ-3).
func handleFlushHandoffTree(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method != http.MethodPost {
		writeFlagsError(w, http.StatusMethodNotAllowed, "POST required")
		return
	}
	var req flushHandoffTreeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeFlagsError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	tree, err := handoff.ParseTreeID(req.TreeID)
	if err != nil {
		writeFlagsError(w, http.StatusBadRequest, "tree_id must be an hft_ tree id")
		return
	}
	svc := handoff.Default()
	if svc == nil {
		writeFlagsError(w, http.StatusServiceUnavailable, "handoff service not running")
		return
	}
	res, err := svc.Flush(r.Context(), handoff.FlushRequest{TreeID: tree})
	if err != nil {
		writeFlagsError(w, handoffErrorStatus(err), err.Error())
		return
	}
	logger.Info("Flushed handoff tree from dashboard", tree.Attr(), "handoffs", res.Handoffs, "children", res.Children)
	json.NewEncoder(w).Encode(map[string]any{"status": "ok", "flushed": res})
}

func handoffErrorStatus(err error) int {
	switch {
	case errs.HasCode(err, errs.CodeInvalidInput):
		return http.StatusBadRequest
	case errs.HasCode(err, handoff.CodeHandoffNotFound):
		return http.StatusNotFound
	default:
		logger.Warn("Failed to flush handoff tree", "error", err)
		return http.StatusInternalServerError
	}
}
