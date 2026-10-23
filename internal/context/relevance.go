package context

import (
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/flags"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
	"github.com/coma-toast/ast-context-cache/internal/render"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

const (
	// maxWithheldEstimates caps how many withheld hits have their skeleton tokens estimated.
	maxWithheldEstimates = 20
	// maxBudgetMisses ends a packing loop after this many consecutive hits over the budget.
	maxBudgetMisses = 5
	// collapseIndexKey carries a candidate's index through CollapseDistractors.
	collapseIndexKey = "_i"

	hybridNoMatchHint   = "No symbol matched the query closely. Try an exact identifier, fewer words, or search_semantic to search by behavior."
	semanticNoMatchHint = "No symbol is semantically close to the query. Try get_context_capsule with an exact identifier, or rephrase."
)

// PrecisionArgs are a code search call's precision arguments.
type PrecisionArgs struct {
	// MinRelative overrides the configured relative floor when positive (min_relative_score).
	MinRelative float64
	// Collapse folds test, mock, vendored and duplicate hits (collapse, default true).
	Collapse bool
}

// ParsePrecisionArgs reads min_relative_score and collapse from a tool call's arguments.
func ParsePrecisionArgs(args map[string]any) PrecisionArgs {
	p := PrecisionArgs{Collapse: true}
	if v, ok := args["min_relative_score"].(float64); ok && v > 0 {
		p.MinRelative = v
	}
	if v, ok := args["collapse"].(bool); ok {
		p.Collapse = v
	}
	return p
}

// Withheld summarizes hits held back by the relevance floor, the token budget or edit mode.
// Tokens is an estimate: skeleton sizes for up to 20 floor-withheld hits plus the exact size
// of each hit the budget skipped.
type Withheld struct {
	Count  int `json:"count"`
	Tokens int `json:"tokens"`
}

// Add counts other into w.
func (w *Withheld) Add(other Withheld) {
	w.Count += other.Count
	w.Tokens += other.Tokens
}

// ApplyTo writes w into resp as "withheld" when anything was withheld.
func (w Withheld) ApplyTo(resp map[string]any) {
	if w.Count > 0 {
		resp["withheld"] = w
	}
}

// Precision is the outcome of the precision pass over a call's ranked candidates.
type Precision struct {
	// Kept are the candidates to pack, in rank order.
	Kept []search.ScoredResult
	// Withheld are the candidates the floor held back (all of them on a no-match).
	Withheld  []search.ScoredResult
	NoMatch   *render.NoMatch
	Collapsed []render.Collapse
}

// ApplyPrecision runs the relevance floor, the weak-match check and distractor collapse over
// scored, in that order. A weak top hit withholds everything and sets NoMatch. semantic
// selects the vector-only weak check. With feature_relevance_floor off, everything is kept.
func ApplyPrecision(scored []search.ScoredResult, query, projectPath string, args PrecisionArgs, semantic bool) Precision {
	if !flags.Enabled(flags.KeyRelevanceFloor) || len(scored) == 0 {
		return Precision{Kept: scored}
	}
	cfg := LoadFloorConfig()
	if args.MinRelative > 0 {
		cfg.MinRelative = args.MinRelative
	}
	kept, withheld := ApplyRelevanceFloor(scored, cfg)
	weak, best := WeakMatch(kept, query, cfg)
	hint := hybridNoMatchHint
	if semantic {
		weak, best = WeakSemanticMatch(kept, cfg)
		hint = semanticNoMatchHint
	}
	if weak {
		return Precision{Withheld: scored, NoMatch: &render.NoMatch{BestScore: best, Hint: hint}}
	}
	p := Precision{Kept: kept, Withheld: withheld}
	p.Kept, p.Collapsed = collapseScored(kept, query, projectPath, args.Collapse)
	return p
}

// collapseScored applies CollapseDistractors to candidates before packing: each is viewed by
// its project-relative file, name, kind and indexed skeleton. A repeat of the same symbol
// gets no skeleton, so session dedup rather than collapse handles it.
func collapseScored(scored []search.ScoredResult, query, projectPath string, enabled bool) ([]search.ScoredResult, []render.Collapse) {
	if !enabled || distractorQueryRe.MatchString(query) {
		return scored, nil
	}
	views := make([]map[string]any, len(scored))
	seen := map[string]bool{}
	conn, connErr := db.IndexReader()
	for i, r := range scored {
		h := hitFromScored(r, projectPath)
		file, name := mapStr(h.Data, "file"), mapStr(h.Data, "name")
		key := SymbolDedupKey(file, name, h.StartLine)
		var skeleton string
		if !seen[key] && connErr == nil {
			conn.QueryRow(selectSymbolSkeletonQuery, file, name, projectlinks.OwningProject(file, projectPath), h.StartLine).Scan(&skeleton)
		}
		seen[key] = true
		views[i] = map[string]any{"file": db.RelPath(file, projectPath), "name": name, "kind": h.Data["kind"], "skeleton": skeleton, collapseIndexKey: i}
	}
	keptViews, collapsed := CollapseDistractors(views, query, true)
	if len(collapsed) == 0 {
		return scored, nil
	}
	kept := make([]search.ScoredResult, 0, len(keptViews))
	for _, v := range keptViews {
		kept = append(kept, scored[v[collapseIndexKey].(int)])
	}
	return kept, collapsed
}

// EstimateWithheld counts hits and estimates their tokens as skeletons, for at most 20 hits.
func EstimateWithheld(hits []search.ScoredResult, projectPath string, fileCache map[string][]string) Withheld {
	w := Withheld{Count: len(hits)}
	for _, r := range hits[:min(len(hits), maxWithheldEstimates)] {
		h := hitFromScored(r, projectPath)
		file, name := mapStr(h.Data, "file"), mapStr(h.Data, "name")
		w.Tokens += WouldSendTokens(file, name, projectlinks.OwningProject(file, projectPath), "skeleton", h.StartLine, h.EndLine, 0, 0, 0, fileCache)
	}
	return w
}
