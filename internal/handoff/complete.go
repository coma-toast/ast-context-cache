package handoff

import (
	"context"
	"database/sql"
	"errors"
	"strings"
	"unicode/utf8"

	"github.com/coma-toast/ast-context-cache/internal/contextnotes"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/memory"
)

const (
	selectCompletingChildQuery = `SELECT c.handoff_ref, c.tree_id, COALESCE(c.project_path, ''), COALESCE(c.label, ''),
		h.parent_session_id, COALESCE(h.label, '')
		FROM handoff_children c JOIN handoffs h ON h.ref = c.handoff_ref WHERE c.child_session_id = ?`
	selectCurrentResultQuery = `SELECT result_ref FROM handoff_results WHERE child_session_id = ? AND superseded_at IS NULL
		ORDER BY id DESC LIMIT 1`
	supersedeResultsQuery = `UPDATE handoff_results SET superseded_at = ? WHERE child_session_id = ? AND superseded_at IS NULL`
	insertResultQuery     = `INSERT INTO handoff_results (child_session_id, result_ref, created_at) VALUES (?, ?, ?)`
	completeChildQuery    = `UPDATE handoff_children SET status = ?, result_ref = ?, summary = ?, summary_source = ?,
		summary_truncated = ?, result_status = ?, last_activity_at = ? WHERE child_session_id = ?`
	touchHandoffQuery = `UPDATE handoffs SET last_access_at = ? WHERE ref = ?`
	touchTreeQuery    = `UPDATE handoff_trees SET last_access_at = ? WHERE tree_id = ?`

	resultLabelPrefix = "result: "
	// truncationMark ends a summary cut at the cap; its bytes count toward the cap.
	truncationMark = "…"
	// A cut moves back at most 1/wordBoundarySlack of the cap to land between words, so one
	// long unbroken run doesn't shrink the summary to nothing.
	wordBoundarySlack = 4
)

// completingChild is the child row a completion updates, with its handoff's parent.
type completingChild struct {
	ref           HandoffRef
	tree          TreeID
	project       string
	label         string
	parent        SessionID
	handoffLabel  string
	resultTokens  int
	supersededRef string
}

// Complete stores a child's result as a handoff_result note and records it as the child's
// current result (RT-1–RT-8). The note is stored before the transaction (contextnotes writes
// through its own pool); if the transaction then fails the note is deleted again, so a failed
// completion leaves nothing behind (NFR-5). FACT:/RULE: lines are promoted into the parent's
// session memory after the commit, so they never outlive a completion that didn't happen.
func (s *realService) Complete(ctx context.Context, req CompleteRequest) (*CompleteResponse, error) {
	content := strings.TrimSpace(req.Content)
	if req.SessionID == "" || content == "" {
		return nil, errs.NewCode(errs.CodeInvalidInput, "session_id and content required")
	}
	status := req.Status
	if status == "" {
		status = StatusDone
	}
	if !status.Completed() {
		return nil, errs.NewCode(errs.CodeInvalidInput, "status must be done, partial, or failed", "status", string(status))
	}
	c, err := completingChildOf(req.SessionID)
	if err != nil {
		return nil, err
	}
	project := req.ProjectPath
	if project == "" {
		project = c.project
	}
	lim := LoadLimits()
	summary, source := strings.TrimSpace(req.Summary), SummarySourceChild
	if summary == "" {
		summary, source = deriveSummary(content, lim.SummaryMaxTokens), SummarySourceDerived
	}
	summary, truncated := truncateToTokens(summary, lim.SummaryMaxTokens)
	stored, err := contextnotes.Store(string(req.SessionID), content, c.resultLabel(), project, nil,
		contextnotes.KindHandoffResult, resultMetadata(c.ref, status, req), s.emb)
	if err != nil {
		return nil, errs.WrapMessage("failed to store handoff result", err, "session", string(req.SessionID))
	}
	c.resultTokens = stored.VirtualTokensStored
	var released []string
	err = db.HandoffTx(func(tx *sql.Tx) error {
		var err error
		released, err = s.completeTx(tx, req.SessionID, &c, stored.Ref, status, summary, source, truncated)
		return err
	})
	if err != nil {
		if _, ferr := contextnotes.Flush(string(req.SessionID), stored.Ref, "", false); ferr != nil {
			s.logger.Warn("Failed to delete result of failed completion", req.SessionID.Attr(), "result", stored.Ref, "error", ferr)
		}
		return nil, err
	}
	promoted := s.promoteResultMemory(c.parent, project, stored.Ref, content)
	s.waiters.notify(c.tree)
	saved := max(0, db.EstimateTokens(content)-db.EstimateTokens(summary))
	s.logger.Info("Completed handoff child", req.SessionID.Attr(), c.ref.Attr(), c.tree.Attr(), "status", status,
		"result", stored.Ref, "summary_source", source, "summary_truncated", truncated, "tokens_saved", saved,
		"promoted_memory", len(promoted), "released_claims", len(released), "superseded", c.supersededRef)
	return &CompleteResponse{
		ResultRef:        stored.Ref,
		Handoff:          c.ref,
		Status:           status,
		Summary:          summary,
		SummarySource:    source,
		SummaryTruncated: truncated,
		Stub:             returnStub(stored.Ref, c.ref, status, summary),
		PromotedMemory:   promoted,
		ReleasedClaims:   released,
		SupersededRef:    c.supersededRef,
		TokensSaved:      saved,
	}, nil
}

// completeTx supersedes the child's current result, records the new one, sets the child's
// status and summary, releases its claims (RT-6, granting them onward), and charges the result
// to the tree. It returns the released claim keys.
func (s *realService) completeTx(tx *sql.Tx, sid SessionID, c *completingChild, resultRef string, status Status,
	summary string, source SummarySource, truncated bool,
) ([]string, error) {
	now := sqlTime(nowFunc())
	err := tx.QueryRow(selectCurrentResultQuery, string(sid)).Scan(&c.supersededRef)
	if err != nil && !errors.Is(err, sql.ErrNoRows) {
		return nil, errs.WrapMessage("failed to read current handoff result", err, "session", string(sid))
	}
	if _, err := tx.Exec(supersedeResultsQuery, now, string(sid)); err != nil {
		return nil, errs.WrapMessage("failed to supersede handoff result", err, "session", string(sid))
	}
	if _, err := tx.Exec(insertResultQuery, string(sid), resultRef, now); err != nil {
		return nil, errs.WrapMessage("failed to record handoff result", err, "session", string(sid))
	}
	res, err := tx.Exec(completeChildQuery, string(status), resultRef, summary, string(source), truncated, string(status), now, string(sid))
	if err != nil {
		return nil, errs.WrapMessage("failed to complete handoff child", err, "session", string(sid))
	}
	// The tree can be flushed between the read in Complete and this transaction.
	if n, _ := res.RowsAffected(); n == 0 {
		return nil, errs.NewCode(CodeHandoffNotFound, "session is not a handoff child", "session", string(sid))
	}
	released, err := s.releaseAllTx(tx, c.tree, sid)
	if err != nil {
		return nil, err
	}
	if err := s.chargeTreeTx(tx, c.tree, c.resultTokens, 1); err != nil {
		return nil, err
	}
	if _, err := tx.Exec(touchHandoffQuery, now, string(c.ref)); err != nil {
		return nil, errs.WrapMessage("failed to touch handoff", err, "handoff", string(c.ref))
	}
	if _, err := tx.Exec(touchTreeQuery, now, string(c.tree)); err != nil {
		return nil, errs.WrapMessage("failed to touch handoff tree", err, "tree", string(c.tree))
	}
	return released, nil
}

// promoteResultMemory stores the result's FACT:/RULE: lines as session memory of the parent,
// sourced to the result note, and returns the new mem_ refs (RT-5). The entries belong to the
// parent and outlive the tree. A failed entry is skipped: the result is already committed.
func (s *realService) promoteResultMemory(parent SessionID, project, resultRef, content string) []string {
	ex := memory.ExtractFromText(content)
	if len(ex.Facts) == 0 && len(ex.Procedures) == 0 {
		return nil
	}
	stored, err := memory.StoreExtracted(string(parent), project, resultRef, ex, memory.ScopeSession)
	if err != nil {
		s.logger.Warn("Failed to promote handoff result memory", parent.Attr(), "result", resultRef, "error", err)
	}
	refs := make([]string, 0, len(stored))
	for _, r := range stored {
		refs = append(refs, r.Ref)
	}
	return refs
}

func completingChildOf(sid SessionID) (completingChild, error) {
	var c completingChild
	if db.ContextDB == nil {
		return c, errNoContextDB
	}
	err := db.ContextDB.QueryRow(selectCompletingChildQuery, string(sid)).Scan(&c.ref, &c.tree, &c.project, &c.label, &c.parent, &c.handoffLabel)
	if errors.Is(err, sql.ErrNoRows) {
		return c, errs.NewCode(CodeHandoffNotFound, "session is not a handoff child", "session", string(sid))
	}
	if err != nil {
		return c, errs.WrapMessage("failed to read handoff child", err, "session", string(sid))
	}
	return c, nil
}

func (c completingChild) resultLabel() string {
	switch {
	case c.label != "":
		return resultLabelPrefix + c.label
	case c.handoffLabel != "":
		return resultLabelPrefix + c.handoffLabel
	}
	return resultLabelPrefix + string(c.ref)
}

// resultMetadata is the result note's metadata; RT-8's structured fields ride along verbatim
// so collect can echo them.
func resultMetadata(ref HandoffRef, status Status, req CompleteRequest) map[string]any {
	meta := map[string]any{"handoff": string(ref), "status": string(status)}
	if len(req.ChangedFiles) > 0 {
		meta["changed_files"] = req.ChangedFiles
	}
	if len(req.OpenQuestions) > 0 {
		meta["open_questions"] = req.OpenQuestions
	}
	return meta
}

// returnStub is what the child prints as its final message for the parent (RT-4).
func returnStub(resultRef string, ref HandoffRef, status Status, summary string) string {
	return "[result " + resultRef + " for " + string(ref) + "] " + string(status) + " — " + summary
}

// deriveSummary builds a summary mechanically when the child gave none (RT-3): the FACT: and
// RULE: lines first, then the content's leading lines. It stops gathering once past the cap;
// the caller truncates to it.
func deriveSummary(content string, maxTokens int) string {
	ex := memory.ExtractFromText(content)
	var lines []string
	for _, f := range ex.Facts {
		lines = append(lines, "FACT: "+f.Subject+" "+f.Predicate+" "+f.Object)
	}
	for _, p := range ex.Procedures {
		lines = append(lines, "RULE: "+p.Rule)
	}
	size, limit := 0, maxTokens*4
	for _, l := range lines {
		size += len(l) + 1
	}
	for _, raw := range strings.Split(content, "\n") {
		if size > limit {
			break
		}
		line := strings.TrimSpace(raw)
		if line == "" || isMarkerLine(line) {
			continue
		}
		lines = append(lines, line)
		size += len(line) + 1
	}
	return strings.Join(lines, "\n")
}

// isMarkerLine reports whether line is a FACT:/RULE: line, which deriveSummary already placed
// first.
func isMarkerLine(line string) bool {
	ex := memory.ExtractFromText(line)
	return len(ex.Facts)+len(ex.Procedures)+len(ex.Skipped) > 0
}

// truncateToTokens cuts s so db.EstimateTokens of the result is at most maxTokens, preferring a
// word boundary and marking the cut. Completion never fails on length (RT-2).
func truncateToTokens(s string, maxTokens int) (string, bool) {
	if db.EstimateTokens(s) <= maxTokens {
		return s, false
	}
	limit := maxTokens*4 - len(truncationMark)
	if limit <= 0 {
		return "", true
	}
	for limit > 0 && !utf8.RuneStart(s[limit]) {
		limit--
	}
	cut := s[:limit]
	if i := strings.LastIndexAny(cut, " \n\t"); i > 0 && i >= limit-limit/wordBoundarySlack {
		cut = cut[:i]
	}
	return strings.TrimRight(cut, " \n\t") + truncationMark, true
}
