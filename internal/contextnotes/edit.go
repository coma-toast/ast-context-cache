package contextnotes

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/search"
)

// Edit applies an in-place mutation to a stored note. Context-as-a-file agents
// (Context Language Models, arXiv 2609.37725) treat their own context as a file
// they can rewrite at will; until now a ctx_* note was write-once, so the only
// way to change one was to flush it (invalidating the stub already written into
// chat) and store a new one (new ref, re-debited quota).
//
// Every edit is reversible: the body being replaced is kept as a revision, and
// an edit that grows a note is charged against the same store quotas as a new
// note so in-place growth cannot bypass them.

const (
	updateNoteContentQuery = `UPDATE context_notes SET content = ?, content_hash = ?, token_est = ?, revision = ?
		WHERE ref = ?`
	selectNoteRevisionQuery = `SELECT COALESCE(revision, 1) FROM context_notes WHERE ref = ?`
)

// Edit actions.
const (
	EditAppend   = "append"
	EditReplace  = "replace"
	EditDelete   = "delete"
	EditRewrite  = "rewrite"
	EditRevert   = "revert"
	maxEditMatch = 1000
)

// EditInput is one edit_context call. Pattern and line addressing are mutually
// exclusive; StartLine/EndLine are 1-indexed and inclusive.
type EditInput struct {
	Action          string
	Ref             string
	SessionID       string
	ExpectRevision  int
	Content         string
	Pattern         string
	Replacement     string
	StartLine       int
	EndLine         int
	MaxReplacements int
	ToRevision      int
	DryRun          bool
}

// EditResult reports the edit and what it did to the note's token footprint.
// TokensReclaimed is the token analogue of the paper's prefix-reuse FLOPs: it is
// the only evidence that a context-management edit actually paid for itself.
type EditResult struct {
	Ref              string                 `json:"ref"`
	Action           string                 `json:"action"`
	Revision         int                    `json:"revision"`
	PreviousRevision int                    `json:"previous_revision"`
	Changed          bool                   `json:"changed"`
	MatchedRegions   int                    `json:"matched_regions"`
	TokensBefore     int                    `json:"tokens_before"`
	TokensAfter      int                    `json:"tokens_after"`
	TokensReclaimed  int                    `json:"tokens_reclaimed"`
	DryRun           bool                   `json:"dry_run,omitempty"`
	SessionID        string                 `json:"session_id"`
	Stats            map[string]interface{} `json:"stats"`
}

// Edit mutates one note in place under the store's own limits.
func Edit(in EditInput, emb embedder.Interface) (*EditResult, error) {
	action := strings.ToLower(strings.TrimSpace(in.Action))
	ref := strings.TrimSpace(in.Ref)
	if ref == "" {
		return nil, errs.NewCode(errs.CodeInvalidInput, "ref required")
	}
	if !validEditAction(action) {
		return nil, errs.NewCode(errs.CodeInvalidInput,
			fmt.Sprintf("unknown action %q: use append, replace, delete, rewrite, or revert", in.Action))
	}
	note, err := noteByRef(ref)
	if err != nil {
		return nil, errs.NewCode(errs.CodeNotFound, fmt.Sprintf("no such context ref: %s", ref))
	}
	if in.SessionID != "" && note.SessionID != in.SessionID {
		return nil, errs.NewCode(errs.CodeNotFound, fmt.Sprintf("no such context ref: %s", ref))
	}
	current, err := noteRevision(ref)
	if err != nil {
		return nil, errs.WrapMessage("failed to read note revision", err, "ref", ref)
	}
	if in.ExpectRevision > 0 && in.ExpectRevision != current {
		return nil, errs.NewCode(errs.CodeConflict, "revision_conflict",
			"ref", ref, "expected_revision", in.ExpectRevision, "current_revision", current,
			"detail", "another agent edited this note; fetch_context to re-read before editing again")
	}

	next, matched, err := applyEdit(action, note, in)
	if err != nil {
		return nil, err
	}
	next = strings.TrimSpace(next)
	before := note.TokenEst
	if before == 0 {
		before = db.EstimateTokens(note.Content)
	}
	res := &EditResult{
		Ref:              ref,
		Action:           action,
		Revision:         current,
		PreviousRevision: current,
		MatchedRegions:   matched,
		TokensBefore:     before,
		TokensAfter:      before,
		SessionID:        note.SessionID,
	}
	if next == strings.TrimSpace(note.Content) {
		// A pattern that matched nothing is a no-op, not an error: report it so the
		// caller can see the edit did not land without burning a revision.
		res.Stats = BuildStatsBlock(note.SessionID, LoadLimits())
		return res, nil
	}
	if next == "" {
		return nil, errs.NewCode(errs.CodeInvalidInput, "edit would empty the note",
			"suggestion", "use flush_context(refs=[...]) to delete a note entirely")
	}
	after := db.EstimateTokens(next)
	if err := checkEditGrowth(note.SessionID, before, after); err != nil {
		return nil, err
	}
	res.TokensAfter = after
	res.TokensReclaimed = before - after
	if in.DryRun {
		res.DryRun = true
		res.Stats = BuildStatsBlock(note.SessionID, LoadLimits())
		return res, nil
	}
	if err := commitEdit(note, ref, current, next, after, action); err != nil {
		return nil, err
	}
	res.Changed = true
	res.Revision = current + 1
	if emb != nil {
		go EmbedNote(ref, note.SessionID, note.Label, next, emb)
	}
	logger.Info("Edited context note in place", "ref", ref, "action", action,
		"revision", res.Revision, "matched", matched, "tokens_before", before, "tokens_after", after)
	res.Stats = BuildStatsBlock(note.SessionID, LoadLimits())
	return res, nil
}

func validEditAction(action string) bool {
	switch action {
	case EditAppend, EditReplace, EditDelete, EditRewrite, EditRevert:
		return true
	}
	return false
}

func noteRevision(ref string) (int, error) {
	var rev int
	if err := db.ContextDB.QueryRow(selectNoteRevisionQuery, ref).Scan(&rev); err != nil {
		return 0, err
	}
	if rev < 1 {
		rev = 1
	}
	return rev, nil
}

func applyEdit(action string, note Note, in EditInput) (string, int, error) {
	switch action {
	case EditAppend:
		if strings.TrimSpace(in.Content) == "" {
			return "", 0, errs.NewCode(errs.CodeInvalidInput, "content required for append")
		}
		sep := "\n"
		if strings.HasSuffix(note.Content, "\n") {
			sep = ""
		}
		return note.Content + sep + in.Content, 1, nil
	case EditRewrite:
		if strings.TrimSpace(in.Content) == "" {
			return "", 0, errs.NewCode(errs.CodeInvalidInput, "content required for rewrite")
		}
		return in.Content, 1, nil
	case EditReplace, EditDelete:
		return applyRegionEdit(action, note.Content, in)
	case EditRevert:
		return revertTarget(note.Ref, in.ToRevision)
	}
	return "", 0, errs.NewCode(errs.CodeInvalidInput, "unknown action")
}

// applyRegionEdit does the CLM-style surgical edit: rewrite or drop the matched
// regions of a body, addressed either by regex or by an inclusive line range.
func applyRegionEdit(action, content string, in EditInput) (string, int, error) {
	hasPattern := strings.TrimSpace(in.Pattern) != ""
	hasRange := in.StartLine > 0 && in.EndLine > 0
	if hasPattern == hasRange {
		return "", 0, errs.NewCode(errs.CodeInvalidInput,
			"replace and delete need exactly one of pattern, or start_line with end_line")
	}
	limit := in.MaxReplacements
	if limit <= 0 || limit > maxEditMatch {
		limit = maxEditMatch
	}

	if hasRange {
		lines := strings.Split(content, "\n")
		if in.EndLine > len(lines) {
			return "", 0, errs.NewCode(errs.CodeInvalidInput,
				fmt.Sprintf("end_line %d is past the end of the note (%d lines)", in.EndLine, len(lines)))
		}
		if action == EditReplace {
			lines = append(lines[:in.StartLine-1], append(splitLines(in.Replacement), lines[in.EndLine:]...)...)
		} else {
			lines = append(lines[:in.StartLine-1], lines[in.EndLine:]...)
		}
		return strings.Join(lines, "\n"), 1, nil
	}

	// Go's regexp is RE2: no backreferences or lookaround, and matching is linear
	// in the input. A model-supplied pattern cannot blow up the edit.
	re, err := regexp.Compile(in.Pattern)
	if err != nil {
		return "", 0, errs.NewCode(errs.CodeInvalidInput, "invalid pattern: "+err.Error())
	}
	locations := re.FindAllStringIndex(content, limit)
	if len(locations) == 0 {
		return content, 0, nil
	}
	var b strings.Builder
	prev := 0
	for _, loc := range locations {
		b.WriteString(content[prev:loc[0]])
		if action == EditReplace {
			match := content[loc[0]:loc[1]]
			b.Write(re.ExpandString(nil, in.Replacement, match, re.FindStringSubmatchIndex(match)))
		}
		prev = loc[1]
	}
	b.WriteString(content[prev:])
	return b.String(), len(locations), nil
}

func splitLines(s string) []string {
	if s == "" {
		return nil
	}
	return strings.Split(s, "\n")
}

// revertTarget resolves the body to restore. Revisions are monotonic, so reverting
// writes a new revision rather than rewinding the counter.
func revertTarget(ref string, toRevision int) (string, int, error) {
	if toRevision > 0 {
		content, _, ok := revisionAt(ref, toRevision)
		if !ok {
			return "", 0, errs.NewCode(errs.CodeNotFound,
				fmt.Sprintf("no revision %d for %s", toRevision, ref))
		}
		return content, 1, nil
	}
	prev, ok := latestRevision(ref)
	if !ok {
		return "", 0, errs.NewCode(errs.CodeNotFound, "no earlier revision to revert to for "+ref)
	}
	return prev.Content, 1, nil
}

// checkEditGrowth charges an edit's growth against the same caps as a new note, so
// repeated appends can't grow a note past what store_context would have accepted.
func checkEditGrowth(sessionID string, before, after int) error {
	lim := LoadLimits()
	if after > lim.MaxTokensSession {
		return newLimitError("single_note_tokens", after, lim.MaxTokensSession, after)
	}
	delta := after - before
	if delta <= 0 {
		return nil
	}
	if err := checkLimits(sessionID, delta, lim); err != nil {
		return err
	}
	return nil
}

// commitEdit writes the replaced body to the revision log, swaps in the new one,
// and reindexes. The note vector is deleted before the row changes: an upsert
// alone would leave the pre-edit embedding behind (vectors are keyed by id, not
// content_hash), which fails loudly here instead of quietly recalling old text.
func commitEdit(note Note, ref string, revision int, content string, tokenEst int, action string) error {
	if err := search.Cache.DeleteRefs("note", []string{noteVectorKey(ref)}); err != nil {
		return errs.WrapMessage("failed to replace note vector", err, "ref", ref)
	}
	if err := writeRevision(ref, revision, note.Content, action, note.TokenEst); err != nil {
		return errs.WrapMessage("failed to record note revision", err, "ref", ref)
	}
	hash := search.ContentHash(content)
	if _, err := db.ContextDB.Exec(updateNoteContentQuery, content, hash, tokenEst, revision+1, ref); err != nil {
		return errs.WrapMessage("failed to update note content", err, "ref", ref)
	}
	reindexNoteFTS(ref, note.SessionID, note.Label, content)
	adjustSessionStore(note.SessionID, 0, tokenEst-note.TokenEst)
	pruneRevisions(ref)
	return nil
}
