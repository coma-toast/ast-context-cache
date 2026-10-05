package handoff

import (
	"context"
	"os"
	"strings"

	astcontext "github.com/coma-toast/ast-context-cache/internal/context"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
	"github.com/coma-toast/ast-context-cache/internal/trail"
)

const (
	// defaultExpandBudget matches the search tools' default token budget.
	defaultExpandBudget = 4000

	modeSkeleton = "skeleton"
	modeAuto     = "auto"
	modeFull     = "full"
)

// pointerView is a pointer as it is now, next to its snapshot.
type pointerView struct {
	item     ExpandedItem
	returned *astcontext.ReturnedSymbol
}

// Expand returns snapshot items in full within the token budget (OP-4). Pointers are resolved
// in the child's worktree when it is a sibling of the snapshot's (OP-8), compared with their
// snapshot fingerprint (OP-7), and recorded as returned to the child (OP-6).
func (s *realService) Expand(ctx context.Context, req ExpandRequest) (*ExpandResponse, error) {
	ref, err := ParseHandoffRef(strings.TrimSpace(string(req.Handoff)))
	if err != nil {
		return nil, err
	}
	sid := SessionID(strings.TrimSpace(string(req.SessionID)))
	if sid == "" {
		return nil, errs.NewCode(errs.CodeInvalidInput, "session_id required: open the handoff first", "handoff", string(ref))
	}
	if !req.Section.Valid() {
		return nil, errs.NewCode(errs.CodeInvalidInput, "unknown snapshot section", "section", string(req.Section))
	}
	mode := strings.TrimSpace(req.Mode)
	if mode == "" {
		mode = modeAuto
	}
	if mode != modeSkeleton && mode != modeAuto && mode != modeFull {
		return nil, errs.NewCode(errs.CodeInvalidInput, "mode must be skeleton, auto, or full", "mode", mode)
	}
	if db.ContextDB == nil {
		return nil, errNoContextDB
	}
	h, err := loadLiveHandoff(db.ContextDB, ref, LoadLimits())
	if err != nil {
		return nil, err
	}
	child, err := loadChild(db.ContextDB, ref, sid)
	if err != nil {
		return nil, err
	}
	items, err := loadSnapshotItems(ref, req.Section)
	if err != nil {
		return nil, err
	}
	items = selectItems(items, req.Items, req.All)
	childProject := projectlinks.NormalizePath(req.ProjectPath)
	if childProject == "" {
		childProject = child.project
	}
	budget := req.TokenBudget
	if budget <= 0 {
		budget = defaultExpandBudget
	}
	resp := s.expandItems(h, sid, req.Section, items, mappedProject(h.project, childProject), mode, budget, req.Next)
	if resp.TokensAvailable, resp.TokensDelivered, err = addDelivered(sid, resp.TokensUsed); err != nil {
		return nil, err
	}
	return resp, nil
}

// expandItems renders items from the cursor on until the budget is spent. The first item is
// always returned, its content cut to fit when it alone is over budget.
func (s *realService) expandItems(h *handoffRow, sid SessionID, section Section, items []snapshotItem, root, mode string, budget int, next *PageCursor) *ExpandResponse {
	// Measure with tokens_used at its largest, and room left for the paging fields, so filling
	// them in afterwards can't overflow.
	resp := &ExpandResponse{Handoff: h.ref, Section: section, Items: []ExpandedItem{}, TokensUsed: budget}
	limit := budget - pagingReserveTokens
	offset := 0
	if next != nil {
		offset = max(0, next.Offset)
	}
	fileCache := map[string][]string{}
	fullCount := 0
	var delivered []astcontext.ReturnedSymbol
	for i := offset; i < len(items); i++ {
		v := expandItem(items[i], root, mode, &fullCount, fileCache)
		resp.Items = append(resp.Items, v.item)
		if responseTokens(resp) > limit {
			if len(resp.Items) > 1 {
				resp.Items = resp.Items[:len(resp.Items)-1]
				resp.Truncated, resp.Next = true, &PageCursor{Section: section, Offset: i}
				break
			}
			cutToBudget(resp, limit)
		}
		// A cut item wasn't delivered whole, so a later search may still send it.
		if v.returned != nil && !resp.Items[len(resp.Items)-1].Truncated {
			v.returned.Tokens = v.item.TokenEst
			delivered = append(delivered, *v.returned)
		}
	}
	astcontext.MarkReturned(string(sid), delivered...)
	resp.TokensUsed = responseTokens(resp)
	return resp
}

// cutToBudget shortens the response's only item until the response fits. JSON escaping makes
// the encoded content longer than the raw text, so it cuts by the measured overflow and re-checks.
func cutToBudget(resp *ExpandResponse, budget int) {
	it := &resp.Items[0]
	it.Truncated = true
	for over := responseTokens(resp) - budget; over > 0 && it.Content != ""; over = responseTokens(resp) - budget {
		it.Content = truncateBytes(it.Content, max(0, len(it.Content)-over*4-16))
		if len(it.Content) <= len("…") {
			it.Content = ""
		}
	}
	it.TokenEst = db.EstimateTokens(it.Content)
}

// expandItem renders one snapshot item in full.
func expandItem(it snapshotItem, root, mode string, fullCount *int, fileCache map[string][]string) pointerView {
	out := ExpandedItem{ID: it.id, Section: it.section, Key: it.key, Label: it.label}
	switch it.section {
	case SectionPointer:
		return expandPointer(it, root, mode, fullCount, fileCache)
	case SectionManifest:
		// The relative hit form is shorter than the stored absolute dedup key.
		out.Key, out.FileRel = trail.HitRef(it.fileRel, it.label, it.startLine), it.fileRel
	case SectionMemory:
		// The stored content is the clone payload; the agent reads the entry's compact line.
		out.Content, out.Label = it.label, ""
	default:
		out.Content = it.content
	}
	out.TokenEst = db.EstimateTokens(out.Content)
	return pointerView{item: out}
}

// expandPointer resolves a pointer under root and classifies how it changed since the snapshot:
// fresh, moved (same text, new lines), modified, deleted (file there, symbol gone), or
// file_missing. Only fresh is not stale; a gone symbol is a result, not an error (OP-7).
func expandPointer(it snapshotItem, root, mode string, fullCount *int, fileCache map[string][]string) pointerView {
	out := ExpandedItem{ID: it.id, Section: SectionPointer, Key: it.key, Label: it.label, FileRel: it.fileRel, FQN: it.fqn, Kind: it.kind}
	old := &LineRange{Start: it.startLine, End: it.endLine}
	file := absUnder(root, it.fileRel)
	if _, err := os.Stat(file); file == "" || err != nil {
		out.Stale, out.Change, out.OldLines = true, ChangeFileMissing, old
		return pointerView{item: out}
	}
	if it.kind == pointerKindFile {
		return expandFilePointer(out, it, file, mode)
	}
	cur := astcontext.SymbolRow{
		File: file, ProjectPath: projectlinks.OwningProject(file, root), Name: it.content, FQN: it.fqn,
		Kind: it.kind, StartLine: it.startLine, EndLine: it.endLine,
	}
	row, err := astcontext.LookupSymbol(root, it.fileRel, it.fqn, it.content)
	switch {
	case err == nil:
		cur = *row
	case sameText(file, it):
		// The index has no such symbol here (an unindexed worktree, say), but the snapshot's
		// lines still hold the same text, so the symbol is where it was.
	default:
		out.Stale, out.Change, out.OldLines = true, ChangeDeleted, old
		return pointerView{item: out}
	}
	out.NewLines = &LineRange{Start: cur.StartLine, End: cur.EndLine}
	fp, _ := astcontext.SymbolFingerprint(cur.File, cur.StartLine, cur.EndLine)
	switch {
	case fp == it.fingerprint && cur.StartLine == it.startLine && cur.EndLine == it.endLine:
		out.Change = ChangeFresh
	case fp == it.fingerprint:
		out.Stale, out.Change, out.OldLines = true, ChangeMoved, old
	default:
		out.Stale, out.Change, out.OldLines = true, ChangeModified, old
	}
	effective := astcontext.EffectiveMode(mode, 0, 0, *fullCount)
	data := map[string]any{"kind": cur.Kind}
	astcontext.ApplyMode(data, effective, cur.File, cur.Name, cur.ProjectPath, cur.StartLine, cur.EndLine, fileCache)
	out.Content = renderedContent(data)
	if effective == modeFull {
		*fullCount++
	}
	out.TokenEst = db.EstimateTokens(out.Content)
	return pointerView{item: out, returned: &astcontext.ReturnedSymbol{
		File: cur.File, Name: cur.Name, ProjectPath: cur.ProjectPath, StartLine: cur.StartLine, Mode: effective,
	}}
}

// expandFilePointer returns a whole-file pointer, as a skeleton in skeleton mode.
func expandFilePointer(out ExpandedItem, it snapshotItem, file, mode string) pointerView {
	data, err := os.ReadFile(file)
	if err != nil {
		out.Stale, out.Change, out.OldLines = true, ChangeFileMissing, &LineRange{Start: it.startLine, End: it.endLine}
		return pointerView{item: out}
	}
	lines := lineCount(data)
	out.NewLines = &LineRange{Start: 1, End: lines}
	if fp, _ := astcontext.SymbolFingerprint(file, 1, lines); fp != it.fingerprint {
		out.Stale, out.Change, out.OldLines = true, ChangeModified, &LineRange{Start: it.startLine, End: it.endLine}
	} else {
		out.Change = ChangeFresh
	}
	out.Content = string(data)
	if mode == modeSkeleton {
		out.Content = indexer.ExtractSkeleton(out.Content, indexer.GetLanguage(file), pointerKindFile)
	}
	out.TokenEst = db.EstimateTokens(out.Content)
	return pointerView{item: out}
}

// sameText reports whether file still holds the snapshot's text at the snapshot's lines.
func sameText(file string, it snapshotItem) bool {
	fp, err := astcontext.SymbolFingerprint(file, it.startLine, it.endLine)
	return err == nil && fp == it.fingerprint
}

// renderedContent picks the text ApplyMode produced.
func renderedContent(data map[string]any) string {
	for _, k := range []string{"source", "skeleton", "summary"} {
		if v, _ := data[k].(string); v != "" {
			return v
		}
	}
	return ""
}

// selectItems keeps the requested ids in snapshot order; no ids (or all) keeps every item.
func selectItems(items []snapshotItem, ids []int64, all bool) []snapshotItem {
	if all || len(ids) == 0 {
		return items
	}
	want := make(map[int64]bool, len(ids))
	for _, id := range ids {
		want[id] = true
	}
	out := items[:0:0]
	for _, it := range items {
		if want[it.id] {
			out = append(out, it)
		}
	}
	return out
}
