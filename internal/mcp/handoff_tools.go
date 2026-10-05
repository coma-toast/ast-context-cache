package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"strconv"
	"strings"
	"time"

	astcontext "github.com/coma-toast/ast-context-cache/internal/context"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/flags"
	"github.com/coma-toast/ast-context-cache/internal/handoff"
	"github.com/coma-toast/ast-context-cache/internal/sys"
)

// The three handoff tools (TS-1). Each takes an action.
const (
	toolHandoff     = "handoff"
	toolOpenHandoff = "open_handoff"
	toolScratchpad  = "scratchpad"

	actionAll = "all"
	// maxGrantNoticeTokens caps the claims_granted content item (CL-5, NFR-3).
	maxGrantNoticeTokens = 30
)

// errHandoffNotStarted is reported, as a plain {"error"}, when no service is installed.
var errHandoffNotStarted = errs.New("handoff service not started")

// handoffCall is one handoff tool call: its parsed arguments and what it logs.
type handoffCall struct {
	tool        string
	action      string
	args        map[string]any
	projectPath string
	// savings is what logToolQuery records for the call (OB-2, OB-3).
	savings astcontext.SavingsMeta
}

// handoffTool is the handoff tool definition: create, complete, and fan-in (TS-1).
func handoffTool() Tool {
	return Tool{
		Name: toolHandoff,
		Description: "Delegate to a subagent and collect its result. create (parent): snapshot this session; put the returned stub " +
			"in the subagent's prompt. complete (child): store your result, get a short stub. collect/list/status/flush: fan in, list, inspect, delete.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"action":              enumProp("", "create", "complete", "collect", "list", "status", "flush"),
				"session_id":          prop("string", "Your session id"),
				"project_path":        prop("string", "Project root"),
				"brief":               prop("string", "create: the subagent's task"),
				"label":               prop("string", "create: short name"),
				"pointers":            pointersProp(),
				"ctx_refs":            arrayProp("string", "create: ctx_* notes to include"),
				"mem_refs":            arrayProp("string", "create: mem_* entries to include"),
				"mode":                enumProp("create: fork if the subagent shares your context", string(handoff.ModeFresh), string(handoff.ModeFork)),
				"exclude_trail":       prop("array", `create: trail indexes to drop, or ["all"]`),
				"exclude_trail_query": prop("string", "create: drop matching trail entries"),
				"include_manifest":    prop("boolean", "create: default true"),
				"handoff":             prop("string", "hof_ ref"),
				"content":             prop("string", "complete: full result"),
				"summary":             prop("string", "complete: short summary"),
				"status":              enumProp("complete", string(handoff.StatusDone), string(handoff.StatusPartial), string(handoff.StatusFailed)),
				"recursive":           prop("boolean", "collect: nested handoffs too"),
				"wait_seconds":        prop("integer", "collect: long-poll, max 60"),
				"token_budget":        prop("integer", "Max tokens"),
				"changed_files":       arrayProp("string", "complete: files you changed"),
				"open_questions":      arrayProp("string", "complete: unresolved questions"),
			},
			"required": []string{"action"},
		},
		Tier: TierCore,
	}
}

// openHandoffTool is the child's entry point. Its first sentence is the TS-4 instruction.
func openHandoffTool() Tool {
	return Tool{
		Name: toolOpenHandoff,
		Description: "If your prompt contains [handoff hof_…], call open_handoff before any search. " +
			"open: join as a child and get the digest; pass the returned session_id on every later call. " +
			"expand: full digest items. resume: reopen your child session.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"action":       enumProp("default open", "open", "expand", "resume"),
				"handoff":      prop("string", "hof_ ref from the stub"),
				"session_id":   prop("string", "expand, resume: your child session id"),
				"project_path": prop("string", "Your project root"),
				"section":      enumProp("expand: snapshot section", string(handoff.SectionPointer), string(handoff.SectionNote), string(handoff.SectionMemory), string(handoff.SectionTrail), string(handoff.SectionManifest)),
				"items":        prop("array", `expand: item ids, or ["all"]`),
				"mode":         enumProp("expand: pointer detail", "skeleton", "auto", "full"),
				"token_budget": prop("integer", "Max tokens"),
				"next":         prop("object", "Cursor from a truncated response"),
			},
			"required": []string{"handoff"},
		},
		Tier: TierCore,
	}
}

// scratchpadTool shares findings and advisory claims across a handoff tree (SP, CL). CL-7
// requires saying that claims are advisory.
func scratchpadTool() Tool {
	return Tool{
		Name: toolScratchpad,
		Description: "Notes shared across a handoff tree. post: finding or dead_end. read: others' entries since a cursor, dead ends, claims. " +
			"retract: hide your entry. claim/release: FIFO claim on a file or symbol key; a later grant arrives as [claims_granted]. " +
			"Claims are advisory: nothing blocks edits.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"action":       enumProp("", "post", "read", "retract", "claim", "release"),
				"session_id":   prop("string", "Your session id"),
				"type":         enumProp("post: entry type", string(handoff.EntryTypeFinding), string(handoff.EntryTypeDeadEnd)),
				"text":         prop("string", "post: entry text"),
				"refs":         arrayProp("string", "post: related files or refs"),
				"since":        prop("integer", "read: next_cursor from the last read"),
				"types":        arrayProp("string", "read: entry types"),
				"author":       prop("string", "read: one author's entries"),
				"include_own":  prop("boolean", "read: include your entries"),
				"key":          prop("string", "claim, release: file path or symbol key"),
				"reason":       prop("string", "claim: why"),
				"entry":        prop("integer", "retract: entry id"),
				"token_budget": prop("integer", "Max tokens"),
			},
			"required": []string{"action", "session_id"},
		},
		Tier: TierCore,
	}
}

// handleHandoffTool routes the handoff, open_handoff, and scratchpad tools, reporting false for
// any other tool. Successful calls return the service's response; failures return
// handoff.ErrorMap (TS-5), so the envelope sets isError. Every call is logged here.
func handleHandoffTool(toolName string, toolArgs, args map[string]any, start time.Time, cpuStart sys.CPUSample, projectPath string) (any, bool, error) {
	if toolName != toolHandoff && toolName != toolOpenHandoff && toolName != toolScratchpad {
		return nil, false, nil
	}
	c := &handoffCall{tool: toolName, action: strArg(toolArgs, "action"), args: toolArgs, projectPath: projectPath}
	if c.action == "" && toolName == toolOpenHandoff {
		c.action = "open"
	}
	result, err := c.run()
	errMsg := ""
	switch {
	case errors.Is(err, errHandoffNotStarted):
		result, errMsg = map[string]string{"error": err.Error()}, err.Error()
	case err != nil:
		result, errMsg = handoff.ErrorMap(err), err.Error()
	}
	resultJSON, _ := json.Marshal(result)
	outTokens := db.EstimateTokens(string(resultJSON))
	if c.savings.TokensUsed == 0 {
		c.savings.TokensUsed = outTokens
	}
	logToolQuery(toolName, args, len(resultJSON), 0, outTokens, c.savings, start, cpuStart, projectPath, errMsg)
	return result, true, err
}

// run gates the action and calls the service.
func (c *handoffCall) run() (any, error) {
	if !flags.ActionEnabled(c.tool, c.action) {
		return nil, errActionDisabled(c.tool, c.action)
	}
	svc := handoff.Default()
	if svc == nil {
		return nil, errHandoffNotStarted
	}
	ctx := context.Background()
	switch c.tool {
	case toolHandoff:
		return c.runHandoff(ctx, svc)
	case toolOpenHandoff:
		return c.runOpenHandoff(ctx, svc)
	default:
		return c.runScratchpad(ctx, svc)
	}
}

func (c *handoffCall) runHandoff(ctx context.Context, svc handoff.Service) (any, error) {
	sid := handoff.SessionID(strArg(c.args, "session_id"))
	ref := handoff.HandoffRef(strArg(c.args, "handoff"))
	switch c.action {
	case "create":
		req := handoff.CreateRequest{
			SessionID: sid, ProjectPath: c.projectPath, Brief: strArg(c.args, "brief"), Label: strArg(c.args, "label"),
			Pointers: pointersArg(c.args["pointers"]), CtxRefs: parseStringList(c.args["ctx_refs"]), MemRefs: parseStringList(c.args["mem_refs"]),
			Mode: handoff.Mode(strArg(c.args, "mode")), ExcludeTrailQuery: strArg(c.args, "exclude_trail_query"),
		}
		ids, all, err := idsArg(c.args["exclude_trail"])
		if err != nil {
			return nil, err
		}
		req.ExcludeAllTrail = all
		for _, id := range ids {
			req.ExcludeTrail = append(req.ExcludeTrail, int(id))
		}
		if v, ok := c.args["include_manifest"].(bool); ok {
			req.IncludeManifest = &v
		}
		return svc.Create(ctx, req)
	case "complete":
		resp, err := svc.Complete(ctx, handoff.CompleteRequest{
			SessionID: sid, ProjectPath: c.projectPath, Content: strArg(c.args, "content"), Summary: strArg(c.args, "summary"),
			Status: handoff.Status(strArg(c.args, "status")), ChangedFiles: textList(c.args["changed_files"]), OpenQuestions: textList(c.args["open_questions"]),
		})
		if err != nil {
			return nil, err
		}
		c.savings.TokensSaved = resp.TokensSaved
		return resp, nil
	case "collect":
		return svc.Collect(ctx, handoff.CollectRequest{
			SessionID: sid, Handoff: ref, Recursive: boolArg(c.args, "recursive"),
			WaitSeconds: intArg(c.args, "wait_seconds"), TokenBudget: intArg(c.args, "token_budget"),
		})
	case "list":
		return svc.List(ctx, handoff.ListRequest{SessionID: sid})
	case "status":
		return svc.Status(ctx, handoff.StatusRequest{SessionID: sid, Handoff: ref})
	case "flush":
		return svc.Flush(ctx, handoff.FlushRequest{SessionID: sid, Handoff: ref})
	}
	return nil, errUnknownAction(c.tool, c.action)
}

func (c *handoffCall) runOpenHandoff(ctx context.Context, svc handoff.Service) (any, error) {
	ref := handoff.HandoffRef(strArg(c.args, "handoff"))
	sid := handoff.SessionID(strArg(c.args, "session_id"))
	budget := intArg(c.args, "token_budget")
	next := cursorArg(c.args["next"])
	switch c.action {
	case "open", "resume":
		req := handoff.OpenRequest{Handoff: ref, ProjectPath: c.projectPath, TokenBudget: budget, Next: next}
		// A session id makes Open resume that child, so a plain open ignores the caller's own
		// id unless it is paging a digest it already opened.
		if c.action == "resume" || next != nil {
			req.SessionID = sid
		}
		if c.action == "resume" && sid == "" {
			return nil, errs.NewCode(errs.CodeInvalidInput, "session_id required to resume", "handoff", string(ref))
		}
		resp, err := svc.Open(ctx, req)
		if err != nil {
			return nil, err
		}
		c.recordDelivery(!resp.Resumed, resp.TokensAvailable, resp.TokensDelivered, resp.TokensUsed)
		return resp, nil
	case "expand":
		ids, all, err := idsArg(c.args["items"])
		if err != nil {
			return nil, err
		}
		resp, err := svc.Expand(ctx, handoff.ExpandRequest{
			Handoff: ref, SessionID: sid, ProjectPath: c.projectPath, Section: handoff.Section(strArg(c.args, "section")),
			Items: ids, All: all, Mode: strArg(c.args, "mode"), TokenBudget: budget, Next: next,
		})
		if err != nil {
			return nil, err
		}
		c.recordDelivery(false, resp.TokensAvailable, resp.TokensDelivered, resp.TokensUsed)
		return resp, nil
	}
	return nil, errUnknownAction(c.tool, c.action)
}

func (c *handoffCall) runScratchpad(ctx context.Context, svc handoff.Service) (any, error) {
	sid := handoff.SessionID(strArg(c.args, "session_id"))
	switch c.action {
	case "post":
		return svc.Post(ctx, handoff.PostRequest{
			SessionID: sid, Type: handoff.EntryType(strArg(c.args, "type")), Text: strArg(c.args, "text"), Refs: parseStringList(c.args["refs"]),
		})
	case "read":
		req := handoff.ReadRequest{
			SessionID: sid, Since: int64(intArg(c.args, "since")), Author: handoff.SessionID(strArg(c.args, "author")),
			IncludeOwn: boolArg(c.args, "include_own"), TokenBudget: intArg(c.args, "token_budget"),
		}
		for _, t := range parseStringList(c.args["types"]) {
			req.Types = append(req.Types, handoff.EntryType(t))
		}
		return svc.Read(ctx, req)
	case "retract":
		return svc.Retract(ctx, handoff.RetractRequest{SessionID: sid, Entry: int64(intArg(c.args, "entry"))})
	case "claim":
		return svc.Claim(ctx, handoff.ClaimRequest{SessionID: sid, Key: strArg(c.args, "key"), Reason: strArg(c.args, "reason")})
	case "release":
		return svc.Release(ctx, handoff.ReleaseRequest{SessionID: sid, Key: strArg(c.args, "key")})
	}
	return nil, errUnknownAction(c.tool, c.action)
}

// recordDelivery sets the call's OB-2 savings. The service reports the child's running totals,
// so logging available−delivered on every call would count the same snapshot again on each
// expand. Instead each call logs its change to that total: a newly opened child's whole
// available−delivered, and every later open or expand the tokens it delivered, negated. The
// logged rows then sum to the child's available−delivered.
func (c *handoffCall) recordDelivery(firstOpen bool, available, delivered, used int) {
	c.savings.TokensUsed = used
	if firstOpen {
		c.savings.TokensSaved = available - delivered
		return
	}
	c.savings.TokensSaved = -used
}

// touchHandoffSession records MCP activity for a tree session (FI-3); other sessions cost one
// in-memory lookup.
func touchHandoffSession(sid string) {
	if sid == "" {
		return
	}
	if svc := handoff.Default(); svc != nil {
		svc.Touch(handoff.SessionID(sid))
	}
}

// claimsGrantedNotice returns the claims_granted text for claims granted to sid since its last
// call (CL-4, CL-5), or "". Fetching the grants marks them told.
func claimsGrantedNotice(sid string) string {
	if sid == "" {
		return ""
	}
	svc := handoff.Default()
	if svc == nil || !svc.IsTreeSession(handoff.SessionID(sid)) {
		return ""
	}
	grants, err := svc.PendingGrants(handoff.SessionID(sid))
	if err != nil {
		logger.Warn("Failed to read pending claim grants", "session", sid, "error", err)
		return ""
	}
	if len(grants) == 0 {
		return ""
	}
	keys := make([]string, 0, len(grants))
	for _, g := range grants {
		keys = append(keys, g.Key)
	}
	return grantNotice(keys)
}

// grantNotice lists as many keys as fit in maxGrantNoticeTokens and counts the rest.
func grantNotice(keys []string) string {
	for n := len(keys); n > 0; n-- {
		if text := grantNoticeText(keys[:n], len(keys)-n); db.EstimateTokens(text) <= maxGrantNoticeTokens {
			return text
		}
	}
	return grantNoticeText(nil, len(keys))
}

func grantNoticeText(shown []string, more int) string {
	list := strings.Join(shown, ", ")
	if more > 0 {
		if list != "" {
			list += ", "
		}
		list += "+" + strconv.Itoa(more) + " more"
	}
	return "[claims_granted] " + list + " (scratchpad)"
}

// errActionDisabled is the FF-7 feature_disabled error, naming the flag that is off.
func errActionDisabled(tool, action string) error {
	return errs.NewCode(handoff.CodeFeatureDisabled, tool+" "+action+" is turned off by a feature flag",
		"tool", tool, "action", action, "flag", disablingFlag(tool, action))
}

func errUnknownAction(tool, action string) error {
	return errs.NewCode(errs.CodeInvalidInput, "unknown action", "tool", tool, "action", action)
}

// disablingFlag is the flag that turns off tool, or else action on it.
func disablingFlag(tool, action string) string {
	if key := flags.ToolDisabledBy(tool); key != "" {
		return key
	}
	for _, f := range flags.All() {
		if flags.Enabled(f.Key) {
			continue
		}
		for _, a := range f.Actions[tool] {
			if a == action {
				return f.Key
			}
		}
	}
	return ""
}

// pointersArg reads create's pointers: objects {key, note}, or bare key strings.
func pointersArg(raw any) []handoff.PointerInput {
	items, _ := raw.([]any)
	var out []handoff.PointerInput
	for _, it := range items {
		switch v := it.(type) {
		case string:
			if k := strings.TrimSpace(v); k != "" {
				out = append(out, handoff.PointerInput{Key: k})
			}
		case map[string]any:
			if k := strArg(v, "key"); k != "" {
				out = append(out, handoff.PointerInput{Key: k, Note: strArg(v, "note")})
			}
		}
	}
	if s, ok := raw.(string); ok {
		for _, k := range parseStringList(s) {
			out = append(out, handoff.PointerInput{Key: k})
		}
	}
	return out
}

// idsArg reads a list of numeric ids, where "all" (alone or as an element) selects everything.
func idsArg(raw any) (ids []int64, all bool, err error) {
	var items []any
	switch v := raw.(type) {
	case nil:
		return nil, false, nil
	case []any:
		items = v
	case string, float64:
		items = []any{v}
	default:
		return nil, false, errs.NewCode(errs.CodeInvalidInput, "ids must be an array of numbers or \"all\"")
	}
	for _, it := range items {
		switch v := it.(type) {
		case float64:
			ids = append(ids, int64(v))
		case string:
			s := strings.TrimSpace(v)
			if strings.EqualFold(s, actionAll) {
				all = true
				continue
			}
			n, err := strconv.ParseInt(s, 10, 64)
			if err != nil {
				return nil, false, errs.NewCode(errs.CodeInvalidInput, "ids must be numbers or \"all\"", "id", s)
			}
			ids = append(ids, n)
		}
	}
	return ids, all, nil
}

// cursorArg reads a {section, offset} paging cursor, or nil.
func cursorArg(raw any) *handoff.PageCursor {
	m, ok := raw.(map[string]any)
	if !ok {
		return nil
	}
	return &handoff.PageCursor{Section: handoff.Section(strArg(m, "section")), Offset: intArg(m, "offset")}
}

// intArg reads a JSON number (or numeric string) argument, or 0.
func intArg(m map[string]any, key string) int {
	switch v := m[key].(type) {
	case float64:
		return int(v)
	case int:
		return v
	case string:
		n, _ := strconv.Atoi(strings.TrimSpace(v))
		return n
	}
	return 0
}

// textList reads free-text list items, which unlike parseStringList are never split on commas.
func textList(raw any) []string {
	var out []string
	add := func(s string) {
		if t := strings.TrimSpace(s); t != "" {
			out = append(out, t)
		}
	}
	switch v := raw.(type) {
	case string:
		add(v)
	case []any:
		for _, it := range v {
			if s, ok := it.(string); ok {
				add(s)
			}
		}
	}
	return out
}

func prop(typ, desc string) map[string]any {
	return map[string]any{"type": typ, "description": desc}
}

func arrayProp(itemType, desc string) map[string]any {
	return map[string]any{"type": "array", "items": map[string]string{"type": itemType}, "description": desc}
}

// enumProp is a string enum; an empty desc is omitted, since the values say enough.
func enumProp(desc string, values ...string) map[string]any {
	p := map[string]any{"type": "string", "enum": values}
	if desc != "" {
		p["description"] = desc
	}
	return p
}

func pointersProp() map[string]any {
	return map[string]any{
		"type": "array",
		"items": map[string]any{
			"type":       "object",
			"properties": map[string]any{"key": map[string]string{"type": "string"}, "note": map[string]string{"type": "string"}},
			"required":   []string{"key"},
		},
		"description": "create: symbol keys or file paths to hand over",
	}
}
