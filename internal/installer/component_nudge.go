package installer

import (
	_ "embed"
	"encoding/json"
	"slices"
	"strings"

	"github.com/tailscale/hujson"

	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/hooks"
)

// The search-first nudge outside Claude Code: an OpenCode plugin file and a Cursor preToolUse
// hook. Both are gated by feature_search_nudge and also read it at run time from the server's
// /health, so the dashboard toggle takes effect without reinstalling.

//go:embed assets/opencode-search-nudge.js
var openCodeNudgeTemplate string

const (
	openCodeNudgeFile  = "ast-context-cache-search-nudge.js"
	cursorHookEvent    = "preToolUse"
	cursorNudgeMatcher = "Read|Grep|Glob|MCP:.*"
	cursorHooksVersion = 1
)

// openCodeNudgePlugin renders the plugin with the shared nudge wording and this install's
// /health URL.
func openCodeNudgePlugin(mcpURL string) ([]byte, error) {
	vals := []struct {
		key string
		val any
	}{
		{"__HEALTH_URL__", hooks.HealthURL(mcpURL)},
		{"__MESSAGES__", hooks.NudgeMessages},
		{"__CODE_EXTENSIONS__", hooks.CodeExtensions},
		{"__AST_TOOLS__", hooks.ASTToolNames},
	}
	out := openCodeNudgeTemplate
	for _, v := range vals {
		b, err := marshalNoEscape(v.val)
		if err != nil {
			return nil, errs.WrapMessage("failed to render OpenCode nudge plugin", err, "placeholder", v.key)
		}
		out = strings.Replace(out, v.key, string(b), 1)
	}
	return []byte(out), nil
}

// nudgePluginUnit owns one whole file the installer creates and deletes: the OpenCode plugin.
type nudgePluginUnit struct {
	file    string
	content []byte
}

func (u nudgePluginUnit) path() string {
	return u.file
}

func (u nudgePluginUnit) disabledReason(cfg Config) (string, bool) {
	return nudgeDisabledReason, !cfg.SearchNudgeEnabled()
}

func (u nudgePluginUnit) status(e *env) ComponentStatus {
	row := e.row(u.file)
	data, err := readOptional(u.file)
	switch {
	case err != nil:
		return e.cs(absentStatus(row), u.file, err.Error())
	case data == nil && row == nil && !e.s.cfg.SearchNudgeEnabled():
		return e.cs(StatusUnsupported, u.file, nudgeDisabledReason)
	case data == nil:
		return e.cs(absentStatus(row), u.file, "")
	case string(data) == string(u.content):
		return e.cs(StatusInstalled, u.file, "")
	case row == nil:
		return e.cs(StatusExternallyManaged, u.file, e.s.displayPath(u.file)+" exists and was not created by the installer")
	case shortHash(data) == row.EntryHash:
		return e.cs(StatusOutdated, u.file, "")
	}
	return e.cs(StatusModifiedByUser, u.file, "")
}

func (u nudgePluginUnit) plan(e *env) ([]FileChange, []string, error) {
	data, err := readOptional(u.file)
	if err != nil {
		return nil, nil, err
	}
	row := e.row(u.file)
	if e.action == ActionUninstall {
		switch {
		case row == nil || data == nil:
			return []FileChange{e.forget(u.file, "not installed")}, nil, nil
		case row.Created&createdFile == 0:
			return []FileChange{e.forget(u.file, "not created by the installer; left in place")}, nil, nil
		}
		c := e.change(u.file, KindDelete, data, nil)
		c.deletes = []stateKey{e.stateKey(u.file)}
		if shortHash(data) != row.EntryHash {
			return []FileChange{c}, []string{"Deletes edits made to " + e.s.displayPath(u.file) + " (a backup is taken first)"}, nil
		}
		return []FileChange{c}, nil, nil
	}
	if !e.s.cfg.SearchNudgeEnabled() {
		return []FileChange{e.skip(u.file, "unsupported: "+nudgeDisabledReason)}, nil, nil
	}
	hash, created := shortHash(u.content), 0
	if row != nil {
		created = row.Created
	}
	switch {
	case data == nil:
		c := e.change(u.file, KindCreate, nil, u.content)
		c.upserts = []stateRow{e.stateRow(u.file, hash, createdFile)}
		return []FileChange{c}, nil, nil
	case string(data) == string(u.content):
		c := e.skip(u.file, "already installed")
		c.upserts = []stateRow{e.stateRow(u.file, hash, created)}
		return []FileChange{c}, nil, nil
	case row == nil && !e.replaceExternal:
		return []FileChange{e.skip(u.file, "externally managed: "+e.s.displayPath(u.file)+" exists and was not created by the installer")}, nil, nil
	}
	c := e.change(u.file, KindModify, data, u.content)
	c.upserts = []stateRow{e.stateRow(u.file, hash, created)}
	if row != nil && shortHash(data) != row.EntryHash {
		return []FileChange{c}, []string{"Replaces edits to " + e.s.displayPath(u.file) + " (a backup is taken first)"}, nil
	}
	return []FileChange{c}, nil, nil
}

// cursorHooksUnit owns our entry in ~/.cursor/hooks.json "hooks.preToolUse": one
// "<ast-mcp> hook cursor-search-nudge" definition, appended beside the user's own hooks.
type cursorHooksUnit struct {
	file string
	exe  string
}

// cursorHook is one of our definitions in the preToolUse array.
type cursorHook struct {
	idx int
	js  []byte
}

func (u cursorHooksUnit) path() string {
	return u.file
}

func (u cursorHooksUnit) disabledReason(cfg Config) (string, bool) {
	return nudgeDisabledReason, !cfg.SearchNudgeEnabled()
}

func (u cursorHooksUnit) entry() jsonObj {
	return jsonObj{
		{"command", shellQuote(u.exe) + " hook " + hooks.EventCursorSearchNudge},
		{"matcher", cursorNudgeMatcher},
		{"timeout", hookTimeoutSeconds},
	}
}

func (u cursorHooksUnit) desired() ([]byte, error) {
	return canonicalJSON(u.entry())
}

func (u cursorHooksUnit) status(e *env) ComponentStatus {
	row := e.row(u.file)
	data, err := readOptional(u.file)
	if err != nil {
		return e.cs(absentStatus(row), u.file, err.Error())
	}
	var found []cursorHook
	if data != nil {
		doc, err := parseJSONDoc(data)
		if err != nil {
			return e.cs(absentStatus(row), u.file, e.s.displayPath(u.file)+" could not be parsed: "+err.Error())
		}
		if found, err = u.scan(doc); err != nil {
			return e.cs(absentStatus(row), u.file, err.Error())
		}
	}
	if len(found) == 0 {
		if row == nil && !e.s.cfg.SearchNudgeEnabled() {
			return e.cs(StatusUnsupported, u.file, nudgeDisabledReason)
		}
		return e.cs(absentStatus(row), u.file, "")
	}
	want, err := u.desired()
	if err != nil {
		return e.cs(StatusNotInstalled, u.file, err.Error())
	}
	switch {
	case len(found) == 1 && string(found[0].js) == string(want):
		return e.cs(StatusInstalled, u.file, "")
	case row != nil && row.EntryHash == shortHash(found[0].js):
		return e.cs(StatusOutdated, u.file, "")
	case row == nil:
		return e.cs(StatusOutdated, u.file, "ast-mcp hook entries exist that do not match this version")
	}
	return e.cs(StatusModifiedByUser, u.file, "")
}

func (u cursorHooksUnit) plan(e *env) ([]FileChange, []string, error) {
	if e.action == ActionInstall && !e.s.cfg.SearchNudgeEnabled() {
		return []FileChange{e.skip(u.file, "unsupported: "+nudgeDisabledReason)}, nil, nil
	}
	data, err := readOptional(u.file)
	if err != nil {
		return nil, nil, err
	}
	if data == nil && e.action == ActionUninstall {
		return []FileChange{e.forget(u.file, "not installed")}, nil, nil
	}
	doc, err := parseJSONDoc(data)
	if err != nil {
		return nil, nil, errs.WrapCodeMessage(errs.CodeInvalidInput, e.s.displayPath(u.file)+" could not be parsed, so it was left unchanged", err, "path", u.file)
	}
	found, err := u.scan(doc)
	if err != nil {
		return nil, nil, err
	}
	row := e.row(u.file)
	if e.action == ActionUninstall {
		return u.planUninstall(e, data, doc, found, row)
	}
	want, err := u.desired()
	if err != nil {
		return nil, nil, err
	}
	dh := shortHash(want)
	if len(found) == 1 && string(found[0].js) == string(want) {
		c := e.skip(u.file, "already installed")
		c.upserts = []stateRow{e.stateRow(u.file, dh, 0)}
		if row != nil {
			c.upserts[0].Created = row.Created
		}
		return []FileChange{c}, nil, nil
	}
	if err := u.removeOurs(doc, found); err != nil {
		return nil, nil, err
	}
	created := 0
	if row != nil {
		created = row.Created
	}
	if _, ok := doc.find("version"); !ok {
		if _, err := doc.set([]string{"version"}, cursorHooksVersion); err != nil {
			return nil, nil, errs.Wrap(err, "path", u.file)
		}
	}
	if n, ok := doc.find(hooksKey, cursorHookEvent); ok {
		if err := doc.appendElem(n, u.entry()); err != nil {
			return nil, nil, errs.Wrap(err, "path", u.file)
		}
	} else {
		createdParent, err := doc.set([]string{hooksKey, cursorHookEvent}, []any{u.entry()})
		if err != nil {
			return nil, nil, errs.Wrap(err, "path", u.file)
		}
		if createdParent {
			created |= createdBlock
		}
	}
	kind := KindModify
	if data == nil {
		kind, created = KindCreate, created|createdFile
	}
	c := e.change(u.file, kind, data, doc.bytes())
	c.upserts = []stateRow{e.stateRow(u.file, dh, created)}
	return []FileChange{c}, nil, nil
}

func (u cursorHooksUnit) planUninstall(e *env, data []byte, doc *jsonDoc, found []cursorHook, row *stateRow) ([]FileChange, []string, error) {
	if len(found) == 0 {
		return []FileChange{e.forget(u.file, "not installed")}, nil, nil
	}
	if err := u.removeOurs(doc, found); err != nil {
		return nil, nil, err
	}
	if row != nil && row.Created&createdBlock != 0 && doc.emptyObject(hooksKey) {
		if _, err := doc.remove(hooksKey); err != nil {
			return nil, nil, err
		}
	}
	c := e.change(u.file, KindRemoveBlock, data, doc.bytes())
	if row != nil && row.Created&createdFile != 0 && onlyVersionLeft(doc) {
		c.Kind, c.After = KindDelete, nil
	}
	c.deletes = []stateKey{e.stateKey(u.file)}
	return []FileChange{c}, nil, nil
}

// scan finds our definitions in hooks.preToolUse.
func (u cursorHooksUnit) scan(doc *jsonDoc) ([]cursorHook, error) {
	n, ok := doc.find(hooksKey, cursorHookEvent)
	if !ok {
		return nil, nil
	}
	arr, ok := n.v.Value.(*hujson.Array)
	if !ok {
		return nil, errs.NewCode(errs.CodeInvalidInput, `"hooks.preToolUse" is not an array`)
	}
	var out []cursorHook
	for i := range arr.Elements {
		js, err := standardJSON(arr.Elements[i])
		if err != nil {
			return nil, err
		}
		var h struct {
			Command string `json:"command"`
		}
		if json.Unmarshal(js, &h) == nil && isOurHookCommand(u.exe, h.Command) {
			out = append(out, cursorHook{idx: i, js: js})
		}
	}
	return out, nil
}

// removeOurs deletes our definitions, and the preToolUse array when that leaves it empty.
func (u cursorHooksUnit) removeOurs(doc *jsonDoc, found []cursorHook) error {
	if len(found) == 0 {
		return nil
	}
	n, ok := doc.find(hooksKey, cursorHookEvent)
	if !ok {
		return nil
	}
	arr := n.v.Value.(*hujson.Array)
	idx := make([]int, 0, len(found))
	for _, h := range found {
		idx = append(idx, h.idx)
	}
	// Highest indexes first so earlier indexes stay valid.
	slices.Sort(idx)
	for _, i := range slices.Backward(idx) {
		removeElem(arr, i)
	}
	if len(arr.Elements) == 0 {
		if _, err := doc.remove(hooksKey, cursorHookEvent); err != nil {
			return err
		}
	}
	return nil
}

// onlyVersionLeft reports whether a hooks.json holds nothing but its "version".
func onlyVersionLeft(doc *jsonDoc) bool {
	obj, ok := doc.root.Value.(*hujson.Object)
	if !ok || hasComment(obj.AfterExtra) || hasComment(doc.root.BeforeExtra) || hasComment(doc.root.AfterExtra) {
		return false
	}
	return len(obj.Members) == 0 || len(obj.Members) == 1 && memberIndex(obj, "version") == 0
}
