package installer

import (
	"encoding/json"
	"path/filepath"
	"slices"
	"strings"

	"github.com/tailscale/hujson"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// hooksUnit owns our entries in ~/.claude/settings.json "hooks". It appends matcher groups to
// each event's array and never replaces the user's own hooks; our hooks are recognized by their
// command, "<ast-mcp> hook <event>", wherever they sit.
type hooksUnit struct {
	file string
	exe  string
}

// hookSpec is one Claude Code hook event the handoff hooks listen on (docs/spikes/claude-code-hooks.md).
type hookSpec struct {
	event   string
	matcher string
	sub     string
}

// ourHook locates one of our hook commands: group gi of an event's array, inner hook hi.
type ourHook struct {
	event   string
	group   int
	hook    int
	alone   bool // the group holds only our hooks, so the whole group is ours to remove
	groupJS []byte
}

const (
	hooksKey            = "hooks"
	hookTimeoutSeconds  = 3
	hooksDisabledReason = "enable the feature_handoff_hooks flag to install Claude Code handoff hooks"
)

var claudeHookSpecs = []hookSpec{
	{event: "SessionStart", matcher: "startup|resume|compact", sub: "session-start"},
	{event: "SubagentStart", sub: "subagent-start"},
	{event: "SubagentStop", sub: "subagent-stop"},
	{event: "PreToolUse", matcher: "Agent", sub: "pre-tool-use-agent"},
}

func (u hooksUnit) path() string {
	return u.file
}

// group is the matcher group the installer appends for spec.
func (u hooksUnit) group(spec hookSpec) jsonObj {
	hook := jsonObj{{"type", "command"}, {"command", shellQuote(u.exe) + " hook " + spec.sub}, {"timeout", hookTimeoutSeconds}}
	if spec.matcher == "" {
		return jsonObj{{"hooks", []any{hook}}}
	}
	return jsonObj{{"matcher", spec.matcher}, {"hooks", []any{hook}}}
}

func (u hooksUnit) desiredHash() (string, error) {
	var all []byte
	for _, spec := range claudeHookSpecs {
		b, err := canonicalJSON(u.group(spec))
		if err != nil {
			return "", err
		}
		all = append(append(all, spec.event...), b...)
	}
	return shortHash(all), nil
}

func (u hooksUnit) status(e *env) ComponentStatus {
	row := e.row(u.file)
	data, err := readOptional(u.file)
	if err != nil {
		return e.cs(absentStatus(row), u.file, err.Error())
	}
	var found []ourHook
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
		if row == nil && !e.s.cfg.HooksEnabled() {
			return e.cs(StatusUnsupported, u.file, hooksDisabledReason)
		}
		return e.cs(absentStatus(row), u.file, "")
	}
	ok, err := u.installed(found)
	if err != nil {
		return e.cs(StatusNotInstalled, u.file, err.Error())
	}
	if ok {
		return e.cs(StatusInstalled, u.file, "")
	}
	if row != nil && row.EntryHash == foundHash(found) {
		return e.cs(StatusOutdated, u.file, "")
	}
	if row == nil {
		return e.cs(StatusOutdated, u.file, "ast-mcp hook entries exist that do not match this version")
	}
	return e.cs(StatusModifiedByUser, u.file, "")
}

func (u hooksUnit) plan(e *env) ([]FileChange, []string, error) {
	if e.action == ActionInstall && !e.s.cfg.HooksEnabled() {
		return []FileChange{e.skip(u.file, "unsupported: "+hooksDisabledReason)}, nil, nil
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
	dh, err := u.desiredHash()
	if err != nil {
		return nil, nil, err
	}
	row := e.row(u.file)
	if e.action == ActionUninstall {
		return u.planUninstall(e, data, doc, found, row)
	}
	if ok, err := u.installed(found); err != nil || ok {
		c := e.skip(u.file, "already installed")
		c.upserts = []stateRow{e.stateRow(u.file, dh, 0)}
		return []FileChange{c}, nil, err
	}
	keep := map[string]bool{}
	for _, spec := range claudeHookSpecs {
		keep[spec.event] = true
	}
	if err := removeOurHooks(doc, found, keep); err != nil {
		return nil, nil, err
	}
	created := 0
	for _, spec := range claudeHookSpecs {
		n, ok := doc.find(hooksKey, spec.event)
		if !ok {
			createdParent, err := doc.set([]string{hooksKey, spec.event}, []any{u.group(spec)})
			if err != nil {
				return nil, nil, errs.Wrap(err, "path", u.file)
			}
			if createdParent {
				created |= createdBlock
			}
			continue
		}
		if err := doc.appendElem(n, u.group(spec)); err != nil {
			return nil, nil, errs.Wrap(err, "path", u.file, "event", spec.event)
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

func (u hooksUnit) planUninstall(e *env, data []byte, doc *jsonDoc, found []ourHook, row *stateRow) ([]FileChange, []string, error) {
	if len(found) == 0 {
		return []FileChange{e.forget(u.file, "not installed")}, nil, nil
	}
	if err := removeOurHooks(doc, found, nil); err != nil {
		return nil, nil, err
	}
	if row != nil && row.Created&createdBlock != 0 && doc.emptyObject(hooksKey) {
		if _, err := doc.remove(hooksKey); err != nil {
			return nil, nil, err
		}
	}
	c := e.change(u.file, KindRemoveBlock, data, doc.bytes())
	if row != nil && row.Created&createdFile != 0 && doc.emptyObject() {
		c.Kind, c.After = KindDelete, nil
	}
	c.deletes = []stateKey{e.stateKey(u.file)}
	return []FileChange{c}, nil, nil
}

// scan finds our hook commands in every event array.
func (u hooksUnit) scan(doc *jsonDoc) ([]ourHook, error) {
	n, ok := doc.find(hooksKey)
	if !ok {
		return nil, nil
	}
	obj, ok := n.v.Value.(*hujson.Object)
	if !ok {
		return nil, errs.NewCode(errs.CodeInvalidInput, `"hooks" is not an object`)
	}
	var out []ourHook
	for _, m := range obj.Members {
		arr, ok := m.Value.Value.(*hujson.Array)
		if !ok {
			continue
		}
		event := m.Name.Value.(hujson.Literal).String()
		for gi := range arr.Elements {
			js, err := standardJSON(arr.Elements[gi])
			if err != nil {
				return nil, err
			}
			var g struct {
				Hooks []struct {
					Command string `json:"command"`
				} `json:"hooks"`
			}
			if json.Unmarshal(js, &g) != nil {
				continue
			}
			var ours []int
			for hi, h := range g.Hooks {
				if u.isOurs(h.Command) {
					ours = append(ours, hi)
				}
			}
			for _, hi := range ours {
				out = append(out, ourHook{event: event, group: gi, hook: hi, alone: len(ours) == len(g.Hooks), groupJS: js})
			}
		}
	}
	return out, nil
}

// installed reports whether found is exactly one current group per hook event.
func (u hooksUnit) installed(found []ourHook) (bool, error) {
	if len(found) != len(claudeHookSpecs) {
		return false, nil
	}
	for _, spec := range claudeHookSpecs {
		want, err := canonicalJSON(u.group(spec))
		if err != nil {
			return false, err
		}
		match := slices.IndexFunc(found, func(h ourHook) bool { return h.event == spec.event && h.alone && string(h.groupJS) == string(want) })
		if match < 0 {
			return false, nil
		}
	}
	return true, nil
}

// isOurs recognizes "<path>/ast-mcp hook <event>", including an ast-mcp at an older path.
func (u hooksUnit) isOurs(cmd string) bool {
	f := shellFields(cmd)
	if len(f) < 2 || f[1] != "hook" {
		return false
	}
	return f[0] == u.exe || filepath.Base(f[0]) == "ast-mcp"
}

// removeOurHooks deletes our hooks: whole groups that hold only ours, otherwise just our inner
// hook. An event array left empty is removed unless keep names it (install re-adds it).
func removeOurHooks(doc *jsonDoc, found []ourHook, keep map[string]bool) error {
	byEvent := map[string][]ourHook{}
	var events []string
	for _, h := range found {
		if _, ok := byEvent[h.event]; !ok {
			events = append(events, h.event)
		}
		byEvent[h.event] = append(byEvent[h.event], h)
	}
	for _, event := range events {
		n, ok := doc.find(hooksKey, event)
		if !ok {
			continue
		}
		arr := n.v.Value.(*hujson.Array)
		hs := byEvent[event]
		// Highest indexes first so earlier indexes stay valid.
		slices.SortFunc(hs, func(a, b ourHook) int {
			if a.group != b.group {
				return b.group - a.group
			}
			return b.hook - a.hook
		})
		removedGroup := map[int]bool{}
		for _, h := range hs {
			if removedGroup[h.group] {
				continue
			}
			if h.alone {
				removeElem(arr, h.group)
				removedGroup[h.group] = true
				continue
			}
			if err := removeInnerHook(&arr.Elements[h.group], h.hook); err != nil {
				return err
			}
		}
		if len(arr.Elements) == 0 && !keep[event] {
			if _, err := doc.remove(hooksKey, event); err != nil {
				return err
			}
		}
	}
	return nil
}

func removeInnerHook(group *hujson.Value, idx int) error {
	obj, ok := group.Value.(*hujson.Object)
	if !ok {
		return errs.NewCode(errs.CodeInvalidInput, "hook group is not an object")
	}
	i := memberIndex(obj, hooksKey)
	if i < 0 {
		return errs.NewCode(errs.CodeInvalidInput, "hook group has no hooks array")
	}
	arr, ok := obj.Members[i].Value.Value.(*hujson.Array)
	if !ok || idx >= len(arr.Elements) {
		return errs.NewCode(errs.CodeInvalidInput, "hook group has no hooks array")
	}
	removeElem(arr, idx)
	return nil
}

// foundHash fingerprints our hook groups as found, for telling Outdated from user-edited.
func foundHash(found []ourHook) string {
	var all []byte
	for _, h := range found {
		all = append(append(all, h.event...), h.groupJS...)
	}
	return shortHash(all)
}

// shellQuote single-quotes s when it holds characters a POSIX shell would interpret.
func shellQuote(s string) string {
	safe := s != "" && strings.IndexFunc(s, func(r rune) bool {
		return !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || strings.ContainsRune("/._-+:@%,=", r))
	}) < 0
	if safe {
		return s
	}
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}

// shellFields splits a command line on spaces, honoring single and double quotes.
func shellFields(s string) []string {
	var out []string
	var cur strings.Builder
	inField := false
	var quote rune
	for _, r := range s {
		switch {
		case quote != 0 && r == quote:
			quote = 0
		case quote != 0:
			cur.WriteRune(r)
		case r == '\'' || r == '"':
			quote, inField = r, true
		case r == ' ' || r == '\t':
			if inField {
				out = append(out, cur.String())
				cur.Reset()
				inField = false
			}
		default:
			cur.WriteRune(r)
			inField = true
		}
	}
	if inField {
		out = append(out, cur.String())
	}
	return out
}
