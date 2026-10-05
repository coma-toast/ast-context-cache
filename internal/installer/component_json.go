package installer

import (
	"github.com/tailscale/hujson"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// jsonEntryUnit owns one server entry, <block>.ast-context-cache, in a JSON or JSONC config.
type jsonEntryUnit struct {
	file  string
	block string
	entry jsonObj
	// manual is appended to a parse error: the steps to register the server by hand.
	manual string
	// warnings are shown whenever the plan writes this file.
	warnings []string
}

func (u jsonEntryUnit) path() string {
	return u.file
}

func (u jsonEntryUnit) status(e *env) ComponentStatus {
	row := e.row(u.file)
	data, err := readOptional(u.file)
	if err != nil {
		return e.cs(absentStatus(row), u.file, err.Error())
	}
	if data == nil {
		return e.cs(absentStatus(row), u.file, "")
	}
	doc, err := parseJSONDoc(data)
	if err != nil {
		return e.cs(absentStatus(row), u.file, u.parseReason(e, err))
	}
	cur, ok, err := doc.standard(u.block, serverName)
	if err != nil || !ok {
		return e.cs(absentStatus(row), u.file, "")
	}
	desired, err := canonicalJSON(u.entry)
	if err != nil {
		return e.cs(StatusNotInstalled, u.file, err.Error())
	}
	st := entryStatus(shortHash(cur), shortHash(desired), row)
	reason := ""
	if st == StatusModifiedByUser && row == nil {
		reason = "an ast-context-cache entry exists that the installer did not write"
	}
	return e.cs(st, u.file, reason)
}

func (u jsonEntryUnit) plan(e *env) ([]FileChange, []string, error) {
	data, err := readOptional(u.file)
	if err != nil {
		return nil, nil, err
	}
	if data == nil && e.action == ActionUninstall {
		return []FileChange{e.forget(u.file, "not installed")}, nil, nil
	}
	doc, err := parseJSONDoc(data)
	if err != nil {
		return nil, nil, errs.WrapCodeMessage(errs.CodeInvalidInput, u.parseReason(e, err), err, "path", u.file)
	}
	if e.action == ActionUninstall {
		return u.planUninstall(e, data, doc)
	}
	return u.planInstall(e, data, doc)
}

func (u jsonEntryUnit) planInstall(e *env, data []byte, doc *jsonDoc) ([]FileChange, []string, error) {
	row := e.row(u.file)
	desired, err := canonicalJSON(u.entry)
	if err != nil {
		return nil, nil, err
	}
	dh := shortHash(desired)
	cur, present, err := doc.standard(u.block, serverName)
	if err != nil {
		return nil, nil, err
	}
	if present && shortHash(cur) == dh {
		c := e.skip(u.file, "already installed")
		c.upserts = []stateRow{e.stateRow(u.file, dh, 0)}
		return []FileChange{c}, nil, nil
	}
	createdParent, err := doc.set([]string{u.block, serverName}, u.entry)
	if err != nil {
		return nil, nil, errs.Wrap(err, "path", u.file)
	}
	kind, created := KindModify, 0
	if data == nil {
		kind, created = KindCreate, createdFile
	}
	if createdParent {
		created |= createdBlock
	}
	c := e.change(u.file, kind, data, doc.bytes())
	c.upserts = []stateRow{e.stateRow(u.file, dh, created)}
	warnings := append([]string(nil), u.warnings...)
	if present && entryStatus(shortHash(cur), dh, row) == StatusModifiedByUser {
		warnings = append(warnings, "Replaces an ast-context-cache entry edited outside the installer in "+e.s.displayPath(u.file)+" (a backup is taken first)")
	}
	return []FileChange{c}, warnings, nil
}

func (u jsonEntryUnit) planUninstall(e *env, data []byte, doc *jsonDoc) ([]FileChange, []string, error) {
	row := e.row(u.file)
	removed, err := doc.remove(u.block, serverName)
	if err != nil {
		return nil, nil, errs.Wrap(err, "path", u.file)
	}
	if !removed {
		return []FileChange{e.forget(u.file, "not installed")}, nil, nil
	}
	if row != nil && row.Created&createdBlock != 0 && doc.emptyObject(u.block) {
		if _, err := doc.remove(u.block); err != nil {
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

func (u jsonEntryUnit) parseReason(e *env, err error) string {
	msg := e.s.displayPath(u.file) + " could not be parsed, so it was left unchanged: " + err.Error()
	if u.manual != "" {
		msg += ". " + u.manual
	}
	return msg
}

// emptyObject reports whether the object at path has no members and no comments, so removing
// it (or deleting a file we created) loses nothing.
func (d *jsonDoc) emptyObject(path ...string) bool {
	n, ok := d.find(path...)
	if !ok {
		return false
	}
	obj, ok := n.v.Value.(*hujson.Object)
	if !ok || len(obj.Members) > 0 || hasComment(obj.AfterExtra) {
		return false
	}
	if len(path) == 0 {
		return !hasComment(d.root.BeforeExtra) && !hasComment(d.root.AfterExtra)
	}
	return true
}
