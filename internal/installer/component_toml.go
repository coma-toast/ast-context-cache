package installer

import (
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// tomlTableUnit owns the [mcp_servers.ast-context-cache] table in a TOML config (Codex).
type tomlTableUnit struct {
	file   string
	parent string
	url    string
}

func (u tomlTableUnit) path() string {
	return u.file
}

func (u tomlTableUnit) key() string {
	return u.parent + "." + serverName
}

// block is the table the installer writes, preceded by its managed-by comment.
func (u tomlTableUnit) block(nl string) string {
	s := tomlManagedPrefix + " v" + currentVersion() + "\n[" + u.key() + "]\nurl = " + tomlString(u.url) + "\n"
	return strings.ReplaceAll(s, "\n", nl)
}

func (u tomlTableUnit) desiredHash() (string, error) {
	b, err := canonicalJSON(map[string]any{"url": u.url})
	return shortHash(b), err
}

func (u tomlTableUnit) status(e *env) ComponentStatus {
	row := e.row(u.file)
	data, err := readOptional(u.file)
	if err != nil {
		return e.cs(absentStatus(row), u.file, err.Error())
	}
	if data == nil {
		return e.cs(absentStatus(row), u.file, "")
	}
	doc, err := decodeTOML(string(data))
	if err != nil {
		return e.cs(absentStatus(row), u.file, e.s.displayPath(u.file)+" could not be parsed: "+err.Error())
	}
	tbl, ok := tomlTable(doc, u.parent, serverName)
	if !ok {
		return e.cs(absentStatus(row), u.file, "")
	}
	cur, err := canonicalJSON(tbl)
	if err != nil {
		return e.cs(StatusNotInstalled, u.file, err.Error())
	}
	dh, err := u.desiredHash()
	if err != nil {
		return e.cs(StatusNotInstalled, u.file, err.Error())
	}
	st := entryStatus(shortHash(cur), dh, row)
	reason := ""
	if st == StatusModifiedByUser && row == nil {
		reason = "an ast-context-cache table exists that the installer did not write"
	}
	return e.cs(st, u.file, reason)
}

func (u tomlTableUnit) plan(e *env) ([]FileChange, []string, error) {
	row := e.row(u.file)
	data, err := readOptional(u.file)
	if err != nil {
		return nil, nil, err
	}
	if data == nil && e.action == ActionUninstall {
		return []FileChange{e.forget(u.file, "not installed")}, nil, nil
	}
	text := string(data)
	doc, err := decodeTOML(text)
	if err != nil {
		return nil, nil, errs.WrapCodeMessage(errs.CodeInvalidInput, e.s.displayPath(u.file)+" could not be parsed, so it was left unchanged", err, "path", u.file)
	}
	tbl, present := tomlTable(doc, u.parent, serverName)
	_, hasHeader := findTOMLTable(text, u.key())
	if present && !hasHeader {
		return nil, nil, errs.NewCode(errs.CodeConflict, "ast-context-cache is defined with dotted keys or an inline table in "+e.s.displayPath(u.file)+"; edit it by hand", "path", u.file)
	}
	dh, err := u.desiredHash()
	if err != nil {
		return nil, nil, err
	}
	if e.action == ActionUninstall {
		return u.planUninstall(e, data, row)
	}
	if present {
		cur, err := canonicalJSON(tbl)
		if err != nil {
			return nil, nil, err
		}
		if shortHash(cur) == dh {
			c := e.skip(u.file, "already installed")
			c.upserts = []stateRow{e.stateRow(u.file, dh, 0)}
			return []FileChange{c}, nil, nil
		}
	}
	after := upsertTOMLTable(text, u.key(), u.block(newlineOf(text)))
	if _, err := decodeTOML(after); err != nil {
		return nil, nil, errs.WrapCodeMessage(errs.CodeInternal, "edited TOML failed validation", err, "path", u.file)
	}
	kind, created := KindModify, 0
	if data == nil {
		kind, created = KindCreate, createdFile
	}
	c := e.change(u.file, kind, data, []byte(after))
	c.upserts = []stateRow{e.stateRow(u.file, dh, created)}
	return []FileChange{c}, nil, nil
}

func (u tomlTableUnit) planUninstall(e *env, data []byte, row *stateRow) ([]FileChange, []string, error) {
	after, removed := removeTOMLTable(string(data), u.key())
	if !removed {
		return []FileChange{e.forget(u.file, "not installed")}, nil, nil
	}
	if _, err := decodeTOML(after); err != nil {
		return nil, nil, errs.WrapCodeMessage(errs.CodeInternal, "edited TOML failed validation", err, "path", u.file)
	}
	c := e.change(u.file, KindRemoveBlock, data, []byte(after))
	if row != nil && row.Created&createdFile != 0 && strings.TrimSpace(after) == "" {
		c.Kind, c.After = KindDelete, nil
	}
	c.deletes = []stateKey{e.stateKey(u.file)}
	return []FileChange{c}, nil, nil
}
