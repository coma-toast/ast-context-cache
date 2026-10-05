package installer

import (
	"strings"
)

// mdBlockUnit owns our marker block in a markdown file (IN-10). With frontmatter set the unit
// owns a whole file (a Cursor .mdc rule): it creates frontmatter plus block, and a same-named
// file without our block is someone else's.
type mdBlockUnit struct {
	file        string
	body        string
	frontmatter string
	warnings    []string
}

func (u mdBlockUnit) path() string {
	return u.file
}

func (u mdBlockUnit) status(e *env) ComponentStatus {
	row := e.row(u.file)
	data, err := readOptional(u.file)
	if err != nil {
		return e.cs(absentStatus(row), u.file, err.Error())
	}
	if data == nil {
		return e.cs(absentStatus(row), u.file, "")
	}
	b, found, err := findBlock(string(data))
	switch {
	case err != nil:
		return e.cs(StatusModifiedByUser, u.file, err.Error())
	case !found && u.frontmatter != "":
		return e.cs(StatusExternallyManaged, u.file, e.s.displayPath(u.file)+" exists and was not created by the installer")
	case !found:
		return e.cs(absentStatus(row), u.file, "")
	}
	return e.cs(blockStatus(b, u.body, currentVersion()), u.file, "")
}

func (u mdBlockUnit) plan(e *env) ([]FileChange, []string, error) {
	data, err := readOptional(u.file)
	if err != nil {
		return nil, nil, err
	}
	if data == nil && e.action == ActionUninstall {
		return []FileChange{e.forget(u.file, "not installed")}, nil, nil
	}
	text := string(data)
	b, found, err := findBlock(text)
	if err != nil {
		return nil, nil, err
	}
	if e.action == ActionUninstall {
		return u.planUninstall(e, data, found)
	}
	nl := newlineOf(text)
	block := renderBlock(currentVersion(), u.body, nl)
	hash := blockHash(u.body)
	switch {
	case data == nil:
		c := e.change(u.file, KindCreate, nil, []byte(u.wholeFile(block, nl)))
		c.upserts = []stateRow{e.stateRow(u.file, hash, createdFile)}
		return []FileChange{c}, u.warnings, nil
	case found && blockStatus(b, u.body, currentVersion()) == StatusInstalled:
		c := e.skip(u.file, "already installed")
		c.upserts = []stateRow{e.stateRow(u.file, hash, 0)}
		return []FileChange{c}, nil, nil
	case !found && u.frontmatter != "" && !e.replaceExternal:
		return []FileChange{e.skip(u.file, "externally managed: "+e.s.displayPath(u.file)+" exists and was not created by the installer; replace it explicitly to overwrite (a backup is taken first)")}, nil, nil
	case !found && u.frontmatter != "":
		c := e.change(u.file, KindModify, data, []byte(u.wholeFile(block, nl)))
		c.upserts = []stateRow{e.stateRow(u.file, hash, 0)}
		return []FileChange{c}, append(append([]string(nil), u.warnings...), "Replaces "+e.s.displayPath(u.file)+" (a backup is taken first)"), nil
	}
	after, err := upsertBlock(text, block, nl)
	if err != nil {
		return nil, nil, err
	}
	c := e.change(u.file, KindModify, data, []byte(after))
	c.upserts = []stateRow{e.stateRow(u.file, hash, 0)}
	warnings := append([]string(nil), u.warnings...)
	if found && blockStatus(b, u.body, currentVersion()) == StatusModifiedByUser {
		warnings = append(warnings, "Replaces edits made inside the ast-context-cache block in "+e.s.displayPath(u.file)+" (a backup is taken first)")
	}
	return []FileChange{c}, warnings, nil
}

func (u mdBlockUnit) planUninstall(e *env, data []byte, found bool) ([]FileChange, []string, error) {
	if !found {
		return []FileChange{e.forget(u.file, "not installed")}, nil, nil
	}
	after, _, err := removeBlock(string(data))
	if err != nil {
		return nil, nil, err
	}
	c := e.change(u.file, KindRemoveBlock, data, []byte(after))
	row := e.row(u.file)
	rest := strings.TrimSpace(strings.TrimPrefix(after, u.frontmatter))
	if row != nil && row.Created&createdFile != 0 && rest == "" {
		c.Kind, c.After = KindDelete, nil
	}
	c.deletes = []stateKey{e.stateKey(u.file)}
	return []FileChange{c}, nil, nil
}

// wholeFile is a new file's content: the frontmatter (if any) then the block.
func (u mdBlockUnit) wholeFile(block, nl string) string {
	if u.frontmatter == "" {
		return block + nl
	}
	return strings.ReplaceAll(u.frontmatter, "\n", nl) + nl + block + nl
}
