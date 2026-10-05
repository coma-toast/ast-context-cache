package installer

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/coma-toast/ast-context-cache/skills"
)

// skillsUnit owns <root>/ast-context-cache-<name>/SKILL.md for every canonical skill.
type skillsUnit struct {
	root string
	// coverRoots are skill directories the host also loads (including root itself for hosts
	// whose install root is shared). When our skills are already in one, the component is
	// Covered and nothing is written, so the host doesn't load duplicates.
	coverRoots []string
	host       string
}

type skillFile struct {
	path    string
	content []byte
}

// externalPath is a path in root that blocks installing: a symlink resolving outside
// ~/.astcache, or the legacy un-suffixed ast-context-cache skill directory.
type externalPath struct {
	path string
	link string
}

func (u skillsUnit) path() string {
	return u.root
}

func (u skillsUnit) files() ([]skillFile, error) {
	all, err := skills.All()
	if err != nil {
		return nil, err
	}
	out := make([]skillFile, 0, len(all))
	for _, sk := range all {
		out = append(out, skillFile{path: filepath.Join(u.root, sk.DirName(), "SKILL.md"), content: []byte(sk.Content())})
	}
	return out, nil
}

func (u skillsUnit) status(e *env) ComponentStatus {
	files, err := u.files()
	if err != nil {
		return e.cs(StatusNotInstalled, u.root, err.Error())
	}
	owned := u.ownedRows(e, files)
	if !owned {
		if p := u.coveredBy(e); p != "" {
			return e.cs(StatusCovered, u.root, u.coveredReason(e, p))
		}
		if ext := u.externals(e); len(ext) > 0 {
			return e.cs(StatusExternallyManaged, ext[0].path, u.externalReason(e, ext))
		}
	}
	var statuses []Status
	for _, f := range files {
		statuses = append(statuses, u.fileStatus(e, f))
	}
	st := foldStatus(statuses)
	reason := ""
	if st == StatusExternallyManaged {
		reason = "skill files exist that the installer did not create"
	}
	return e.cs(st, u.root, reason)
}

func (u skillsUnit) plan(e *env) ([]FileChange, []string, error) {
	files, err := u.files()
	if err != nil {
		return nil, nil, err
	}
	if e.action == ActionUninstall {
		return u.planUninstall(e, files)
	}
	owned := u.ownedRows(e, files)
	if !owned {
		if p := u.coveredBy(e); p != "" {
			return []FileChange{e.skip(u.root, "covered: "+u.coveredReason(e, p))}, nil, nil
		}
	}
	var changes []FileChange
	var warnings []string
	linked := map[string]bool{}
	if ext := u.externals(e); len(ext) > 0 {
		if !e.replaceExternal {
			return []FileChange{e.skip(ext[0].path, "externally managed: "+u.externalReason(e, ext)+"; replace it explicitly to install (a backup is taken first)")}, nil, nil
		}
		for _, x := range ext {
			if x.link == "" {
				changes = append(changes, e.skip(x.path, "not a symlink; remove "+e.s.displayPath(x.path)+" by hand to avoid duplicate skills"))
				warnings = append(warnings, e.s.displayPath(x.path)+" stays in place and may duplicate the installed skills")
				continue
			}
			c := e.change(x.path, KindDelete, nil, nil)
			c.linkTarget, c.BeforeHash = x.link, symlinkHashPrefix+x.link
			c.Diff = "symlink " + x.path + " -> " + x.link + " (externally managed) will be removed; the backup records the link\n"
			changes = append(changes, c)
			linked[x.path] = true
		}
	}
	for _, f := range files {
		c, warn := u.planFile(e, f, linked[filepath.Dir(f.path)])
		changes = append(changes, c)
		warnings = append(warnings, warn...)
	}
	return changes, warnings, nil
}

// planFile plans one SKILL.md. viaLink means its directory is an external symlink an earlier
// change deletes, so the file is created fresh.
func (u skillsUnit) planFile(e *env, f skillFile, viaLink bool) (FileChange, []string) {
	hash := shortHash(f.content)
	if viaLink {
		c := e.change(f.path, KindCreate, nil, f.content)
		c.viaLink = true
		c.upserts = []stateRow{e.stateRow(f.path, hash, createdFile)}
		return c, nil
	}
	data, err := readOptional(f.path)
	if err != nil {
		return e.skip(f.path, err.Error()), nil
	}
	row := u.anyRow(e, f.path)
	switch {
	case data == nil:
		c := e.change(f.path, KindCreate, nil, f.content)
		c.upserts = []stateRow{e.stateRow(f.path, hash, createdFile)}
		return c, nil
	case string(data) == string(f.content):
		c := e.skip(f.path, "already installed")
		created := 0
		if row != nil {
			created = row.Created
		}
		c.upserts = []stateRow{e.stateRow(f.path, hash, created)}
		return c, nil
	case row == nil && !e.replaceExternal:
		return e.skip(f.path, "externally managed: "+e.s.displayPath(f.path)+" exists and was not created by the installer"), nil
	}
	c := e.change(f.path, KindModify, data, f.content)
	created := 0
	if row != nil {
		created = row.Created
	}
	c.upserts = []stateRow{e.stateRow(f.path, hash, created)}
	if row != nil && shortHash(data) != row.EntryHash {
		return c, []string{"Replaces edits to " + e.s.displayPath(f.path) + " (a backup is taken first)"}
	}
	return c, nil
}

func (u skillsUnit) planUninstall(e *env, files []skillFile) ([]FileChange, []string, error) {
	var changes []FileChange
	var warnings []string
	for _, f := range files {
		row := e.row(f.path)
		if row == nil {
			continue
		}
		if others := e.state.otherOwners(e.target, f.path); len(others) > 0 {
			c := e.forget(f.path, "still installed for "+joinTargets(others))
			changes = append(changes, c)
			continue
		}
		data, err := readOptional(f.path)
		if err != nil {
			return nil, nil, err
		}
		switch {
		case data == nil:
			changes = append(changes, e.forget(f.path, "already absent"))
		case row.Created&createdFile == 0:
			changes = append(changes, e.forget(f.path, "not created by the installer; left in place"))
		default:
			c := e.change(f.path, KindDelete, data, nil)
			c.deletes = []stateKey{e.stateKey(f.path)}
			changes = append(changes, c)
			if shortHash(data) != row.EntryHash {
				warnings = append(warnings, "Deletes edits made to "+e.s.displayPath(f.path)+" (a backup is taken first)")
			}
		}
	}
	if len(changes) == 0 {
		return []FileChange{e.skip(u.root, "not installed")}, nil, nil
	}
	return changes, warnings, nil
}

// fileStatus classifies one installed skill file.
func (u skillsUnit) fileStatus(e *env, f skillFile) Status {
	row := u.anyRow(e, f.path)
	data, err := readOptional(f.path)
	switch {
	case err != nil || data == nil:
		return absentStatus(e.row(f.path))
	case string(data) == string(f.content):
		return StatusInstalled
	case row == nil:
		return StatusExternallyManaged
	case shortHash(data) == row.EntryHash:
		return StatusOutdated
	default:
		return StatusModifiedByUser
	}
}

// anyRow returns this target's row for path, else any target's (skills in ~/.agents/skills are
// shared by Cursor, OpenCode, and Codex).
func (u skillsUnit) anyRow(e *env, path string) *stateRow {
	if r := e.row(path); r != nil {
		return r
	}
	if rows := e.state.byPath[path]; len(rows) > 0 {
		r := rows[0]
		return &r
	}
	return nil
}

// ownedRows reports whether this target recorded any of the skill files.
func (u skillsUnit) ownedRows(e *env, files []skillFile) bool {
	for _, f := range files {
		if e.row(f.path) != nil {
			return true
		}
	}
	return false
}

// coveredBy returns the first existing skill path, in a directory the host also loads, that
// already provides our skills.
func (u skillsUnit) coveredBy(e *env) string {
	names := []string{skills.DirPrefix}
	all, _ := skills.All()
	for _, sk := range all {
		names = append(names, sk.DirName())
	}
	for _, root := range u.coverRoots {
		for _, n := range names {
			p := filepath.Join(root, n)
			if _, err := os.Lstat(p); err == nil {
				return p
			}
		}
	}
	return ""
}

func (u skillsUnit) coveredReason(e *env, p string) string {
	return u.host + " already loads ast-context-cache skills from " + e.s.displayPath(p)
}

// externals lists paths in root that block a clean install (IN-9).
func (u skillsUnit) externals(e *env) []externalPath {
	var out []externalPath
	legacy := filepath.Join(u.root, skills.DirPrefix)
	if link, ok := e.s.external(legacy); ok {
		out = append(out, externalPath{path: legacy, link: link})
	} else if fi, err := os.Lstat(legacy); err == nil && fi.Mode()&os.ModeSymlink == 0 {
		out = append(out, externalPath{path: legacy})
	}
	all, _ := skills.All()
	for _, sk := range all {
		p := filepath.Join(u.root, sk.DirName())
		if link, ok := e.s.external(p); ok {
			out = append(out, externalPath{path: p, link: link})
		}
	}
	return out
}

func (u skillsUnit) externalReason(e *env, ext []externalPath) string {
	parts := make([]string, 0, len(ext))
	for _, x := range ext {
		if x.link != "" {
			parts = append(parts, e.s.displayPath(x.path)+" is a symlink to "+x.link)
			continue
		}
		parts = append(parts, e.s.displayPath(x.path)+" already exists")
	}
	return strings.Join(parts, "; ")
}

func joinTargets(ts []Target) string {
	parts := make([]string, len(ts))
	for i, t := range ts {
		parts[i] = string(t)
	}
	return strings.Join(parts, ", ")
}
