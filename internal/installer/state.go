package installer

import (
	"database/sql"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
)

const (
	selectInstallerStateQuery = "SELECT target, component, path, entry_hash, version, created, COALESCE(installed_at, '') FROM installer_state"
	upsertInstallerStateQuery = `INSERT INTO installer_state (target, component, path, entry_hash, version, created) VALUES (?, ?, ?, ?, ?, ?)
		ON CONFLICT(target, component, path) DO UPDATE SET entry_hash = excluded.entry_hash, version = excluded.version,
		created = installer_state.created | excluded.created, installed_at = datetime('now')`
	deleteInstallerStateQuery = "DELETE FROM installer_state WHERE target = ? AND component = ? AND path = ?"
)

// Bits of installer_state.created.
const (
	createdFile  = 1 // the installer created the file, so uninstall may delete it
	createdBlock = 2 // the installer created the containing JSON object (e.g. mcpServers, hooks)
)

// stateRow is one installer_state row: what the installer last wrote at path for a component.
type stateRow struct {
	Target      Target
	Component   Component
	Path        string
	EntryHash   string
	Version     string
	Created     int
	InstalledAt string
}

type stateKey struct {
	target    Target
	component Component
	path      string
}

// stateIndex is installer_state loaded for one Plan or Verify call.
type stateIndex struct {
	rows   map[stateKey]stateRow
	byPath map[string][]stateRow
}

func (r stateRow) key() stateKey {
	return stateKey{r.Target, r.Component, r.Path}
}

// loadState reads installer_state. With no database open it returns an empty index, so status
// still works from disk alone.
func loadState() (stateIndex, error) {
	idx := stateIndex{rows: map[stateKey]stateRow{}, byPath: map[string][]stateRow{}}
	if db.DB == nil {
		return idx, nil
	}
	rows, err := db.DB.Query(selectInstallerStateQuery)
	if err != nil {
		return idx, errs.WrapMessage("failed to read installer state", err)
	}
	defer rows.Close()
	for rows.Next() {
		var r stateRow
		if err := rows.Scan(&r.Target, &r.Component, &r.Path, &r.EntryHash, &r.Version, &r.Created, &r.InstalledAt); err != nil {
			return idx, errs.WrapMessage("failed to scan installer state", err)
		}
		idx.rows[r.key()] = r
		idx.byPath[r.Path] = append(idx.byPath[r.Path], r)
	}
	return idx, errs.WrapMessage("failed to read installer state", rows.Err())
}

// get returns the row for a target × component × path, or nil.
func (x stateIndex) get(t Target, c Component, path string) *stateRow {
	r, ok := x.rows[stateKey{t, c, path}]
	if !ok {
		return nil
	}
	return &r
}

// otherOwners lists targets other than t that recorded the same path, so uninstalling one host
// doesn't delete a shared skill another host still uses.
func (x stateIndex) otherOwners(t Target, path string) []Target {
	var out []Target
	for _, r := range x.byPath[path] {
		if r.Target != t {
			out = append(out, r.Target)
		}
	}
	return out
}

// applyState writes a change's state rows in one transaction.
func applyState(upserts []stateRow, deletes []stateKey) error {
	if len(upserts) == 0 && len(deletes) == 0 {
		return nil
	}
	tx, err := db.DB.Begin()
	if err != nil {
		return errs.WrapMessage("failed to begin installer state transaction", err)
	}
	if err := writeState(tx, upserts, deletes); err != nil {
		tx.Rollback()
		return err
	}
	return errs.WrapMessage("failed to commit installer state", tx.Commit())
}

func writeState(tx *sql.Tx, upserts []stateRow, deletes []stateKey) error {
	for _, k := range deletes {
		if _, err := tx.Exec(deleteInstallerStateQuery, k.target, k.component, k.path); err != nil {
			return errs.WrapMessage("failed to delete installer state", err, "path", k.path)
		}
	}
	for _, r := range upserts {
		if _, err := tx.Exec(upsertInstallerStateQuery, r.Target, r.Component, r.Path, r.EntryHash, r.Version, r.Created); err != nil {
			return errs.WrapMessage("failed to save installer state", err, "path", r.Path)
		}
	}
	return nil
}
