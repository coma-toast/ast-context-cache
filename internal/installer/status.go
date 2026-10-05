package installer

// Status is a component's installed state, computed from the files on disk plus installer_state
// (IN-8): the database alone never decides it.
type Status string

// Component statuses.
const (
	// StatusInstalled means the current version of our entry, block, or files is present.
	StatusInstalled Status = "installed"
	// StatusOutdated means our content is present as the installer last wrote it, but this
	// version would write something different.
	StatusOutdated Status = "outdated"
	// StatusModifiedByUser means our content is present but was edited after the installer wrote it.
	StatusModifiedByUser Status = "modified_by_user"
	// StatusMissing means installer_state records an install that is no longer on disk.
	StatusMissing Status = "missing"
	// StatusNotInstalled means nothing of ours is present and nothing is recorded.
	StatusNotInstalled Status = "not_installed"
	// StatusExternallyManaged means a path we would write already belongs to something else, such
	// as a symlink into another repo (IN-9). It is skipped unless the user opts to replace it.
	StatusExternallyManaged Status = "externally_managed"
	// StatusUnsupported means the host has no supported location for the component.
	StatusUnsupported Status = "unsupported"
	// StatusCovered means the host already loads our skills from a directory another target
	// installs into, so a second copy would only duplicate them.
	StatusCovered Status = "covered"
)

// ComponentStatus is one target × component cell of the status table.
type ComponentStatus struct {
	Target    Target    `json:"target"`
	Component Component `json:"component"`
	Status    Status    `json:"status"`
	Path      string    `json:"path"`
	Reason    string    `json:"reason,omitempty"`
}

// statusRank orders per-file statuses by how much attention they need, for folding a
// multi-file component into one status.
var statusRank = map[Status]int{
	StatusInstalled:         0,
	StatusNotInstalled:      1,
	StatusOutdated:          2,
	StatusMissing:           3,
	StatusModifiedByUser:    4,
	StatusExternallyManaged: 5,
}

// foldStatus combines per-file statuses: all installed is Installed, all absent is
// NotInstalled, and otherwise the status needing the most attention wins.
func foldStatus(statuses []Status) Status {
	if len(statuses) == 0 {
		return StatusNotInstalled
	}
	out := statuses[0]
	mixed := false
	for _, st := range statuses[1:] {
		if st != out {
			mixed = true
		}
		if statusRank[st] > statusRank[out] {
			out = st
		}
	}
	if mixed && out == StatusNotInstalled {
		// Some files present and current, others absent: a partial install.
		return StatusOutdated
	}
	return out
}

// entryStatus classifies a present entry by its canonical hash: current, as we last wrote it, or
// edited since. An entry we never recorded that differs from ours counts as user-modified. The
// recorded hash may be either form (see entryHashes).
func entryStatus(cur entryHashes, desiredHash string, row *stateRow) Status {
	switch {
	case cur.norm == desiredHash:
		return StatusInstalled
	case row != nil && (row.EntryHash == cur.norm || row.EntryHash == cur.raw):
		return StatusOutdated
	default:
		return StatusModifiedByUser
	}
}

// absentStatus is the status of a component with nothing on disk.
func absentStatus(row *stateRow) Status {
	if row != nil {
		return StatusMissing
	}
	return StatusNotInstalled
}
