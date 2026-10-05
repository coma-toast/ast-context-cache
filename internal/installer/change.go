package installer

import (
	"crypto/sha256"
	"encoding/hex"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/pmezard/go-difflib/difflib"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// ChangeKind says what applying a FileChange does to its file.
type ChangeKind string

// Change kinds.
const (
	// KindCreate writes a file that does not exist yet.
	KindCreate ChangeKind = "create"
	// KindModify rewrites an existing file with our entry, block, or content updated.
	KindModify ChangeKind = "modify"
	// KindRemoveBlock rewrites an existing file with our entry, block, table, or hooks removed.
	KindRemoveBlock ChangeKind = "remove-block"
	// KindDelete deletes a file the installer created, or an externally managed symlink the user
	// chose to replace.
	KindDelete ChangeKind = "delete"
	// KindNone changes nothing; the FileChange is skipped and Reason says why.
	KindNone ChangeKind = "none"
)

const (
	backupKeepSetting  = "installer_backup_keep"
	defaultBackupKeep  = 5
	backupTimeLayout   = "20060102-150405"
	backupSymlinkExt   = ".symlink"
	backupPathSep      = "%"
	absentHash         = ""
	symlinkHashPrefix  = "symlink:"
	defaultCreatedMode = fs.FileMode(0o644)
	backupFileMode     = fs.FileMode(0o600)
	backupDirMode      = fs.FileMode(0o700)
)

// FileChange is one file the plan would touch. Before/After hold full contents; Diff is the
// unified diff shown in the preview. A skipped change writes nothing.
type FileChange struct {
	Target     Target     `json:"target"`
	Component  Component  `json:"component"`
	Path       string     `json:"path"`
	Kind       ChangeKind `json:"kind"`
	Diff       string     `json:"diff"`
	Skipped    bool       `json:"skipped"`
	Reason     string     `json:"reason,omitempty"`
	BeforeHash string     `json:"before_hash,omitempty"`
	Before     []byte     `json:"-"`
	After      []byte     `json:"-"`

	// linkTarget is set when Path is a symlink this change deletes.
	linkTarget string
	// viaLink marks a file written into a path whose external symlink an earlier change in the
	// same plan deletes; that change's check covers it.
	viaLink bool
	upserts []stateRow
	deletes []stateKey
}

// Backup is one saved copy of a file taken before the installer changed it.
type Backup struct {
	ID        string    `json:"id"`
	Path      string    `json:"path"`
	CreatedAt time.Time `json:"created_at"`
	Size      int64     `json:"size"`
	Symlink   bool      `json:"symlink,omitempty"`
}

func (c FileChange) underLink() bool {
	return c.viaLink
}

// unifiedDiff renders the change as a unified diff with three lines of context.
func unifiedDiff(path string, before, after []byte) string {
	if string(before) == string(after) {
		return ""
	}
	from, to := "a"+path, "b"+path
	if before == nil {
		from = "/dev/null"
	}
	if after == nil {
		to = "/dev/null"
	}
	out, err := difflib.GetUnifiedDiffString(difflib.UnifiedDiff{
		A:        difflib.SplitLines(string(before)),
		B:        difflib.SplitLines(string(after)),
		FromFile: from,
		ToFile:   to,
		Context:  3,
	})
	if err != nil {
		return ""
	}
	return out
}

// contentHash identifies a file's state for the preview/apply check (IN-5).
func contentHash(b []byte) string {
	if b == nil {
		return absentHash
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// currentHash re-reads path the same way the plan did: a symlink being replaced is identified by
// its link target, anything else by its contents (following symlinks).
func currentHash(path string, asLink bool) (string, error) {
	if asLink {
		t, err := os.Readlink(path)
		if os.IsNotExist(err) {
			return absentHash, nil
		}
		if err != nil {
			return "", errs.WrapMessage("failed to read symlink", err, "path", path)
		}
		return symlinkHashPrefix + t, nil
	}
	b, err := readOptional(path)
	if err != nil {
		return "", err
	}
	return contentHash(b), nil
}

// readOptional reads path, returning nil (not an empty slice) when it does not exist.
func readOptional(path string) ([]byte, error) {
	b, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, errs.WrapMessage("failed to read file", err, "path", path)
	}
	if b == nil {
		b = []byte{}
	}
	return b, nil
}

// atomicWrite replaces path with data via a temp file in the same directory, fsync, and rename.
// A symlinked path is resolved first so the link survives and its target is updated. An
// existing file keeps its mode; a new one gets mode.
func atomicWrite(path string, data []byte, mode fs.FileMode) error {
	if real, err := filepath.EvalSymlinks(path); err == nil {
		path = real
	}
	if fi, err := os.Stat(path); err == nil {
		mode = fi.Mode().Perm()
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return errs.WrapMessage("failed to create directory", err, "path", dir)
	}
	tmp, err := os.CreateTemp(dir, "."+filepath.Base(path)+".astcache-*")
	if err != nil {
		return errs.WrapMessage("failed to create temp file", err, "path", path)
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return errs.WrapMessage("failed to write temp file", err, "path", path)
	}
	if err := tmp.Chmod(mode); err != nil {
		tmp.Close()
		return errs.WrapMessage("failed to set file mode", err, "path", path)
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return errs.WrapMessage("failed to sync temp file", err, "path", path)
	}
	if err := tmp.Close(); err != nil {
		return errs.WrapMessage("failed to close temp file", err, "path", path)
	}
	if err := os.Rename(tmpName, path); err != nil {
		return errs.WrapMessage("failed to replace file", err, "path", path)
	}
	syncDir(dir)
	return nil
}

// syncDir makes the rename durable; failure only weakens crash safety, so it's ignored.
func syncDir(dir string) {
	if d, err := os.Open(dir); err == nil {
		d.Sync()
		d.Close()
	}
}

// backupFile copies path (or records its link target) under the backup root before a write and
// prunes older copies of the same path (IN-4). It returns nil when path does not exist.
func backupFile(root, path string, asLink bool, now time.Time) (*Backup, error) {
	var data []byte
	name := encodeBackupName(path)
	if asLink {
		t, err := os.Readlink(path)
		if os.IsNotExist(err) {
			return nil, nil
		}
		if err != nil {
			return nil, errs.WrapMessage("failed to read symlink", err, "path", path)
		}
		data, name = []byte(t), name+backupSymlinkExt
	} else {
		b, err := readOptional(path)
		if err != nil || b == nil {
			return nil, err
		}
		data = b
	}
	dir := uniqueBackupDir(root, now, name)
	if err := os.MkdirAll(dir, backupDirMode); err != nil {
		return nil, errs.WrapMessage("failed to create backup directory", err, "path", dir)
	}
	dest := filepath.Join(dir, name)
	if err := os.WriteFile(dest, data, backupFileMode); err != nil {
		return nil, errs.WrapMessage("failed to write backup", err, "path", dest)
	}
	syncDir(dir)
	if err := pruneBackups(root, name, backupKeep()); err != nil {
		logger.Warn("Failed to prune installer backups", "error", err, "path", path)
	}
	return &Backup{ID: filepath.Base(dir) + "/" + name, Path: path, CreatedAt: now, Size: int64(len(data)), Symlink: asLink}, nil
}

// uniqueBackupDir returns the timestamped backup directory for now, suffixed when the same file
// was already backed up in that second so no earlier copy is overwritten.
func uniqueBackupDir(root string, now time.Time, name string) string {
	base := now.Format(backupTimeLayout)
	for i := 0; ; i++ {
		dir := filepath.Join(root, base)
		if i > 0 {
			dir += "-" + strconv.Itoa(i)
		}
		// Any Lstat error (normally not-exist) ends the search; a real I/O problem surfaces
		// when the backup is written.
		if _, err := os.Lstat(filepath.Join(dir, name)); err != nil {
			return dir
		}
	}
}

// pruneBackups keeps the newest keep copies of one file and removes emptied backup directories.
func pruneBackups(root, name string, keep int) error {
	entries, err := os.ReadDir(root)
	if err != nil {
		return errs.WrapMessage("failed to list backups", err, "path", root)
	}
	var dirs []string
	for _, e := range entries {
		if _, err := os.Lstat(filepath.Join(root, e.Name(), name)); err == nil && e.IsDir() {
			dirs = append(dirs, e.Name())
		}
	}
	slices.SortFunc(dirs, compareBackupDirs)
	for len(dirs) > keep {
		dir := filepath.Join(root, dirs[0])
		if err := os.Remove(filepath.Join(dir, name)); err != nil {
			return errs.WrapMessage("failed to remove old backup", err, "path", dir)
		}
		os.Remove(dir) // only succeeds once the directory is empty
		dirs = dirs[1:]
	}
	return nil
}

// listBackups returns every backup under root, newest first.
func listBackups(root string) ([]Backup, error) {
	entries, err := os.ReadDir(root)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, errs.WrapMessage("failed to list backups", err, "path", root)
	}
	var out []Backup
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		ts, ok := parseBackupDir(e.Name())
		if !ok {
			continue
		}
		files, err := os.ReadDir(filepath.Join(root, e.Name()))
		if err != nil {
			continue
		}
		for _, f := range files {
			info, err := f.Info()
			if err != nil || !info.Mode().IsRegular() {
				continue
			}
			path, link := decodeBackupName(f.Name())
			out = append(out, Backup{ID: e.Name() + "/" + f.Name(), Path: path, CreatedAt: ts, Size: info.Size(), Symlink: link})
		}
	}
	slices.SortFunc(out, func(a, b Backup) int {
		return -compareBackupDirs(strings.Split(a.ID, "/")[0], strings.Split(b.ID, "/")[0])
	})
	return out, nil
}

// compareBackupDirs orders "20060102-150405" and "20060102-150405-N" directories oldest first.
func compareBackupDirs(a, b string) int {
	ta, _ := parseBackupDir(a)
	tb, _ := parseBackupDir(b)
	if c := ta.Compare(tb); c != 0 {
		return c
	}
	return backupSuffix(a) - backupSuffix(b)
}

func parseBackupDir(name string) (time.Time, bool) {
	if len(name) < len(backupTimeLayout) {
		return time.Time{}, false
	}
	t, err := time.ParseInLocation(backupTimeLayout, name[:len(backupTimeLayout)], time.Local)
	return t, err == nil
}

func backupSuffix(name string) int {
	rest := strings.TrimPrefix(name[min(len(name), len(backupTimeLayout)):], "-")
	n, _ := strconv.Atoi(rest)
	return n
}

// encodeBackupName flattens an absolute path into one file name: "/" becomes "%".
func encodeBackupName(path string) string {
	return strings.ReplaceAll(filepath.ToSlash(path), "/", backupPathSep)
}

func decodeBackupName(name string) (string, bool) {
	link := strings.HasSuffix(name, backupSymlinkExt)
	name = strings.TrimSuffix(name, backupSymlinkExt)
	return filepath.FromSlash(strings.ReplaceAll(name, backupPathSep, "/")), link
}

// backupKeep reads installer_backup_keep, falling back to 5 for a missing or invalid value.
func backupKeep() int {
	n, err := strconv.Atoi(strings.TrimSpace(db.GetSetting(backupKeepSetting, strconv.Itoa(defaultBackupKeep))))
	if err != nil || n < 1 {
		return defaultBackupKeep
	}
	return n
}
