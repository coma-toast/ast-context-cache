package sys

import (
	"os"
	"path/filepath"
	"strings"
)

// prevSuffix is what selfupdate renames the running binary to while it installs the new one.
const prevSuffix = ".prev"

// startupExe is resolved at package init, before an update can rename the running binary.
var startupExe, startupExeErr = resolveExecutable(os.Executable, filepath.EvalSymlinks, fileExists)

// Executable is the path ast-mcp was started from, symlinks resolved, as it was at process
// start. Restart and self-update use it instead of os.Executable: an update renames the
// running binary to <exe>.prev and installs the new one at <exe>, and on Linux
// os.Executable reads /proc/self/exe, which follows the renamed inode, so it would name
// the old binary.
func Executable() (string, error) {
	return startupExe, startupExeErr
}

// resolveExecutable resolves the running binary through exe and eval. A path ending in
// .prev is mapped back to the path without it when that exists, so a process started
// after an update-time rename still restarts into the installed binary.
func resolveExecutable(exe func() (string, error), eval func(string) (string, error), exists func(string) bool) (string, error) {
	p, err := exe()
	if err != nil {
		return "", err
	}
	if r, err := eval(p); err == nil {
		p = r
	}
	if base := strings.TrimSuffix(p, prevSuffix); base != p && exists(base) {
		return base, nil
	}
	return p, nil
}

func fileExists(p string) bool {
	fi, err := os.Stat(p)
	return err == nil && fi.Mode().IsRegular()
}
