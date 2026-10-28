package sys

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// An update renames the running binary to .prev and installs the new one in its place.
// On Linux /proc/self/exe follows the renamed inode, so a late os.Executable names .prev;
// the restart target must still be the original path, which now holds the new binary.
func TestResolveExecutableSurvivesUpdateRename(t *testing.T) {
	dir := t.TempDir()
	exe := filepath.Join(dir, "ast-mcp")
	require.NoError(t, os.WriteFile(exe, []byte("old"), 0o755))
	procExe := exe
	resolver := func() (string, error) { return procExe, nil }
	atStartup, err := resolveExecutable(resolver, filepath.EvalSymlinks, fileExists)
	require.NoError(t, err)
	// selfupdate.install: exe -> exe.prev, exe.next -> exe.
	require.NoError(t, os.Rename(exe, exe+prevSuffix))
	require.NoError(t, os.WriteFile(exe, []byte("new"), 0o755))
	procExe = exe + prevSuffix
	late, err := resolveExecutable(resolver, filepath.EvalSymlinks, fileExists)
	require.NoError(t, err)
	want, _ := filepath.EvalSymlinks(exe)
	assert.Equal(t, want, atStartup)
	assert.Equal(t, want, late, "a .prev path maps back to the installed binary")
}

func TestResolveExecutableKeepsPrevWithoutInstalledBinary(t *testing.T) {
	dir := t.TempDir()
	prev := filepath.Join(dir, "ast-mcp"+prevSuffix)
	require.NoError(t, os.WriteFile(prev, []byte("old"), 0o755))
	got, err := resolveExecutable(func() (string, error) { return prev, nil }, filepath.EvalSymlinks, fileExists)
	require.NoError(t, err)
	want, _ := filepath.EvalSymlinks(prev)
	assert.Equal(t, want, got)
}

func TestResolveExecutableFollowsSymlink(t *testing.T) {
	dir := t.TempDir()
	real := filepath.Join(dir, "ast-mcp")
	link := filepath.Join(dir, "link")
	require.NoError(t, os.WriteFile(real, []byte("bin"), 0o755))
	require.NoError(t, os.Symlink(real, link))
	got, err := resolveExecutable(func() (string, error) { return link, nil }, filepath.EvalSymlinks, fileExists)
	require.NoError(t, err)
	want, _ := filepath.EvalSymlinks(real)
	assert.Equal(t, want, got)
}

func TestExecutableIsResolvedAtStartup(t *testing.T) {
	got, err := Executable()
	require.NoError(t, err)
	assert.NotEmpty(t, got)
}
