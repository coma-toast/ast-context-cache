package ortinstall

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

type entry struct {
	name, link, body string
	dir              bool
}

func buildTGZ(t *testing.T, entries []entry) []byte {
	t.Helper()
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)
	for _, e := range entries {
		hdr := &tar.Header{Name: e.name, Mode: 0o755}
		switch {
		case e.dir:
			hdr.Typeflag = tar.TypeDir
		case e.link != "":
			hdr.Typeflag, hdr.Linkname = tar.TypeSymlink, e.link
		default:
			hdr.Typeflag, hdr.Size = tar.TypeReg, int64(len(e.body))
		}
		require.NoError(t, tw.WriteHeader(hdr))
		if hdr.Typeflag == tar.TypeReg {
			_, err := tw.Write([]byte(e.body))
			require.NoError(t, err)
		}
	}
	require.NoError(t, tw.Close())
	require.NoError(t, gz.Close())
	return buf.Bytes()
}

func linuxArchive(t *testing.T, ver string) []byte {
	top := "onnxruntime-linux-x64-" + ver + "/"
	return buildTGZ(t, []entry{
		{name: top, dir: true},
		{name: top + "lib/", dir: true},
		{name: top + "lib/libonnxruntime.so." + ver, body: "elf"},
		{name: top + "lib/libonnxruntime.so.1", link: "libonnxruntime.so." + ver},
		{name: top + "lib/libonnxruntime.so", link: "libonnxruntime.so.1"},
		{name: top + "lib/libonnxruntime_evil.so", link: "/etc/passwd"},
		{name: top + "lib/libonnxruntime_up.so", link: "../../x"},
		{name: top + "lib/cmake/onnxruntime/x.cmake", body: "cmake"},
		{name: top + "include/onnxruntime_c_api.h", body: "h"},
		{name: "../libonnxruntime.so", body: "escape"},
	})
}

func sum(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

func serve(t *testing.T, body []byte) (*httptest.Server, *[]string) {
	t.Helper()
	var paths []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.Path)
		w.Write(body)
	}))
	t.Cleanup(srv.Close)
	return srv, &paths
}

func TestInstallVerifiesAndExtracts(t *testing.T) {
	t.Parallel()
	archive := linuxArchive(t, "1.31.0")
	srv, paths := serve(t, archive)
	dataDir := t.TempDir()
	inst := &Installer{
		DataDir: dataDir, BaseURL: srv.URL, GOOS: "linux", GOARCH: "amd64",
		Checksums: map[string]map[string]string{"1.31.0": {"linux-x64": sum(archive)}},
	}
	got, err := inst.InstallVersion(context.Background(), "1.31.0")
	require.NoError(t, err)
	assert.Equal(t, []string{"/v1.31.0/onnxruntime-linux-x64-1.31.0.tgz"}, *paths)
	assert.Equal(t, "1.31.0", got.Version)
	assert.Equal(t, filepath.Join(dataDir, "lib", "onnxruntime-1.31.0", "libonnxruntime.so"), got.Lib)
	data, err := os.ReadFile(got.Lib)
	require.NoError(t, err)
	assert.Equal(t, "elf", string(data), "symlink chain resolves to the library")
	names := dirNames(t, got.Dir)
	assert.ElementsMatch(t, []string{"libonnxruntime.so", "libonnxruntime.so.1", "libonnxruntime.so.1.31.0"}, names)
	assert.NoFileExists(t, filepath.Join(dataDir, "libonnxruntime.so"))
	assert.Equal(t, []string{"onnxruntime-1.31.0"}, dirNames(t, LibRoot(dataDir)), "no temp files left behind")
	found, ok := installedFor(dataDir, "linux")
	require.True(t, ok)
	assert.Equal(t, got, found)
}

func TestInstallRefusesChecksumMismatch(t *testing.T) {
	t.Parallel()
	archive := linuxArchive(t, "1.31.0")
	srv, _ := serve(t, archive)
	dataDir := t.TempDir()
	inst := &Installer{
		DataDir: dataDir, BaseURL: srv.URL, GOOS: "linux", GOARCH: "amd64",
		Checksums: map[string]map[string]string{"1.31.0": {"linux-x64": sum([]byte("something else"))}},
	}
	_, err := inst.InstallVersion(context.Background(), "1.31.0")
	require.Error(t, err)
	assert.True(t, errs.HasCode(err, errs.CodeConflict))
	assert.Contains(t, err.Error(), "checksum mismatch")
	assert.Empty(t, dirNames(t, LibRoot(dataDir)), "nothing installed or left behind")
	_, ok := installedFor(dataDir, "linux")
	assert.False(t, ok)
}

func TestInstallErrors(t *testing.T) {
	t.Parallel()
	notFound := httptest.NewServer(http.NotFoundHandler())
	t.Cleanup(notFound.Close)
	tests := []struct {
		name   string
		inst   Installer
		ver    string
		substr string
	}{
		{name: "unsupported platform", inst: Installer{GOOS: "windows", GOARCH: "amd64"}, ver: Version, substr: "no managed ONNX Runtime build"},
		{name: "unpinned version", inst: Installer{GOOS: "linux", GOARCH: "amd64"}, ver: "9.9.9", substr: "no pinned checksum"},
		{name: "bad version", inst: Installer{GOOS: "linux", GOARCH: "amd64"}, ver: "../1", substr: "invalid ONNX Runtime version"},
		{name: "http error", inst: Installer{GOOS: "linux", GOARCH: "amd64", BaseURL: notFound.URL}, ver: Version, substr: "HTTP 404"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.inst.DataDir = t.TempDir()
			_, err := tt.inst.InstallVersion(context.Background(), tt.ver)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.substr)
		})
	}
}

func TestReinstallReplacesSameVersion(t *testing.T) {
	t.Parallel()
	archive := linuxArchive(t, "1.31.0")
	srv, _ := serve(t, archive)
	dataDir := t.TempDir()
	inst := &Installer{
		DataDir: dataDir, BaseURL: srv.URL, GOOS: "linux", GOARCH: "amd64",
		Checksums: map[string]map[string]string{"1.31.0": {"linux-x64": sum(archive)}},
	}
	_, err := inst.InstallVersion(context.Background(), "1.31.0")
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(filepath.Join(InstallDir(dataDir, "1.31.0"), "stale"), nil, 0o644))
	_, err = inst.InstallVersion(context.Background(), "1.31.0")
	require.NoError(t, err)
	assert.NoFileExists(t, filepath.Join(InstallDir(dataDir, "1.31.0"), "stale"))
	assert.Equal(t, []string{"onnxruntime-1.31.0"}, dirNames(t, LibRoot(dataDir)))
}

func TestInstalledPicksNewestUsableAndPrune(t *testing.T) {
	t.Parallel()
	dataDir := t.TempDir()
	for _, v := range []string{"1.24.0", "1.25.0", "1.31.0", "1.40.0"} {
		dir := InstallDir(dataDir, v)
		require.NoError(t, os.MkdirAll(dir, 0o755))
		if v != "1.40.0" {
			require.NoError(t, os.WriteFile(filepath.Join(dir, "libonnxruntime.so"), nil, 0o644))
		}
	}
	require.NoError(t, os.MkdirAll(filepath.Join(LibRoot(dataDir), "onnxruntime-junk"), 0o755))
	got, ok := installedFor(dataDir, "linux")
	require.True(t, ok)
	assert.Equal(t, "1.31.0", got.Version, "1.40.0 has no library and 1.24.0 is below the minimum")
	_, ok = installedFor(dataDir, "darwin")
	assert.False(t, ok)
	Prune(dataDir, "1.31.0")
	assert.Equal(t, []string{"onnxruntime-1.31.0"}, dirNames(t, LibRoot(dataDir)))
}

func TestVersionHelpers(t *testing.T) {
	t.Parallel()
	assert.Equal(t, 0, CompareVersions("1.31.0", "1.31"))
	assert.Equal(t, 1, CompareVersions("1.31.0", "1.25.0"))
	assert.Equal(t, -1, CompareVersions("1.9.0", "1.25.0"))
	assert.True(t, UpdateAvailable("1.25.0"))
	assert.False(t, UpdateAvailable(Version))
	assert.False(t, UpdateAvailable(""))
	dataDir := t.TempDir()
	assert.Equal(t, "1.31.0", VersionFromLibPath(dataDir, filepath.Join(InstallDir(dataDir, "1.31.0"), "libonnxruntime.so")))
	assert.Equal(t, "", VersionFromLibPath(dataDir, "/usr/lib/libonnxruntime.so"))
	for _, tt := range []struct{ goos, goarch, want string }{
		{"linux", "amd64", "linux-x64"}, {"linux", "arm64", "linux-aarch64"}, {"darwin", "arm64", "osx-arm64"},
	} {
		p, err := Platform(tt.goos, tt.goarch)
		require.NoError(t, err)
		assert.Equal(t, tt.want, p)
		assert.Len(t, checksums[Version][p], 64, "every supported platform has a pinned digest")
	}
	_, err := Platform("darwin", "amd64")
	assert.Error(t, err)
}

func dirNames(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if os.IsNotExist(err) {
		return nil
	}
	require.NoError(t, err)
	var out []string
	for _, e := range entries {
		out = append(out, e.Name())
	}
	return out
}
