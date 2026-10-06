package selfupdate

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coma-toast/ast-context-cache/internal/version"
)

// fakeRelease serves a GitHub "latest release" document plus its archive and
// checksums.txt, the way api.github.com and its download URLs do.
type fakeRelease struct {
	tag      string
	archive  []byte
	checksum string // overrides the archive's real sha256 when set
	noAsset  bool
	gate     chan struct{} // the archive download waits on it when set
}

func tarGz(t *testing.T, files map[string]string) []byte {
	t.Helper()
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)
	for name, body := range files {
		require.NoError(t, tw.WriteHeader(&tar.Header{Name: name, Mode: 0o755, Size: int64(len(body)), Typeflag: tar.TypeReg}))
		_, err := tw.Write([]byte(body))
		require.NoError(t, err)
	}
	require.NoError(t, tw.Close())
	require.NoError(t, gz.Close())
	return buf.Bytes()
}

func serve(t *testing.T, r fakeRelease) {
	t.Helper()
	name := assetName(r.tag[1:])
	sum := sha256.Sum256(r.archive)
	checksum := hex.EncodeToString(sum[:])
	if r.checksum != "" {
		checksum = r.checksum
	}
	var srv *httptest.Server
	srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		switch req.URL.Path {
		case "/latest":
			assets := []asset{{Name: checksumName, URL: srv.URL + "/checksums.txt"}}
			if !r.noAsset {
				assets = append(assets, asset{Name: name, URL: srv.URL + "/archive"})
			}
			json.NewEncoder(w).Encode(release{TagName: r.tag, HTMLURL: "https://example.test/" + r.tag, Assets: assets})
		case "/checksums.txt":
			fmt.Fprintf(w, "%s  %s\n%s  other_archive.tar.gz\n", checksum, name, checksum)
		case "/archive":
			if r.gate != nil {
				<-r.gate
			}
			w.Write(r.archive)
		default:
			http.NotFound(w, req)
		}
	}))
	t.Cleanup(srv.Close)
	prev := LatestReleaseURL
	LatestReleaseURL = srv.URL + "/latest"
	t.Cleanup(func() { LatestReleaseURL = prev })
}

func runningAs(t *testing.T, v, build string) {
	t.Helper()
	pv, pb := version.Version, version.Build
	version.Version, version.Build = v, build
	t.Cleanup(func() { version.Version, version.Build = pv, pb })
}

func resetSnapshot(t *testing.T) {
	t.Helper()
	set(Snapshot{})
	t.Cleanup(func() { set(Snapshot{}) })
}

func waitDone(t *testing.T) Snapshot {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if s := GetSnapshot(); !s.Active {
			return s
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("update never finished")
	return Snapshot{}
}

func TestCompareVersions(t *testing.T) {
	assert.Equal(t, 1, compareVersions("4.0.10", "4.0.9"))
	assert.Equal(t, -1, compareVersions("v4.0.6", "4.1.0"))
	assert.Equal(t, 0, compareVersions("4.0.6", "4.0.6-rc1"))
	assert.Equal(t, 1, compareVersions("4.0.6", "dev"))
}

func TestCheckOffersNewerRelease(t *testing.T) {
	runningAs(t, "4.0.6", "release")
	serve(t, fakeRelease{tag: "v4.0.7", archive: []byte("x")})
	c := Check()
	assert.Empty(t, c.Error)
	assert.True(t, c.UpdateAvailable)
	assert.False(t, c.SourceBuild)
	assert.Equal(t, "4.0.7", c.LatestVersion)
	assert.Equal(t, "https://example.test/v4.0.7", c.ReleaseURL)
}

func TestCheckReportsUpToDateRelease(t *testing.T) {
	runningAs(t, "4.0.7", "release")
	serve(t, fakeRelease{tag: "v4.0.7", archive: []byte("x")})
	assert.False(t, Check().UpdateAvailable)
}

// A source build can carry the release's version number while running other
// code, so it is always offered the release, with SourceBuild set for the UI
// to warn about.
func TestCheckOffersReleaseToSourceBuild(t *testing.T) {
	runningAs(t, "4.0.7", "source")
	serve(t, fakeRelease{tag: "v4.0.7", archive: []byte("x")})
	c := Check()
	assert.True(t, c.UpdateAvailable)
	assert.True(t, c.SourceBuild)
}

func TestCheckReportsMissingPlatformBuild(t *testing.T) {
	runningAs(t, "4.0.6", "release")
	serve(t, fakeRelease{tag: "v4.0.7", archive: []byte("x"), noAsset: true})
	c := Check()
	assert.Contains(t, c.Error, "has no build for")
	assert.False(t, c.UpdateAvailable)
}

func TestStartInstallsReleaseAndKeepsPrevious(t *testing.T) {
	runningAs(t, "4.0.6", "release")
	resetSnapshot(t)
	serve(t, fakeRelease{tag: "v4.0.7", archive: tarGz(t, map[string]string{"README.md": "docs", "ast-mcp": "new binary"})})
	exe := filepath.Join(t.TempDir(), "ast-mcp")
	require.NoError(t, os.WriteFile(exe, []byte("old binary"), 0o755))

	started, msg := Start(exe)
	require.True(t, started, msg)
	s := waitDone(t)
	require.Empty(t, s.Error)
	assert.True(t, s.Done)
	assert.Equal(t, "installed", s.Phase)
	assert.Equal(t, "4.0.6", s.FromVersion)
	assert.Equal(t, "4.0.7", s.ToVersion)
	got, err := os.ReadFile(exe)
	require.NoError(t, err)
	assert.Equal(t, "new binary", string(got))
	info, err := os.Stat(exe)
	require.NoError(t, err)
	assert.Equal(t, os.FileMode(0o755), info.Mode().Perm())
	prev, err := os.ReadFile(exe + ".prev")
	require.NoError(t, err)
	assert.Equal(t, "old binary", string(prev))
	leftovers, _ := filepath.Glob(filepath.Join(filepath.Dir(exe), ".ast-mcp-download-*"))
	assert.Empty(t, leftovers)
}

func TestStartRejectsChecksumMismatch(t *testing.T) {
	runningAs(t, "4.0.6", "release")
	resetSnapshot(t)
	serve(t, fakeRelease{tag: "v4.0.7", archive: tarGz(t, map[string]string{"ast-mcp": "tampered"}), checksum: hex.EncodeToString(make([]byte, 32))})
	exe := filepath.Join(t.TempDir(), "ast-mcp")
	require.NoError(t, os.WriteFile(exe, []byte("old binary"), 0o755))

	started, msg := Start(exe)
	require.True(t, started, msg)
	s := waitDone(t)
	assert.Equal(t, "error", s.Phase)
	assert.Contains(t, s.Error, "does not match checksums.txt")
	got, _ := os.ReadFile(exe)
	assert.Equal(t, "old binary", string(got))
	assert.NoFileExists(t, exe+".prev")
}

func TestStartRejectsArchiveWithoutBinary(t *testing.T) {
	runningAs(t, "4.0.6", "release")
	resetSnapshot(t)
	serve(t, fakeRelease{tag: "v4.0.7", archive: tarGz(t, map[string]string{"README.md": "docs"})})
	exe := filepath.Join(t.TempDir(), "ast-mcp")
	require.NoError(t, os.WriteFile(exe, []byte("old binary"), 0o755))

	started, _ := Start(exe)
	require.True(t, started)
	s := waitDone(t)
	assert.Contains(t, s.Error, "no ast-mcp binary")
	got, _ := os.ReadFile(exe)
	assert.Equal(t, "old binary", string(got))
	assert.NoFileExists(t, exe+".next")
}

func TestStartRefusesWhenAlreadyUpToDate(t *testing.T) {
	runningAs(t, "4.0.7", "release")
	resetSnapshot(t)
	serve(t, fakeRelease{tag: "v4.0.7", archive: []byte("x")})
	started, msg := Start(filepath.Join(t.TempDir(), "ast-mcp"))
	assert.False(t, started)
	assert.Equal(t, "already up to date", msg)
	assert.False(t, GetSnapshot().Active)
}

func TestStartIsExclusiveUnderConcurrency(t *testing.T) {
	runningAs(t, "4.0.6", "release")
	resetSnapshot(t)
	// The download waits until every Start has returned, so none of them can
	// see the first update finished and start a second.
	gate := make(chan struct{})
	serve(t, fakeRelease{tag: "v4.0.7", archive: tarGz(t, map[string]string{"ast-mcp": "new"}), gate: gate})
	exe := filepath.Join(t.TempDir(), "ast-mcp")
	require.NoError(t, os.WriteFile(exe, []byte("old"), 0o755))
	var wg sync.WaitGroup
	var mu sync.Mutex
	starts := 0
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if ok, _ := Start(exe); ok {
				mu.Lock()
				starts++
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	close(gate)
	waitDone(t)
	assert.Equal(t, 1, starts)
}
