// Package selfupdate lets the dashboard replace the running ast-mcp with the
// latest GitHub release build: download the archive for this platform, verify
// it against the release's checksums.txt, and swap it in next to the current
// executable. Restarting is left to the operator ("Restart now").
package selfupdate

import (
	"archive/tar"
	"bufio"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/realtime"
	"github.com/coma-toast/ast-context-cache/internal/version"
)

const (
	projectName  = "ast-context-cache"
	binaryName   = "ast-mcp"
	checksumName = "checksums.txt"
	// maxArchiveBytes bounds a release download; real archives are ~20 MB.
	maxArchiveBytes = 512 << 20
)

// LatestReleaseURL is the GitHub API endpoint for the newest published
// release (a var so tests can point it at a fake server).
var LatestReleaseURL = "https://api.github.com/repos/coma-toast/ast-context-cache/releases/latest"

var httpClient = &http.Client{Timeout: 5 * time.Minute}

// Snapshot is a point-in-time view of an update for the dashboard.
type Snapshot struct {
	Active      bool
	Done        bool
	Phase       string
	Error       string
	StartedAt   time.Time
	FinishedAt  time.Time
	FromVersion string
	ToVersion   string
}

// CheckResult compares the running build with the latest release.
type CheckResult struct {
	CurrentVersion  string
	Build           string
	LatestVersion   string
	ReleaseURL      string
	PublishedAt     string
	AssetName       string
	UpdateAvailable bool
	// SourceBuild is true when ast-mcp was built locally, so its version
	// number may not match the code it runs; an update still installs the
	// release.
	SourceBuild bool
	Error       string
}

type release struct {
	TagName     string  `json:"tag_name"`
	HTMLURL     string  `json:"html_url"`
	PublishedAt string  `json:"published_at"`
	Assets      []asset `json:"assets"`
}

type asset struct {
	Name string `json:"name"`
	URL  string `json:"browser_download_url"`
}

var (
	mu       sync.Mutex
	snapshot Snapshot
)

// GetSnapshot returns the current update state for the dashboard.
func GetSnapshot() Snapshot {
	mu.Lock()
	defer mu.Unlock()
	return snapshot
}

func set(s Snapshot) {
	mu.Lock()
	snapshot = s
	mu.Unlock()
	realtime.Notify(realtime.Settings)
}

func setPhase(phase string) {
	s := GetSnapshot()
	s.Phase = phase
	set(s)
}

func fail(err error) {
	s := GetSnapshot()
	s.Active = false
	s.Done = false
	s.Phase = "error"
	s.Error = err.Error()
	s.FinishedAt = time.Now()
	set(s)
}

// Check fetches the latest release and reports whether it should replace the
// running build. It changes nothing, so the dashboard can call it on load.
func Check() CheckResult {
	c, _ := check()
	return c
}

func check() (CheckResult, *release) {
	c := CheckResult{CurrentVersion: version.Version, Build: version.Build, SourceBuild: version.Build != "release"}
	rel, err := latestRelease()
	if err != nil {
		c.Error = err.Error()
		return c, nil
	}
	c.LatestVersion = strings.TrimPrefix(rel.TagName, "v")
	c.ReleaseURL = rel.HTMLURL
	c.PublishedAt = rel.PublishedAt
	c.AssetName = assetName(c.LatestVersion)
	if findAsset(rel, c.AssetName) == nil {
		c.Error = fmt.Sprintf("release v%s has no build for %s/%s", c.LatestVersion, runtime.GOOS, runtime.GOARCH)
		return c, rel
	}
	c.UpdateAvailable = c.SourceBuild || compareVersions(c.LatestVersion, c.CurrentVersion) > 0
	return c, rel
}

func latestRelease() (*release, error) {
	req, err := http.NewRequest(http.MethodGet, LatestReleaseURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/vnd.github+json")
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, errs.WrapMessage("failed to reach GitHub releases", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, errs.New("GitHub releases returned an unexpected status", "status", resp.Status)
	}
	var rel release
	if err := json.NewDecoder(resp.Body).Decode(&rel); err != nil {
		return nil, errs.WrapMessage("failed to decode the latest release", err)
	}
	if rel.TagName == "" {
		return nil, errs.New("latest release has no tag")
	}
	return &rel, nil
}

func assetName(v string) string {
	return fmt.Sprintf("%s_%s_%s_%s.tar.gz", projectName, v, runtime.GOOS, runtime.GOARCH)
}

func findAsset(rel *release, name string) *asset {
	for i := range rel.Assets {
		if rel.Assets[i].Name == name {
			return &rel.Assets[i]
		}
	}
	return nil
}

// compareVersions compares dotted numeric versions ("4.0.10" > "4.0.9"),
// ignoring any pre-release or build suffix. Unparseable parts count as 0.
func compareVersions(a, b string) int {
	pa, pb := versionParts(a), versionParts(b)
	for i := range pa {
		if pa[i] != pb[i] {
			if pa[i] > pb[i] {
				return 1
			}
			return -1
		}
	}
	return 0
}

func versionParts(v string) [3]int {
	var out [3]int
	v = strings.TrimPrefix(v, "v")
	if i := strings.IndexAny(v, "-+"); i >= 0 {
		v = v[:i]
	}
	for i, p := range strings.SplitN(v, ".", 3) {
		out[i], _ = strconv.Atoi(p)
	}
	return out
}

// tryClaim atomically checks and sets Active, so two near-simultaneous Start
// calls can't both launch runUpdate.
func tryClaim() bool {
	mu.Lock()
	defer mu.Unlock()
	if snapshot.Active {
		return false
	}
	snapshot = Snapshot{Active: true}
	return true
}

// Start installs the latest release over exePath in the background.
func Start(exePath string) (started bool, errMsg string) {
	if !tryClaim() {
		return false, "an update is already in progress"
	}
	c, rel := check()
	if c.Error != "" {
		set(Snapshot{})
		return false, c.Error
	}
	if !c.UpdateAvailable {
		set(Snapshot{})
		return false, "already up to date"
	}
	set(Snapshot{Active: true, Phase: "downloading", StartedAt: time.Now(), FromVersion: c.CurrentVersion, ToVersion: c.LatestVersion})
	go runUpdate(exePath, rel, c.AssetName)
	return true, ""
}

func runUpdate(exePath string, rel *release, name string) {
	if err := install(exePath, rel, name); err != nil {
		fail(err)
		return
	}
	s := GetSnapshot()
	s.Active = false
	s.Phase = "installed"
	s.Done = true
	s.FinishedAt = time.Now()
	set(s)
}

// install downloads and verifies the archive, then swaps its ast-mcp in for
// exePath, keeping the old binary as exePath+".prev".
func install(exePath string, rel *release, name string) error {
	sums := findAsset(rel, checksumName)
	if sums == nil {
		return errs.New("release has no checksums.txt", "tag", rel.TagName)
	}
	want, err := expectedChecksum(sums.URL, name)
	if err != nil {
		return err
	}
	archive, err := os.CreateTemp(filepath.Dir(exePath), ".ast-mcp-download-*")
	if err != nil {
		return errs.WrapMessage("failed to create a download file next to ast-mcp", err)
	}
	defer os.Remove(archive.Name())
	defer archive.Close()
	got, err := download(findAsset(rel, name).URL, archive)
	if err != nil {
		return err
	}
	setPhase("verifying")
	if got != want {
		return errs.New("downloaded archive does not match checksums.txt", "asset", name, "want", want, "got", got)
	}
	setPhase("installing")
	if _, err := archive.Seek(0, io.SeekStart); err != nil {
		return err
	}
	next, prev := exePath+".next", exePath+".prev"
	if err := extractBinary(archive, next); err != nil {
		os.Remove(next)
		return err
	}
	os.Remove(prev)
	if err := os.Rename(exePath, prev); err != nil && !os.IsNotExist(err) {
		os.Remove(next)
		return errs.WrapMessage("failed to set the current ast-mcp aside", err)
	}
	if err := os.Rename(next, exePath); err != nil {
		os.Rename(prev, exePath)
		os.Remove(next)
		return errs.WrapMessage("failed to move the new ast-mcp into place", err)
	}
	return nil
}

func get(url string) (*http.Response, error) {
	resp, err := httpClient.Get(url)
	if err != nil {
		return nil, errs.WrapMessage("download failed", err, "url", url)
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, errs.New("download returned an unexpected status", "url", url, "status", resp.Status)
	}
	return resp, nil
}

// expectedChecksum reads the sha256 listed for name in a checksums.txt.
func expectedChecksum(url, name string) (string, error) {
	resp, err := get(url)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	sc := bufio.NewScanner(io.LimitReader(resp.Body, 1<<20))
	for sc.Scan() {
		if f := strings.Fields(sc.Text()); len(f) == 2 && f[1] == name {
			return strings.ToLower(f[0]), nil
		}
	}
	return "", errs.New("checksums.txt has no entry for this platform's archive", "asset", name)
}

// download writes url to w and returns the sha256 of what it wrote.
func download(url string, w io.Writer) (string, error) {
	resp, err := get(url)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	h := sha256.New()
	n, err := io.Copy(io.MultiWriter(w, h), io.LimitReader(resp.Body, maxArchiveBytes+1))
	if err != nil {
		return "", errs.WrapMessage("download interrupted", err, "url", url)
	}
	if n > maxArchiveBytes {
		return "", errs.New("release archive is larger than expected", "bytes", n)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// extractBinary writes the archive's top-level ast-mcp to dst as an
// executable.
func extractBinary(r io.Reader, dst string) error {
	gz, err := gzip.NewReader(r)
	if err != nil {
		return errs.WrapMessage("release archive is not gzip", err)
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	for {
		h, err := tr.Next()
		if err == io.EOF {
			return errs.New("release archive has no ast-mcp binary")
		}
		if err != nil {
			return errs.WrapMessage("failed to read the release archive", err)
		}
		if h.Typeflag != tar.TypeReg || filepath.Clean(h.Name) != binaryName {
			continue
		}
		f, err := os.OpenFile(dst, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o755)
		if err != nil {
			return errs.WrapMessage("failed to write the new ast-mcp", err)
		}
		if _, err := io.Copy(f, io.LimitReader(tr, maxArchiveBytes)); err != nil {
			f.Close()
			return errs.WrapMessage("failed to extract ast-mcp", err)
		}
		return f.Close()
	}
}
