// Package ortinstall downloads, verifies and installs a pinned ONNX Runtime release into the
// data directory, so the local (onnx) embedder works without a system-wide ORT install.
package ortinstall

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// Version is the ONNX Runtime release ast-mcp installs. onnxruntime_go v1.30.1 needs C API 25.
const Version = "1.31.0"

// MinVersion is the oldest ONNX Runtime with the C API version onnxruntime_go needs.
const MinVersion = "1.25.0"

// DefaultBaseURL is the GitHub release download prefix; assets live at <base>/v<ver>/<asset>.
const DefaultBaseURL = "https://github.com/microsoft/onnxruntime/releases/download"

const (
	dirPrefix        = "onnxruntime-"
	maxDownloadBytes = 256 << 20
	maxExtractBytes  = 512 << 20
	downloadTimeout  = 10 * time.Minute
)

// checksums pins the sha256 of each release tarball, by version then platform. Values are the
// GitHub release asset digests for microsoft/onnxruntime, re-verified by hashing the downloads.
var checksums = map[string]map[string]string{
	"1.31.0": {
		"linux-x64":     "cc5c72baf5ae5c8238a6841f0897227be2d02826b9cf98eaf02fdefaeeb03a57",
		"linux-aarch64": "c5b8b3cca31f3d643a3b313b8f42e2d0f16a4fcefd02cda343d97bd5306afaab",
		"osx-arm64":     "1031b36dd3bdaa8d976a6130d9cf2a05e9be320b93ae804f8427b91dc3432afc",
	},
}

// installing holds the lib roots with an install in progress.
var installing sync.Map

// Installation is a managed ONNX Runtime install.
type Installation struct {
	Version string `json:"version"`
	Dir     string `json:"dir"`
	Lib     string `json:"lib"`
}

// Installer downloads and installs ONNX Runtime under DataDir/lib. Zero fields use defaults.
type Installer struct {
	DataDir string
	BaseURL string
	Client  *http.Client
	GOOS    string
	GOARCH  string
	// Checksums overrides the pinned digests (tests only).
	Checksums map[string]map[string]string
}

// Platform maps GOOS/GOARCH to the release asset platform name.
func Platform(goos, goarch string) (string, error) {
	switch goos + "/" + goarch {
	case "linux/amd64":
		return "linux-x64", nil
	case "linux/arm64":
		return "linux-aarch64", nil
	case "darwin/arm64":
		return "osx-arm64", nil
	}
	return "", errs.NewCode(errs.CodeUnsupported, "no managed ONNX Runtime build for "+goos+"/"+goarch, "goos", goos, "goarch", goarch)
}

// LibName is the library file name the loader opens on goos.
func LibName(goos string) string {
	if goos == "darwin" {
		return "libonnxruntime.dylib"
	}
	return "libonnxruntime.so"
}

// LibRoot is the directory managed installs live under.
func LibRoot(dataDir string) string {
	return filepath.Join(dataDir, "lib")
}

// InstallDir is where version ver is installed.
func InstallDir(dataDir, ver string) string {
	return filepath.Join(LibRoot(dataDir), dirPrefix+ver)
}

// Supported reports whether this platform has a pinned managed build.
func Supported() bool {
	p, err := Platform(runtime.GOOS, runtime.GOARCH)
	return err == nil && checksums[Version][p] != ""
}

// Installed returns the newest managed install with a library file and a version >= MinVersion.
func Installed(dataDir string) (Installation, bool) {
	return installedFor(dataDir, runtime.GOOS)
}

func installedFor(dataDir, goos string) (Installation, bool) {
	all := listInstalled(dataDir, goos)
	if len(all) == 0 {
		return Installation{}, false
	}
	return all[0], true
}

// listInstalled returns usable managed installs, newest first.
func listInstalled(dataDir, goos string) []Installation {
	entries, err := os.ReadDir(LibRoot(dataDir))
	if err != nil {
		return nil
	}
	var out []Installation
	for _, e := range entries {
		ver, ok := strings.CutPrefix(e.Name(), dirPrefix)
		if !ok || !e.IsDir() || !validVersion(ver) || CompareVersions(ver, MinVersion) < 0 {
			continue
		}
		dir := filepath.Join(LibRoot(dataDir), e.Name())
		lib := filepath.Join(dir, LibName(goos))
		if _, err := os.Stat(lib); err != nil {
			continue
		}
		out = append(out, Installation{Version: ver, Dir: dir, Lib: lib})
	}
	sort.Slice(out, func(i, j int) bool { return CompareVersions(out[i].Version, out[j].Version) > 0 })
	return out
}

// UpdateAvailable reports whether the pinned Version is newer than installed.
func UpdateAvailable(installed string) bool {
	return installed != "" && CompareVersions(Version, installed) > 0
}

// VersionFromLibPath returns the managed version encoded in a library path under dataDir, or "".
func VersionFromLibPath(dataDir, lib string) string {
	rel, err := filepath.Rel(LibRoot(dataDir), lib)
	if err != nil || strings.HasPrefix(rel, "..") {
		return ""
	}
	first, _, _ := strings.Cut(filepath.ToSlash(rel), "/")
	ver, ok := strings.CutPrefix(first, dirPrefix)
	if !ok {
		return ""
	}
	return ver
}

// Prune removes managed installs other than keep. Call it once keep has loaded.
func Prune(dataDir, keep string) {
	entries, err := os.ReadDir(LibRoot(dataDir))
	if err != nil {
		return
	}
	for _, e := range entries {
		ver, ok := strings.CutPrefix(e.Name(), dirPrefix)
		if !ok || ver == keep || !e.IsDir() {
			continue
		}
		p := filepath.Join(LibRoot(dataDir), e.Name())
		if err := os.RemoveAll(p); err != nil {
			logger.Warn("Failed to remove old ONNX Runtime install", "path", p, "error", err)
			continue
		}
		logger.Info("Removed old ONNX Runtime install", "path", p, "kept", keep)
	}
}

// CompareVersions compares dotted numeric versions; missing or non-numeric parts count as 0.
func CompareVersions(a, b string) int {
	pa, pb := strings.Split(a, "."), strings.Split(b, ".")
	for i := 0; i < len(pa) || i < len(pb); i++ {
		x, y := versionPart(pa, i), versionPart(pb, i)
		if x != y {
			if x < y {
				return -1
			}
			return 1
		}
	}
	return 0
}

// validVersion reports whether v is a dotted numeric version like 1.31.0.
func validVersion(v string) bool {
	for _, p := range strings.Split(v, ".") {
		if _, err := strconv.Atoi(p); err != nil || p == "" {
			return false
		}
	}
	return v != ""
}

func versionPart(parts []string, i int) int {
	if i >= len(parts) {
		return 0
	}
	n, _ := strconv.Atoi(parts[i])
	return n
}

// Install downloads the pinned Version, verifies its sha256 and installs it.
func (i *Installer) Install(ctx context.Context) (Installation, error) {
	return i.InstallVersion(ctx, Version)
}

// InstallVersion downloads ver, refuses it unless its sha256 matches the pinned digest, and
// extracts the libraries into DataDir/lib/onnxruntime-<ver>/ through a temp dir and a rename.
// Other installed versions are left in place.
func (i *Installer) InstallVersion(ctx context.Context, ver string) (Installation, error) {
	if strings.TrimSpace(i.DataDir) == "" {
		return Installation{}, errs.NewCode(errs.CodeInvalidInput, "data directory is required")
	}
	root := LibRoot(i.DataDir)
	if _, busy := installing.LoadOrStore(root, struct{}{}); busy {
		return Installation{}, errs.NewCode(errs.CodeConflict, "an ONNX Runtime install is already running", "path", root)
	}
	defer installing.Delete(root)
	goos, goarch := i.platform()
	plat, err := Platform(goos, goarch)
	if err != nil {
		return Installation{}, err
	}
	want := i.checksum(ver, plat)
	if !validVersion(ver) {
		return Installation{}, errs.NewCode(errs.CodeInvalidInput, "invalid ONNX Runtime version "+ver, "version", ver)
	}
	if want == "" {
		return Installation{}, errs.NewCode(errs.CodeUnsupported, fmt.Sprintf("no pinned checksum for ONNX Runtime %s on %s", ver, plat), "version", ver, "platform", plat)
	}
	if err := os.MkdirAll(root, 0o755); err != nil {
		return Installation{}, errs.WrapMessage("failed to create ONNX Runtime directory", err, "path", root)
	}
	asset := "onnxruntime-" + plat + "-" + ver + ".tgz"
	tgz, err := i.download(ctx, ver, asset, want, root)
	if err != nil {
		return Installation{}, err
	}
	defer os.Remove(tgz)
	tmp, err := os.MkdirTemp(root, "."+dirPrefix+ver+".tmp-")
	if err != nil {
		return Installation{}, errs.WrapMessage("failed to create temp install directory", err, "path", root)
	}
	defer os.RemoveAll(tmp)
	if err := extractLibs(tgz, tmp); err != nil {
		return Installation{}, err
	}
	if _, err := os.Stat(filepath.Join(tmp, LibName(goos))); err != nil {
		return Installation{}, errs.WrapMessage("ONNX Runtime archive has no "+LibName(goos), err, "asset", asset)
	}
	final := InstallDir(i.DataDir, ver)
	if err := replaceDir(tmp, final); err != nil {
		return Installation{}, err
	}
	logger.Info("Installed ONNX Runtime", "version", ver, "path", final)
	return Installation{Version: ver, Dir: final, Lib: filepath.Join(final, LibName(goos))}, nil
}

func (i *Installer) platform() (string, string) {
	goos, goarch := i.GOOS, i.GOARCH
	if goos == "" {
		goos = runtime.GOOS
	}
	if goarch == "" {
		goarch = runtime.GOARCH
	}
	return goos, goarch
}

func (i *Installer) checksum(ver, plat string) string {
	if i.Checksums != nil {
		return i.Checksums[ver][plat]
	}
	return checksums[ver][plat]
}

// download streams the asset into a temp file under dir and returns its path once the sha256
// matches want. A mismatch removes the file and fails.
func (i *Installer) download(ctx context.Context, ver, asset, want, dir string) (string, error) {
	base := strings.TrimRight(i.BaseURL, "/")
	if base == "" {
		base = DefaultBaseURL
	}
	url := base + "/v" + ver + "/" + asset
	client := i.Client
	if client == nil {
		client = &http.Client{Timeout: downloadTimeout}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", errs.WrapMessage("failed to build ONNX Runtime download request", err, "url", url)
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", errs.WrapMessage("failed to download ONNX Runtime", err, "url", url)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", errs.New(fmt.Sprintf("failed to download ONNX Runtime: HTTP %d from %s", resp.StatusCode, url), "url", url, "status", resp.StatusCode)
	}
	f, err := os.CreateTemp(dir, "."+asset+".download-")
	if err != nil {
		return "", errs.WrapMessage("failed to create download file", err, "path", dir)
	}
	h := sha256.New()
	n, err := io.Copy(io.MultiWriter(f, h), io.LimitReader(resp.Body, maxDownloadBytes+1))
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err == nil && n > maxDownloadBytes {
		err = errs.New("ONNX Runtime download exceeds size limit", "limit", maxDownloadBytes)
	}
	if err != nil {
		os.Remove(f.Name())
		return "", errs.WrapMessage("failed to download ONNX Runtime", err, "url", url)
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != want {
		os.Remove(f.Name())
		return "", errs.NewCode(errs.CodeConflict, fmt.Sprintf("ONNX Runtime checksum mismatch for %s: got sha256 %s, want %s", asset, got, want), "asset", asset, "got", got, "want", want)
	}
	return f.Name(), nil
}

// extractLibs copies the library files directly under the archive's <top>/lib/ into dst:
// regular files and symlinks whose target is a sibling file. Everything else (headers, cmake,
// dSYM bundles) is skipped, so no entry can write outside dst.
func extractLibs(tgz, dst string) error {
	f, err := os.Open(tgz)
	if err != nil {
		return errs.WrapMessage("failed to open ONNX Runtime archive", err, "path", tgz)
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return errs.WrapMessage("failed to read ONNX Runtime archive", err, "path", tgz)
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	var total int64
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return errs.WrapMessage("failed to read ONNX Runtime archive", err, "path", tgz)
		}
		name, ok := libEntryName(hdr.Name)
		if !ok {
			continue
		}
		target := filepath.Join(dst, name)
		switch hdr.Typeflag {
		case tar.TypeSymlink:
			if hdr.Linkname == "" || strings.ContainsAny(hdr.Linkname, `/\`) || hdr.Linkname == ".." || hdr.Linkname == "." {
				continue
			}
			if err := os.Symlink(hdr.Linkname, target); err != nil {
				return errs.WrapMessage("failed to create ONNX Runtime symlink", err, "path", target)
			}
		case tar.TypeReg:
			total += hdr.Size
			if total > maxExtractBytes {
				return errs.New("ONNX Runtime archive exceeds extract size limit", "limit", maxExtractBytes)
			}
			if err := writeFile(target, tr, hdr.Size); err != nil {
				return err
			}
		}
	}
}

// libEntryName returns the file name of an archive entry that sits directly in <top>/lib/ and
// names an onnxruntime library.
func libEntryName(name string) (string, bool) {
	parts := strings.Split(strings.TrimPrefix(path.Clean(strings.TrimPrefix(name, "./")), "/"), "/")
	if len(parts) != 3 || parts[1] != "lib" || !strings.HasPrefix(parts[2], "libonnxruntime") {
		return "", false
	}
	base := parts[2]
	if strings.HasSuffix(base, ".dSYM") || base == "." || base == ".." {
		return "", false
	}
	return base, true
}

func writeFile(target string, r io.Reader, size int64) error {
	out, err := os.OpenFile(target, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o755)
	if err != nil {
		return errs.WrapMessage("failed to create ONNX Runtime file", err, "path", target)
	}
	_, err = io.CopyN(out, r, size)
	if cerr := out.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return errs.WrapMessage("failed to write ONNX Runtime file", err, "path", target)
	}
	return nil
}

// replaceDir moves src to dst. An existing dst (a reinstall of the same version) is moved
// aside first and removed only after src is in place, and restored if the move fails.
func replaceDir(src, dst string) error {
	if _, err := os.Stat(dst); err != nil {
		if err := os.Rename(src, dst); err != nil {
			return errs.WrapMessage("failed to move ONNX Runtime into place", err, "path", dst)
		}
		return nil
	}
	old := filepath.Join(filepath.Dir(dst), "."+filepath.Base(dst)+".old-"+strconv.FormatInt(time.Now().UnixNano(), 36))
	if err := os.Rename(dst, old); err != nil {
		return errs.WrapMessage("failed to move previous ONNX Runtime aside", err, "path", dst)
	}
	if err := os.Rename(src, dst); err != nil {
		os.Rename(old, dst)
		return errs.WrapMessage("failed to move ONNX Runtime into place", err, "path", dst)
	}
	os.RemoveAll(old)
	return nil
}
