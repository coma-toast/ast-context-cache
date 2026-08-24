package projectmeta

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/repokey"
	"github.com/coma-toast/ast-context-cache/internal/watcher"
	"gopkg.in/yaml.v3"
)

const enrichCacheTTL = 45 * time.Second

var (
	enrichMu    sync.Mutex
	enrichCache = map[string]enrichCacheEntry{}
)

type enrichCacheEntry struct {
	at   time.Time
	info Info
}

// Info describes a repo checkout for dashboard display and WTG grouping.
type Info struct {
	Path      string
	RepoName  string
	Branch    string
	Workspace string
	RepoKey   string
	Label     string
}

type wtgConfig struct {
	Spaces struct {
		RootDir string `yaml:"root_dir"`
	} `yaml:"spaces"`
	Discovery struct {
		RootDir  string   `yaml:"root_dir"`
		MaxDepth int      `yaml:"max_depth"`
		Exclude  []string `yaml:"exclude"`
	} `yaml:"discovery"`
}

// Enrich returns display metadata for an indexed project path.
func Enrich(projectPath string) Info {
	path := watcher.NormalizeProjectPath(projectPath)
	if path == "" {
		return Info{}
	}
	enrichMu.Lock()
	e, ok := enrichCache[path]
	fresh := ok && time.Since(e.at) < enrichCacheTTL
	var info Info
	if fresh {
		info = e.info
		enrichMu.Unlock()
	} else {
		enrichMu.Unlock()
		info = enrichFresh(path)
		enrichMu.Lock()
		enrichCache[path] = enrichCacheEntry{at: time.Now(), info: info}
		enrichMu.Unlock()
	}
	if custom := displayNameOverride(path); custom != "" {
		info.Label = custom
	} else if fresh {
		// Custom may have been cleared while cache still holds an old overridden label;
		// rebuild auto label from cached repo metadata.
		info.Label = autoLabel(info.RepoName, info.Workspace, info.Branch)
	}
	return info
}

// Invalidate clears cached enrich metadata for a path (or all if path is empty).
func Invalidate(projectPath string) {
	path := watcher.NormalizeProjectPath(projectPath)
	enrichMu.Lock()
	defer enrichMu.Unlock()
	if path == "" {
		enrichCache = map[string]enrichCacheEntry{}
		return
	}
	delete(enrichCache, path)
}

// SetDisplayNameOverrideFunc registers how custom labels are loaded (set from db at init).
func SetDisplayNameOverrideFunc(fn func(projectPath string) string) {
	displayNameMu.Lock()
	displayNameFn = fn
	displayNameMu.Unlock()
}

var (
	displayNameMu sync.Mutex
	displayNameFn func(string) string
)

func displayNameOverride(path string) string {
	displayNameMu.Lock()
	fn := displayNameFn
	displayNameMu.Unlock()
	if fn == nil {
		return ""
	}
	return strings.TrimSpace(fn(path))
}

func enrichFresh(path string) Info {
	repoName := filepath.Base(path)
	branch := gitBranch(path)
	workspace := workspaceForPath(path)
	repoKey := repokey.Key(path)
	label := repoName
	if workspace != "" {
		label = repoName + " · " + workspace
	} else if branch != "" && branch != "HEAD" {
		label = repoName + " · " + branch
	}
	return Info{
		Path:      path,
		RepoName:  repoName,
		Branch:    branch,
		Workspace: workspace,
		RepoKey:   repoKey,
		Label:     label,
	}
}

func autoLabel(repoName, workspace, branch string) string {
	if repoName == "" {
		return ""
	}
	if workspace != "" {
		return repoName + " · " + workspace
	}
	if branch != "" && branch != "HEAD" {
		return repoName + " · " + branch
	}
	return repoName
}

// SpacesRoot returns the configured WTG spaces root directory (default ~/spaces).
func SpacesRoot() string {
	root := expandHome(loadWTGConfig().Spaces.RootDir)
	if root == "" {
		root = expandHome("~/spaces")
	}
	return root
}

// ListSpaceRepoPaths returns every repo checkout directly inside a WTG space,
// whether or not it has ever been indexed or watched. WTG worktrees carry .git as
// a file rather than a directory, so presence alone marks a checkout.
func ListSpaceRepoPaths(spaceName string) ([]string, error) {
	spaceName = strings.TrimSpace(spaceName)
	if spaceName == "" {
		return nil, fmt.Errorf("space required")
	}
	if spaceName != filepath.Base(spaceName) || spaceName == "." || spaceName == ".." {
		return nil, fmt.Errorf("invalid space name: %s", spaceName)
	}
	root := SpacesRoot()
	if root == "" {
		return nil, fmt.Errorf("no WTG spaces root configured")
	}
	spaceDir := filepath.Join(root, spaceName)
	entries, err := os.ReadDir(spaceDir)
	if err != nil {
		return nil, fmt.Errorf("space not found: %s", spaceName)
	}
	var out []string
	for _, e := range entries {
		if !e.IsDir() || strings.HasPrefix(e.Name(), ".") {
			continue
		}
		p := watcher.NormalizeProjectPath(filepath.Join(spaceDir, e.Name()))
		if p == "" || !isGitRepo(p) {
			continue
		}
		out = append(out, p)
	}
	sort.Strings(out)
	return out, nil
}

// DiscoverPaths returns repo roots from WTG spaces and the configured discovery root.
func DiscoverPaths() []string {
	cfg := loadWTGConfig()
	seen := map[string]bool{}
	var out []string
	add := func(p string) {
		p = watcher.NormalizeProjectPath(p)
		if p == "" || seen[p] || IsExcluded(p) {
			return
		}
		if st, err := os.Stat(p); err != nil || !st.IsDir() {
			return
		}
		if !isGitRepo(p) {
			return
		}
		seen[p] = true
		out = append(out, p)
	}
	spacesRoot := SpacesRoot()
	if entries, err := os.ReadDir(spacesRoot); err == nil {
		for _, ws := range entries {
			if !ws.IsDir() || strings.HasPrefix(ws.Name(), ".") {
				continue
			}
			wsPath := filepath.Join(spacesRoot, ws.Name())
			repos, err := os.ReadDir(wsPath)
			if err != nil {
				continue
			}
			for _, repo := range repos {
				if !repo.IsDir() || strings.HasPrefix(repo.Name(), ".") {
					continue
				}
				add(filepath.Join(wsPath, repo.Name()))
			}
		}
	}
	discoveryRoot := expandHome(cfg.Discovery.RootDir)
	if discoveryRoot == "" {
		discoveryRoot = expandHome("~/git")
	}
	maxDepth := cfg.Discovery.MaxDepth
	if maxDepth <= 0 {
		maxDepth = 2
	}
	walkDiscovery(discoveryRoot, maxDepth, add)
	return out
}

func walkDiscovery(root string, maxDepth int, add func(string)) {
	root = watcher.NormalizeProjectPath(root)
	if root == "" {
		return
	}
	filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() && d.Name() == ".git" {
			candidate := filepath.Dir(path)
			if IsExcluded(candidate) {
				return filepath.SkipDir
			}
			add(candidate)
			return filepath.SkipDir
		}
		if d.IsDir() && IsExcluded(path) {
			return filepath.SkipDir
		}
		if !d.IsDir() {
			return nil
		}
		rel, _ := filepath.Rel(root, path)
		depth := 0
		if rel != "." {
			depth = strings.Count(rel, string(os.PathSeparator)) + 1
		}
		if depth > maxDepth {
			return filepath.SkipDir
		}
		return nil
	})
}

func workspaceForPath(path string) string {
	cfg := loadWTGConfig()
	spacesRoot := expandHome(cfg.Spaces.RootDir)
	if spacesRoot == "" {
		spacesRoot = expandHome("~/spaces")
	}
	spacesRoot = watcher.NormalizeProjectPath(spacesRoot)
	if spacesRoot == "" || !strings.HasPrefix(path, spacesRoot+string(os.PathSeparator)) {
		return ""
	}
	rel, err := filepath.Rel(spacesRoot, path)
	if err != nil {
		return ""
	}
	parts := strings.Split(rel, string(os.PathSeparator))
	if len(parts) >= 2 {
		return parts[0]
	}
	return ""
}

func gitBranch(dir string) string {
	out, err := exec.Command("git", "-C", dir, "rev-parse", "--abbrev-ref", "HEAD").Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func isGitRepo(dir string) bool {
	_, err := os.Stat(filepath.Join(dir, ".git"))
	return err == nil
}

func loadWTGConfig() wtgConfig {
	var cfg wtgConfig
	for _, p := range wtgConfigPaths() {
		data, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		if yaml.Unmarshal(data, &cfg) == nil {
			return cfg
		}
	}
	return cfg
}

func wtgConfigPaths() []string {
	if p := strings.TrimSpace(os.Getenv("WTG_CONFIG")); p != "" {
		return []string{expandHome(p)}
	}
	home, _ := os.UserHomeDir()
	if home == "" {
		return nil
	}
	return []string{
		filepath.Join(home, ".config", "wtg", "config.yaml"),
		filepath.Join(home, ".wtg", "config.yaml"),
	}
}

func expandHome(p string) string {
	p = strings.TrimSpace(p)
	if p == "" {
		return ""
	}
	if strings.HasPrefix(p, "~/") {
		home, _ := os.UserHomeDir()
		if home == "" {
			return ""
		}
		return filepath.Clean(filepath.Join(home, p[2:]))
	}
	return filepath.Clean(p)
}
