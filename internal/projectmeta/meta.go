package projectmeta

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/watcher"
	"gopkg.in/yaml.v3"
)

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
		RootDir  string `yaml:"root_dir"`
		MaxDepth int    `yaml:"max_depth"`
	} `yaml:"discovery"`
}

// Enrich returns display metadata for an indexed project path.
func Enrich(projectPath string) Info {
	path := watcher.NormalizeProjectPath(projectPath)
	if path == "" {
		return Info{}
	}
	repoName := filepath.Base(path)
	branch, commonDir := gitMeta(path)
	workspace := workspaceForPath(path)
	repoKey := path
	if commonDir != "" {
		if abs, err := filepath.Abs(filepath.Join(path, commonDir)); err == nil {
			repoKey = filepath.Clean(abs)
		}
	}
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

// DiscoverPaths returns repo roots from WTG spaces and the configured discovery root.
func DiscoverPaths() []string {
	cfg := loadWTGConfig()
	seen := map[string]bool{}
	var out []string
	add := func(p string) {
		p = watcher.NormalizeProjectPath(p)
		if p == "" || seen[p] {
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
	spacesRoot := expandHome(cfg.Spaces.RootDir)
	if spacesRoot == "" {
		spacesRoot = expandHome("~/spaces")
	}
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
			add(filepath.Dir(path))
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

func gitMeta(dir string) (branch, commonDir string) {
	if out, err := exec.Command("git", "-C", dir, "rev-parse", "--abbrev-ref", "HEAD").Output(); err == nil {
		branch = strings.TrimSpace(string(out))
	}
	if out, err := exec.Command("git", "-C", dir, "rev-parse", "--git-common-dir").Output(); err == nil {
		commonDir = strings.TrimSpace(string(out))
	}
	return branch, commonDir
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
