// Package ignorefiles evaluates a project's in-tree ignore files — .gitignore
// (nested, git semantics), .astignore (nested, gitignore syntax, ast-context-cache
// specific) and a root .stignore (Syncthing semantics) — plus an optional list of
// per-project patterns (gitignore syntax, anchored at the project root).
//
// Precedence, highest first:
//  1. per-project patterns (dashboard Settings → Projects → Excludes)
//  2. .astignore files, deepest directory first
//  3. .gitignore files, deepest directory first
//  4. root .stignore
//
// A match in (1) or (2) is decisive, so "!generated/" in .astignore re-includes a
// gitignored directory. Otherwise a path is ignored when .gitignore or .stignore
// ignores it. Within one gitignore-syntax file the last matching line wins; within
// .stignore the first matching line wins (Syncthing semantics).
//
// Callers walking a tree should prune a directory as soon as MatchEntry reports it
// ignored; like git, contents of an ignored directory cannot be re-included.
package ignorefiles

import (
	"bufio"
	"os"
	"path"
	"path/filepath"
	"strings"
	"sync"
)

const (
	GitIgnore       = ".gitignore"
	AstIgnore       = ".astignore"
	SyncthingIgnore = ".stignore"
)

// IsIgnoreFileName reports whether base is one of the ignore file names this
// package reads, so file watchers can invalidate cached matchers when one changes.
func IsIgnoreFileName(base string) bool {
	return base == GitIgnore || base == AstIgnore || base == SyncthingIgnore
}

type rule struct {
	segs     []string // pattern split on "/", may contain "**"
	negate   bool
	dirOnly  bool
	anchored bool // false: match against the basename only
	fold     bool // case-insensitive (.stignore "(?i)")
}

type level struct {
	git []rule
	ast []rule
}

// Matcher answers ignore queries for paths under one project root. It lazily
// loads per-directory ignore files and caches them for its lifetime; build a new
// Matcher (or drop a cached one) after an ignore file changes.
type Matcher struct {
	root  string
	extra []rule
	st    []rule
	mu    sync.Mutex
	dirs  map[string]*level // slash rel dir ("" = root) → rules; nil entry = none
}

// New returns a Matcher for root. extra are per-project patterns in gitignore
// syntax, anchored at root like a root-level .gitignore.
func New(root string, extra []string) *Matcher {
	root = filepath.Clean(root)
	m := &Matcher{root: root, dirs: map[string]*level{}}
	for _, p := range extra {
		if r, ok := parseGitLine(p); ok {
			m.extra = append(m.extra, r)
		}
	}
	m.st = loadStignore(filepath.Join(root, SyncthingIgnore), 0)
	return m
}

// Root returns the project root this Matcher evaluates paths against.
func (m *Matcher) Root() string { return m.root }

// MatchEntry reports whether absPath is ignored by the rules that apply to it,
// without checking whether an ancestor directory is ignored. Use it inside a walk
// that already pruned ignored directories.
func (m *Matcher) MatchEntry(absPath string, isDir bool) bool {
	rel, ok := m.rel(absPath)
	if !ok || rel == "" {
		return false
	}
	return m.match(rel, isDir)
}

// Excluded reports whether absPath or any of its ancestor directories below the
// root is ignored. Use it for one-off checks (file events, purges) outside a walk.
func (m *Matcher) Excluded(absPath string, isDir bool) bool {
	rel, ok := m.rel(absPath)
	if !ok || rel == "" {
		return false
	}
	parts := strings.Split(rel, "/")
	for i := 1; i < len(parts); i++ {
		if m.match(strings.Join(parts[:i], "/"), true) {
			return true
		}
	}
	return m.match(rel, isDir)
}

func (m *Matcher) rel(absPath string) (string, bool) {
	r, err := filepath.Rel(m.root, filepath.Clean(absPath))
	if err != nil {
		return "", false
	}
	r = filepath.ToSlash(r)
	if r == "." {
		return "", true
	}
	if r == ".." || strings.HasPrefix(r, "../") {
		return "", false
	}
	return r, true
}

func (m *Matcher) match(rel string, isDir bool) bool {
	if hit, ign := lastMatch(m.extra, rel, isDir); hit {
		return ign
	}
	levels := m.levelsFor(path.Dir(rel))
	for _, lv := range levels {
		if hit, ign := lastMatch(lv.rules.ast, relTo(rel, lv.dir), isDir); hit {
			return ign
		}
	}
	for _, lv := range levels {
		if hit, ign := lastMatch(lv.rules.git, relTo(rel, lv.dir), isDir); hit {
			if ign {
				return true
			}
			break
		}
	}
	for _, r := range m.st {
		if r.matches(rel, isDir) {
			return !r.negate
		}
	}
	return false
}

type dirLevel struct {
	dir   string
	rules *level
}

// levelsFor returns rule levels for dir and its ancestors, deepest first.
func (m *Matcher) levelsFor(dir string) []dirLevel {
	if dir == "." {
		dir = ""
	}
	var out []dirLevel
	for {
		if lv := m.load(dir); lv != nil {
			out = append(out, dirLevel{dir: dir, rules: lv})
		}
		if dir == "" {
			return out
		}
		dir = path.Dir(dir)
		if dir == "." {
			dir = ""
		}
	}
}

func (m *Matcher) load(dir string) *level {
	m.mu.Lock()
	defer m.mu.Unlock()
	if lv, ok := m.dirs[dir]; ok {
		return lv
	}
	abs := filepath.Join(m.root, filepath.FromSlash(dir))
	lv := &level{git: loadGitFile(filepath.Join(abs, GitIgnore)), ast: loadGitFile(filepath.Join(abs, AstIgnore))}
	if len(lv.git) == 0 && len(lv.ast) == 0 {
		lv = nil
	}
	m.dirs[dir] = lv
	return lv
}

func relTo(rel, dir string) string {
	if dir == "" {
		return rel
	}
	return strings.TrimPrefix(rel, dir+"/")
}

func lastMatch(rules []rule, rel string, isDir bool) (hit, ignored bool) {
	for i := len(rules) - 1; i >= 0; i-- {
		if rules[i].matches(rel, isDir) {
			return true, !rules[i].negate
		}
	}
	return false, false
}

func (r rule) matches(rel string, isDir bool) bool {
	if r.dirOnly && !isDir {
		return false
	}
	if r.fold {
		rel = strings.ToLower(rel)
	}
	if !r.anchored {
		return matchSegs(r.segs, []string{path.Base(rel)})
	}
	return matchSegs(r.segs, strings.Split(rel, "/"))
}

func matchSegs(pat, parts []string) bool {
	for len(pat) > 0 {
		if pat[0] == "**" {
			if len(pat) == 1 {
				return len(parts) > 0
			}
			for i := 0; i <= len(parts); i++ {
				if matchSegs(pat[1:], parts[i:]) {
					return true
				}
			}
			return false
		}
		if len(parts) == 0 {
			return false
		}
		if ok, err := path.Match(pat[0], parts[0]); err != nil || !ok {
			return false
		}
		pat, parts = pat[1:], parts[1:]
	}
	return len(parts) == 0
}

func readLines(file string) []string {
	f, err := os.Open(file)
	if err != nil {
		return nil
	}
	defer f.Close()
	var out []string
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		out = append(out, sc.Text())
	}
	return out
}

func loadGitFile(file string) []rule {
	var out []rule
	for _, line := range readLines(file) {
		if r, ok := parseGitLine(line); ok {
			out = append(out, r)
		}
	}
	return out
}

// parseGitLine parses one gitignore-syntax line (see gitignore(5)).
func parseGitLine(line string) (rule, bool) {
	line = strings.TrimSuffix(line, "\r")
	if strings.HasPrefix(line, "#") {
		return rule{}, false
	}
	line = trimTrailingSpaces(line)
	var r rule
	if strings.HasPrefix(line, "!") {
		r.negate = true
		line = line[1:]
	} else if strings.HasPrefix(line, `\!`) || strings.HasPrefix(line, `\#`) {
		line = line[1:]
	}
	if strings.HasSuffix(line, "/") {
		r.dirOnly = true
		line = strings.TrimRight(line, "/")
	}
	if strings.HasPrefix(line, "/") {
		r.anchored = true
		line = strings.TrimLeft(line, "/")
	}
	if line == "" {
		return rule{}, false
	}
	if strings.Contains(line, "/") {
		r.anchored = true
	}
	r.segs = compileSegs(line)
	return r, true
}

// trimTrailingSpaces drops unescaped trailing spaces ("foo\ " keeps one space).
func trimTrailingSpaces(s string) string {
	t := strings.TrimRight(s, " ")
	if len(t) < len(s) && strings.HasSuffix(t, `\`) {
		return t[:len(t)-1] + " "
	}
	return t
}

func compileSegs(p string) []string {
	var out []string
	for _, s := range strings.Split(p, "/") {
		if s == "" || (s == "**" && len(out) > 0 && out[len(out)-1] == "**") {
			continue
		}
		// gitignore bracket negation "[!x]" → path.Match "[^x]".
		out = append(out, strings.ReplaceAll(s, "[!", "[^"))
	}
	return out
}

// loadStignore parses a Syncthing .stignore: "//" comments, "#include file",
// "!" / "(?i)" / "(?d)" prefixes, "/" anchors to the folder root, otherwise a
// pattern matches at any depth. depth guards #include recursion.
func loadStignore(file string, depth int) []rule {
	var out []rule
	for _, line := range readLines(file) {
		line = strings.TrimSpace(strings.TrimSuffix(line, "\r"))
		if line == "" || strings.HasPrefix(line, "//") {
			continue
		}
		if strings.HasPrefix(line, "#include ") {
			if depth < 4 {
				inc := strings.TrimSpace(strings.TrimPrefix(line, "#include "))
				if !filepath.IsAbs(inc) {
					inc = filepath.Join(filepath.Dir(file), inc)
				}
				out = append(out, loadStignore(inc, depth+1)...)
			}
			continue
		}
		if strings.HasPrefix(line, "#") {
			continue
		}
		var r rule
		for {
			switch {
			case strings.HasPrefix(line, "!"):
				r.negate = true
				line = line[1:]
				continue
			case strings.HasPrefix(line, "(?i)"):
				r.fold = true
				line = line[4:]
				continue
			case strings.HasPrefix(line, "(?d)"):
				line = line[4:]
				continue
			}
			break
		}
		line = strings.TrimRight(line, "/")
		anchored := strings.HasPrefix(line, "/")
		line = strings.TrimLeft(line, "/")
		if line == "" {
			continue
		}
		if r.fold {
			line = strings.ToLower(line)
		}
		r.anchored = true
		r.segs = compileSegs(line)
		if !anchored {
			r.segs = append([]string{"**"}, r.segs...)
		}
		out = append(out, r)
	}
	return out
}
