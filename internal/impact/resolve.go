package impact

import (
	"path/filepath"
	"strings"
)

// moduleMatch says how an import specifier relates to a file defining a symbol.
type moduleMatch int

const (
	matchNone moduleMatch = iota
	// matchPackage: the specifier names a package/directory that contains the
	// defining file (`from model_manager import x`, `import {x} from './components'`),
	// so only an explicit import of the symbol's name ties it to the definition.
	matchPackage
	// matchFile: the specifier loads the defining file (or, for Go/HCL, its
	// package directory) itself.
	matchFile
)

// resolveModule compares an import specifier written in srcFile against defFile
// by whole path segments, the way the source language resolves imports. Nothing
// is matched by substring: "eslint/config" never reaches organize/config.py, and a
// Python module path only reaches a file whose trailing directories spell it.
// Absolute specifiers are matched as a trailing-segment suffix because the import
// root (sys.path, a Go module path, a TS path alias) is not known to the index.
func resolveModule(module, srcFile, defFile string) moduleMatch {
	if module == "" || defFile == "" {
		return matchNone
	}
	srcLang, defLang := langFamily(srcFile), langFamily(defFile)
	switch srcLang {
	case "python":
		if defLang != "python" {
			return matchNone
		}
		return resolvePython(module, srcFile, defFile)
	case "js":
		if defLang != "js" {
			return matchNone
		}
		return resolveJS(module, srcFile, defFile)
	case "go":
		if defLang != "go" || module == "C" {
			return matchNone
		}
		// A Go import names a package directory: its trailing path segments
		// must be the defining file's directory.
		if hasSegmentSuffix(splitSegs(filepath.Dir(defFile)), lastSegs(splitSegs(module), 1)) {
			return matchFile
		}
		return matchNone
	case "hcl":
		if defLang != "hcl" || !isRelativePath(module) {
			return matchNone
		}
		if filepath.Clean(filepath.Join(filepath.Dir(srcFile), module)) == filepath.Dir(defFile) {
			return matchFile
		}
		return matchNone
	}
	return resolvePath(module, srcFile, defFile)
}

func resolvePython(module, srcFile, defFile string) moduleMatch {
	defNoExt := strings.TrimSuffix(defFile, filepath.Ext(defFile))
	isInit := filepath.Base(defNoExt) == "__init__"
	dots := len(module) - len(strings.TrimLeft(module, "."))
	rest := module[dots:]
	var restSegs []string
	if rest != "" {
		restSegs = strings.Split(rest, ".")
	}
	if dots > 0 {
		base := filepath.Dir(srcFile)
		for i := 1; i < dots; i++ {
			base = filepath.Dir(base)
		}
		target := filepath.Join(append([]string{base}, restSegs...)...)
		if defNoExt == target || (isInit && filepath.Dir(defFile) == target) {
			return matchFile
		}
		if isAncestorDir(target, filepath.Dir(defFile)) {
			return matchPackage
		}
		return matchNone
	}
	if len(restSegs) == 0 {
		return matchNone
	}
	if hasSegmentSuffix(splitSegs(defNoExt), restSegs) ||
		(isInit && hasSegmentSuffix(splitSegs(filepath.Dir(defFile)), restSegs)) {
		return matchFile
	}
	if ancestorHasSuffix(filepath.Dir(defFile), restSegs) {
		return matchPackage
	}
	return matchNone
}

func resolveJS(module, srcFile, defFile string) moduleMatch {
	defNoExt := strings.TrimSuffix(defFile, filepath.Ext(defFile))
	isIndex := filepath.Base(defNoExt) == "index"
	if isRelativePath(module) || strings.HasPrefix(module, "/") {
		target := module
		if !strings.HasPrefix(module, "/") {
			target = filepath.Join(filepath.Dir(srcFile), module)
		}
		target = stripJSExt(filepath.Clean(target))
		if defNoExt == target || (isIndex && filepath.Dir(defFile) == target) {
			return matchFile
		}
		if isAncestorDir(target, filepath.Dir(defFile)) {
			return matchPackage
		}
		return matchNone
	}
	// Bare specifier: a package ("react", "eslint/config") or a path alias
	// ("@/lib/api", "~/lib/api", "#lib/api"). Only a local file whose trailing
	// segments spell the whole specifier can be what it loads.
	spec := module
	for _, prefix := range []string{"@/", "~/", "#", "~"} {
		if strings.HasPrefix(spec, prefix) {
			spec = spec[len(prefix):]
			break
		}
	}
	segs := splitSegs(stripJSExt(spec))
	if len(segs) == 0 {
		return matchNone
	}
	if hasSegmentSuffix(splitSegs(defNoExt), segs) ||
		(isIndex && hasSegmentSuffix(splitSegs(filepath.Dir(defFile)), segs)) {
		return matchFile
	}
	if ancestorHasSuffix(filepath.Dir(defFile), segs) {
		return matchPackage
	}
	return matchNone
}

// resolvePath handles file-path specifiers (bash/fish `source`, Ansible
// include_tasks/roles): relative to the importing file, or else a trailing
// segment suffix of the defining file. Segments before a shell variable
// ("$DIR/lib/common.sh") are dropped since their value is unknown.
func resolvePath(module, srcFile, defFile string) moduleMatch {
	segs := splitSegs(module)
	for i := len(segs) - 1; i >= 0; i-- {
		if strings.ContainsAny(segs[i], "$~{}") {
			segs = segs[i+1:]
			break
		}
	}
	if len(segs) == 0 {
		return matchNone
	}
	if !strings.ContainsAny(module, "$~{}") && !strings.HasPrefix(module, "/") {
		if filepath.Clean(filepath.Join(filepath.Dir(srcFile), module)) == defFile {
			return matchFile
		}
	}
	if hasSegmentSuffix(splitSegs(defFile), segs) {
		return matchFile
	}
	// An Ansible role or directory include pulls in every file below it.
	if filepath.Ext(module) == "" && ancestorHasSuffix(filepath.Dir(defFile), segs) {
		return matchFile
	}
	return matchNone
}

// moduleNamesSymbol reports whether symbol is one of the module path's own
// segments, for callers asking about a module or package rather than a
// declaration ("switch" in model_manager.switch, "impact" in .../internal/impact).
func moduleNamesSymbol(module, srcFile, symbol string) bool {
	var segs []string
	if langFamily(srcFile) == "python" {
		segs = strings.Split(strings.TrimLeft(module, "."), ".")
	} else {
		segs = splitSegs(stripJSExt(module))
	}
	for _, s := range segs {
		if s == symbol {
			return true
		}
	}
	return false
}

func langFamily(file string) string {
	switch strings.ToLower(filepath.Ext(file)) {
	case ".py", ".pyi":
		return "python"
	case ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts":
		return "js"
	case ".go":
		return "go"
	case ".tf", ".tfvars":
		return "hcl"
	}
	return ""
}

func stripJSExt(p string) string {
	switch strings.ToLower(filepath.Ext(p)) {
	case ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts":
		return strings.TrimSuffix(p, filepath.Ext(p))
	}
	return p
}

func isRelativePath(p string) bool {
	return p == "." || p == ".." || strings.HasPrefix(p, "./") || strings.HasPrefix(p, "../")
}

func splitSegs(p string) []string {
	var out []string
	for _, s := range strings.Split(filepath.ToSlash(p), "/") {
		if s != "" && s != "." {
			out = append(out, s)
		}
	}
	return out
}

func lastSegs(segs []string, n int) []string {
	if len(segs) <= n {
		return segs
	}
	return segs[len(segs)-n:]
}

// hasSegmentSuffix reports whether path ends with exactly the segments suffix.
func hasSegmentSuffix(path, suffix []string) bool {
	if len(suffix) == 0 || len(suffix) > len(path) {
		return false
	}
	off := len(path) - len(suffix)
	for i, s := range suffix {
		if path[off+i] != s {
			return false
		}
	}
	return true
}

// ancestorHasSuffix reports whether dir or one of its ancestors ends with segs.
func ancestorHasSuffix(dir string, segs []string) bool {
	path := splitSegs(dir)
	for n := len(path); n >= len(segs); n-- {
		if hasSegmentSuffix(path[:n], segs) {
			return true
		}
	}
	return false
}

func isAncestorDir(ancestor, dir string) bool {
	if ancestor == dir {
		return true
	}
	rel, err := filepath.Rel(ancestor, dir)
	return err == nil && rel != "." && !strings.HasPrefix(rel, "..")
}
