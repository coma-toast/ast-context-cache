package codescripts

import (
	"embed"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"

	"github.com/coma-toast/ast-context-cache/internal/db"
)

//go:embed builtin/*
var builtinFS embed.FS

const (
	repoDirName     = "scripts/code-mode"
	maxScriptBytes  = 32 * 1024
	maxHints        = 2
	builtinManifest = "builtin/manifest.json"
)

// MatchRules controls when a script is suggested inline on search tools.
type MatchRules struct {
	Tools      []string `json:"tools"`
	QueryRegex string   `json:"query_regex"`
	MinResults int      `json:"min_results"`
}

// ManifestEntry is one row in manifest.json (builtin or repo).
type ManifestEntry struct {
	ID          string     `json:"id"`
	Title       string     `json:"title"`
	Description string     `json:"description"`
	Match       MatchRules `json:"match"`
	CodeFile    string     `json:"code_file"`
	Extends     string     `json:"extends"`
	Code        string     `json:"-"`
}

// Hint is attached to search tool responses as code_script_hints.
type Hint struct {
	ScriptID        string            `json:"script_id"`
	Title           string            `json:"title"`
	Why             string            `json:"why"`
	EstDataTokens   int               `json:"est_data_tokens"`
	EstOutputTokens int               `json:"est_output_tokens"`
	EstTokensSaved  int               `json:"est_tokens_saved"`
	Execute         map[string]string `json:"execute"`
}

type script struct {
	ManifestEntry
	queryRe *regexp.Regexp
}

var (
	builtinOnce sync.Once
	builtinList []script
	builtinByID map[string]script
	repoCache   sync.Map // projectPath -> []script
)

func loadBuiltins() {
	builtinOnce.Do(func() {
		raw, err := builtinFS.ReadFile(builtinManifest)
		if err != nil {
			return
		}
		var entries []ManifestEntry
		if err := json.Unmarshal(raw, &entries); err != nil {
			return
		}
		builtinByID = make(map[string]script, len(entries))
		for _, e := range entries {
			code, err := readBuiltinCode(e.CodeFile)
			if err != nil {
				continue
			}
			e.Code = code
			s := script{ManifestEntry: e}
			if e.Match.QueryRegex != "" {
				s.queryRe, _ = regexp.Compile(e.Match.QueryRegex)
			}
			builtinList = append(builtinList, s)
			builtinByID[e.ID] = s
		}
	})
}

func readBuiltinCode(codeFile string) (string, error) {
	if codeFile == "" {
		return "", fmt.Errorf("empty code_file")
	}
	path := filepath.Join("builtin", filepath.Base(codeFile))
	raw, err := builtinFS.ReadFile(path)
	if err != nil {
		return "", err
	}
	if len(raw) > maxScriptBytes {
		return "", fmt.Errorf("script too large")
	}
	return string(raw), nil
}

// ResolveScript returns JS source for script_id from repo override or builtin.
func ResolveScript(scriptID, projectPath string) (string, error) {
	loadBuiltins()
	if projectPath != "" {
		for _, s := range loadRepoScripts(projectPath) {
			if s.ID == scriptID && s.Code != "" {
				return s.Code, nil
			}
		}
	}
	if s, ok := builtinByID[scriptID]; ok && s.Code != "" {
		return s.Code, nil
	}
	return "", fmt.Errorf("unknown script_id %q", scriptID)
}

func loadRepoScripts(projectPath string) []script {
	if projectPath == "" {
		return nil
	}
	if v, ok := repoCache.Load(projectPath); ok {
		return v.([]script)
	}
	list := parseRepoManifest(projectPath)
	repoCache.Store(projectPath, list)
	return list
}

func parseRepoManifest(projectPath string) []script {
	dir := filepath.Join(projectPath, repoDirName)
	manifestPath := filepath.Join(dir, "manifest.json")
	raw, err := os.ReadFile(manifestPath)
	if err != nil {
		return nil
	}
	var entries []ManifestEntry
	if err := json.Unmarshal(raw, &entries); err != nil {
		return nil
	}
	var out []script
	for _, e := range entries {
		if e.Extends != "" {
			loadBuiltins()
			if base, ok := builtinByID[e.Extends]; ok {
				if e.Title == "" {
					e.Title = base.Title
				}
				if e.Description == "" {
					e.Description = base.Description
				}
				if len(e.Match.Tools) == 0 {
					e.Match = base.Match
				}
				if e.ID == "" {
					e.ID = base.ID
				}
			}
		}
		if e.ID == "" {
			continue
		}
		if e.CodeFile != "" {
			codePath, err := jailScriptPath(dir, e.CodeFile)
			if err != nil {
				continue
			}
			body, err := os.ReadFile(codePath)
			if err != nil || len(body) > maxScriptBytes {
				continue
			}
			e.Code = string(body)
		} else if e.Extends != "" {
			loadBuiltins()
			if base, ok := builtinByID[e.Extends]; ok {
				e.Code = base.Code
			}
		}
		if e.Code == "" {
			continue
		}
		s := script{ManifestEntry: e}
		if e.Match.QueryRegex != "" {
			s.queryRe, _ = regexp.Compile(e.Match.QueryRegex)
		}
		out = append(out, s)
	}
	return out
}

func jailScriptPath(dir, codeFile string) (string, error) {
	cleanDir := filepath.Clean(dir)
	joined := filepath.Clean(filepath.Join(cleanDir, codeFile))
	rel, err := filepath.Rel(cleanDir, joined)
	if err != nil || strings.HasPrefix(rel, "..") {
		return "", fmt.Errorf("path escapes scripts/code-mode")
	}
	return joined, nil
}

func allScripts(projectPath string) []script {
	loadBuiltins()
	byID := make(map[string]script, len(builtinList))
	for _, s := range builtinList {
		byID[s.ID] = s
	}
	for _, s := range loadRepoScripts(projectPath) {
		byID[s.ID] = s
	}
	out := make([]script, 0, len(byID))
	for _, s := range byID {
		out = append(out, s)
	}
	return out
}

func toolAllowed(rules MatchRules, tool string) bool {
	if len(rules.Tools) == 0 {
		return true
	}
	for _, t := range rules.Tools {
		if strings.EqualFold(t, tool) {
			return true
		}
	}
	return false
}

func scoreScript(s script, tool, query string, nResults int) int {
	if !toolAllowed(s.Match, tool) {
		return -1
	}
	if s.Match.MinResults > 0 && nResults < s.Match.MinResults {
		return -1
	}
	score := 0
	if s.queryRe != nil {
		if s.queryRe.MatchString(query) {
			score += 10
		} else {
			return -1
		}
	}
	if s.Match.MinResults > 0 && nResults >= s.Match.MinResults {
		score += 5 + (nResults - s.Match.MinResults)
	}
	return score
}

func estOutputTokens(scriptID string, results []map[string]interface{}) int {
	n := len(results)
	switch scriptID {
	case "compact-symbol-list":
		return n * 25
	case "group-by-file":
		files := map[string]struct{}{}
		for _, r := range results {
			if f, ok := r["file"].(string); ok && f != "" {
				files[f] = struct{}{}
			}
		}
		return len(files)*45 + 40
	case "filter-by-kind":
		return 120 + n*8
	case "exports-only":
		return n * 18
	case "dedupe-by-file":
		files := map[string]struct{}{}
		for _, r := range results {
			if f, ok := r["file"].(string); ok && f != "" {
				files[f] = struct{}{}
			}
		}
		return len(files) * 30
	case "impact-candidates":
		if n > 20 {
			n = 20
		}
		return n * 22
	default:
		return n * 20
	}
}

type scored struct {
	script script
	score  int
}

// MatchHints returns top script suggestions for inline search responses.
func MatchHints(tool, query, projectPath string, results []map[string]interface{}) []Hint {
	if len(results) == 0 {
		return nil
	}
	dataJSON, _ := json.Marshal(results)
	dataTokens := db.EstimateTokens(string(dataJSON))
	var ranked []scored
	for _, s := range allScripts(projectPath) {
		sc := scoreScript(s, tool, query, len(results))
		if sc < 0 {
			continue
		}
		ranked = append(ranked, scored{script: s, score: sc})
	}
	if len(ranked) == 0 {
		return nil
	}
	sort.Slice(ranked, func(i, j int) bool {
		if ranked[i].score != ranked[j].score {
			return ranked[i].score > ranked[j].score
		}
		return ranked[i].script.ID < ranked[j].script.ID
	})
	limit := maxHints
	if len(ranked) < limit {
		limit = len(ranked)
	}
	hints := make([]Hint, 0, limit)
	for i := 0; i < limit; i++ {
		s := ranked[i].script
		outTok := estOutputTokens(s.ID, results)
		saved := dataTokens - outTok
		if saved < 0 {
			saved = 0
		}
		why := fmt.Sprintf("%d results — script returns ~%d tokens vs ~%d in DATA", len(results), outTok, dataTokens)
		if s.Description != "" {
			why = s.Description + "; " + why
		}
		hints = append(hints, Hint{
			ScriptID:        s.ID,
			Title:           s.Title,
			Why:             why,
			EstDataTokens:   dataTokens,
			EstOutputTokens: outTok,
			EstTokensSaved:  saved,
			Execute: map[string]string{
				"script_id": s.ID,
				"note":      "Pass prior results JSON as execute_code data",
			},
		})
	}
	return hints
}

// AttachHints adds code_script_hints to a response map when matches exist.
func AttachHints(resp map[string]interface{}, tool, query, projectPath string, results []map[string]interface{}) {
	hints := MatchHints(tool, query, projectPath, results)
	if len(hints) > 0 {
		resp["code_script_hints"] = hints
	}
}

// InvalidateRepoCache drops cached repo scripts (e.g. after manifest edit).
func InvalidateRepoCache(projectPath string) {
	repoCache.Delete(projectPath)
}

// BuiltinIDs returns built-in script ids (for docs/tests).
func BuiltinIDs() []string {
	loadBuiltins()
	ids := make([]string, 0, len(builtinList))
	for _, s := range builtinList {
		ids = append(ids, s.ID)
	}
	sort.Strings(ids)
	return ids
}
