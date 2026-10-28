package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// The index and prune-projects subcommands drive the running server over its
// dashboard API rather than opening the databases: the server owns the index
// writer, and a second process writing the same SQLite files would race it.
const (
	indexProjectPath     = "/api/index-project"
	pruneSpaceRootsPath  = "/api/prune-space-roots"
	cliIndexTimeout      = 60 * time.Minute
	cliPruneTimeout      = 10 * time.Minute
	defaultDashboardPort = 7830
)

// dashboardURL is $AST_DASHBOARD_URL verbatim, else the loopback dashboard on
// $AST_DASHBOARD_PORT or the default port.
func dashboardURL() string {
	if u := strings.TrimSpace(os.Getenv("AST_DASHBOARD_URL")); u != "" {
		return strings.TrimSuffix(u, "/")
	}
	port, _ := strconv.Atoi(strings.TrimSpace(os.Getenv(envDashboardPort)))
	if port <= 0 {
		port = defaultDashboardPort
	}
	return "http://127.0.0.1:" + strconv.Itoa(port)
}

// runIndex handles "ast-mcp index <dir>...": index each directory through the
// running server, which copies unchanged files from an already-indexed checkout
// of the same repo and indexes a wtg space root as its separate checkouts.
func runIndex(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("index", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "print the server's JSON response for each directory")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: ast-mcp index [--json] <dir>...")
		fmt.Fprintln(stderr, "Indexes each directory (a repo checkout or a wtg space root) through the running ast-mcp server.")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return exitError
	}
	if fs.NArg() == 0 {
		fs.Usage()
		return exitError
	}
	code := exitOK
	for _, dir := range fs.Args() {
		abs, err := filepath.Abs(dir)
		if err != nil {
			fmt.Fprintf(stderr, "%s: %v\n", dir, err)
			code = exitError
			continue
		}
		var out struct {
			Symbols  int      `json:"symbols"`
			Projects []string `json:"projects"`
		}
		raw, err := postDashboard(indexProjectPath, map[string]interface{}{"project_path": abs}, cliIndexTimeout, &out)
		if err != nil {
			fmt.Fprintf(stderr, "%s: %v\n", abs, err)
			code = exitError
			continue
		}
		if *asJSON {
			fmt.Fprintln(stdout, strings.TrimSpace(string(raw)))
			continue
		}
		fmt.Fprintf(stdout, "indexed %s: %d symbols\n", abs, out.Symbols)
		if len(out.Projects) > 1 || (len(out.Projects) == 1 && out.Projects[0] != abs) {
			for _, p := range out.Projects {
				fmt.Fprintf(stdout, "  checkout %s\n", p)
			}
		}
	}
	return code
}

// runPruneProjects handles "ast-mcp prune-projects --space-roots [--apply]". It is
// a dry run unless --apply is given.
func runPruneProjects(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("prune-projects", flag.ContinueOnError)
	fs.SetOutput(stderr)
	spaceRoots := fs.Bool("space-roots", false, "list (and with --apply, prune) indexed wtg space roots covered by their own checkouts")
	apply := fs.Bool("apply", false, "delete the index rows of covered space roots (default: dry run)")
	asJSON := fs.Bool("json", false, "print the server's JSON response")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: ast-mcp prune-projects --space-roots [--apply] [--json]")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return exitError
	}
	if !*spaceRoots {
		fs.Usage()
		return exitError
	}
	var out struct {
		SpaceRoots []struct {
			Path      string   `json:"path"`
			Files     int      `json:"files"`
			Symbols   int      `json:"symbols"`
			Checkouts []string `json:"checkouts"`
			Unindexed []string `json:"unindexed"`
			Covered   bool     `json:"covered"`
			Pruned    bool     `json:"pruned"`
			Error     string   `json:"error"`
		} `json:"space_roots"`
	}
	raw, err := postDashboard(pruneSpaceRootsPath, map[string]interface{}{"apply": *apply}, cliPruneTimeout, &out)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return exitError
	}
	if *asJSON {
		fmt.Fprintln(stdout, strings.TrimSpace(string(raw)))
		return exitOK
	}
	if len(out.SpaceRoots) == 0 {
		fmt.Fprintln(stdout, "no indexed space roots")
		return exitOK
	}
	code := exitOK
	covered := 0
	for _, r := range out.SpaceRoots {
		state := "covered"
		switch {
		case r.Error != "":
			state = "error: " + r.Error
			code = exitError
		case r.Pruned:
			state = "pruned"
		case !r.Covered:
			state = fmt.Sprintf("skipped, %d checkout(s) not indexed yet", len(r.Unindexed))
		}
		if r.Covered {
			covered++
		}
		fmt.Fprintf(stdout, "%s  files=%d symbols=%d checkouts=%d  %s\n", r.Path, r.Files, r.Symbols, len(r.Checkouts), state)
		for _, u := range r.Unindexed {
			fmt.Fprintf(stdout, "  not indexed: %s (run: ast-mcp index %s)\n", u, u)
		}
	}
	if !*apply && covered > 0 {
		fmt.Fprintf(stdout, "dry run: re-run with --apply to prune %d covered space root(s)\n", covered)
	}
	return code
}

// postDashboard POSTs body as JSON to the dashboard API, decodes the response
// into out, and returns the raw response.
func postDashboard(path string, body map[string]interface{}, timeout time.Duration, out interface{}) ([]byte, error) {
	data, err := json.Marshal(body)
	if err != nil {
		return nil, errs.WrapMessage("failed to encode request", err)
	}
	url := dashboardURL() + path
	client := &http.Client{Timeout: timeout}
	resp, err := client.Post(url, "application/json", bytes.NewReader(data))
	if err != nil {
		return nil, errs.WrapMessage("ast-mcp server not reachable; start it first", err, "url", url)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, errs.WrapMessage("failed to read response", err, "url", url)
	}
	if resp.StatusCode != http.StatusOK {
		var e struct {
			Error string `json:"error"`
		}
		json.Unmarshal(raw, &e)
		return raw, errs.New("server returned an error", "url", url, "status", resp.StatusCode, "error", e.Error)
	}
	if err := json.Unmarshal(raw, out); err != nil {
		return raw, errs.WrapMessage("failed to decode response", err, "url", url)
	}
	return raw, nil
}
