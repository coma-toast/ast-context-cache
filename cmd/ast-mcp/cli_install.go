package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"text/tabwriter"

	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/errs"
	"github.com/coma-toast/ast-context-cache/internal/flags"
	"github.com/coma-toast/ast-context-cache/internal/installer"
)

// installerOpts are the parsed flags shared by the installer subcommands.
type installerOpts struct {
	targets         listFlag
	components      listFlag
	mcpURL          string
	mcpPort         int
	dryRun          bool
	yes             bool
	jsonOut         bool
	replaceExternal bool
	args            []string
}

// listFlag accepts repeated flags and comma-separated lists.
type listFlag []string

// cliOutput is the --json shape mcp-local parses; keep the field set stable.
type cliOutput struct {
	Changes  []cliChange `json:"changes"`
	Status   []cliStatus `json:"status"`
	Warnings []string    `json:"warnings"`
}

type cliChange struct {
	Path    string `json:"path"`
	Kind    string `json:"kind"`
	Diff    string `json:"diff"`
	Skipped bool   `json:"skipped"`
	Reason  string `json:"reason"`
}

type cliStatus struct {
	Target    string `json:"target"`
	Component string `json:"component"`
	Status    string `json:"status"`
	Path      string `json:"path"`
}

// String joins the values.
func (l *listFlag) String() string {
	return strings.Join(*l, ",")
}

// Set appends each comma-separated value.
func (l *listFlag) Set(v string) error {
	for _, p := range strings.Split(v, ",") {
		if p = strings.TrimSpace(p); p != "" {
			*l = append(*l, p)
		}
	}
	return nil
}

// runInstaller runs one installer subcommand in-process. It opens only the usage database, so it
// works beside a running server and starts no listeners or watchers.
func runInstaller(cmd string, args []string, stdout, stderr io.Writer) int {
	opts, err := parseInstallerFlags(cmd, args, stderr)
	if errors.Is(err, flag.ErrHelp) {
		return exitOK
	}
	if err != nil {
		return exitError
	}
	if err := db.InitUsage(); err != nil {
		fmt.Fprintln(stderr, "error: failed to open the ast-context-cache database:", err)
		return exitError
	}
	defer db.Close()
	flags.Reload()
	svc, err := installer.New(installer.Config{MCPURL: opts.resolveURL()})
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return exitError
	}
	switch cmd {
	case "install":
		return runPlanCommand(svc, installer.ActionInstall, opts, stdout, stderr)
	case "uninstall":
		return runPlanCommand(svc, installer.ActionUninstall, opts, stdout, stderr)
	case "verify":
		return runVerify(svc, opts, stdout, stderr)
	case "backups":
		return runBackups(svc, opts, stdout, stderr)
	default:
		return runRestore(svc, opts, stdout, stderr)
	}
}

func parseInstallerFlags(cmd string, args []string, stderr io.Writer) (*installerOpts, error) {
	opts := &installerOpts{}
	fs := flag.NewFlagSet("ast-mcp "+cmd, flag.ContinueOnError)
	fs.SetOutput(stderr)
	switch cmd {
	case "install", "uninstall", "verify":
		fs.Var(&opts.targets, "target", "target host (repeatable or comma list): "+targetNames()+", or all")
		fs.StringVar(&opts.mcpURL, "mcp-url", "", "MCP URL to register, verbatim (overrides --mcp-port)")
		fs.IntVar(&opts.mcpPort, "mcp-port", 0, "MCP port for http://127.0.0.1:<port>/mcp (default $AST_MCP_PORT, then 7821)")
	}
	switch cmd {
	case "install", "uninstall":
		fs.Var(&opts.components, "component", "components (comma list): mcp,skills,rules,hooks (default: all supported)")
		fs.BoolVar(&opts.dryRun, "dry-run", false, "print the diff and statuses without writing")
		fs.BoolVar(&opts.replaceExternal, "replace-external", false, "replace externally managed skill or rule paths (backed up first)")
		fs.BoolVar(&opts.yes, "yes", false, "apply the changes")
	case "restore":
		fs.BoolVar(&opts.yes, "yes", false, "restore the backup")
	}
	fs.BoolVar(&opts.jsonOut, "json", false, "machine-readable output")
	if err := fs.Parse(args); err != nil {
		return nil, err
	}
	opts.args = fs.Args()
	return opts, nil
}

// resolveURL applies --mcp-url, then --mcp-port, then $AST_MCP_PORT, then the default port.
func (o *installerOpts) resolveURL() string {
	if o.mcpURL != "" {
		return o.mcpURL
	}
	port := o.mcpPort
	if port <= 0 {
		port, _ = strconv.Atoi(strings.TrimSpace(os.Getenv("AST_MCP_PORT")))
	}
	if port <= 0 {
		port = installer.DefaultMCPPort
	}
	return installer.MCPURL(port)
}

// resolveTargets expands "all" and validates names. explicit is false for "all".
func (o *installerOpts) resolveTargets(required bool) ([]installer.Target, bool, error) {
	if len(o.targets) == 0 {
		if required {
			return nil, false, errs.NewCode(errs.CodeInvalidInput, "--target is required ("+targetNames()+", or all)")
		}
		return installer.AllTargets(), false, nil
	}
	var out []installer.Target
	for _, name := range o.targets {
		if name == "all" {
			return installer.AllTargets(), false, nil
		}
		t, err := installer.ParseTarget(name)
		if err != nil {
			return nil, false, err
		}
		out = append(out, t)
	}
	return out, true, nil
}

func runPlanCommand(svc installer.Service, action installer.Action, opts *installerOpts, stdout, stderr io.Writer) int {
	targets, explicit, err := opts.resolveTargets(true)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return exitError
	}
	req := installer.PlanRequest{Targets: targets, Action: action, ReplaceExternal: opts.replaceExternal}
	for _, c := range opts.components {
		comp, err := installer.ParseComponent(c)
		if err != nil {
			fmt.Fprintln(stderr, "error:", err)
			return exitError
		}
		req.Components = append(req.Components, comp)
	}
	plan, err := svc.Plan(req)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return exitError
	}
	warnings := append(planErrorWarnings(plan), plan.Warnings...)
	warnings = append(warnings, svc.LegacyWarnings()...)
	unsupported := explicit && unsupportedTarget(plan, targets)
	pending := pendingWrites(plan)
	status := plan.Status
	if opts.yes {
		res, err := svc.Apply(plan.ID)
		if err != nil {
			writeResult(stdout, opts.jsonOut, plan.Changes, status, append(warnings, "error: "+err.Error()))
			if errs.HasCode(err, errs.CodeConflict) || errs.HasCode(err, errs.CodeExpired) {
				return exitConflict
			}
			return exitError
		}
		status = res.Status
	}
	writeResult(stdout, opts.jsonOut, plan.Changes, status, warnings)
	switch {
	case plan.HasErrors():
		return exitConflict
	case unsupported:
		return exitUnsupported
	case opts.yes || opts.dryRun || pending == 0:
		return exitOK
	}
	if !opts.jsonOut {
		fmt.Fprintln(stderr, "Preview only: re-run with --yes to apply, or --dry-run to preview without this exit code.")
	}
	return exitConfirm
}

func runVerify(svc installer.Service, opts *installerOpts, stdout, stderr io.Writer) int {
	targets, _, err := opts.resolveTargets(false)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return exitError
	}
	status, err := svc.Verify(targets)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return exitError
	}
	writeResult(stdout, opts.jsonOut, nil, status, svc.LegacyWarnings())
	return exitOK
}

func runBackups(svc installer.Service, opts *installerOpts, stdout, stderr io.Writer) int {
	backups, err := svc.Backups()
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return exitError
	}
	if opts.jsonOut {
		writeJSON(stdout, backups)
		return exitOK
	}
	if len(backups) == 0 {
		fmt.Fprintln(stdout, "No backups.")
		return exitOK
	}
	tw := tabwriter.NewWriter(stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "ID\tCREATED\tSIZE\tPATH")
	for _, b := range backups {
		fmt.Fprintf(tw, "%s\t%s\t%d\t%s\n", b.ID, b.CreatedAt.Format("2006-01-02 15:04:05"), b.Size, b.Path)
	}
	tw.Flush()
	return exitOK
}

func runRestore(svc installer.Service, opts *installerOpts, stdout, stderr io.Writer) int {
	if len(opts.args) != 1 {
		fmt.Fprintln(stderr, "usage: ast-mcp restore [--yes] <backup-id>   (list ids with: ast-mcp backups)")
		return exitError
	}
	id := opts.args[0]
	backups, err := svc.Backups()
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return exitError
	}
	var target *installer.Backup
	for i := range backups {
		if backups[i].ID == id {
			target = &backups[i]
		}
	}
	if target == nil {
		fmt.Fprintln(stderr, "error: backup not found:", id)
		return exitError
	}
	if !opts.yes {
		fmt.Fprintf(stdout, "Would restore %s from backup %s (the current file is backed up first).\n", target.Path, id)
		fmt.Fprintln(stderr, "Re-run with --yes to restore.")
		return exitConfirm
	}
	if err := svc.Restore(id); err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return exitError
	}
	fmt.Fprintf(stdout, "Restored %s from backup %s.\n", target.Path, id)
	return exitOK
}

// writeResult prints changes, statuses, and warnings as text or as the --json shape.
func writeResult(w io.Writer, asJSON bool, changes []installer.FileChange, status []installer.ComponentStatus, warnings []string) {
	out := cliOutput{Changes: []cliChange{}, Status: []cliStatus{}, Warnings: []string{}}
	for _, c := range changes {
		out.Changes = append(out.Changes, cliChange{Path: c.Path, Kind: string(c.Kind), Diff: c.Diff, Skipped: c.Skipped, Reason: c.Reason})
	}
	for _, s := range status {
		out.Status = append(out.Status, cliStatus{Target: string(s.Target), Component: string(s.Component), Status: string(s.Status), Path: s.Path})
	}
	out.Warnings = append(out.Warnings, warnings...)
	if asJSON {
		writeJSON(w, out)
		return
	}
	for _, c := range changes {
		label := string(c.Target) + "/" + string(c.Component)
		if c.Skipped {
			fmt.Fprintf(w, "skip  %s %s: %s\n", label, c.Path, c.Reason)
			continue
		}
		fmt.Fprintf(w, "%-6s %s %s\n", c.Kind, label, c.Path)
		if c.Diff != "" {
			fmt.Fprint(w, c.Diff)
			if !strings.HasSuffix(c.Diff, "\n") {
				fmt.Fprintln(w)
			}
		}
	}
	if len(status) > 0 {
		if len(changes) > 0 {
			fmt.Fprintln(w)
		}
		tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
		fmt.Fprintln(tw, "TARGET\tCOMPONENT\tSTATUS\tPATH\tNOTE")
		for _, s := range status {
			fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\n", s.Target, s.Component, s.Status, s.Path, s.Reason)
		}
		tw.Flush()
	}
	if len(warnings) > 0 {
		fmt.Fprintln(w, "\nWarnings:")
		for _, wn := range warnings {
			fmt.Fprintln(w, "  - "+wn)
		}
	}
}

func writeJSON(w io.Writer, v any) {
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	enc.Encode(v)
}

// planErrorWarnings renders aborted targets as warnings for the output.
func planErrorWarnings(p *installer.Plan) []string {
	var out []string
	for _, e := range p.Errors {
		out = append(out, "error: "+string(e.Target)+"/"+string(e.Component)+": "+e.Message)
	}
	return out
}

// unsupportedTarget reports whether some named target has no supported requested component.
func unsupportedTarget(p *installer.Plan, targets []installer.Target) bool {
	for _, t := range targets {
		supported := false
		for _, s := range p.Status {
			if s.Target == t && s.Status != installer.StatusUnsupported {
				supported = true
			}
		}
		if !supported {
			return true
		}
	}
	return false
}

// pendingWrites counts changes that would write.
func pendingWrites(p *installer.Plan) int {
	n := 0
	for _, c := range p.Changes {
		if !c.Skipped {
			n++
		}
	}
	return n
}

func targetNames() string {
	var names []string
	for _, t := range installer.AllTargets() {
		names = append(names, string(t))
	}
	return strings.Join(names, ", ")
}
