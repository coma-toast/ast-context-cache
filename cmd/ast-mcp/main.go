package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	ctxpkg "github.com/coma-toast/ast-context-cache/internal/context"
	"github.com/coma-toast/ast-context-cache/internal/contextnotes"
	"github.com/coma-toast/ast-context-cache/internal/dashboard"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/docs"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/embedqueue"
	"github.com/coma-toast/ast-context-cache/internal/flags"
	"github.com/coma-toast/ast-context-cache/internal/handoff"
	"github.com/coma-toast/ast-context-cache/internal/httpguard"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/coma-toast/ast-context-cache/internal/installer"
	"github.com/coma-toast/ast-context-cache/internal/instance"
	"github.com/coma-toast/ast-context-cache/internal/logging"
	"github.com/coma-toast/ast-context-cache/internal/logretention"
	"github.com/coma-toast/ast-context-cache/internal/mcp"
	"github.com/coma-toast/ast-context-cache/internal/memory"
	"github.com/coma-toast/ast-context-cache/internal/netlisten"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
	"github.com/coma-toast/ast-context-cache/internal/projectmeta"
	"github.com/coma-toast/ast-context-cache/internal/purge"
	"github.com/coma-toast/ast-context-cache/internal/search"
	"github.com/coma-toast/ast-context-cache/internal/startup"
	"github.com/coma-toast/ast-context-cache/internal/sys"
	"github.com/coma-toast/ast-context-cache/internal/transcripts"
	"github.com/coma-toast/ast-context-cache/internal/version"
	"github.com/coma-toast/ast-context-cache/internal/watcher"
)

// Overridable (-mcp-port / -dashboard-port, or AST_MCP_PORT / AST_DASHBOARD_PORT) so a second
// instance can run beside the shared one, e.g. with HOME pointed at a scratch data dir.
var (
	mcpPort       = 7821
	dashboardPort = 7830
)

// instanceName holds the resolved instance name for use in handlers and health payload.

// Port env vars. AST_MCP_PORT is also what the installer CLI and the hooks read, so one export
// moves the server, its registrations, and the hooks together.
const (
	envMCPPort       = "AST_MCP_PORT"
	envDashboardPort = "AST_DASHBOARD_PORT"
)

// listenAddr is the host both servers bind to (-listen / AST_LISTEN). Loopback by
// default: loopback clients need no credentials, and the MCP spec says local servers
// SHOULD bind to 127.0.0.1. Docker sets 0.0.0.0 so published ports work. Extra
// addresses (a Tailscale IP) come from the listen_extra_addrs setting or
// AST_LISTEN_EXTRA and are managed live by netlisten.
var listenAddr = defaultListenAddr

const (
	defaultListenAddr = "127.0.0.1"
	ipv6LoopbackAddr  = "::1"
)

// extraListenRetry is how often extra listeners that failed to bind are retried.
const extraListenRetry = 30 * time.Second

// shutdownFlushTimeout bounds the final analytics flush so a wedged SQLite write
// can't stop the process from exiting on SIGTERM.
const shutdownFlushTimeout = 2 * time.Second

const (
	deleteFromQueryPrefix        = "DELETE FROM "
	whereDotProjectQuerySuffix   = " WHERE project_path = '.'"
	deleteDotProjectQueriesQuery = "DELETE FROM queries WHERE project_path = '.'"
	selectSymbolProjectsQuery    = "SELECT DISTINCT project_path FROM symbols WHERE project_path IS NOT NULL AND project_path != ''"
)

var (
	startTime = time.Now()
	exiting   = atomic.Bool{}
)

func GetStartTime() time.Time {
	return startTime
}

func main() {
	if code, handled := runCLI(os.Args[1:]); handled {
		os.Exit(code)
	}
	logging.Setup(os.Stderr)
	tierFlag := flag.String("tier", "", "Tool tier: core, extended, complete (default: from AST_MCP_TIER env or complete)")
	codeModeFlag := flag.Bool("code-mode", true, "Enable execute_code sandbox tool (default: true)")
	embedWorkersFlag := flag.Int("embed-workers", -1, "Embed worker count at startup (-1 = auto/DB)")
	flag.IntVar(&mcpPort, "mcp-port", envPort(envMCPPort, mcpPort), "MCP HTTP port (default: from AST_MCP_PORT env or 7821)")
	flag.IntVar(&dashboardPort, "dashboard-port", envPort(envDashboardPort, dashboardPort), "Dashboard HTTP port (default: from AST_DASHBOARD_PORT env or 7830)")
	nameFlag := flag.String("name", "", "Instance name for tab title and cookie/localStorage scoping (default: from AST_INSTANCE_NAME env or settings)")
	if v := strings.TrimSpace(os.Getenv("AST_LISTEN")); v != "" {
		listenAddr = v
	}
	flag.StringVar(&listenAddr, "listen", listenAddr, "Host/IP for the MCP and dashboard servers to bind (default: from AST_LISTEN env or [IPV4_REDACTED])")
	flag.Parse()
	// JoinHostPort adds IPv6 brackets itself; "[::1]" would otherwise become "[[::1]]:7821".
	listenAddr = strings.Trim(strings.TrimSpace(listenAddr), "[]")
	readRestartedFrom()
	// Cancelled on SIGINT/SIGTERM so background loops stop before the final flush.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	cfg := mcp.DefaultConfig()
	if *tierFlag != "" {
		cfg.ActiveTier = mcp.ParseTier(*tierFlag)
	}
	if !*codeModeFlag {
		cfg.CodeMode = false
	}
	mcp.SetConfig(cfg)
	logger.Info("Config", "tier", cfg.ActiveTier, "code_mode", cfg.CodeMode, "listen", listenAddr)
	if !httpguard.IsLoopbackHost(listenAddr) {
		logger.Warn("Listening on non-loopback address", "listen", listenAddr)
	}
	if listenAddr == defaultListenAddr {
		netlisten.SetBase(listenAddr, ipv6LoopbackAddr)
	} else {
		netlisten.SetBase(listenAddr)
	}
	// Before db.Init only the env vars resolve, so an AST_ACCESS_TOKEN guards the very
	// first request; the settings-backed values are applied once the databases open.
	netlisten.Apply()

	logger.Info("Initializing", "pid", os.Getpid())
	if restartedFrom > 0 {
		logger.Info("Started by a restart", "previous_pid", restartedFrom)
	}
	if fi, err := os.Stdout.Stat(); err == nil && fi.Mode()&os.ModeCharDevice != 0 {
		logPath := db.DefaultLogPath()
		_ = os.MkdirAll(filepath.Dir(logPath), 0o755)
		if f, err := os.OpenFile(logPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644); err == nil {
			logging.Setup(f)
			logger.Info("Logging to file", "path", logPath)
		}
	}
	if u := sys.FileDescriptorUsage(); u.SoftLimit > 0 {
		logger.Info("File descriptors", "open", u.Open, "soft_limit", u.SoftLimit, "hard_limit", u.HardLimit, "watcher_backend", watcher.DefaultBackendName())
	}
	watcher.ContainerRootsFunc = projectmeta.ContainerRoots
	startup.SetMessage("Opening databases…")

	exePath, _ := os.Executable()
	exeDir := filepath.Dir(exePath)

	// Resolve the instance name early (flag > env > fallback hash) so the
	// dashboard serves with a scoped cookie before the DB opens. The DB
	// setting is picked up again after db.Init below.
	instance.Init(strings.TrimSpace(*nameFlag))
	applyInstanceName("")

	dashHandler := dashboard.NewHandler(listenAddr)
	// Constructed here (before the db.Init() goroutine below reads it via restartHook)
	// rather than at its ListenAndServe call further down, so that read has a clear
	// happens-before edge and isn't a data race with this assignment.
	dashSrv := &http.Server{Addr: net.JoinHostPort(listenAddr, strconv.Itoa(dashboardPort)), Handler: dashHandler}
	dashListeners := netlisten.NewManager("dashboard", dashSrv, dashboardPort)
	dbReady := make(chan error, 1)

	go func() {
		if err := db.Init(); err != nil {
			startup.MarkFailed(err.Error())
			logger.Error("Failed to initialize databases", "error", err)
			dbReady <- err
			return
		}
		// The DB setting is readable now: re-resolve (flag > env > setting)
		// and re-scope the cookie before serving authenticated traffic.
		applyInstanceName(strings.TrimSpace(db.GetSetting("instance_name", "")))
		// If data directory is unavailable (override path missing/not-writable),
		// still start MCP and dashboard in degraded mode, but skip post-init work.
		if startup.DataDirUnavailable() {
			logger.Warn("Starting in degraded mode: data directory unavailable")
			projectmeta.SetDisplayNameOverrideFunc(db.ProjectDisplayName)
			dbReady <- nil
		} else {
			projectmeta.SetDisplayNameOverrideFunc(db.ProjectDisplayName)
			mcp.Init()
			if svc, err := installer.New(installer.Config{MCPURL: installer.MCPURL(mcpPort)}); err != nil {
				logger.Warn("Failed to start installer", "error", err)
			} else {
				dashboard.SetInstaller(svc)
			}
			dbReady <- nil

			db.BeforeForceCheckpoint = func() {
				embedqueue.PauseAllForMaintenance(2 * time.Minute)
				search.Cache.Unload()
			}
			db.AfterForceCheckpoint = embedqueue.RestoreAfterMaintenance
			db.WALInFlightHook = embedqueue.InFlight
			db.EmbedQueueIdleHook = embedqueue.QueueIdleForWAL
			if embedqueue.BeginRunLock() {
				logger.Warn("Embed queue previous run exited abnormally; using persisted worker count from DB")
			}
			watcher.EnsureDefaultIgnoreGlobs()
			go db.StartWALCheckpoint()
			go db.StartDriveMonitor()
		}

		// Set up MCP listener and RestartProcess in ALL cases (degraded or not).
		// In degraded mode we skip mcp.Init()/installer/embed queue/WAL setup.
		mcpMux := http.NewServeMux()
		mcpMux.Handle("/mcp", httpguard.MCPMiddleware(listenAddr, mcp.NewHandler()))
		mcpMux.HandleFunc("/health", handleMCPHealth)
		mcpMux.HandleFunc("/embed", handleEmbedHTTP)
		mcpMux.HandleFunc("/embed/health", handleEmbedHealthHTTP)
		mcpMux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
			if strings.HasPrefix(r.URL.Path, "/api/") || strings.HasPrefix(r.URL.Path, "/mcp") {
				return
			}
			json.NewEncoder(w).Encode(map[string]interface{}{"service": "AST MCP", "dashboard": serverURL(dashboardPort)})
		})

		// The health probe stays open so remote monitors and the installer can check liveness.
		mcpSrv := &http.Server{Addr: net.JoinHostPort(listenAddr, strconv.Itoa(mcpPort)), Handler: httpguard.RequireMCPAuth(mcpMux, "/health")}
		// Shutdown waits for idle connections, which open SSE streams never become.
		mcpSrv.RegisterOnShutdown(mcp.CloseStreams)
		db.RestartProcess = restartProcess(mcpSrv, dashSrv, cancel)
		netlisten.Register(netlisten.NewManager("mcp", mcpSrv, mcpPort), dashListeners)
		if cfg := netlisten.Apply(); len(cfg.ExtraAddrs) > 0 && cfg.Token == "" {
			logger.Warn("Extra listen addresses have no access token; anyone who can reach them can use the server", "extra", cfg.ExtraAddrs)
		}
		netlisten.StartRetry(ctx, extraListenRetry)

		go func() {
			logger.Info("Starting MCP server", "url", serverURL(mcpPort)+"/mcp")
			if err := listenAndServe(mcpSrv, mcpPort); err != nil {
				logger.Error("MCP server failed", "error", err)
				os.Exit(1)
			}
		}()

		embedder.MarkLoading()
		finishStartup(ctx, exeDir, *embedWorkersFlag)
	}()

	go func() {
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
		<-sig
		exiting.Store(true)
		logger.Info("Shutting down")
		cancel()
		netlisten.CloseAll()
		db.RequestShutdown()
		flushWriteBuffers(shutdownFlushTimeout)
		embedqueue.EndRunLock()
		os.Exit(0)
	}()

	logger.Info("Starting dashboard server", "url", serverURL(dashboardPort))
	serveWithRestartOutcome(
		func() error { return listenAndServe(dashSrv, dashboardPort) },
		exiting.Load,
		os.Exit,
		func() { select {} },
	)
}

// serveWithRestartOutcome runs serve and decides what main does when it returns: an error
// is fatal; a nil return during an exit (restart or signal) blocks so the owner —
// restartProcess (spawn, exec or exit), or the signal goroutine's os.Exit(0) after the
// flush — can finish. A nil return otherwise is an unexpected shutdown and exits 1 so supervisors
// restart us. The logic is extracted here for unit testing with fakes.
func serveWithRestartOutcome(serve func() error, exiting func() bool, exit func(int), block func()) {
	err := serve()
	switch {
	case err != nil:
		logger.Error("Dashboard server failed", "error", err)
		exit(1)
	case exiting():
		logger.Info("Exit in progress; waiting for owner")
		block()
	default:
		logger.Error("Dashboard server stopped unexpectedly")
		exit(1)
	}
}

// listenAndServe serves srv on listenAddr:port and, for the default IPv4 loopback, on
// [::1]:port too: clients configured with http://localhost:7821/mcp otherwise fail
// wherever localhost resolves to ::1 first. Both listeners share srv, so Shutdown closes
// both. It returns nil once srv is shut down, or the first listen or serve error.
func listenAndServe(srv *http.Server, port int) error {
	lns, err := listeners(port)
	if err != nil {
		return err
	}
	errc := make(chan error, len(lns))
	for _, ln := range lns {
		go func() { errc <- srv.Serve(ln) }()
	}
	for range lns {
		if err := <-errc; !errors.Is(err, http.ErrServerClosed) {
			return err
		}
	}
	return nil
}

// listeners opens listenAddr:port, adding [::1]:port when listenAddr is the default. A
// host without IPv6 loopback just keeps the IPv4 listener. After a restart each bind is
// retried briefly (listenRetry) while the previous process's sockets close.
func listeners(port int) ([]net.Listener, error) {
	p := strconv.Itoa(port)
	primary, err := listenRetry(net.JoinHostPort(listenAddr, p))
	if err != nil {
		return nil, err
	}
	if listenAddr != defaultListenAddr {
		return []net.Listener{primary}, nil
	}
	v6, err := listenRetry(net.JoinHostPort(ipv6LoopbackAddr, p))
	if err != nil {
		logger.Debug("Not listening on IPv6 loopback", "port", port, "error", err)
		return []net.Listener{primary}, nil
	}
	return []net.Listener{primary, v6}, nil
}

// envPort is the port in env var name, or def when it is unset. An invalid value is ignored with
// a warning rather than failing startup, matching how the installer and hooks read AST_MCP_PORT.
func envPort(name string, def int) int {
	v := strings.TrimSpace(os.Getenv(name))
	if v == "" {
		return def
	}
	port, err := strconv.Atoi(v)
	if err != nil || port <= 0 || port > 65535 {
		logger.Warn("Ignoring invalid port env", "env", name, "value", v, "default", def)
		return def
	}
	return port
}

// serverURL is the address to show for a server on port: listenAddr, or
// localhost when it binds every interface.
func serverURL(port int) string {
	host := listenAddr
	if ip := net.ParseIP(host); host == "" || (ip != nil && ip.IsUnspecified()) {
		host = "localhost"
	}
	return "http://" + net.JoinHostPort(host, strconv.Itoa(port))
}

// flushWriteBuffers commits buffered query and session analytics before exit, giving
// up after timeout. The flush keeps running in its goroutine on timeout, but os.Exit
// follows immediately, so at worst that batch is lost, as it was before this flush.
func flushWriteBuffers(timeout time.Duration) {
	done := make(chan struct{})
	go func() {
		defer close(done)
		db.FlushWriteBuffers()
	}()
	select {
	case <-done:
		logger.Info("Flushed write buffers")
	case <-time.After(timeout):
		logger.Warn("Timed out flushing write buffers", "timeout", timeout)
	}
}

func finishStartup(ctx context.Context, exeDir string, embedWorkersFlag int) {
	defer func() {
		if startup.Ready() {
			return
		}
		if r := recover(); r != nil {
			startup.MarkFailed(fmt.Sprint(r))
			logger.Error("Startup panic", "panic", r)
		}
	}()

	if conn, err := db.IndexReader(); err == nil {
		for _, tbl := range []string{"symbols", "edges", "vectors", "summaries"} {
			conn.Exec(deleteFromQueryPrefix + tbl + whereDotProjectQuerySuffix)
		}
	}
	if db.DB != nil {
		db.DB.Exec(deleteDotProjectQueriesQuery)
	}

	modelDir := strings.TrimSpace(embedder.EffectiveEnv("MODEL_DIR"))
	if modelDir == "" {
		modelDir = filepath.Join(exeDir, "model")
	}
	embedder.SetRuntimeHooks(embedder.RuntimeHooks{
		OnSwap: func(tracked embedder.Interface) {
			mcp.SetEmbedder(tracked)
			ctxpkg.Emb = tracked
			embedqueue.SetEmbedder(tracked)
			if embedqueue.Ready() {
				embedqueue.RestoreWorkersAfterSwap()
				go embedqueue.RecoverAfterEmbedder()
				go embedqueue.FlushPendingIfReady()
			}
		},
	})
	embedder.SetOnBeforeSwap(func() {
		embedqueue.PrepareForEmbedderSwap(2 * time.Minute)
	})

	startup.SetMessage("Loading embedder…")
	if err := embedder.InitRuntime(modelDir); err != nil {
		startup.MarkFailed(err.Error())
		logger.Warn("Failed to initialize embedder; dashboard and MCP up, embeddings unavailable", "error", err)
		return
	}
	emb := embedder.Tracked()
	wb, wm, _, _, wd := embedder.WiredSnapshot()
	logger.Info("Embedder configured", "backend", wb, "model", wm, "dims", wd)
	if n := resolveStartupWorkers(embedWorkersFlag); n >= 0 {
		embedqueue.SetStartupWorkers(n)
		logger.Info("Embed queue startup workers override", "workers", n)
	}
	startup.SetMessage("Starting embed queue…")
	// Purged files (deleted from disk, symlink aliases) must leave the pending retry set too.
	indexer.OnFilePurged = embedqueue.ForgetFile
	embedqueue.Start(emb)
	if err := embedder.InitAuxRuntime(modelDir); err != nil {
		logger.Warn("Failed to initialize aux embedder; aux catch-up workers disabled", "error", err)
	} else {
		auxEmb := embedder.RawAux()
		if embedder.AuxSharesPrimary() {
			auxEmb = embedder.Tracked()
		}
		embedqueue.StartAux(auxEmb)
	}
	embedder.SetOnRecovery(embedqueue.RecoverAfterEmbedder)
	embedder.SetOnReady(embedqueue.FlushPendingIfReady)
	embedder.SetOnError(embedqueue.OnEmbedderError)
	projectlinks.SetOnLinkCleanup(func(parent, child string) {
		embedqueue.RemoveProjectFilesUnder(parent, child)
	})
	embedqueue.StartErrorScanLoop()
	embedqueue.StartPendingReconciler()
	watcher.PostIndexHook = func(filePath, projectPath string, removed bool) {
		if removed {
			search.Cache.DeleteByFile(filePath, projectPath)
		} else {
			if indexer.ShouldSkipEmbed(filePath) {
				return
			}
			if db.ShouldThrottleHeavyWork() && !db.IsPinnedProject(projectPath) {
				return
			}
			embedqueue.SubmitPriority(filePath, projectPath, db.IsPinnedProject(projectPath))
		}
	}

	startup.SetMessage("Starting background services…")
	startBackgroundServices(ctx)
	startup.MarkReady()
	logger.Info("Startup complete", "mcp_port", mcpPort, "dashboard_port", dashboardPort)
}

func handleMCPHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(buildHealthPayload())
}

func handleEmbedHealthHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	payload := buildHealthPayload()
	embedState, _, embedErr := embedder.HealthSnapshot()
	status := "ok"
	if startup.Starting() {
		status = "starting"
		embedState = "loading"
	} else if startup.Failed() {
		status = "error"
		if embedErr == "" {
			embedErr = startup.Error()
		}
	} else if embedState == "error" {
		status = "error"
	}
	activeBackend, activeModel, activeRuntime, _, activeDim := embedder.ActiveSnapshot()
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":          status,
		"state":           embedState,
		"error":           embedErr,
		"model":           activeModel,
		"dimensions":      activeDim,
		"loaded":          embedder.IsLoaded(),
		"backend":         activeBackend,
		"runtime":         activeRuntime,
		"startup_phase":   payload["startup_phase"],
		"startup_message": payload["startup_message"],
	})
}

// applyInstanceName resolves the effective instance name from dbSetting
// (flag > env > setting > empty), scopes the session cookie and the login
// page to it. An empty dbSetting keeps the flag/env/fallback resolution.
func applyInstanceName(dbSetting string) {
	n := instance.Update(dbSetting, db.GetDataDir())
	httpguard.SetCookieScope(n.Scope)
	httpguard.SetInstanceName(n.Value)
}

func buildHealthPayload() map[string]interface{} {
	// Read the live value: a dashboard rename updates instance.Current
	// without a restart.
	cur := instance.Current()
	embedState, _, embedErr := embedder.HealthSnapshot()
	status := "healthy"
	if startup.Starting() {
		status = "starting"
		embedState = "loading"
	} else if startup.Failed() {
		status = "failed"
		if embedErr == "" {
			embedErr = startup.Error()
		}
	} else if startup.DataDirUnavailable() {
		status = "data_dir_unavailable"
		if embedErr == "" {
			embedErr = startup.Error()
		}
	} else if embedState == "error" {
		status = "degraded"
	}
	embedBackend, embedModel, _, _, _ := embedder.WiredSnapshot()
	payload := map[string]interface{}{
		"status":                status,
		"service":               "ast-context-cache",
		"version":               version.Version,
		"pid":                   os.Getpid(),
		"instance":              cur.Value,
		"instance_name":         cur.Value,
		"embedder":              embedder.IsLoaded(),
		"embed_state":           embedState,
		"embed_error":           embedErr,
		"embed_mode":            embedBackend,
		"embed_model":           embedModel,
		"startup_phase":         string(startup.CurrentPhase()),
		"startup_message":       startup.Message(),
		"abnormal_previous_run": embedqueue.AbnormalPreviousRun(),
		// search_nudge lets "ast-mcp hook search-nudge" and the OpenCode plugin read the flag cheaply.
		"search_nudge": flags.Enabled(flags.KeySearchNudge),
	}
	return payload
}

func handleEmbedHTTP(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Texts []string `json:"texts"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":"invalid JSON"}`, http.StatusBadRequest)
		return
	}
	if len(req.Texts) == 0 {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"embeddings": [][]float32{}})
		return
	}
	if startup.Starting() {
		http.Error(w, `{"error":"embedder starting"}`, http.StatusServiceUnavailable)
		return
	}
	if startup.Failed() {
		http.Error(w, fmt.Sprintf(`{"error":%q}`, startup.Error()), http.StatusServiceUnavailable)
		return
	}
	emb := embedder.Tracked()
	if emb == nil {
		http.Error(w, `{"error":"embedder not configured"}`, http.StatusServiceUnavailable)
		return
	}
	embeddings, err := emb.Embed(req.Texts)
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"%s"}`, err.Error()), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"embeddings": embeddings})
}

func resolveStartupWorkers(flagVal int) int {
	if flagVal >= 0 {
		return flagVal
	}
	if raw := strings.TrimSpace(os.Getenv("AST_EMBED_WORKERS")); raw != "" {
		if n, err := strconv.Atoi(raw); err == nil && n >= 0 {
			return n
		}
	}
	return -1
}

// startBackgroundServices starts the periodic jobs; the tickers owned here stop when
// ctx is cancelled at shutdown.
func startBackgroundServices(ctx context.Context) {
	// Run memory lifecycle jobs once at startup so a server that restarts daily still prunes
	memory.RunDailyPrune()
	memory.RunExpiryJob()
	memory.RunStaleAnchorJob()

	go docs.EmbedAllSources()
	purge.StartDeletedProjectSweep()
	db.StartFTSSelfCheck()
	go runEvery(ctx, time.Hour, logretention.RunOnce)
	go runEvery(ctx, 24*time.Hour, docs.UpdateAllSources)
	go runEvery(ctx, time.Hour, contextnotes.RunOffloadPurge)
	go runEvery(ctx, time.Hour, transcripts.RunOnce)
	// MEM-2 / MEM-8: background maintenance jobs for structured memory.
	go runEvery(ctx, 24*time.Hour, memory.RunDailyPrune)
	go runEvery(ctx, time.Hour, memory.RunExpiryJob)
	go runEvery(ctx, time.Hour, memory.RunStaleAnchorJob)
	ctxpkg.StartSessionStoreEviction(ctx)
	handoff.Start(ctx, embedder.Tracked())
	seen := map[string]bool{}
	if conn, err := db.IndexReader(); err == nil {
		restoreRows, err := conn.Query(selectSymbolProjectsQuery)
		if err == nil {
			for restoreRows.Next() {
				var pp string
				restoreRows.Scan(&pp)
				if projectmeta.IsExcluded(pp) {
					continue
				}
				seen[pp] = true
			}
			restoreRows.Close()
		}
	}
	for _, pp := range projectmeta.DiscoverPaths() {
		if projectmeta.IsExcluded(pp) {
			continue
		}
		seen[pp] = true
	}
	watcher.RegisterAllKnownProjects()
	for _, pp := range projectmeta.DiscoverPaths() {
		if projectmeta.IsExcluded(pp) {
			continue
		}
		watcher.RegisterKnownProject(pp)
		seen[pp] = true
	}
	for pp := range seen {
		maybeStartPinnedWatcher(pp)
	}
}

// runEvery calls fn every interval until ctx is cancelled.
func runEvery(ctx context.Context, interval time.Duration, fn func()) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			fn()
		}
	}
}

func maybeStartPinnedWatcher(projectPath string) {
	projectPath = watcher.NormalizeProjectPath(projectPath)
	if projectPath == "" || !db.IsPinnedProject(projectPath) {
		return
	}
	if info, err := os.Stat(projectPath); err != nil || !info.IsDir() {
		return
	}
	go watcher.EnsureWatcher(projectPath)
}
