package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	ctxpkg "github.com/coma-toast/ast-context-cache/internal/context"
	"github.com/coma-toast/ast-context-cache/internal/dashboard"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/docs"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/embedqueue"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/coma-toast/ast-context-cache/internal/logging"
	"github.com/coma-toast/ast-context-cache/internal/logretention"
	"github.com/coma-toast/ast-context-cache/internal/mcp"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
	"github.com/coma-toast/ast-context-cache/internal/projectmeta"
	"github.com/coma-toast/ast-context-cache/internal/purge"
	"github.com/coma-toast/ast-context-cache/internal/search"
	"github.com/coma-toast/ast-context-cache/internal/startup"
	"github.com/coma-toast/ast-context-cache/internal/sys"
	"github.com/coma-toast/ast-context-cache/internal/version"
	"github.com/coma-toast/ast-context-cache/internal/watcher"
)

// Overridable (-mcp-port / -dashboard-port) so a second instance can run
// beside the shared one, e.g. with HOME pointed at a scratch data dir.
var (
	mcpPort       = 7821
	dashboardPort = 7830
)

const (
	deleteFromQueryPrefix        = "DELETE FROM "
	whereDotProjectQuerySuffix   = " WHERE project_path = '.'"
	deleteDotProjectQueriesQuery = "DELETE FROM queries WHERE project_path = '.'"
	selectSymbolProjectsQuery    = "SELECT DISTINCT project_path FROM symbols WHERE project_path IS NOT NULL AND project_path != ''"
)

var startTime = time.Now()

func GetStartTime() time.Time {
	return startTime
}

func main() {
	logging.Setup(os.Stderr)
	tierFlag := flag.String("tier", "", "Tool tier: core, extended, complete (default: from AST_MCP_TIER env or complete)")
	codeModeFlag := flag.Bool("code-mode", true, "Enable execute_code sandbox tool (default: true)")
	embedWorkersFlag := flag.Int("embed-workers", -1, "Embed worker count at startup (-1 = auto/DB)")
	flag.IntVar(&mcpPort, "mcp-port", mcpPort, "MCP HTTP port")
	flag.IntVar(&dashboardPort, "dashboard-port", dashboardPort, "Dashboard HTTP port")
	flag.Parse()

	cfg := mcp.DefaultConfig()
	if *tierFlag != "" {
		cfg.ActiveTier = mcp.ParseTier(*tierFlag)
	}
	if !*codeModeFlag {
		cfg.CodeMode = false
	}
	mcp.SetConfig(cfg)
	logger.Info("Config", "tier", cfg.ActiveTier, "code_mode", cfg.CodeMode)

	logger.Info("Initializing")
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

	dashHandler := dashboard.NewHandler("")
	// Constructed here (before the db.Init() goroutine below reads it via restartHook)
	// rather than at its ListenAndServe call further down, so that read has a clear
	// happens-before edge and isn't a data race with this assignment.
	dashSrv := &http.Server{Addr: fmt.Sprintf(":%d", dashboardPort), Handler: dashHandler}
	dbReady := make(chan error, 1)

	go func() {
		if err := db.Init(); err != nil {
			startup.MarkFailed(err.Error())
			logger.Error("Failed to initialize databases", "error", err)
			dbReady <- err
			return
		}
		projectmeta.SetDisplayNameOverrideFunc(db.ProjectDisplayName)
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

		mcpMux := http.NewServeMux()
		mcpMux.HandleFunc("/mcp", mcp.NewHandler())
		mcpMux.HandleFunc("/health", handleMCPHealth)
		mcpMux.HandleFunc("/embed", handleEmbedHTTP)
		mcpMux.HandleFunc("/embed/health", handleEmbedHealthHTTP)
		mcpMux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
			if strings.HasPrefix(r.URL.Path, "/api/") || strings.HasPrefix(r.URL.Path, "/mcp") {
				return
			}
			json.NewEncoder(w).Encode(map[string]interface{}{"service": "AST MCP", "dashboard": fmt.Sprintf("http://localhost:%d", dashboardPort)})
		})

		mcpSrv := &http.Server{Addr: fmt.Sprintf(":%d", mcpPort), Handler: mcpMux}
		db.RestartProcess = restartProcess(mcpSrv, dashSrv)

		go func() {
			logger.Info("Starting MCP server", "url", "http://localhost"+mcpSrv.Addr+"/mcp")
			if err := mcpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
				logger.Error("MCP server failed", "error", err)
				os.Exit(1)
			}
		}()

		embedder.MarkLoading()
		finishStartup(exeDir, *embedWorkersFlag)
	}()

	go func() {
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
		<-sig
		logger.Info("Shutting down")
		db.RequestShutdown()
		embedqueue.EndRunLock()
		os.Exit(0)
	}()

	logger.Info("Starting dashboard server", "url", "http://localhost"+dashSrv.Addr)
	if err := dashSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		logger.Error("Dashboard server failed", "error", err)
		os.Exit(1)
	}
}

// restartProcess returns the hook wired to db.RestartProcess: drain both HTTP servers
// (stop accepting new connections, let in-flight requests finish), pause the embed
// queue the same way WAL maintenance already does before a risky operation, then
// re-exec the current binary in place so a fresh main() and db.Init() pick up whatever
// just changed on disk (a moved data directory, an updated binary after `git pull` +
// rebuild).
func restartProcess(mcpSrv, dashSrv *http.Server) func() {
	return func() {
		logger.Info("Draining connections to restart")
		// db.RequestShutdown aborts any in-progress WAL checkpoint quickly, but it also
		// unconditionally calls AfterForceCheckpoint (wired below to
		// embedqueue.RestoreAfterMaintenance) as a side effect — harmless for its
		// original SIGINT/SIGTERM use (the process exits right after), but it would
		// silently undo the embed-queue pause below if called after it. Call it first
		// so PauseAllForMaintenance has the final say and workers stay paused through
		// the drain and into the exec.
		db.RequestShutdown()
		embedqueue.PauseAllForMaintenance(2 * time.Minute)
		logger.Info("Embed queue paused, shutting down servers")

		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if err := mcpSrv.Shutdown(ctx); err != nil {
			logger.Warn("Failed to shut down MCP server", "error", err)
		}
		if err := dashSrv.Shutdown(ctx); err != nil {
			logger.Warn("Failed to shut down dashboard server", "error", err)
		}
		logger.Info("Servers shut down")

		exe, err := os.Executable()
		if err != nil {
			logger.Error("Failed to resolve executable to restart", "error", err)
			os.Exit(1)
		}
		logger.Info("Restarting", "exe", exe)
		if err := syscall.Exec(exe, os.Args, os.Environ()); err != nil {
			logger.Error("Failed to restart", "exe", exe, "error", err)
			os.Exit(1)
		}
	}
}

func finishStartup(exeDir string, embedWorkersFlag int) {
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
	startBackgroundServices()
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

func buildHealthPayload() map[string]interface{} {
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
	} else if embedState == "error" {
		status = "degraded"
	}
	embedBackend, embedModel, _, _, _ := embedder.WiredSnapshot()
	return map[string]interface{}{
		"status":                status,
		"service":               "ast-context-cache",
		"version":               version.Version,
		"embedder":              embedder.IsLoaded(),
		"embed_state":           embedState,
		"embed_error":           embedErr,
		"embed_mode":            embedBackend,
		"embed_model":           embedModel,
		"startup_phase":         string(startup.CurrentPhase()),
		"startup_message":       startup.Message(),
		"abnormal_previous_run": embedqueue.AbnormalPreviousRun(),
	}
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

func startBackgroundServices() {
	go docs.EmbedAllSources()
	purge.StartDeletedProjectSweep()
	db.StartFTSSelfCheck()
	go func() {
		ticker := time.NewTicker(1 * time.Hour)
		defer ticker.Stop()
		for range ticker.C {
			logretention.RunOnce()
		}
	}()
	go func() {
		ticker := time.NewTicker(24 * time.Hour)
		defer ticker.Stop()
		for range ticker.C {
			docs.UpdateAllSources()
		}
	}()
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
