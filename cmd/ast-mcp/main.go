package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
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
	"github.com/coma-toast/ast-context-cache/internal/logretention"
	"github.com/coma-toast/ast-context-cache/internal/mcp"
	"github.com/coma-toast/ast-context-cache/internal/projectlinks"
	"github.com/coma-toast/ast-context-cache/internal/projectmeta"
	"github.com/coma-toast/ast-context-cache/internal/search"
	"github.com/coma-toast/ast-context-cache/internal/startup"
	"github.com/coma-toast/ast-context-cache/internal/watcher"
	"github.com/coma-toast/ast-context-cache/internal/version"
)

const (
	mcpPort       = 7821
	dashboardPort = 7830
)

var startTime = time.Now()

func GetStartTime() time.Time {
	return startTime
}

func main() {
	tierFlag := flag.String("tier", "", "Tool tier: core, extended, complete (default: from AST_MCP_TIER env or complete)")
	codeModeFlag := flag.Bool("code-mode", true, "Enable execute_code sandbox tool (default: true)")
	embedWorkersFlag := flag.Int("embed-workers", -1, "Embed worker count at startup (-1 = auto/DB)")
	flag.Parse()

	cfg := mcp.DefaultConfig()
	if *tierFlag != "" {
		cfg.ActiveTier = mcp.ParseTier(*tierFlag)
	}
	if !*codeModeFlag {
		cfg.CodeMode = false
	}
	mcp.SetConfig(cfg)
	log.Printf("Config: tier=%s code_mode=%v", cfg.ActiveTier, cfg.CodeMode)

	log.Println("Initializing...")
	if fi, err := os.Stdout.Stat(); err == nil && fi.Mode()&os.ModeCharDevice != 0 {
		logPath := db.DefaultLogPath()
		_ = os.MkdirAll(filepath.Dir(logPath), 0755)
		if f, err := os.OpenFile(logPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644); err == nil {
			log.SetOutput(f)
			log.Printf("Logging to %s", logPath)
		}
	}
	startup.SetMessage("Opening databases…")

	exePath, _ := os.Executable()
	exeDir := filepath.Dir(exePath)

	dashHandler := dashboard.NewHandler("")
	dbReady := make(chan error, 1)

	go func() {
		if err := db.Init(); err != nil {
			startup.MarkFailed(err.Error())
			log.Printf("DB error: %v", err)
			dbReady <- err
			return
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
			log.Printf("embedqueue: previous run exited abnormally; using persisted worker count from DB")
		}
		watcher.EnsureDefaultIgnoreGlobs()
		go db.StartWALCheckpoint()

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

		go func() {
			addr := fmt.Sprintf(":%d", mcpPort)
			log.Printf("MCP: http://localhost%s/mcp (starting)", addr)
			log.Fatal(http.ListenAndServe(addr, mcpMux))
		}()

		embedder.MarkLoading()
		finishStartup(exeDir, *embedWorkersFlag)
	}()

	go func() {
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
		<-sig
		log.Println("ast-mcp: shutting down")
		db.RequestShutdown()
		embedqueue.EndRunLock()
		os.Exit(0)
	}()

	addr := fmt.Sprintf(":%d", dashboardPort)
	log.Printf("Dashboard: http://localhost%s (starting)", addr)
	log.Fatal(http.ListenAndServe(addr, dashHandler))
}

func finishStartup(exeDir string, embedWorkersFlag int) {
	defer func() {
		if startup.Ready() {
			return
		}
		if r := recover(); r != nil {
			startup.MarkFailed(fmt.Sprint(r))
			log.Printf("startup panic: %v", r)
		}
	}()

	for _, tbl := range []string{"symbols", "edges", "vectors", "summaries"} {
		if db.IndexDB != nil {
			db.IndexDB.Exec("DELETE FROM "+tbl+" WHERE project_path = '.'")
		}
	}
	if db.DB != nil {
		db.DB.Exec("DELETE FROM queries WHERE project_path = '.'")
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
		log.Printf("embedder: %v (dashboard and MCP up; embeddings unavailable)", err)
		return
	}
	emb := embedder.Tracked()
	wb, wm, _, _, wd := embedder.WiredSnapshot()
	log.Printf("Embedder configured: backend=%s model=%s dims=%d", wb, wm, wd)
	if n := resolveStartupWorkers(embedWorkersFlag); n >= 0 {
		embedqueue.SetStartupWorkers(n)
		log.Printf("embedqueue: startup workers override: %d", n)
	}
	startup.SetMessage("Starting embed queue…")
	embedqueue.Start(emb)
	if err := embedder.InitAuxRuntime(modelDir); err != nil {
		log.Printf("aux embedder: %v (aux catch-up workers disabled)", err)
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
	log.Printf("Startup complete (MCP :%d  Dashboard :%d)", mcpPort, dashboardPort)
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
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":          status,
		"state":           embedState,
		"error":           embedErr,
		"model":           embedder.ActiveModel,
		"dimensions":      embedder.ActiveDim,
		"loaded":          embedder.IsLoaded(),
		"backend":         embedder.ActiveBackend,
		"runtime":         embedder.ActiveRuntime,
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
		"status":          status,
		"service":         "ast-context-cache",
		"version":         version.Version,
		"embedder":        embedder.IsLoaded(),
		"embed_state":     embedState,
		"embed_error":     embedErr,
		"embed_mode":      embedBackend,
		"embed_model":     embedModel,
		"startup_phase":   string(startup.CurrentPhase()),
		"startup_message": startup.Message(),
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
	if db.IndexDB != nil {
		restoreRows, err := db.IndexDB.Query("SELECT DISTINCT project_path FROM symbols WHERE project_path IS NOT NULL AND project_path != ''")
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
