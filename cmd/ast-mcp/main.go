package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
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
	"github.com/coma-toast/ast-context-cache/internal/search"
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

	exePath, _ := os.Executable()
	exeDir := filepath.Dir(exePath)

	if err := db.Init(); err != nil {
		log.Fatalf("DB error: %v", err)
	}

	go db.StartWALCheckpoint()

	for _, tbl := range []string{"symbols", "edges", "queries", "vectors", "summaries"} {
		db.DB.Exec("DELETE FROM " + tbl + " WHERE project_path = '.'")
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
				go embedqueue.RecoverAfterEmbedder()
				go embedqueue.FlushPendingIfReady()
			}
		},
	})
	if err := embedder.InitRuntime(modelDir); err != nil {
		log.Fatalf("embedder: %v", err)
	}
	emb := embedder.Tracked()
	wb, wm, _, _, wd := embedder.WiredSnapshot()
	log.Printf("Embedder configured: backend=%s model=%s dims=%d", wb, wm, wd)
	embedqueue.Start(emb)
	embedder.SetOnRecovery(embedqueue.RecoverAfterEmbedder)
	embedder.SetOnReady(embedqueue.FlushPendingIfReady)
	embedder.SetOnError(embedqueue.OnEmbedderError)
	embedqueue.StartErrorScanLoop()
	embedqueue.StartPendingReconciler()
	watcher.PostIndexHook = func(filePath, projectPath string, removed bool) {
		if removed {
			search.Cache.DeleteByFile(filePath, projectPath)
		} else {
			if indexer.ShouldSkipEmbed(filePath) {
				return
			}
			embedqueue.SubmitPriority(filePath, projectPath, db.IsPinnedProject(projectPath))
		}
	}

	mcpMux := http.NewServeMux()
	mcpMux.HandleFunc("/mcp", mcp.NewHandler())
	mcpMux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		embedState, _, embedErr := embedder.HealthSnapshot()
		status := "healthy"
		if embedState == "error" {
			status = "degraded"
		}
		embedBackend, embedModel, _, _, _ := embedder.WiredSnapshot()
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status":         status,
			"service":        "ast-context-cache",
			"version":        version.Version,
			"embedder":       embedder.IsLoaded(),
			"embed_state":    embedState,
			"embed_error":    embedErr,
			"embed_mode":     embedBackend,
			"embed_model":    embedModel,
		})
	})
	mcpMux.HandleFunc("/embed", func(w http.ResponseWriter, r *http.Request) {
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
		embeddings, err := emb.Embed(req.Texts)
		if err != nil {
			http.Error(w, fmt.Sprintf(`{"error":"%s"}`, err.Error()), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"embeddings": embeddings})
	})
	mcpMux.HandleFunc("/embed/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		embedState, _, embedErr := embedder.HealthSnapshot()
		status := "ok"
		if embedState == "error" {
			status = "error"
		}
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status":     status,
			"state":      embedState,
			"error":      embedErr,
			"model":      embedder.ActiveModel,
			"dimensions": embedder.ActiveDim,
			"loaded":     embedder.IsLoaded(),
			"backend":    embedder.ActiveBackend,
			"runtime":    embedder.ActiveRuntime,
		})
	})
	mcpMux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/api/") || strings.HasPrefix(r.URL.Path, "/mcp") {
			return
		}
		json.NewEncoder(w).Encode(map[string]interface{}{"service": "AST MCP", "dashboard": fmt.Sprintf("http://localhost:%d", dashboardPort)})
	})

	dashHandler := dashboard.NewHandler("")

	go func() {
		addr := fmt.Sprintf(":%d", mcpPort)
		log.Printf("MCP: http://localhost%s/mcp", addr)
		log.Fatal(http.ListenAndServe(addr, mcpMux))
	}()

	go startBackgroundServices()

	addr := fmt.Sprintf(":%d", dashboardPort)
	log.Printf("Dashboard: http://localhost%s", addr)
	log.Fatal(http.ListenAndServe(addr, dashHandler))
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
	restoreRows, err := db.DB.Query("SELECT DISTINCT project_path FROM symbols WHERE project_path IS NOT NULL AND project_path != ''")
	if err == nil {
		for restoreRows.Next() {
			var pp string
			restoreRows.Scan(&pp)
			if info, sErr := os.Stat(pp); sErr == nil && info.IsDir() {
				go watcher.EnsureWatcher(pp)
			}
		}
		restoreRows.Close()
	}
}
