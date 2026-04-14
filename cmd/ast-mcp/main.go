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
	"github.com/coma-toast/ast-context-cache/internal/mcp"
	"github.com/coma-toast/ast-context-cache/internal/search"
	"github.com/coma-toast/ast-context-cache/internal/watcher"
)

const (
	mcpPort       = 7821
	dashboardPort = 7830
)

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

	modelDir := os.Getenv("MODEL_DIR")
	if modelDir == "" {
		modelDir = filepath.Join(exeDir, "model")
	}
	if err := embedder.EnsureModel(modelDir); err != nil {
		log.Printf("WARNING: Could not ensure embedder model files: %v", err)
	}
	emb := embedder.NewLazy(modelDir)
	log.Printf("Embedder configured (lazy-load from %s)", modelDir)
	mcp.SetEmbedder(emb)
	ctxpkg.Emb = emb
	embedqueue.Start(emb)
	watcher.PostIndexHook = func(filePath, projectPath string, removed bool) {
		if removed {
			search.Cache.DeleteByFile(filePath, projectPath)
		} else {
			embedqueue.SubmitPriority(filePath, projectPath, db.IsPinnedProject(projectPath))
		}
	}

	restoreRows, err := db.DB.Query("SELECT DISTINCT project_path FROM symbols WHERE project_path IS NOT NULL AND project_path != ''")
	if err == nil {
		for restoreRows.Next() {
			var pp string
			restoreRows.Scan(&pp)
			if info, sErr := os.Stat(pp); sErr == nil && info.IsDir() {
				go watcher.StartWatcher(pp)
			}
		}
		restoreRows.Close()
	}

	go func() {
		ticker := time.NewTicker(1 * time.Hour)
		defer ticker.Stop()
		for range ticker.C {
			docs.UpdateAllSources()
		}
	}()

	mcpMux := http.NewServeMux()
	mcpMux.HandleFunc("/mcp", mcp.NewHandler())
	mcpMux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"status": "healthy", "service": "ast-context-cache", "version": "2.0.0", "embedder": emb.IsLoaded()})
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
		json.NewEncoder(w).Encode(map[string]interface{}{"status": "ok", "model": embedder.ModelName, "dimensions": embedder.Dimensions, "loaded": emb.IsLoaded()})
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

	addr := fmt.Sprintf(":%d", dashboardPort)
	log.Printf("Dashboard: http://localhost%s", addr)
	log.Fatal(http.ListenAndServe(addr, dashHandler))
}
