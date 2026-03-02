package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	ctxpkg "github.com/coma-toast/ast-context-cache/internal/context"
	"github.com/coma-toast/ast-context-cache/internal/dashboard"
	"github.com/coma-toast/ast-context-cache/internal/db"
	"github.com/coma-toast/ast-context-cache/internal/embedder"
	"github.com/coma-toast/ast-context-cache/internal/indexer"
	"github.com/coma-toast/ast-context-cache/internal/mcp"
	"github.com/coma-toast/ast-context-cache/internal/search"
	"github.com/coma-toast/ast-context-cache/internal/watcher"
)

const (
	mcpPort       = 7821
	dashboardPort = 7830
)

func main() {
	log.Println("Initializing...")

	exePath, _ := os.Executable()
	exeDir := filepath.Dir(exePath)
	frontendDir := filepath.Join(exeDir, "dist")

	if err := db.Init(); err != nil {
		log.Fatalf("DB error: %v", err)
	}

	go db.StartWALCheckpoint()

	if err := search.Cache.Load(); err != nil {
		log.Printf("WARNING: Failed to load vector cache: %v", err)
	}

	modelDir := os.Getenv("MODEL_DIR")
	if modelDir == "" {
		modelDir = filepath.Join(exeDir, "model")
	}
	if err := embedder.EnsureModel(modelDir); err != nil {
		log.Printf("WARNING: Could not ensure embedder model files: %v", err)
	}
	emb, err := embedder.New(modelDir)
	if err != nil {
		log.Printf("WARNING: Embedder init failed (vector features disabled): %v", err)
		log.Printf("Embedder tip: need model/ (tokenizer.json + model.onnx) next to binary or MODEL_DIR set, and ONNXRUNTIME_LIB set to libonnxruntime path. See README.")
	} else {
		log.Printf("Embedder loaded: %s (%d dims)", embedder.ModelName, embedder.Dimensions)
		mcp.SetEmbedder(emb)
		ctxpkg.Emb = emb
		watcher.PostIndexHook = func(filePath, projectPath string, removed bool) {
			if removed {
				search.Cache.DeleteByFile(filePath, projectPath)
			} else {
				indexer.EmbedFileSymbols(emb, filePath, projectPath)
			}
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

	mcpMux := http.NewServeMux()
	mcpMux.HandleFunc("/mcp", mcp.NewHandler())
	mcpMux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"status": "healthy", "service": "ast-context-cache", "version": "2.0.0", "embedder": emb != nil})
	})
	if emb != nil {
		mcpMux.HandleFunc("/embed", emb.HandleEmbed)
		mcpMux.HandleFunc("/embed/health", emb.HandleHealth)
	}
	mcpMux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/api/") || strings.HasPrefix(r.URL.Path, "/mcp") {
			return
		}
		json.NewEncoder(w).Encode(map[string]interface{}{"service": "AST MCP", "dashboard": fmt.Sprintf("http://localhost:%d", dashboardPort)})
	})

	dashHandler := dashboard.NewHandler(frontendDir)

	go func() {
		addr := fmt.Sprintf(":%d", mcpPort)
		log.Printf("MCP: http://localhost%s/mcp", addr)
		log.Fatal(http.ListenAndServe(addr, mcpMux))
	}()

	addr := fmt.Sprintf(":%d", dashboardPort)
	log.Printf("Dashboard: http://localhost%s", addr)
	log.Fatal(http.ListenAndServe(addr, dashHandler))
}
