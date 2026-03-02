package mcp

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/search"
)

func handleSyncRemote(toolArgs map[string]interface{}, projectPath string) string {
	direction, _ := toolArgs["direction"].(string)
	if direction == "" {
		direction = "push"
	}
	collection := "configsync_docs"
	if c, ok := toolArgs["collection"].(string); ok && c != "" {
		collection = c
	}

	remoteURL := os.Getenv("REMOTE_VECTORDB_URL")
	if remoteURL == "" {
		remoteURL = "https://ai.jasondale.me/vectordb/mcp"
	}

	switch direction {
	case "push":
		return syncPush(projectPath, collection, remoteURL)
	case "pull":
		return syncPull(projectPath, collection, remoteURL)
	default:
		return `{"error": "direction must be 'push' or 'pull'"}`
	}
}

func syncPush(projectPath, collection, remoteURL string) string {
	entries := search.Cache.GetAll(projectPath)
	if len(entries) == 0 {
		data, _ := json.Marshal(map[string]string{"status": "nothing to push", "project_path": projectPath})
		return string(data)
	}

	batchSize := 10
	pushed := 0
	errors := 0

	for i := 0; i < len(entries); i += batchSize {
		end := i + batchSize
		if end > len(entries) {
			end = len(entries)
		}
		batch := entries[i:end]

		var ids []int
		var contents, sourceFiles, docTypes, names, kinds []string
		var denseVectors [][]float32

		for j, e := range batch {
			ids = append(ids, i+j+1)
			contents = append(contents, e.ContentHash)
			sourceFiles = append(sourceFiles, e.SourceFile)
			docTypes = append(docTypes, e.DocType)
			names = append(names, e.Name)
			kinds = append(kinds, e.Kind)
			denseVectors = append(denseVectors, e.Vector)
		}

		insertReq := map[string]interface{}{
			"jsonrpc": "2.0",
			"id":      fmt.Sprintf("push-%d", i/batchSize),
			"method":  "tools/call",
			"params": map[string]interface{}{
				"name": "insert_data",
				"arguments": map[string]interface{}{
					"collection_name": collection,
					"data": map[string]interface{}{
						"id":           ids,
						"content":      contents,
						"source_file":  sourceFiles,
						"doc_type":     docTypes,
						"name":         names,
						"dense_vector": denseVectors,
					},
				},
			},
		}

		body, _ := json.Marshal(insertReq)
		resp, err := ssePost(remoteURL, body)
		if err != nil {
			log.Printf("sync push batch %d error: %v", i/batchSize, err)
			errors++
			continue
		}
		_ = resp
		pushed += len(batch)
	}

	data, _ := json.Marshal(map[string]interface{}{
		"status":  "pushed",
		"pushed":  pushed,
		"errors":  errors,
		"total":   len(entries),
		"collection": collection,
	})
	return string(data)
}

func syncPull(projectPath, collection, remoteURL string) string {
	queryReq := map[string]interface{}{
		"jsonrpc": "2.0",
		"id":      "pull-query",
		"method":  "tools/call",
		"params": map[string]interface{}{
			"name": "query_data",
			"arguments": map[string]interface{}{
				"collection_name": collection,
				"limit":           1000,
			},
		},
	}

	body, _ := json.Marshal(queryReq)
	resp, err := ssePost(remoteURL, body)
	if err != nil {
		data, _ := json.Marshal(map[string]string{"error": "pull failed: " + err.Error()})
		return string(data)
	}

	var rpcResp map[string]interface{}
	if err := json.Unmarshal(resp, &rpcResp); err != nil {
		data, _ := json.Marshal(map[string]string{"error": "parse response: " + err.Error()})
		return string(data)
	}

	data, _ := json.Marshal(map[string]interface{}{
		"status":     "pulled",
		"collection": collection,
		"response":   rpcResp,
	})
	return string(data)
}

func ssePost(url string, body []byte) ([]byte, error) {
	req, err := http.NewRequest("POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")

	token := os.Getenv("REMOTE_VECTORDB_TOKEN")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	ct := resp.Header.Get("Content-Type")
	if strings.Contains(ct, "text/event-stream") {
		for _, line := range strings.Split(string(data), "\n") {
			if strings.HasPrefix(line, "data: ") {
				return []byte(strings.TrimPrefix(line, "data: ")), nil
			}
		}
		return data, nil
	}
	return data, nil
}
