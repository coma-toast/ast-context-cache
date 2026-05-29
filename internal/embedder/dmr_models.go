package embedder

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"
)

type openAIModelsListResponse struct {
	Data []struct {
		ID string `json:"id"`
	} `json:"data"`
}

type ollamaTagsResponse struct {
	Models []struct {
		Name string `json:"name"`
	} `json:"models"`
}

// ListDMRModels returns model ids from Docker Model Runner at rawURL (host or /engines/v1 root).
func ListDMRModels(rawURL string) ([]string, error) {
	base := normalizeDMRBase(strings.TrimSpace(rawURL))
	client := &http.Client{Timeout: 15 * time.Second}
	if ids, err := ListOpenAIModelsWithClient(client, base, ""); err == nil && len(ids) > 0 {
		return ids, nil
	} else if err != nil && !isHTTPReachableErr(err) {
		return nil, err
	}
	host := dmrHostRoot(base)
	if ids, err := ListOllamaModelsWithClient(client, host); err == nil && len(ids) > 0 {
		return ids, nil
	} else if err != nil {
		return nil, err
	}
	return nil, fmt.Errorf("no models returned from %s (is Model Runner running?)", host)
}

// ListOpenAIModels lists models from an OpenAI-compatible GET {base}/models endpoint.
func ListOpenAIModels(baseURL, apiKey string) ([]string, error) {
	client := &http.Client{Timeout: 15 * time.Second}
	return ListOpenAIModelsWithClient(client, baseURL, apiKey)
}

func ListOpenAIModelsWithClient(client *http.Client, baseURL, apiKey string) ([]string, error) {
	base := strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if base == "" {
		base = "https://api.openai.com/v1"
	}
	req, err := http.NewRequest(http.MethodGet, base+"/models", nil)
	if err != nil {
		return nil, err
	}
	if k := strings.TrimSpace(apiKey); k != "" {
		req.Header.Set("Authorization", "Bearer "+k)
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("models: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
	}
	var list openAIModelsListResponse
	if err := json.NewDecoder(resp.Body).Decode(&list); err != nil {
		return nil, err
	}
	return uniqueSortedModelIDs(collectOpenAIModelIDs(list)), nil
}

func collectOpenAIModelIDs(list openAIModelsListResponse) []string {
	var ids []string
	for _, m := range list.Data {
		if id := strings.TrimSpace(m.ID); id != "" {
			ids = append(ids, id)
		}
	}
	return ids
}

// ListOllamaModels lists models from Ollama GET /api/tags.
func ListOllamaModels(host string) ([]string, error) {
	client := &http.Client{Timeout: 15 * time.Second}
	return ListOllamaModelsWithClient(client, host)
}

func ListOllamaModelsWithClient(client *http.Client, host string) ([]string, error) {
	host = strings.TrimRight(strings.TrimSpace(host), "/")
	if host == "" {
		host = "http://127.0.0.1:11434"
	}
	if !strings.HasPrefix(host, "http://") && !strings.HasPrefix(host, "https://") {
		host = "http://" + host
	}
	resp, err := client.Get(host + "/api/tags")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("ollama /api/tags: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
	}
	var tags ollamaTagsResponse
	if err := json.NewDecoder(resp.Body).Decode(&tags); err != nil {
		return nil, err
	}
	var ids []string
	for _, m := range tags.Models {
		if name := strings.TrimSpace(m.Name); name != "" {
			ids = append(ids, name)
		}
	}
	return uniqueSortedModelIDs(ids), nil
}

func dmrHostRoot(base string) string {
	host := strings.TrimSuffix(strings.TrimRight(base, "/"), "/engines/v1")
	if host == "" {
		return DefaultDockerURL
	}
	return host
}

func uniqueSortedModelIDs(ids []string) []string {
	seen := make(map[string]struct{}, len(ids))
	var out []string
	for _, id := range ids {
		id = strings.TrimSpace(id)
		if id == "" {
			continue
		}
		if _, ok := seen[id]; ok {
			continue
		}
		seen[id] = struct{}{}
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}

func isHTTPReachableErr(err error) bool {
	if err == nil {
		return false
	}
	s := err.Error()
	return strings.Contains(s, "connection refused") ||
		strings.Contains(s, "no such host") ||
		strings.Contains(s, "timeout") ||
		strings.Contains(s, "context deadline exceeded")
}
