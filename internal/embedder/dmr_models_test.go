package embedder

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestListDMRModels_OpenAI(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/engines/v1/models" {
			t.Fatalf("path %s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(openAIModelsListResponse{
			Data: []struct {
				ID string `json:"id"`
			}{
				{ID: "docker.io/ai/qwen3-embedding:latest"},
				{ID: "docker.io/ai/nomic-embed-text-v2-moe:latest"},
			},
		})
	}))
	defer srv.Close()
	host := srv.URL[len("http://"):]
	ids, err := ListDMRModels("http://" + host)
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 2 || ids[0] != "docker.io/ai/nomic-embed-text-v2-moe:latest" {
		t.Fatalf("got %v", ids)
	}
}

func TestListDMRModels_OllamaFallback(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/engines/v1/models":
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"object":"list","data":[]}`))
		case "/api/tags":
			_ = json.NewEncoder(w).Encode(ollamaTagsResponse{
				Models: []struct {
					Name string `json:"name"`
				}{{Name: "ai/smollm2"}},
			})
		default:
			t.Fatalf("path %s", r.URL.Path)
		}
	}))
	defer srv.Close()
	host := srv.URL[len("http://"):]
	ids, err := ListDMRModels("http://" + host)
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 1 || ids[0] != "ai/smollm2" {
		t.Fatalf("got %v", ids)
	}
}
