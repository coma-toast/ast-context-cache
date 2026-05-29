package embedder

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestListModels_Ollama(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/tags" {
			t.Fatalf("path %s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(ollamaTagsResponse{
			Models: []struct {
				Name string `json:"name"`
			}{{Name: "nomic-embed-text"}},
		})
	}))
	defer srv.Close()
	ids, err := ListModels(Settings{Backend: "ollama", OllamaHost: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 1 || ids[0] != "nomic-embed-text" {
		t.Fatalf("got %v", ids)
	}
}

func TestListModels_OpenAI(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			t.Fatalf("path %s", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer sk-test" {
			t.Fatalf("auth %q", got)
		}
		_ = json.NewEncoder(w).Encode(openAIModelsListResponse{
			Data: []struct {
				ID string `json:"id"`
			}{{ID: "text-embedding-3-small"}},
		})
	}))
	defer srv.Close()
	ids, err := ListModels(Settings{
		Backend:       "openai",
		OpenAIBaseURL: srv.URL + "/v1",
		OpenAIAPIKey:  "sk-test",
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 1 || ids[0] != "text-embedding-3-small" {
		t.Fatalf("got %v", ids)
	}
}

func TestListModels_Unsupported(t *testing.T) {
	_, err := ListModels(Settings{Backend: "http"})
	if err == nil {
		t.Fatal("expected error")
	}
}
