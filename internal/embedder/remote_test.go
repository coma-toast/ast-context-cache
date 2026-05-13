package embedder

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestHTTPEmbedder_Success(t *testing.T) {
	ActiveDim = 768
	vec := make([]float32, 768)
	for i := range vec {
		vec[i] = 1
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(httpEmbedResp{Embeddings: [][]float32{vec}})
	}))
	defer srv.Close()
	SetActive("http", "test", 768, "test", srv.URL)
	h := NewHTTPEmbedder(srv.URL, "")
	out, err := h.Embed([]string{"a"})
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 1 || len(out[0]) != 768 {
		t.Fatalf("bad shape: %d x %d", len(out), len(out[0]))
	}
}

func TestHTTPEmbedder_WrongDim(t *testing.T) {
	ActiveDim = 768
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(httpEmbedResp{Embeddings: [][]float32{make([]float32, 12)}})
	}))
	defer srv.Close()
	SetActive("http", "test", 768, "test", srv.URL)
	h := NewHTTPEmbedder(srv.URL, "")
	_, err := h.Embed([]string{"a"})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestOllamaEmbedder_Parse(t *testing.T) {
	ActiveDim = 768
	vec := make([]float32, 768)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(ollamaEmbedResp{Embeddings: [][]float32{vec}})
	}))
	defer srv.Close()
	SetActive("ollama", "nomic-embed-text", 768, "ollama", srv.URL)
	o := NewOllamaEmbedder(srv.URL, "m")
	out, err := o.postEmbed([]string{"x"})
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 1 {
		t.Fatal(len(out))
	}
}
