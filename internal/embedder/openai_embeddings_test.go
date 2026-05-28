package embedder

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestOpenAIEmbedder_Success(t *testing.T) {
	ActiveDim = 768
	vec := make([]float64, 768)
	for i := range vec {
		vec[i] = 0.25
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" {
			t.Fatalf("path %s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(openAIEmbedAPIResponse{
			Data: []openAIEmbedDatum{{Embedding: vec, Index: 0}},
		})
	}))
	defer srv.Close()
	SetActive("openai", "test-model", 768, "openai", srv.URL+"/embeddings")
	o := NewOpenAIEmbedder(srv.URL, "", "test-model", 0)
	out, err := o.Embed([]string{"hello"})
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 1 || len(out[0]) != 768 {
		t.Fatalf("bad shape: %d x %d", len(out), len(out[0]))
	}
}

func TestOpenAIEmbedder_WrongDim(t *testing.T) {
	ActiveDim = 768
	small := make([]float64, 12)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(openAIEmbedAPIResponse{
			Data: []openAIEmbedDatum{{Embedding: small, Index: 0}},
		})
	}))
	defer srv.Close()
	SetActive("openai", "m", 768, "openai", srv.URL+"/embeddings")
	o := NewOpenAIEmbedder(srv.URL, "", "m", 0)
	_, err := o.Embed([]string{"a"})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestOpenAIEmbedder_ReorderByIndex(t *testing.T) {
	ActiveDim = 768
	v0 := make([]float64, 768)
	v0[0] = 1 // "first" — dominant axis 0
	v1 := make([]float64, 768)
	v1[1] = 1 // "second" — dominant axis 1
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(openAIEmbedAPIResponse{
			Data: []openAIEmbedDatum{
				{Embedding: v1, Index: 1},
				{Embedding: v0, Index: 0},
			},
		})
	}))
	defer srv.Close()
	SetActive("openai", "m", 768, "openai", srv.URL+"/embeddings")
	o := NewOpenAIEmbedder(srv.URL, "", "m", 0)
	out, err := o.Embed([]string{"first", "second"})
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 {
		t.Fatalf("len %d", len(out))
	}
	// L2-normalized: first input should align with axis 0, second with axis 1
	if out[0][0] <= out[0][1] || out[1][1] <= out[1][0] {
		t.Fatalf("order wrong after norm: first axis0/1=%v/%v second=%v/%v", out[0][0], out[0][1], out[1][0], out[1][1])
	}
}

func TestOpenAIEmbedder_RequestIncludesDimensions(t *testing.T) {
	ActiveDim = 768
	vec := make([]float64, 768)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req openAIEmbedAPIRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatal(err)
		}
		if req.Dimensions == nil || *req.Dimensions != 768 {
			t.Fatalf("dimensions: %#v", req.Dimensions)
		}
		_ = json.NewEncoder(w).Encode(openAIEmbedAPIResponse{
			Data: []openAIEmbedDatum{{Embedding: vec, Index: 0}},
		})
	}))
	defer srv.Close()
	o := NewOpenAIEmbedder(srv.URL, "", "m", 768)
	_, err := o.Embed([]string{"x"})
	if err != nil {
		t.Fatal(err)
	}
}
