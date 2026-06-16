package embedder

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// OllamaEmbedder uses Ollama’s POST /api/embed. Default model nomic-embed-text is 768-dim, matching the vector store.
// See: https://docs.ollama.com/api/embed
type OllamaEmbedder struct {
	BaseURL string
	Model   string
	client  *http.Client
}

// NewOllamaEmbedder builds a client. baseURL is e.g. http://127.0.0.1:11434 (no path).
func NewOllamaEmbedder(baseURL, model string) *OllamaEmbedder {
	return newOllamaEmbedder(baseURL, model, 120*time.Second)
}

func newOllamaEmbedder(baseURL, model string, timeout time.Duration) *OllamaEmbedder {
	b := strings.TrimSpace(baseURL)
	if b == "" {
		b = "http://127.0.0.1:11434"
	}
	b = strings.TrimRight(b, "/")
	m := strings.TrimSpace(model)
	if m == "" {
		m = "nomic-embed-text"
	}
	if timeout <= 0 {
		timeout = 120 * time.Second
	}
	return &OllamaEmbedder{
		BaseURL: b,
		Model:   m,
		client:  &http.Client{Timeout: timeout},
	}
}

type ollamaEmbedReq struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type ollamaEmbedResp struct {
	Embeddings [][]float32 `json:"embeddings"`
}

const ollamaChunk = 32

func (o *OllamaEmbedder) embedURL() string { return o.BaseURL + "/api/embed" }

func (o *OllamaEmbedder) Embed(texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	var all [][]float32
	for start := 0; start < len(texts); start += ollamaChunk {
		end := start + ollamaChunk
		if end > len(texts) {
			end = len(texts)
		}
		chunk := texts[start:end]
		vecs, err := o.postEmbed(chunk)
		if err != nil {
			return nil, err
		}
		all = append(all, vecs...)
	}
	return all, nil
}

func (o *OllamaEmbedder) postEmbed(texts []string) ([][]float32, error) {
	body, err := json.Marshal(ollamaEmbedReq{Model: o.Model, Input: texts})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequest(http.MethodPost, o.embedURL(), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := o.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama embed: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("ollama %s: %s", resp.Status, truncateForErr(raw, 200))
	}
	var out ollamaEmbedResp
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("ollama response json: %w", err)
	}
	if len(out.Embeddings) != len(texts) {
		return nil, fmt.Errorf("ollama: got %d embeddings for %d inputs", len(out.Embeddings), len(texts))
	}
	if err := checkDims(out.Embeddings, ActiveDim); err != nil {
		return nil, err
	}
	return out.Embeddings, nil
}

func (o *OllamaEmbedder) EmbedSingle(text string) ([]float32, error) {
	vecs, err := o.Embed([]string{text})
	if err != nil {
		return nil, err
	}
	if len(vecs) == 0 {
		return nil, fmt.Errorf("ollama: empty result")
	}
	return vecs[0], nil
}
