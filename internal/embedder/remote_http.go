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

// HTTPEmbedder calls a service that matches ast-mcp’s POST /embed JSON: request {"texts":[...]} response {"embeddings":[[...float32]...]}.
// Use with a local sidecar, a proxy to OpenAI, Text Embeddings Inference, etc. Vectors must be 768 dimensions to match the vector index.
type HTTPEmbedder struct {
	URL    string
	Bearer string
	client *http.Client
}

// NewHTTPEmbedder creates a remote embedder. url should be a full URL (e.g. http://127.0.0.1:9000/embed).
func NewHTTPEmbedder(url, bearer string) *HTTPEmbedder {
	return newHTTPEmbedder(url, bearer, ResolveRemoteTimeout())
}

func newHTTPEmbedder(url, bearer string, timeout time.Duration) *HTTPEmbedder {
	u := strings.TrimSpace(url)
	if u == "" {
		u = "http://127.0.0.1:8080/embed"
	}
	if timeout <= 0 {
		timeout = DefaultHTTPEmbedTimeout
	}
	return &HTTPEmbedder{
		URL:    u,
		Bearer: strings.TrimSpace(bearer),
		client: &http.Client{Timeout: timeout},
	}
}

type httpEmbedReq struct {
	Texts []string `json:"texts"`
}

type httpEmbedResp struct {
	Embeddings [][]float32 `json:"embeddings"`
}

func (h *HTTPEmbedder) Embed(texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	body, err := json.Marshal(httpEmbedReq{Texts: texts})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequest(http.MethodPost, h.URL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if h.Bearer != "" {
		req.Header.Set("Authorization", "Bearer "+h.Bearer)
	}
	resp, err := h.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embed http: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("embed http %s: %s", resp.Status, truncateForErr(raw, 200))
	}
	var out httpEmbedResp
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("embed response json: %w", err)
	}
	if len(out.Embeddings) != len(texts) {
		return nil, fmt.Errorf("embed: got %d vectors for %d inputs", len(out.Embeddings), len(texts))
	}
	if err := checkDims(out.Embeddings, ActiveDim); err != nil {
		return nil, err
	}
	return out.Embeddings, nil
}

func (h *HTTPEmbedder) EmbedSingle(text string) ([]float32, error) {
	vecs, err := h.Embed([]string{text})
	if err != nil {
		return nil, err
	}
	if len(vecs) == 0 {
		return nil, fmt.Errorf("embed: empty result")
	}
	return vecs[0], nil
}

func truncateForErr(b []byte, n int) string {
	s := string(b)
	if len(s) > n {
		return s[:n] + "…"
	}
	return s
}

func checkDims(emb [][]float32, want int) error {
	for i, v := range emb {
		if len(v) != want {
			return fmt.Errorf("embedding[%d] has %d dimensions (expected %d for this index; re-embed or match backend to %d-dim model)",
				i, len(v), want, want)
		}
		NormalizeL2(emb[i])
	}
	return nil
}
