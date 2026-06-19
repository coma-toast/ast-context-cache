package embedder

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"
)

const openAIChunk = 32

// OpenAIEmbedder calls an OpenAI-compatible POST {base}/embeddings (LiteLLM, OpenAI, etc.).
// Vectors must match ActiveDim (768) after normalization; see checkDims.
type OpenAIEmbedder struct {
	embeddingsURL  string
	apiKey         string
	model          string
	jsonDimensions int // if > 0, include "dimensions" in request JSON; 0 = omit
	errLabel       string
	client         *http.Client
}

// NewOpenAIEmbedder builds a remote embedder. baseURL is the API root including /v1, e.g.
// https://litellm.example.com/v1 (trailing slashes trimmed). jsonDimensions: 0 omits the
// dimensions field; >0 sends it (e.g. 768 for text-embedding-3-small).
func NewOpenAIEmbedder(baseURL, apiKey, model string, jsonDimensions int) *OpenAIEmbedder {
	return newOpenAIEmbedder(baseURL, apiKey, model, jsonDimensions, "openai embed", ResolveRemoteTimeout())
}

// NewOpenAIEmbedderWithLabel is like NewOpenAIEmbedder but uses label in error messages (e.g. "dmr embed" for Docker Model Runner).
func NewOpenAIEmbedderWithLabel(baseURL, apiKey, model string, jsonDimensions int, label string) *OpenAIEmbedder {
	return newOpenAIEmbedder(baseURL, apiKey, model, jsonDimensions, label, ResolveRemoteTimeout())
}

func newOpenAIEmbedder(baseURL, apiKey, model string, jsonDimensions int, label string, timeout time.Duration) *OpenAIEmbedder {
	b := strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if b == "" {
		b = "https://api.openai.com/v1"
	}
	label = strings.TrimSpace(label)
	if label == "" {
		label = "openai embed"
	}
	if timeout <= 0 {
		timeout = DefaultHTTPEmbedTimeout
	}
	return &OpenAIEmbedder{
		embeddingsURL:  b + "/embeddings",
		apiKey:         strings.TrimSpace(apiKey),
		model:          strings.TrimSpace(model),
		jsonDimensions: jsonDimensions,
		errLabel:       label,
		client:         &http.Client{Timeout: timeout},
	}
}

type openAIEmbedAPIRequest struct {
	Model       string      `json:"model"`
	Input       []string    `json:"input"`
	Dimensions  *int        `json:"dimensions,omitempty"`
}

type openAIEmbedAPIResponse struct {
	Data []openAIEmbedDatum `json:"data"`
}

type openAIEmbedDatum struct {
	Embedding []float64 `json:"embedding"`
	Index     int       `json:"index"`
}

func (o *OpenAIEmbedder) Embed(texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	var out [][]float32
	for start := 0; start < len(texts); start += openAIChunk {
		end := start + openAIChunk
		if end > len(texts) {
			end = len(texts)
		}
		chunk := texts[start:end]
		vecs, err := o.embedBatch(chunk)
		if err != nil {
			return nil, err
		}
		out = append(out, vecs...)
	}
	return out, nil
}

func (o *OpenAIEmbedder) embedBatch(texts []string) ([][]float32, error) {
	var dimPtr *int
	if o.jsonDimensions > 0 {
		d := o.jsonDimensions
		dimPtr = &d
	}
	body, err := json.Marshal(openAIEmbedAPIRequest{Model: o.model, Input: texts, Dimensions: dimPtr})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequest(http.MethodPost, o.embeddingsURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if o.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+o.apiKey)
	}
	resp, err := o.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", o.errLabel, err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("%s %s: %s", o.errLabel, resp.Status, truncateForErr(raw, 200))
	}
	var apiOut openAIEmbedAPIResponse
	if err := json.Unmarshal(raw, &apiOut); err != nil {
		return nil, fmt.Errorf("%s response json: %w", o.errLabel, err)
	}
	if len(apiOut.Data) != len(texts) {
		return nil, fmt.Errorf("%s: got %d data rows for %d inputs", o.errLabel, len(apiOut.Data), len(texts))
	}
	sort.Slice(apiOut.Data, func(i, j int) bool { return apiOut.Data[i].Index < apiOut.Data[j].Index })
	vecs := make([][]float32, len(apiOut.Data))
	for i, d := range apiOut.Data {
		v := make([]float32, len(d.Embedding))
		for j, x := range d.Embedding {
			v[j] = float32(x)
		}
		vecs[i] = v
	}
	if err := checkDims(vecs, ActiveDim); err != nil {
		return nil, err
	}
	return vecs, nil
}

func (o *OpenAIEmbedder) EmbedSingle(text string) ([]float32, error) {
	vecs, err := o.Embed([]string{text})
	if err != nil {
		return nil, err
	}
	if len(vecs) == 0 {
		return nil, fmt.Errorf("%s: empty result", o.errLabel)
	}
	return vecs[0], nil
}
