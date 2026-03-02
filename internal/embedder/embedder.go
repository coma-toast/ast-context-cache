package embedder

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"github.com/daulet/tokenizers"
	ort "github.com/yalue/onnxruntime_go"
)

const (
	ModelName  = "all-mpnet-base-v2"
	Dimensions = 768
	maxSeqLen  = 128
)

type Embedder struct {
	mu        sync.Mutex
	tokenizer *tokenizers.Tokenizer
	session   *ort.AdvancedSession
	inputIDs  *ort.Tensor[int64]
	attnMask  *ort.Tensor[int64]
	output    *ort.Tensor[float32]
}

type embedRequest struct {
	Texts []string `json:"texts"`
}

type embedResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
}

type healthResponse struct {
	Status     string `json:"status"`
	Model      string `json:"model"`
	Device     string `json:"device"`
	Dimensions int    `json:"dimensions"`
	Runtime    string `json:"runtime"`
	Ready      bool   `json:"ready"`
}

func New(modelDir string) (*Embedder, error) {
	ortLib := os.Getenv("ONNXRUNTIME_LIB")
	if ortLib == "" {
		ortLib = "/opt/homebrew/lib/libonnxruntime.dylib"
		if runtime.GOOS == "linux" {
			ortLib = "/usr/lib/libonnxruntime.so"
		}
	}

	ort.SetSharedLibraryPath(ortLib)
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("init ONNX Runtime: %w", err)
	}

	tokPath := filepath.Join(modelDir, "tokenizer.json")
	tk, err := tokenizers.FromFile(tokPath)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer from %s: %w", tokPath, err)
	}

	modelPath := filepath.Join(modelDir, "model.onnx")

	inputIDs, err := ort.NewEmptyTensor[int64](ort.NewShape(1, maxSeqLen))
	if err != nil {
		return nil, fmt.Errorf("create input_ids tensor: %w", err)
	}
	attnMask, err := ort.NewEmptyTensor[int64](ort.NewShape(1, maxSeqLen))
	if err != nil {
		return nil, fmt.Errorf("create attention_mask tensor: %w", err)
	}
	output, err := ort.NewEmptyTensor[float32](ort.NewShape(1, Dimensions))
	if err != nil {
		return nil, fmt.Errorf("create output tensor: %w", err)
	}

	session, err := ort.NewAdvancedSession(modelPath,
		[]string{"input_ids", "attention_mask"},
		[]string{"sentence_embedding"},
		[]ort.ArbitraryTensor{inputIDs, attnMask},
		[]ort.ArbitraryTensor{output},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("create ONNX session: %w", err)
	}

	return &Embedder{
		tokenizer: tk,
		session:   session,
		inputIDs:  inputIDs,
		attnMask:  attnMask,
		output:    output,
	}, nil
}

func (e *Embedder) Close() {
	e.session.Destroy()
	e.inputIDs.Destroy()
	e.attnMask.Destroy()
	e.output.Destroy()
	e.tokenizer.Close()
	ort.DestroyEnvironment()
}

func (e *Embedder) Embed(texts []string) ([][]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	results := make([][]float32, len(texts))
	for i, text := range texts {
		emb, err := e.embedSingle(text)
		if err != nil {
			return nil, fmt.Errorf("embed text %d: %w", i, err)
		}
		results[i] = emb
	}
	return results, nil
}

func (e *Embedder) EmbedSingle(text string) ([]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.embedSingle(text)
}

func (e *Embedder) embedSingle(text string) ([]float32, error) {
	enc := e.tokenizer.EncodeWithOptions(text, true,
		tokenizers.WithReturnAttentionMask(),
	)
	ids := enc.IDs
	mask := enc.AttentionMask

	idsBuf := e.inputIDs.GetData()
	maskBuf := e.attnMask.GetData()
	for i := range idsBuf {
		idsBuf[i] = 0
	}
	for i := range maskBuf {
		maskBuf[i] = 0
	}

	n := len(ids)
	if n > maxSeqLen {
		n = maxSeqLen
	}
	for i := 0; i < n; i++ {
		idsBuf[i] = int64(ids[i])
		maskBuf[i] = int64(mask[i])
	}

	if err := e.session.Run(); err != nil {
		return nil, fmt.Errorf("ONNX inference: %w", err)
	}

	raw := e.output.GetData()
	result := make([]float32, Dimensions)
	copy(result, raw)

	normalize(result)
	return result, nil
}

func normalize(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	norm := float32(math.Sqrt(sum))
	if norm > 0 {
		for i := range v {
			v[i] /= norm
		}
	}
}

// HandleEmbed provides backward-compatible /embed HTTP endpoint
func (e *Embedder) HandleEmbed(w http.ResponseWriter, r *http.Request) {
	var req embedRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":"invalid JSON"}`, http.StatusBadRequest)
		return
	}
	if len(req.Texts) == 0 {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(embedResponse{Embeddings: [][]float32{}})
		return
	}

	start := time.Now()
	embeddings, err := e.Embed(req.Texts)
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"%s"}`, err.Error()), http.StatusInternalServerError)
		return
	}
	elapsed := time.Since(start)
	log.Printf("Embedded %d texts in %v (%.1fms/text)", len(req.Texts), elapsed, float64(elapsed.Milliseconds())/float64(len(req.Texts)))

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(embedResponse{Embeddings: embeddings})
}

func (e *Embedder) HandleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(healthResponse{
		Status:     "ok",
		Model:      ModelName,
		Device:     "cpu",
		Dimensions: Dimensions,
		Runtime:    "onnxruntime",
		Ready:      true,
	})
}
